"""
Employee Attendance & Analytics API  (FastAPI + MongoDB)

Run:  uvicorn app.main:app --port 8000
Env:  MONGO_URI, MONGO_DB   (a local .env is loaded for convenience; real environment variables win)

Everything lives in this one file, in this order:
  1. configuration / database / indexes
  2. time and rule helpers (R1 - R5, R8)
  3. request models and validation helpers
  4. employees, attendance (punch-in / punch-out / regularize / list)
  5. analytics (aggregation pipelines), then the explain endpoint
"""
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import date, datetime, time, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple

from bson import Decimal128, json_util
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, StrictInt, model_validator, field_validator
from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError, OperationFailure, PyMongoError

load_dotenv()  # override=False: real environment variables take priority over .env

log = logging.getLogger("attendance")

# --------------------------------------------------------------------------- #
# 1. Configuration, database, indexes
# --------------------------------------------------------------------------- #
# tz_aware=True: PyMongo hands back aware UTC datetimes instead of naive ones.
client = MongoClient(
    os.getenv("MONGO_URI", "mongodb://localhost:27017"),
    tz_aware=True,
    serverSelectionTimeoutMS=5000,
)
db = client[os.getenv("MONGO_DB", "attendance_db")]

IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

PRESENCE_STATUSES = ["PRESENT", "WFH", "ON_DUTY"]
ALL_STATUSES = ["PRESENT", "WFH", "ON_DUTY", "ABSENT", "LEAVE"]
Status = Literal["PRESENT", "WFH", "ON_DUTY", "ABSENT", "LEAVE"]
PresenceStatus = Literal["PRESENT", "WFH", "ON_DUTY"]

GRACE_SECONDS = 10 * 60
MIN_OVERTIME_MINUTES = 30
HALF_DAY_HOURS = 4.5
MAX_SHIFT_SECONDS = 24 * 3600
MS_MIN, MS_MAX = 100_000_000_000, 4_102_444_800_000
MONTH_RE = r"^[0-9]{4}-(0[1-9]|1[0-2])$"
DATE_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
HHMM_RE = r"^([01][0-9]|2[0-3]):[0-5][0-9]$"

_indexes_ready = False


def ensure_indexes() -> None:
    """Create every index we rely on. Idempotent: create_index is a no-op if it already exists."""
    global _indexes_ready
    logs, emps = db.attendance_logs, db.employees

    # One record per (emp_code, date): the DB enforces it, so concurrent punch-ins cannot both win.
    # Also serves: emp_code (+ date range) filters, employee monthly analytics, trend log scan.
    try:
        logs.create_index([("emp_code", ASCENDING), ("date", ASCENDING)], unique=True, name="uq_emp_date")
    except OperationFailure:
        # Seeded data that already violates uniqueness must not stop the service from starting.
        log.exception("could not build UNIQUE (emp_code, date); falling back to a plain index")
        logs.create_index([("emp_code", ASCENDING), ("date", ASCENDING)], name="emp_date")
    # GET /attendance default sort (date desc, emp_code asc) + date-range filters + leaderboard month scan.
    logs.create_index([("date", DESCENDING), ("emp_code", ASCENDING)], name="date_emp")
    # GET /attendance?status=...  (equality on status, then the same sort).
    logs.create_index([("status", ASCENDING), ("date", DESCENDING), ("emp_code", ASCENDING)], name="status_date_emp")
    # Punch-out: "latest record of this employee with punch_in <= t".
    logs.create_index([("emp_code", ASCENDING), ("punch_in", DESCENDING)], name="emp_punch_in")

    try:
        emps.create_index([("emp_code", ASCENDING)], unique=True, name="uq_emp_code")
    except OperationFailure:
        log.exception("could not build UNIQUE emp_code; falling back to a plain index")
        emps.create_index([("emp_code", ASCENDING)], name="emp_code")
    emps.create_index([("department", ASCENDING), ("emp_code", ASCENDING)], name="dept_emp_code")
    emps.create_index([("department", ASCENDING), ("joined_on", ASCENDING)], name="dept_joined_on")
    emps.create_index([("joined_on", ASCENDING)], name="joined_on")
    _indexes_ready = True


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        ensure_indexes()
    except PyMongoError:
        # Mongo not reachable yet: /health answers 503 and retries index creation once it is.
        log.exception("index creation at startup failed")
    yield


app = FastAPI(title="Employee Attendance & Analytics API", version="2.0.0", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# 2. Time and rule helpers
# --------------------------------------------------------------------------- #
def now_utc() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def ms_to_dt(ms: int) -> datetime:
    """Epoch milliseconds -> aware UTC datetime, truncated to whole seconds (R1)."""
    return (EPOCH + timedelta(milliseconds=ms)).replace(microsecond=0)


def dt_to_ms(dt: Optional[datetime]) -> Optional[int]:
    """BSON datetime (naive = UTC, or aware) -> epoch milliseconds."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return (dt - EPOCH) // timedelta(milliseconds=1)


def as_aware(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def round_half_up(value, places: int) -> float:
    """R8: half-up rounding (Python's round() and MongoDB's $round are banker's rounding)."""
    if isinstance(value, Decimal128):
        d = value.to_decimal()
    else:
        d = Decimal(repr(value)) if isinstance(value, float) else Decimal(value)
    return float(d.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def _hhmm(s: str) -> Tuple[int, int]:
    h, m = s.split(":")
    return int(h), int(m)


def is_overnight(emp: dict) -> bool:
    return emp["shift_end"] <= emp["shift_start"]  # zero-padded HH:MM compares correctly as text


def ist_at(day: date, hhmm: str) -> datetime:
    h, m = _hhmm(hhmm)
    return datetime.combine(day, time(h, m), tzinfo=IST)


def attendance_date(punch_in: datetime, emp: dict) -> str:
    """R1: IST calendar date of the punch-in; an overnight shift's early-morning punch-in belongs to the day before."""
    ist = punch_in.astimezone(IST)
    day = ist.date()
    if is_overnight(emp):
        h, m = _hhmm(emp["shift_end"])
        if ist.hour * 3600 + ist.minute * 60 + ist.second < h * 3600 + m * 60:
            day -= timedelta(days=1)
    return day.isoformat()


def compute_late_minutes(punch_in: datetime, shift_start: str, date_str: str) -> int:
    """R2: late only when strictly more than 10:00 after shift_start; then whole minutes since shift_start."""
    start = ist_at(date.fromisoformat(date_str), shift_start)
    seconds = int((punch_in - start).total_seconds())
    return seconds // 60 if seconds > GRACE_SECONDS else 0


def compute_work_hours(punch_in: datetime, punch_out: datetime) -> float:
    """R4: seconds / 3600, 2 decimals half-up (exact Decimal arithmetic, no float rounding)."""
    seconds = int((punch_out - punch_in).total_seconds())
    return float((Decimal(seconds) / Decimal(3600)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def compute_overtime(punch_out: datetime, emp: dict, date_str: str) -> int:
    """R3: whole minutes after shift_end, only when >= 30. An overnight shift ends on the next calendar day."""
    end_day = date.fromisoformat(date_str) + (timedelta(days=1) if is_overnight(emp) else timedelta(0))
    seconds = int((punch_out - ist_at(end_day, emp["shift_end"])).total_seconds())
    minutes = seconds // 60
    return minutes if minutes >= MIN_OVERTIME_MINUTES else 0


def derive(emp: dict, date_str: str, status: str, punch_in: Optional[datetime], punch_out: Optional[datetime]) -> dict:
    """All four derived fields from the final state of a record (DATA_MODEL.md table)."""
    if status not in PRESENCE_STATUSES or punch_in is None:
        return {"work_hours": None, "late_minutes": 0, "overtime_minutes": 0, "half_day": False}
    late = compute_late_minutes(punch_in, emp["shift_start"], date_str)
    if punch_out is None:
        return {"work_hours": None, "late_minutes": late, "overtime_minutes": 0, "half_day": False}
    wh = compute_work_hours(punch_in, punch_out)
    return {
        "work_hours": wh,
        "late_minutes": late,
        "overtime_minutes": compute_overtime(punch_out, emp, date_str),
        "half_day": wh < HALF_DAY_HOURS,
    }


# --------------------------------------------------------------------------- #
# 3. Models, validation helpers, serialisation
# --------------------------------------------------------------------------- #
EpochMs = Annotated[StrictInt, Field(ge=MS_MIN, le=MS_MAX)]  # StrictInt: floats / numeric strings -> 422


def parse_date(value: str, field: str = "date") -> date:
    """Strict YYYY-MM-DD (rejects 20260706, 2026-7-6, 2026-02-30 ...)."""
    if not DATE_RE.match(value):
        raise _bad(field, "must be YYYY-MM-DD")
    try:
        d = date.fromisoformat(value)
    except ValueError:
        raise _bad(field, "not a real calendar date")
    if not 1900 <= d.year <= 2200:
        raise _bad(field, "year out of supported range")
    return d


def _bad(field: str, msg: str, where: str = "query") -> RequestValidationError:
    return RequestValidationError([{"loc": [where, field], "msg": msg, "type": "value_error"}])


def _missing(field: str) -> RequestValidationError:
    return RequestValidationError([{"loc": ["query", field], "msg": "Field required", "type": "missing"}])


def month_bounds(month: str) -> Tuple[date, date]:
    if not re.match(MONTH_RE, month):
        raise _bad("month", "must be YYYY-MM")
    year, mon = int(month[:4]), int(month[5:])
    if not 1900 <= year <= 2200:
        raise _bad("month", "year out of supported range")
    first = date(year, mon, 1)
    nxt = date(year + (mon == 12), mon % 12 + 1, 1)
    return first, nxt - timedelta(days=1)


def weekdays_between(start: date, end: date) -> int:
    """R7: Monday-Friday days in [start, end] (no holiday calendar)."""
    n, d = 0, start
    while d <= end:
        n += d.weekday() < 5
        d += timedelta(days=1)
    return n


class EmployeeIn(BaseModel):
    emp_code: str = Field(pattern=r"^EMP[0-9]{4,6}$")
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(min_length=1, max_length=50)
    shift_start: str = Field(default="09:30", pattern=HHMM_RE)
    shift_end: str = Field(default="18:30", pattern=HHMM_RE)
    joined_on: str

    @field_validator("joined_on")
    @classmethod
    def _joined_on(cls, v: str) -> str:
        if not DATE_RE.match(v):
            raise ValueError("must be YYYY-MM-DD")
        date.fromisoformat(v)  # ValueError -> 422
        return v

    @model_validator(mode="after")
    def _shift(self):
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start must differ from shift_end")
        return self


class PunchInIn(BaseModel):
    emp_code: str
    punched_at: Optional[EpochMs] = None
    status: PresenceStatus = "PRESENT"


class PunchOutIn(BaseModel):
    emp_code: str
    punched_at: Optional[EpochMs] = None


class RegularizeIn(BaseModel):
    status: Optional[Status] = None
    punch_in: Optional[EpochMs] = None
    punch_out: Optional[EpochMs] = None
    reason: str = Field(min_length=5, max_length=200)
    regularized_by: str = Field(min_length=1, max_length=50)

    @model_validator(mode="before")
    @classmethod
    def _no_explicit_null(cls, data):
        # Omitted means "unchanged"; an explicit null is not a valid value in the contract.
        if isinstance(data, dict):
            for k in ("status", "punch_in", "punch_out"):
                if k in data and data[k] is None:
                    raise ValueError(f"{k} must not be null")
        return data


def employee_out(doc: dict) -> dict:
    return {
        "emp_code": doc["emp_code"],
        "name": doc["name"],
        "email": doc["email"],
        "department": doc["department"],
        "shift_start": doc["shift_start"],
        "shift_end": doc["shift_end"],
        "joined_on": doc["joined_on"],
        "created_at": dt_to_ms(doc.get("created_at")),
    }


def _history_value(key: str, value):
    return dt_to_ms(value) if key in ("punch_in", "punch_out") else value


def record_out(doc: dict) -> dict:
    """Stored attendance document -> API shape. Never leaks _id; fills the documented gaps in legacy data."""
    history = []
    for h in doc.get("history") or []:
        history.append({
            "at": dt_to_ms(h.get("at")),
            "by": h.get("by"),
            "reason": h.get("reason"),
            "changes": {
                k: {"from": _history_value(k, v.get("from")), "to": _history_value(k, v.get("to"))}
                for k, v in (h.get("changes") or {}).items()
            },
        })
    return {
        "emp_code": doc["emp_code"],
        "date": doc["date"],
        "status": doc["status"],
        "punch_in": dt_to_ms(doc.get("punch_in")),
        "punch_out": dt_to_ms(doc.get("punch_out")),
        "work_hours": doc.get("work_hours"),
        "late_minutes": doc.get("late_minutes") or 0,
        "overtime_minutes": doc.get("overtime_minutes") or 0,
        "half_day": bool(doc.get("half_day", False)),
        "history": history,
    }


def get_employee(emp_code: str) -> dict:
    emp = db.employees.find_one({"emp_code": emp_code})
    if emp is None:
        raise HTTPException(404, "employee not found")
    return emp


def paginate_args(page: int, page_size: int) -> Tuple[int, int]:
    skip = (page - 1) * page_size
    return skip, page_size


# --------------------------------------------------------------------------- #
# 4. Endpoints: system, employees, attendance
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    try:
        client.admin.command("ping")
    except PyMongoError:
        raise HTTPException(503, "database unavailable")
    if not _indexes_ready:
        try:
            ensure_indexes()
        except PyMongoError:
            log.exception("index creation failed")
    return {"status": "ok"}


@app.post("/employees", status_code=201)
def create_employee(body: EmployeeIn):
    doc = body.model_dump()
    doc["created_at"] = now_utc()
    try:
        db.employees.insert_one(doc)  # the unique index decides; no check-then-insert
    except DuplicateKeyError:
        raise HTTPException(409, "emp_code already exists")
    return employee_out(doc)


@app.get("/employees")
def list_employees(
    department: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    q: Dict[str, Any] = {}
    if department:
        q["department"] = department
    skip, limit = paginate_args(page, page_size)
    total = db.employees.count_documents(q)
    items = []
    if skip < 2**31:
        items = [employee_out(d) for d in db.employees.find(q).sort("emp_code", ASCENDING).skip(skip).limit(limit)]
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInIn):
    emp = get_employee(body.emp_code)
    ts = ms_to_dt(body.punched_at) if body.punched_at is not None else now_utc()
    d = attendance_date(ts, emp)
    doc = {
        "emp_code": body.emp_code,
        "date": d,
        "status": body.status,
        "punch_in": ts,
        "punch_out": None,
        "work_hours": None,
        "late_minutes": compute_late_minutes(ts, emp["shift_start"], d),
        "overtime_minutes": 0,
        "half_day": False,
        "history": [],
    }
    try:
        db.attendance_logs.insert_one(doc)  # unique (emp_code, date): exactly one concurrent request wins
    except DuplicateKeyError:
        raise HTTPException(409, "already punched in for this date")
    return record_out(doc)


@app.post("/attendance/punch-out")
def punch_out(body: PunchOutIn):
    emp = get_employee(body.emp_code)
    ts = ms_to_dt(body.punched_at) if body.punched_at is not None else now_utc()
    # Most recent record that started at or before the punch-out; works across midnight because it
    # looks at punch_in instants, not at the calendar date. ABSENT/LEAVE (punch_in null) never match $lte.
    rec = db.attendance_logs.find_one(
        {"emp_code": body.emp_code, "punch_in": {"$lte": ts}},
        sort=[("punch_in", DESCENDING)],
    )
    if rec is None:
        raise HTTPException(404, "no punch-in found for this employee")
    if rec.get("punch_out") is not None:
        raise HTTPException(409, "already punched out")
    pin = as_aware(rec["punch_in"])
    if ts <= pin:
        raise HTTPException(422, "punched_at must be after punch_in")
    if (ts - pin).total_seconds() > MAX_SHIFT_SECONDS:
        raise HTTPException(422, "punched_at is more than 24 hours after punch_in")
    fields = derive(emp, rec["date"], rec["status"], pin, ts)
    fields["punch_out"] = ts
    # Compare-and-set on punch_out: of N simultaneous punch-outs exactly one matches.
    updated = db.attendance_logs.find_one_and_update(
        {"_id": rec["_id"], "punch_out": None},
        {"$set": fields},
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise HTTPException(409, "already punched out")
    return record_out(updated)


@app.patch("/attendance/{emp_code}/{date}")
def regularize(emp_code: str, date: str, body: RegularizeIn):
    parse_date(date, "date")
    emp = get_employee(emp_code)
    for _ in range(5):  # optimistic concurrency: retry if somebody else wrote in between
        rec = db.attendance_logs.find_one({"emp_code": emp_code, "date": date})
        if rec is None:
            raise HTTPException(404, "no attendance record for that date")

        old_status = rec["status"]
        old_in, old_out = as_aware(rec.get("punch_in")), as_aware(rec.get("punch_out"))
        new_status = body.status or old_status
        new_in = ms_to_dt(body.punch_in) if body.punch_in is not None else None
        new_out = ms_to_dt(body.punch_out) if body.punch_out is not None else None

        if new_status in ("ABSENT", "LEAVE"):
            if new_in is not None or new_out is not None:
                raise HTTPException(422, f"punch_in / punch_out cannot be set with status {new_status}")
            final_in = final_out = None
        else:
            final_in = new_in or old_in
            final_out = new_out or old_out
            if final_in is None:
                raise HTTPException(422, "a presence status requires a punch_in")
            if attendance_date(final_in, emp) != date:
                raise HTTPException(422, "punch_in must stay on the record's attendance date")
            if final_out is not None:
                if final_out <= final_in:
                    raise HTTPException(422, "punch_out must be after punch_in")
                if (final_out - final_in).total_seconds() > MAX_SHIFT_SECONDS:
                    raise HTTPException(422, "punch_out must be within 24 hours of punch_in")

        # "Changes nothing" is judged on what the client can set, not on derived fields.
        if (new_status, final_in, final_out) == (old_status, old_in, old_out):
            raise HTTPException(422, "request changes nothing")

        new_vals = {"status": new_status, "punch_in": final_in, "punch_out": final_out}
        new_vals.update(derive(emp, date, new_status, final_in, final_out))
        old_vals = {
            "status": old_status,
            "punch_in": old_in,
            "punch_out": old_out,
            "work_hours": rec.get("work_hours"),
            "late_minutes": rec.get("late_minutes") or 0,
            "overtime_minutes": rec.get("overtime_minutes") or 0,
            "half_day": bool(rec.get("half_day", False)),
        }
        changes = {k: {"from": old_vals[k], "to": v} for k, v in new_vals.items() if old_vals[k] != v}
        entry = {"at": now_utc(), "by": body.regularized_by, "reason": body.reason, "changes": changes}

        n = len(rec.get("history") or [])
        guard: Dict[str, Any] = {
            "_id": rec["_id"],
            "status": old_status,
            "punch_in": rec.get("punch_in"),
            "punch_out": rec.get("punch_out"),
            f"history.{n}": {"$exists": False},  # history still has exactly n entries
        }
        if n:
            guard[f"history.{n - 1}"] = {"$exists": True}
        updated = db.attendance_logs.find_one_and_update(
            guard, {"$set": new_vals, "$push": {"history": entry}}, return_document=ReturnDocument.AFTER
        )
        if updated is not None:
            return record_out(updated)
    raise HTTPException(409, "concurrent modification, please retry")


def build_attendance_query(emp_code, date_from, date_to, status, page, page_size):
    """Shared by GET /attendance and its explain, so the explained query is the real one."""
    if date_from:
        parse_date(date_from, "date_from")
    if date_to:
        parse_date(date_to, "date_to")
    if date_from and date_to and date_from > date_to:
        raise _bad("date_from", "date_from must not be after date_to")
    q: Dict[str, Any] = {}
    if emp_code:
        q["emp_code"] = emp_code
    if date_from or date_to:
        q["date"] = {}
        if date_from:
            q["date"]["$gte"] = date_from
        if date_to:
            q["date"]["$lte"] = date_to
    if status:
        q["status"] = status
    sort = [("date", DESCENDING), ("emp_code", ASCENDING)]
    skip, limit = paginate_args(page, page_size)
    return q, sort, skip, limit


@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[Status] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    q, sort, skip, limit = build_attendance_query(emp_code, date_from, date_to, status, page, page_size)
    total = db.attendance_logs.count_documents(q)
    items = []
    if skip < 2**31:
        items = [record_out(d) for d in db.attendance_logs.find(q).sort(sort).skip(skip).limit(limit)]
    return {"items": items, "total": total, "page": page, "page_size": page_size}


# --------------------------------------------------------------------------- #
# 5. Analytics: aggregation pipelines
# --------------------------------------------------------------------------- #
D0 = Decimal128("0")
D_HALF = Decimal128("0.5")
D_1E4 = Decimal128("10000")


def _is_null(expr) -> dict:
    return {"$in": [{"$type": expr}, ["null", "missing"]]}


def _halfup4(expr) -> dict:
    """R8 inside the database: floor(x * 10^4 + 0.5) / 10^4 in Decimal128 (MongoDB's $round is half-to-even)."""
    return {"$divide": [{"$floor": {"$add": [{"$multiply": [expr, D_1E4]}, D_HALF]}}, D_1E4]}


def _record_flags_stage() -> dict:
    """Per-record helper fields shared by the monthly and department pipelines. Stored values are read, never recomputed."""
    return {"$set": {
        "_p": {"$in": ["$status", PRESENCE_STATUSES]},                                   # R6
        "_wd": {"$lte": [{"$isoDayOfWeek": {"$dateFromString": {"dateString": "$date", "format": "%Y-%m-%d"}}}, 5]},  # Mon-Fri
        "_late": {"$ifNull": ["$late_minutes", 0]},
        "_ot": {"$ifNull": ["$overtime_minutes", 0]},
        "_half": {"$ifNull": ["$half_day", False]},
        "_has_wh": {"$in": [{"$type": "$work_hours"}, ["double", "int", "long", "decimal"]]},
    }}


def _stats_group(group_id) -> dict:
    return {"$group": {
        "_id": group_id,
        # R5 + R6 + R7: presence, Mon-Fri only, half day = 0.5
        "present_days": {"$sum": {"$cond": [{"$and": ["$_p", "$_wd"]}, {"$cond": ["$_half", 0.5, 1]}, 0]}},
        "leave_count": {"$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}},
        "on_duty_count": {"$sum": {"$cond": [{"$eq": ["$status", "ON_DUTY"]}, 1, 0]}},
        "late_count": {"$sum": {"$cond": [{"$gt": ["$_late", 0]}, 1, 0]}},
        "total_late_minutes": {"$sum": {"$cond": [{"$gt": ["$_late", 0]}, "$_late", 0]}},
        "total_overtime_minutes": {"$sum": "$_ot"},
        # Decimal128 sums keep the average exact; the division happens once, in Python, on the totals.
        "wh_sum": {"$sum": {"$cond": [{"$and": ["$_p", "$_has_wh"]}, {"$toDecimal": "$work_hours"}, D0]}},
        "wh_count": {"$sum": {"$cond": [{"$and": ["$_p", "$_has_wh"]}, 1, 0]}},
    }}


def employee_monthly_pipeline(emp_code: str, first: str, last: str) -> list:
    return [
        {"$match": {"emp_code": emp_code, "date": {"$gte": first, "$lte": last}}},  # IXSCAN uq_emp_date
        _record_flags_stage(),
        _stats_group(None),
    ]


def department_summary_pipeline(first: str, last: str, department: Optional[str]) -> Tuple[list, dict]:
    """Runs on employees (so people with zero logs are still counted) and left-joins their month of logs."""
    match: Dict[str, Any] = {"joined_on": {"$lte": last}}  # R9
    hint: Dict[str, int] = {"joined_on": 1}
    if department:
        match["department"] = department
        hint = {"department": 1, "joined_on": 1}
    pipeline = [
        {"$match": match},
        {"$lookup": {
            "from": "attendance_logs",
            "localField": "emp_code",
            "foreignField": "emp_code",  # uses uq_emp_date for the join
            "pipeline": [
                {"$match": {"date": {"$gte": first, "$lte": last}}},
                _record_flags_stage(),
                _stats_group(None),
            ],
            "as": "s",
        }},
        # no logs -> empty array -> {} -> every counter falls back to 0
        {"$set": {"s": {"$ifNull": [{"$first": "$s"}, {}]}}},
        {"$group": {
            "_id": "$department",
            "headcount": {"$sum": 1},
            "present_days": {"$sum": {"$ifNull": ["$s.present_days", 0]}},
            "late_count": {"$sum": {"$ifNull": ["$s.late_count", 0]}},
            "total_late_minutes": {"$sum": {"$ifNull": ["$s.total_late_minutes", 0]}},
            "leave_count": {"$sum": {"$ifNull": ["$s.leave_count", 0]}},
            "on_duty_count": {"$sum": {"$ifNull": ["$s.on_duty_count", 0]}},
            "wh_sum": {"$sum": {"$ifNull": ["$s.wh_sum", D0]}},
            "wh_count": {"$sum": {"$ifNull": ["$s.wh_count", 0]}},
        }},
        {"$sort": {"_id": 1}},
    ]
    return pipeline, hint


def late_leaderboard_pipeline(first: str, last: str, limit: int, department: Optional[str]) -> list:
    pipeline: list = [
        {"$match": {"date": {"$gte": first, "$lte": last}}},  # IXSCAN date_emp
        {"$group": {
            "_id": "$emp_code",
            "total_late_minutes": {"$sum": {"$ifNull": ["$late_minutes", 0]}},
            "late_count": {"$sum": {"$cond": [{"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}, 1, 0]}},
        }},
        {"$match": {"total_late_minutes": {"$gt": 0}}},
        {"$lookup": {"from": "employees", "localField": "_id", "foreignField": "emp_code", "as": "emp"}},
        {"$unwind": "$emp"},  # logs whose employee does not exist are dropped here
    ]
    if department:
        pipeline.append({"$match": {"emp.department": department}})  # rank inside the department only
    pipeline += [
        # $rank = standard competition ranking (1, 2, 2, 4): ties share a rank, the next rank is skipped.
        {"$setWindowFields": {"sortBy": {"total_late_minutes": -1}, "output": {"rank": {"$rank": {}}}}},
        {"$match": {"rank": {"$lte": limit}}},  # limit applies AFTER ranking
        {"$sort": {"total_late_minutes": -1, "_id": 1}},
        {"$project": {
            "_id": 0, "rank": 1, "emp_code": "$_id", "name": "$emp.name", "department": "$emp.department",
            "total_late_minutes": 1, "late_count": 1,
        }},
    ]
    return pipeline


def department_trend_pipeline(department: str, codes: List[str], d_from: date, d_to: date) -> list:
    first, last = d_from.isoformat(), d_to.isoformat()
    lo = datetime(d_from.year, d_from.month, d_from.day, tzinfo=UTC)
    hi = datetime(d_to.year, d_to.month, d_to.day, tzinfo=UTC) + timedelta(days=1)  # upper bound is exclusive
    return [
        {"$match": {"emp_code": {"$in": codes}, "date": {"$gte": first, "$lte": last}}},  # IXSCAN uq_emp_date
        {"$project": {
            "_id": 0, "date": 1,
            "p": {"$cond": [{"$in": ["$status", PRESENCE_STATUSES]},
                            {"$cond": [{"$ifNull": ["$half_day", False]}, 0.5, 1]}, 0]},
            "l": {"$cond": [{"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}, 1, 0]},
        }},
        # One neutral row for day 1 so $densify always has input, even if the department has no logs in range.
        {"$unionWith": {"pipeline": [{"$documents": [{"date": first, "p": 0, "l": 0}]}]}},
        {"$group": {"_id": "$date", "present": {"$sum": "$p"}, "late": {"$sum": "$l"}}},
        {"$set": {"d": {"$dateFromString": {"dateString": "$_id", "format": "%Y-%m-%d"}}}},
        {"$sort": {"d": 1}},
        # Gap filling in the database: one row per calendar day in [from, to].
        {"$densify": {"field": "d", "range": {"step": 1, "unit": "day", "bounds": [lo, hi]}}},
        {"$set": {
            "date": {"$dateToString": {"date": "$d", "format": "%Y-%m-%d"}},
            "present": {"$ifNull": ["$present", 0]},
            "late": {"$ifNull": ["$late", 0]},
            "is_working_day": {"$lte": [{"$isoDayOfWeek": "$d"}, 5]},
        }},
        # R9: headcount on that day = department employees with joined_on <= day (index dept_joined_on).
        {"$lookup": {
            "from": "employees",
            "let": {"day": "$date"},
            "pipeline": [
                {"$match": {"department": department, "$expr": {"$lte": ["$joined_on", "$$day"]}}},
                {"$count": "n"},
            ],
            "as": "hc",
        }},
        {"$set": {"headcount": {"$ifNull": [{"$first": "$hc.n"}, 0]}}},
        {"$set": {"rate": {"$cond": [
            {"$and": ["$is_working_day", {"$gt": ["$headcount", 0]}]},
            _halfup4({"$divide": [{"$toDecimal": "$present"}, {"$toDecimal": "$headcount"}]}),
            None,
        ]}}},
        # Mean of the non-null rates in the current row + up to 6 preceding rows (shorter window at the start).
        {"$setWindowFields": {"sortBy": {"d": 1}, "output": {"ma": {"$avg": "$rate", "window": {"documents": [-6, 0]}}}}},
        {"$sort": {"d": 1}},
        {"$project": {
            "_id": 0,
            "date": 1,
            "is_working_day": 1,
            "headcount": 1,
            "present_count": {"$toDouble": "$present"},
            "late_count": "$late",
            "attendance_rate": {"$cond": [_is_null("$rate"), None, {"$toDouble": "$rate"}]},
            "moving_avg_7d": {"$cond": [_is_null("$ma"), None, {"$toDouble": _halfup4("$ma")}]},
        }},
    ]


def _num(value) -> float:
    if isinstance(value, Decimal128):
        return float(value.to_decimal())
    return float(value)


@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(emp_code: str, month: str = Query(..., pattern=MONTH_RE)):
    first, last = month_bounds(month)
    emp = get_employee(emp_code)
    rows = list(db.attendance_logs.aggregate(employee_monthly_pipeline(emp_code, first.isoformat(), last.isoformat())))
    s = rows[0] if rows else {}
    start = max(first, date.fromisoformat(emp["joined_on"]))  # R7: count from joined_on for mid-month joiners
    working_days = weekdays_between(start, last) if start <= last else 0
    present = float(s.get("present_days", 0))
    pct = None
    if working_days:
        pct = round_half_up(Decimal(repr(present)) / Decimal(working_days) * 100, 2)
    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working_days,
        "present_days": round_half_up(present, 2),
        "leave_days": int(s.get("leave_count", 0)),
        "late_count": int(s.get("late_count", 0)),
        "total_late_minutes": int(s.get("total_late_minutes", 0)),
        "total_overtime_minutes": int(s.get("total_overtime_minutes", 0)),
        "attendance_pct": pct,
    }


@app.get("/analytics/departments/summary")
def department_summary(month: str = Query(..., pattern=MONTH_RE), department: Optional[str] = None):
    first, last = month_bounds(month)
    pipeline, hint = department_summary_pipeline(first.isoformat(), last.isoformat(), department)
    items = []
    for r in db.employees.aggregate(pipeline, hint=hint):
        wh_count = int(r["wh_count"])
        avg = None
        if wh_count:
            avg = round_half_up(r["wh_sum"].to_decimal() / Decimal(wh_count), 2)  # mean over records
        items.append({
            "department": r["_id"],
            "headcount": int(r["headcount"]),
            "present_days": round_half_up(float(r["present_days"]), 2),
            "avg_work_hours": avg,
            "late_count": int(r["late_count"]),
            "total_late_minutes": int(r["total_late_minutes"]),
            "leave_count": int(r["leave_count"]),
            "on_duty_count": int(r["on_duty_count"]),
        })
    return {"month": month, "items": items}


@app.get("/analytics/leaderboard/late")
def late_leaderboard(
    month: str = Query(..., pattern=MONTH_RE),
    limit: int = Query(10, ge=1, le=50),
    department: Optional[str] = None,
):
    first, last = month_bounds(month)
    rows = db.attendance_logs.aggregate(late_leaderboard_pipeline(first.isoformat(), last.isoformat(), limit, department))
    items = [{
        "rank": int(r["rank"]),
        "emp_code": r["emp_code"],
        "name": r["name"],
        "department": r["department"],
        "total_late_minutes": int(r["total_late_minutes"]),
        "late_count": int(r["late_count"]),
    } for r in rows]
    return {"month": month, "items": items}


def trend_inputs(department: str, date_from: str, date_to: str) -> Tuple[List[str], date, date]:
    d_from, d_to = parse_date(date_from, "from"), parse_date(date_to, "to")
    if d_to < d_from:
        raise _bad("to", "to must not be before from")
    if (d_to - d_from).days + 1 > 92:
        raise _bad("to", "range must not exceed 92 days")
    codes = [e["emp_code"] for e in db.employees.find({"department": department}, {"emp_code": 1, "_id": 0})]
    return codes, d_from, d_to


@app.get("/analytics/departments/{department}/trend")
def department_trend(department: str, from_: str = Query(..., alias="from"), to: str = Query(...)):
    codes, d_from, d_to = trend_inputs(department, from_, to)
    if not codes:
        raise HTTPException(404, "unknown department")
    rows = db.attendance_logs.aggregate(department_trend_pipeline(department, codes, d_from, d_to))
    return {"department": department, "items": [dict(r) for r in rows]}


# --------------------------------------------------------------------------- #
# Explain: executionStats of the endpoint's main query, built by the same functions as the endpoint
# --------------------------------------------------------------------------- #
@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: Literal["attendance_list", "employee_monthly", "department_summary", "late_leaderboard", "department_trend"],
    emp_code: Optional[str] = None,
    month: Optional[str] = Query(None, pattern=MONTH_RE),
    department: Optional[str] = None,
    limit: int = Query(10, ge=1, le=50),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[Status] = None,
    from_: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    def need(**kw):
        for name, value in kw.items():
            if value is None or value == "":
                raise _missing(name)

    if endpoint == "attendance_list":
        q, sort, skip, lim = build_attendance_query(emp_code, date_from, date_to, status, page, page_size)
        collection = "attendance_logs"
        cmd = {"find": collection, "filter": q, "sort": dict(sort), "skip": skip, "limit": lim}
    elif endpoint == "employee_monthly":
        need(emp_code=emp_code, month=month)
        first, last = month_bounds(month)
        collection = "attendance_logs"
        cmd = {"aggregate": collection, "pipeline": employee_monthly_pipeline(emp_code, first.isoformat(), last.isoformat()), "cursor": {}}
    elif endpoint == "department_summary":
        need(month=month)
        first, last = month_bounds(month)
        collection = "employees"
        pipeline, hint = department_summary_pipeline(first.isoformat(), last.isoformat(), department)
        cmd = {"aggregate": collection, "pipeline": pipeline, "cursor": {}, "hint": hint}
    elif endpoint == "late_leaderboard":
        need(month=month)
        first, last = month_bounds(month)
        collection = "attendance_logs"
        cmd = {"aggregate": collection, "pipeline": late_leaderboard_pipeline(first.isoformat(), last.isoformat(), limit, department), "cursor": {}}
    else:  # department_trend
        need(department=department, **{"from": from_, "to": to})
        codes, d_from, d_to = trend_inputs(department, from_, to)
        collection = "attendance_logs"
        cmd = {"aggregate": collection, "pipeline": department_trend_pipeline(department, codes, d_from, d_to), "cursor": {}}

    raw = db.command({"explain": cmd, "verbosity": "executionStats"})
    return {
        "endpoint": endpoint,
        "collection": collection,
        "explain": json.loads(json_util.dumps(raw, json_options=json_util.RELAXED_JSON_OPTIONS)),
    }
