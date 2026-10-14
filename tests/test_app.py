"""Offline tests: rule helpers + endpoint flows on mongomock.
mongomock cannot run $densify / $setWindowFields / $lookup-with-pipeline, so analytics pipelines are NOT covered here;
run them against a real MongoDB 6+ (see README)."""
import threading
from datetime import datetime, timezone

import mongomock
import pytest
from fastapi.testclient import TestClient

import app.main as m

UTC = timezone.utc


def ms(y, mo, d, h=0, mi=0, s=0, ist=True):
    dt = datetime(y, mo, d, h, mi, s, tzinfo=m.IST if ist else UTC)
    return int(dt.timestamp() * 1000)


EMP = {"emp_code": "EMP0001", "shift_start": "09:30", "shift_end": "18:30"}
NIGHT = {"emp_code": "EMP0005", "shift_start": "22:00", "shift_end": "06:00"}


def D(y, mo, d, h, mi, s=0):
    return datetime(y, mo, d, h, mi, s, tzinfo=m.IST)


def test_late_rule_examples_from_contract():
    f = lambda h, mi, s: m.compute_late_minutes(D(2026, 7, 6, h, mi, s), "09:30", "2026-07-06")
    assert f(9, 40, 0) == 0 and f(9, 40, 1) == 10 and f(10, 15, 59) == 45 and f(9, 0, 0) == 0


def test_work_hours_half_up_and_half_day():
    base = D(2026, 7, 6, 9, 0)
    # 4h 29m 42s = 16182 s -> 4.495 -> half-up 4.50 (banker's / float round gives 4.49) -> NOT a half day
    from datetime import timedelta
    assert m.compute_work_hours(base, base + timedelta(seconds=16182)) == 4.5
    assert m.derive(EMP, "2026-07-06", "PRESENT", base, base + timedelta(seconds=16182))["half_day"] is False
    assert m.derive(EMP, "2026-07-06", "PRESENT", base, base + timedelta(seconds=16100))["half_day"] is True


def test_overtime_threshold_and_overnight():
    assert m.compute_overtime(D(2026, 7, 6, 19, 0), EMP, "2026-07-06") == 30
    assert m.compute_overtime(D(2026, 7, 6, 18, 59, 59), EMP, "2026-07-06") == 0
    assert m.compute_overtime(D(2026, 7, 7, 6, 40), NIGHT, "2026-07-06") == 40


def test_attendance_date_overnight():
    assert m.attendance_date(D(2026, 7, 6, 21, 55), NIGHT) == "2026-07-06"
    assert m.attendance_date(D(2026, 7, 7, 5, 59, 59), NIGHT) == "2026-07-06"
    assert m.attendance_date(D(2026, 7, 7, 6, 0, 0), NIGHT) == "2026-07-07"
    assert m.attendance_date(D(2026, 7, 7, 0, 30), EMP) == "2026-07-07"


def test_round_half_up():
    assert m.round_half_up(2.675, 2) == 2.68 and m.round_half_up(0.03125, 4) == 0.0313


@pytest.fixture()
def c(monkeypatch):
    mc = mongomock.MongoClient(tz_aware=True)
    monkeypatch.setattr(m, "client", mc)
    monkeypatch.setattr(m, "db", mc["t"])
    m.db.attendance_logs.create_index([("emp_code", 1), ("date", 1)], unique=True)
    m.db.employees.create_index([("emp_code", 1)], unique=True)
    yield TestClient(m.app)


def emp(c, code="EMP0001", **kw):
    body = {"emp_code": code, "name": "A", "email": "a@x.io", "department": "Eng", "joined_on": "2026-01-05", **kw}
    return c.post("/employees", json=body)


def test_employee_create_validation_and_409(c):
    r = emp(c)
    assert r.status_code == 201 and isinstance(r.json()["created_at"], int) and "_id" not in r.json()
    assert emp(c).status_code == 409
    assert emp(c, "bad").status_code == 422
    assert emp(c, "EMP0002", shift_start="09:00", shift_end="09:00").status_code == 422
    assert emp(c, "EMP0002", joined_on="2026-02-30").status_code == 422
    assert emp(c, "EMP0002", shift_start="24:00").status_code == 422


def test_employee_list_pagination(c):
    for i in range(1, 6):
        emp(c, f"EMP000{i}", department="A" if i < 4 else "B")
    r = c.get("/employees", params={"page": 2, "page_size": 2}).json()
    assert [e["emp_code"] for e in r["items"]] == ["EMP0003", "EMP0004"] and r["total"] == 5
    assert c.get("/employees", params={"department": "A"}).json()["total"] == 3
    assert c.get("/employees", params={"page": 0}).status_code == 422
    assert c.get("/employees", params={"page_size": 101}).status_code == 422


def test_punch_in_out_flow(c):
    emp(c)
    t = ms(2026, 7, 6, 10, 15, 59)
    r = c.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": t + 900, "late_minutes": 99})
    assert r.status_code == 201
    j = r.json()
    assert j["date"] == "2026-07-06" and j["late_minutes"] == 45 and j["punch_in"] == t and j["history"] == []
    assert c.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": t}).status_code == 409
    assert c.post("/attendance/punch-in", json={"emp_code": "NOPE"}).status_code == 404
    for bad in (1783312500, 1783312500000.5, "1783312500000", -1, 10**14):
        assert c.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": bad}).status_code == 422
    assert c.post("/attendance/punch-in", json={"emp_code": "EMP0001", "status": "ABSENT"}).status_code == 422
    out = ms(2026, 7, 6, 19, 10)
    r = c.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": out})
    j = r.json()
    assert r.status_code == 200 and j["overtime_minutes"] == 40 and j["work_hours"] == 8.9 and j["half_day"] is False
    assert c.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": out + 60000}).status_code == 409


def test_punch_out_edge_cases(c):
    emp(c, "EMP0005", shift_start="22:00", shift_end="06:00")
    c.post("/attendance/punch-in", json={"emp_code": "EMP0005", "punched_at": ms(2026, 7, 6, 21, 55)})
    assert c.post("/attendance/punch-out", json={"emp_code": "EMP0005", "punched_at": ms(2026, 7, 6, 21, 55)}).status_code == 422
    assert c.post("/attendance/punch-out", json={"emp_code": "EMP0005", "punched_at": ms(2026, 7, 8, 9, 0)}).status_code == 422
    r = c.post("/attendance/punch-out", json={"emp_code": "EMP0005", "punched_at": ms(2026, 7, 7, 6, 40)})
    assert r.status_code == 200 and r.json()["overtime_minutes"] == 40 and r.json()["date"] == "2026-07-06"
    assert c.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ms(2026, 7, 7, 6, 40)}).status_code == 404


def test_regularize(c):
    emp(c)
    c.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ms(2026, 7, 7, 10, 5)})
    url = "/attendance/EMP0001/2026-07-07"
    base = {"reason": "biometric glitch", "regularized_by": "hr"}
    assert c.patch(url, json=base).status_code == 422                                   # changes nothing
    assert c.patch(url, json={**base, "punch_in": ms(2026, 7, 8, 9, 0)}).status_code == 422   # off the date
    assert c.patch(url, json={**base, "punch_in": None}).status_code == 422
    assert c.patch(url, json={**base, "status": "LEAVE", "punch_in": ms(2026, 7, 7, 9, 0)}).status_code == 422
    assert c.patch("/attendance/EMP0001/2026-07-09", json={**base, "status": "LEAVE"}).status_code == 404
    assert c.patch("/attendance/EMP0001/20260707", json={**base, "status": "LEAVE"}).status_code == 422
    r = c.patch(url, json={**base, "punch_in": ms(2026, 7, 7, 9, 28), "late_minutes": 5})
    j = r.json()
    assert r.status_code == 200 and j["late_minutes"] == 0 and len(j["history"]) == 1
    ch = j["history"][0]["changes"]
    assert set(ch) == {"punch_in", "late_minutes"} and ch["late_minutes"] == {"from": 35, "to": 0}
    assert ch["punch_in"]["from"] == ms(2026, 7, 7, 10, 5)
    r = c.patch(url, json={**base, "status": "LEAVE"}).json()
    assert r["punch_in"] is None and len(r["history"]) == 2 and r["history"][1]["changes"]["status"]["to"] == "LEAVE"
    assert c.patch(url, json={**base, "status": "PRESENT"}).status_code == 422          # needs a punch_in


def test_list_attendance_validation_and_shape(c):
    emp(c)
    c.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ms(2026, 7, 7, 9, 0)})
    r = c.get("/attendance", params={"emp_code": "EMP0001"}).json()
    assert r["total"] == 1 and "id" not in r["items"][0] and "_id" not in r["items"][0]
    assert c.get("/attendance", params={"date_from": "2026-07-09", "date_to": "2026-07-01"}).status_code == 422
    assert c.get("/attendance", params={"date_from": "2026-13-01"}).status_code == 422
    assert c.get("/attendance", params={"status": "BOGUS"}).status_code == 422


def test_concurrent_punch_in_exactly_one_201(c):
    emp(c)
    res = []
    def go():
        res.append(c.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ms(2026, 7, 7, 9, 0)}).status_code)
    ts = [threading.Thread(target=go) for _ in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert sorted(res) == [201] + [409] * 7
