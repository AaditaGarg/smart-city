"""API tests for the multi-junction network endpoints."""
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_scenarios_list_network_and_controllers():
    r = client.get("/network/scenarios")
    assert r.status_code == 200
    body = r.json()
    ids = {s["id"] for s in body["scenarios"]}
    assert {"morning_rush", "two_vs_hundred", "over_capacity", "ambulance", "camera_failure"} <= ids
    assert [j["id"] for j in body["network"]["junctions"]] == ["J1", "J2", "J3", "J4"]


def test_simulate_returns_frames_and_metrics():
    r = client.post("/network/simulate", json={"scenario": "balanced", "controller": "max_pressure", "duration_s": 120})
    assert r.status_code == 200
    body = r.json()
    assert body["metrics"]["vehicles_entered"] > 0
    assert len(body["frames"]) == 60
    assert set(body["frames"][0]["signals"]) == {"J1", "J2", "J3", "J4"}


def test_unknown_scenario_is_404():
    r = client.post("/network/simulate", json={"scenario": "nope", "controller": "fixed"})
    assert r.status_code == 404


def test_decide_gives_heavy_road_green_with_explanation():
    r = client.post("/network/decide", json={
        "junction_id": "J2",
        "approaches": {"N": {"counts": {"bike": 60, "car": 30, "auto": 10}, "lanes": 2},
                       "E": {"counts": {"car": 2}, "lanes": 3},
                       "S": {"counts": {"car": 3}, "lanes": 2},
                       "W": {"counts": {}, "lanes": 3}},
    })
    assert r.status_code == 200
    d = r.json()
    assert d["phase_name"] == "N-S through"
    assert "N has" in d["explanation"]


def test_emergency_plans_every_junction_ahead():
    r = client.post("/network/emergency", json={"source": "gps"})
    assert r.status_code == 200
    plan = r.json()
    assert [p["junction"] for p in plan["junctions"]] == ["J1", "J2", "J3", "J4"]
    assert all(p["green_from_s"] <= p["eta_s"] for p in plan["junctions"])


def test_health_reports_fallback_without_camera_frames():
    r = client.get("/network/health")
    assert r.status_code == 200
    modes = {j["junction"]: j["mode"] for j in r.json()["junctions"]}
    assert set(modes) == {"J1", "J2", "J3", "J4"}
