import os

os.environ["DATABASE_URL"] = "sqlite://"

import pytest
from fastapi.testclient import TestClient

from backend.app.api.deps import get_registry
from backend.app.main import app


class FakeRegistry:
    def available(self):
        return {"vehicle": True, "road_damage": True, "congestion": True}

    def model_name(self, task):
        return f"{task}_best.pt"

    def detect(self, task, img, conf):
        return [{"class_name": "car", "confidence": 0.9, "x1": 10, "y1": 10, "x2": 90, "y2": 60}], 4.0

    def congestion(self, dets, w, h):
        return {"index": 35.0, "level": "moderate"}


@pytest.fixture()
def client():
    app.dependency_overrides[get_registry] = lambda: FakeRegistry()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_signal_optimization_dynamic_generic(client):
    payload = {
        "approaches": [
            {"name": "Northbound", "lanes": 2, "counts": {"car": 20, "truck": 3}, "has_emergency": False},
            {"name": "Southbound", "lanes": 2, "counts": {"car": 15, "van": 2}, "has_emergency": False},
            {"name": "Eastbound", "lanes": 1, "counts": {"car": 5, "bike": 2}, "has_emergency": False},
            {"name": "Westbound", "lanes": 1, "counts": {"car": 6}, "has_emergency": False},
        ]
    }
    r = client.post("/signals/optimize-dynamic", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["mode"] == "WEBSTER_DYNAMIC_OPTIMIZED"
    assert data["cycle_length"] >= 45.0
    assert len(data["phase_timings"]) == 4
    assert data["benefits"]["delay_reduction_pct"] > 0
    # Northbound (highest PCU) should receive highest or equal green time
    nb_timing = next(p for p in data["phase_timings"] if p["approach"] == "Northbound")
    eb_timing = next(p for p in data["phase_timings"] if p["approach"] == "Eastbound")
    assert nb_timing["green_time"] > eb_timing["green_time"]


def test_intersection_signal_optimization_with_emergency_override(client):
    # Create intersection in DB
    inter_res = client.post("/intersections", json={"name": "Rajiv Chowk Crossing", "approaches": 4})
    assert inter_res.status_code == 201
    iid = inter_res.json()["id"]

    # Trigger optimization with emergency ambulance on Eastbound
    payload = {
        "intersection_id": iid,
        "approaches": [
            {"name": "Northbound", "lanes": 2, "counts": {"car": 30}, "has_emergency": False},
            {"name": "Eastbound", "lanes": 2, "counts": {"car": 3, "ambulance": 1}, "has_emergency": True},
        ],
    }
    res = client.post(f"/intersections/{iid}/optimize-signal", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] == "EMERGENCY_OVERRIDE"
    assert data["emergency_approach"] == "Eastbound"
    assert data["recommendation_id"] is not None

    # Verify critical emergency alert was logged
    alerts = client.get("/alerts").json()
    em_alerts = [a for a in alerts if a.get("kind") == "emergency_vehicle_green_corridor"]
    assert len(em_alerts) >= 1
    assert "Rajiv Chowk Crossing" in em_alerts[0]["message"]


def test_security_headers_enforced(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.headers.get("x-content-type-options") == "nosniff"
    assert r.headers.get("x-frame-options") == "DENY"
    assert r.headers.get("x-xss-protection") == "1; mode=block"


def test_repair_prioritization_endpoint(client):
    # Create a test road segment
    seg = client.post("/segments", json={"name": "Expressway Corridor 101", "traffic_importance": 4.5}).json()
    sid = seg["id"]

    r = client.post("/segments/prioritize-repairs")
    assert r.status_code == 200
    ranked = r.json()
    assert isinstance(ranked, list)
    assert any(item["segment_id"] == sid for item in ranked)

