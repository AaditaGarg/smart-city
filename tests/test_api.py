"""API plumbing tests. FakeRegistry stands in for the trained models ONLY inside these tests so DB/HTTP behaviour
can be checked without PyTorch; the running API never uses it."""
import os

os.environ["DATABASE_URL"] = "sqlite://"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend.app.api.deps import get_registry  # noqa: E402
from backend.app.main import app  # noqa: E402
from backend.app.services.model_service import ModelRegistry  # noqa: E402
from decision_engine.scoring import damage_score  # noqa: E402


class FakeRegistry:
    def available(self):
        return {"vehicle": True, "road_damage": True, "congestion": True}

    def model_name(self, task):
        return f"{task}_best.pt"

    def detect(self, task, img, conf):
        if task == "vehicle":
            return [{"class_name": "car", "confidence": 0.9, "x1": 10, "y1": 10, "x2": 90, "y2": 60},
                    {"class_name": "truck", "confidence": 0.8, "x1": 100, "y1": 20, "x2": 180, "y2": 90}], 4.0
        return [{"class_name": "pothole", "confidence": 0.7, "x1": 20, "y1": 20, "x2": 80, "y2": 70}], 4.0

    def congestion(self, dets, w, h):
        return {"index": 42.0, "level": "moderate"}


def _png():
    ok, buf = cv2.imencode(".png", np.full((120, 200, 3), 90, np.uint8))
    return buf.tobytes()


@pytest.fixture()
def client():
    app.dependency_overrides[get_registry] = lambda: FakeRegistry()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health_reports_db_and_models(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["database"] is True and set(r["models"]) == {"vehicle", "road_damage", "congestion"}


def test_locations_crud_and_duplicates(client):
    assert client.post("/segments", json={"name": "A", "traffic_importance": 2}).status_code == 201
    assert client.post("/segments", json={"name": "A"}).status_code == 409
    assert client.post("/segments", json={"name": "B", "traffic_importance": 99}).status_code == 422
    assert client.post("/intersections", json={"name": "X"}).status_code == 201
    assert len(client.get("/segments").json()) >= 1 and len(client.get("/intersections").json()) >= 1


def test_vehicle_prediction_is_stored_and_listed(client):
    iid = client.post("/intersections", json={"name": "I-veh"}).json()["id"]
    r = client.post(f"/predict/vehicles?intersection_id={iid}&annotate_image=true",
                    files={"file": ("a.png", _png(), "image/png")})
    assert r.status_code == 200
    j = r.json()
    assert j["counts"] == {"car": 1, "truck": 1} and j["congestion"]["level"] == "moderate" and j["annotated_image_b64"]
    t = [x for x in client.get("/traffic/latest").json() if x["intersection_id"] == iid]
    assert t and t[0]["vehicle_count"] == 2
    assert any(p["id"] == j["prediction_id"] for p in client.get("/predictions?task=vehicle").json())


def test_road_damage_prediction_creates_condition(client):
    sid = client.post("/segments", json={"name": "S-dmg"}).json()["id"]
    r = client.post(f"/predict/road-damage?segment_id={sid}", files={"file": ("b.png", _png(), "image/png")})
    assert r.status_code == 200 and r.json()["road_condition"]["severity"] in {"low", "medium", "high"}
    c = [x for x in client.get("/road-conditions/latest").json() if x["segment_id"] == sid]
    assert c and c[0]["defect_counts"] == {"pothole": 1}


def test_bad_inputs(client):
    assert client.post("/predict/vehicles", files={"file": ("x.txt", b"not an image", "text/plain")}).status_code == 400
    assert client.post("/predict/vehicles?segment_id=9999", files={"file": ("a.png", _png(), "image/png")}).status_code == 404


def test_maintenance_lifecycle(client):
    sid = client.post("/segments", json={"name": "S-mnt"}).json()["id"]
    m = client.post("/maintenance", json={"segment_id": sid, "description": "fill pothole"}).json()
    assert m["status"] == "open" and m["source"] == "manual"
    d = client.patch(f"/maintenance/{m['id']}", json={"status": "done"}).json()
    assert d["status"] == "done" and d["completed_at"]
    assert client.patch(f"/maintenance/{m['id']}", json={"status": "bogus"}).status_code == 422
    assert client.patch("/maintenance/9999", json={"status": "done"}).status_code == 404
    assert any(x["id"] == m["id"] for x in client.get("/maintenance?status=done").json())


def test_missing_weights_give_503_not_fake_predictions(client, tmp_path):
    app.dependency_overrides[get_registry] = lambda: ModelRegistry(tmp_path)   # empty dir, real class
    r = client.post("/predict/vehicles", files={"file": ("a.png", _png(), "image/png")})
    assert r.status_code == 503 and "not found" in r.json()["detail"]


def test_damage_score_properties():
    none = damage_score([], 640, 480)
    assert none["severity"] == "none" and none["score"] == 0
    small = damage_score([{"class_name": "transverse_crack", "confidence": 0.3, "x1": 0, "y1": 0, "x2": 20, "y2": 20}], 640, 480)
    big = damage_score([{"class_name": "pothole", "confidence": 0.9, "x1": 0, "y1": 0, "x2": 300, "y2": 300}] * 3, 640, 480)
    assert 0 < small["score"] < big["score"] <= 100 and big["severity"] == "high"
