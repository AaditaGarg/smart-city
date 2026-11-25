"""Security behaviour: file signatures, headers, API-key enforcement, rate limiting, config validation."""
import os

os.environ["DATABASE_URL"] = "sqlite://"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend.app.api.deps import get_registry  # noqa: E402
from backend.app.core.config import Settings, settings  # noqa: E402
from backend.app.core.security import RateLimitMiddleware, detect_image_type  # noqa: E402
from backend.app.main import app  # noqa: E402


class FakeRegistry:
    def available(self):
        return {"vehicle": True, "road_damage": True, "congestion": True}

    def model_name(self, task):
        return f"{task}_best.pt"

    def detect(self, task, img, conf):
        return [], 1.0

    def congestion(self, dets, w, h):
        return {"index": 0.0, "level": "free"}


@pytest.fixture()
def client():
    app.dependency_overrides[get_registry] = lambda: FakeRegistry()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _img(ext=".png"):
    ok, buf = cv2.imencode(ext, np.full((64, 64, 3), 90, np.uint8))
    return buf.tobytes()


def test_detect_image_type_signatures():
    assert detect_image_type(_img(".png")) == "png"
    assert detect_image_type(_img(".jpg")) == "jpeg"
    assert detect_image_type(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "webp"
    assert detect_image_type(b"<script>alert(1)</script>" * 3) is None
    assert detect_image_type(b"GIF89a" + b"\x00" * 20) is None


def test_upload_with_wrong_signature_is_rejected(client):
    r = client.post("/predict/vehicles", files={"file": ("evil.png", b"<?php echo 1; ?>" * 4, "image/png")})
    assert r.status_code == 415


def test_valid_jpeg_and_png_are_accepted(client):
    for name, data, mime in (("a.png", _img(".png"), "image/png"), ("a.jpg", _img(".jpg"), "image/jpeg")):
        assert client.post("/predict/vehicles", files={"file": (name, data, mime)}).status_code == 200


def test_security_headers_present(client):
    h = client.get("/health").headers
    assert h["x-content-type-options"] == "nosniff"
    assert h["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in h["content-security-policy"]
    assert "default-src 'none'" in h["content-security-policy"]
    assert "max-age=" in h["strict-transport-security"]
    assert "permissions-policy" in h


def test_docs_get_relaxed_csp_so_swagger_works(client):
    csp = client.get("/docs").headers["content-security-policy"]
    assert "cdn.jsdelivr.net" in csp


def test_api_key_enforced_when_enabled(client, monkeypatch):
    monkeypatch.setattr(settings, "require_api_key", True)
    monkeypatch.setattr(settings, "api_key", "test-key-123")
    assert client.post("/segments", json={"name": "SEC-1"}).status_code == 401
    assert client.post("/segments", json={"name": "SEC-1"}, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/segments", json={"name": "SEC-1"}, headers={"X-API-Key": "test-key-123"}).status_code == 201
    assert client.get("/segments").status_code == 200          # reads stay open


def test_api_key_not_required_by_default(client):
    assert settings.api_key == "" and settings.require_api_key is False
    assert client.post("/segments", json={"name": "SEC-open"}).status_code == 201


def test_require_api_key_without_key_is_a_config_error():
    with pytest.raises(ValueError):
        Settings(require_api_key=True, api_key="")


def test_rate_limiter_returns_429_after_limit():
    mini = FastAPI()
    mini.add_middleware(RateLimitMiddleware, max_requests=3, window_seconds=60)

    @mini.get("/ping")
    def ping():
        return {"ok": True}

    c = TestClient(mini)
    codes = [c.get("/ping").status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    assert "retry-after" in c.get("/ping").headers
