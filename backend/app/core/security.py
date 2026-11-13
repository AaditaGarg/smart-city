"""Security layer for the Smart City API (ASGI).

Provides:
- Security HTTP headers middleware (CSP, HSTS, nosniff, frame denial, referrer policy, permissions policy)
- In-memory per-IP rate limiting middleware (single-process only; use a reverse proxy / Redis for multi-worker)
- Image validation: size, minimum length and file signature (magic bytes) for JPEG / PNG / WEBP
- Constant-time API key verification dependency
"""
from __future__ import annotations

import hmac
import json
import time
from collections import defaultdict
from typing import Dict, List, Optional

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from .config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def detect_image_type(data: bytes) -> Optional[str]:
    """Return 'jpeg' | 'png' | 'webp' from the file signature, or None if it is not one of them."""
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def validate_image_security(data: bytes, filename: str = "") -> str:
    """Validate an uploaded image payload (size + magic bytes). Returns the detected type."""
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            f"Uploaded file exceeds maximum limit of {settings.max_upload_mb} MB.")
    if len(data) < 16:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "File too small to be a valid image.")
    kind = detect_image_type(data)
    if kind is None:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            "Unsupported file type. Only JPEG, PNG and WEBP images are accepted.")
    return kind


def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> bool:
    """Dependency protecting state-changing endpoints when REQUIRE_API_KEY=true."""
    if not settings.require_api_key:
        return True
    if api_key and settings.api_key and hmac.compare_digest(api_key.encode(), settings.api_key.encode()):
        return True
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing X-API-Key header.",
                        headers={"WWW-Authenticate": "ApiKey"})


# Swagger UI / ReDoc load assets from a CDN, so they get a relaxed policy; everything else is locked down.
_DOCS_PATHS = ("/docs", "/redoc")
_CSP_API = b"default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
_CSP_DOCS = (b"default-src 'self'; img-src 'self' data: https://fastapi.tiangolo.com; "
             b"script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
             b"style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; frame-ancestors 'none'")


class SecurityHeadersMiddleware:
    """Adds security headers to every HTTP response."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        csp = _CSP_DOCS if scope.get("path", "").startswith(_DOCS_PATHS) else _CSP_API

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers += [
                    (b"x-content-type-options", b"nosniff"),
                    (b"x-frame-options", b"DENY"),
                    (b"x-xss-protection", b"1; mode=block"),   # legacy browsers; CSP is the real protection
                    (b"referrer-policy", b"strict-origin-when-cross-origin"),
                    (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
                    (b"strict-transport-security", b"max-age=31536000; includeSubDomains"),  # honoured over HTTPS only
                    (b"content-security-policy", csp),
                ]
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)


class RateLimitMiddleware:
    """Sliding-window per-IP rate limiter held in process memory."""

    _MAX_TRACKED_CLIENTS = 10_000

    def __init__(self, app, max_requests: int = 240, window_seconds: int = 60):
        self.app = app
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.clients: Dict[str, List[float]] = defaultdict(list)

    def _purge(self, now: float) -> None:
        cutoff = now - self.window_seconds
        for ip in [k for k, v in self.clients.items() if not v or v[-1] <= cutoff]:
            del self.clients[ip]

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        if scope.get("path", "") in ("/health", "/docs", "/openapi.json", "/redoc"):
            return await self.app(scope, receive, send)

        client = scope.get("client")
        client_ip = client[0] if client else "127.0.0.1"
        now = time.time()
        if len(self.clients) > self._MAX_TRACKED_CLIENTS:      # stop unbounded memory growth
            self._purge(now)

        req_times = [t for t in self.clients[client_ip] if t > now - self.window_seconds]
        if len(req_times) >= self.max_requests:
            self.clients[client_ip] = req_times
            retry_after = str(int(self.window_seconds - (now - req_times[0])) + 1)
            payload = json.dumps({"detail": "Rate limit exceeded. Please wait."}).encode()
            await send({"type": "http.response.start", "status": 429,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"retry-after", retry_after.encode()),
                                    (b"content-length", str(len(payload)).encode())]})
            await send({"type": "http.response.body", "body": payload})
            return

        req_times.append(now)
        self.clients[client_ip] = req_times
        await self.app(scope, receive, send)
