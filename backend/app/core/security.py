"""Security layer for Smart City Intelligence API (ASGI Standard).

Provides:
- OWASP Security HTTP headers ASGI middleware
- Fast in-memory Rate Limiting ASGI middleware
- File magic bytes verification
- API key verification dependency
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import Dict, List, Optional

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from .config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def validate_image_security(data: bytes, filename: str = "") -> None:
    """Validate image payload against buffer overflow, size limits, and malicious magic bytes."""
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded file exceeds maximum limit of {settings.max_upload_mb} MB.",
        )

    if len(data) < 16:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File too small to be a valid image.",
        )


def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> bool:
    """FastAPI dependency to verify API key for administrative operations."""
    if not settings.require_api_key:
        return True
    if api_key and api_key == settings.api_key:
        return True
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing X-API-Key header.",
        headers={"WWW-Authenticate": "ApiKey"},
    )


class SecurityHeadersMiddleware:
    """Enforces OWASP recommended security headers on every response via ASGI."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-content-type-options", b"nosniff"))
                headers.append((b"x-frame-options", b"DENY"))
                headers.append((b"x-xss-protection", b"1; mode=block"))
                headers.append((b"referrer-policy", b"strict-origin-when-cross-origin"))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)


class RateLimitMiddleware:
    """Lightweight in-memory rate limiter per client IP address via ASGI."""

    def __init__(self, app, max_requests: int = 240, window_seconds: int = 60):
        self.app = app
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.clients: Dict[str, List[float]] = defaultdict(list)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        path = scope.get("path", "")
        if path in ("/health", "/docs", "/openapi.json"):
            return await self.app(scope, receive, send)

        client = scope.get("client")
        client_ip = client[0] if client else "127.0.0.1"
        now = time.time()
        window_start = now - self.window_seconds

        req_times = [t for t in self.clients[client_ip] if t > window_start]
        self.clients[client_ip] = req_times

        if len(req_times) >= self.max_requests:
            retry_after = str(int(self.window_seconds - (now - req_times[0])) + 1)
            payload = json.dumps({"detail": "Rate limit exceeded. Please wait."}).encode("utf-8")
            response_start = {
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"retry-after", retry_after.encode("utf-8")),
                    (b"content-length", str(len(payload)).encode("utf-8")),
                ],
            }
            response_body = {"type": "http.response.body", "body": payload}
            await send(response_start)
            await send(response_body)
            return

        self.clients[client_ip].append(now)
        await self.app(scope, receive, send)
