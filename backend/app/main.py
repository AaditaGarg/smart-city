from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from .api.deps import get_registry
from .api.routes import router
from .core.config import settings
from .core.security import RateLimitMiddleware, SecurityHeadersMiddleware
from .db.session import engine, init_db
from .services.model_service import ModelUnavailable


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_db()
    except Exception as e:  # keep the API up so /health can explain what is wrong
        print(f"[startup] database not ready: {str(e).splitlines()[0]}")
    yield


app = FastAPI(
    title="Smart City Intelligence & Dynamic Traffic System",
    description="SIH 2025 AI-Powered Dynamic Traffic Light Optimization, Road Damage Detection, and Surveillance API.",
    version="0.3.0",
    lifespan=lifespan,
)

# Attach Security Middlewares
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware, max_requests=settings.rate_limit_per_minute, window_seconds=60)

# CORS Configuration
origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.exception_handler(ModelUnavailable)
async def model_unavailable(_: Request, exc: ModelUnavailable):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health")
def health(registry=Depends(get_registry)):
    try:
        with engine.connect() as c:
            c.execute(text("select 1"))
        db_ok = True
    except Exception:
        db_ok = False
    models = registry.available()
    weights_ok = bool(models.get("vehicle") and models.get("road_damage"))
    return {
        "status": "ok" if (db_ok and weights_ok) else "degraded",
        "database": db_ok,
        "models": models,
        "security": {
            "rate_limiter": "active",
            "security_headers": "enforced",
            "auth_mode": "enforced" if settings.require_api_key else "permissive_dev",
            "csp": "enabled",
            "hsts": "enabled",
        },
    }
