from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from .api.deps import get_registry
from .api.routes import router
from .core.config import settings
from .db.session import engine, init_db
from .services.model_service import ModelUnavailable


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_db()
    except Exception as e:  # keep the API up so /health can explain what is wrong
        print(f"[startup] database not ready: {str(e).splitlines()[0]}")
    yield


app = FastAPI(title="Smart City Intelligence System", version="0.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins.split(","), allow_methods=["*"],
                   allow_headers=["*"])
app.include_router(router)


@app.exception_handler(ModelUnavailable)
async def model_unavailable(_: Request, exc: ModelUnavailable):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health")
def health():
    try:
        with engine.connect() as c:
            c.execute(text("select 1"))
        db_ok = True
    except Exception:
        db_ok = False
    return {"status": "ok", "database": db_ok, "models": get_registry().available()}
