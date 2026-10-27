from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import settings

app = FastAPI(title="Smart City Intelligence System", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}
