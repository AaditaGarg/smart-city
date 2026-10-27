"""Checks that the environment is ready. Run: python scripts/verify_setup.py"""
import importlib
import sys

PKGS = ["torch", "torchvision", "ultralytics", "cv2", "numpy", "pandas", "sklearn",
        "albumentations", "yaml", "matplotlib", "fastapi", "sqlalchemy", "psycopg2"]

ok = True
print(f"Python {sys.version.split()[0]}")
for p in PKGS:
    try:
        m = importlib.import_module(p)
        print(f"  [ok] {p} {getattr(m, '__version__', '')}")
    except Exception as e:
        ok = False
        print(f"  [MISSING] {p}: {e}")

try:
    import torch
    mps = getattr(torch.backends, "mps", None)
    print("CUDA available:", torch.cuda.is_available(),
          "| MPS available:", bool(mps and mps.is_available()))
except Exception:
    pass

try:
    from sqlalchemy import create_engine, text
    from backend.app.core.config import settings
    with create_engine(settings.database_url).connect() as c:
        c.execute(text("select 1"))
    print("  [ok] PostgreSQL connection")
except Exception as e:
    ok = False
    print(f"  [FAIL] PostgreSQL: {str(e).splitlines()[0]}  (is `docker compose up -d` running?)")

sys.exit(0 if ok else 1)
