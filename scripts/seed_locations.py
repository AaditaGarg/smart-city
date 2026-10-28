"""Create placeholder road segments and intersections so you can attach uploads to them.
These are configuration rows (names only) - no predictions or measurements are invented.
    python scripts/seed_locations.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from backend.app.db import models as M  # noqa: E402
from backend.app.db.session import SessionLocal, init_db  # noqa: E402

init_db()
with SessionLocal() as db:
    if db.scalar(select(M.RoadSegment)) is None:
        for i, imp in enumerate([3.0, 2.5, 2.0, 1.5, 1.0], start=1):
            db.add(M.RoadSegment(name=f"Segment {i}", traffic_importance=imp))
        for i in range(1, 4):
            db.add(M.Intersection(name=f"Intersection {i}", approaches=4))
        db.commit()
        print("seeded 5 segments and 3 intersections")
    else:
        print("locations already exist; nothing to do")
