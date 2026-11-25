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

DEFAULT_SEGMENTS = [
    ("Outer Ring Road (NH-44 Expressway)", 4.8),
    ("Mahatma Gandhi Marg (Central Arterial)", 4.2),
    ("Tech Zone Blvd (IT Corridor)", 3.5),
    ("Subhash Nagar Link Road", 2.1),
    ("Connaught Circus Outer Ring", 3.8),
]

DEFAULT_INTERSECTIONS = [
    "Rajiv Chowk Junction",
    "Bangalore Silk Board Crossing",
    "Mumbai Western Express Interchange",
]

init_db()
with SessionLocal() as db:
    if db.scalar(select(M.RoadSegment)) is None:
        for name, imp in DEFAULT_SEGMENTS:
            db.add(M.RoadSegment(name=name, traffic_importance=imp))
        for name in DEFAULT_INTERSECTIONS:
            db.add(M.Intersection(name=name, approaches=4))
        db.commit()
        print("seeded 5 segments and 3 intersections")
    else:
        print("locations already exist; nothing to do")
