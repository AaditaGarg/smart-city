"""Road-damage severity scoring (HEURISTIC).

The weights and thresholds below are engineering assumptions, NOT calibrated against pavement-engineer
assessments or repair-cost data. They exist to turn detector output into a comparable 0-100 number; treat
rankings as indicative. Detector recall for road damage is low (see reports/RESULTS.md), so absent detections
do not prove a road is undamaged.
"""
from __future__ import annotations

import math

SEVERITY_WEIGHT = {"pothole": 1.0, "alligator_crack": 0.8, "longitudinal_crack": 0.4, "transverse_crack": 0.4}
AREA_GAIN = 20.0          # a box covering 5% of the frame counts double
AREA_CAP = 0.25           # box area fractions above this are capped
SATURATION = 3.0          # raw score at which the 0-100 score reaches ~63
BINS = [(50.0, "high"), (25.0, "medium"), (0.0, "low")]


def damage_score(detections: list[dict], width: int, height: int) -> dict:
    """detections: [{'class_name', 'confidence', 'x1','y1','x2','y2'}] in pixels."""
    img_area = float(max(width * height, 1))
    raw, counts = 0.0, {}
    for d in detections:
        c = d["class_name"]
        counts[c] = counts.get(c, 0) + 1
        area = max(0.0, (d["x2"] - d["x1"]) * (d["y2"] - d["y1"])) / img_area
        raw += SEVERITY_WEIGHT.get(c, 0.5) * float(d["confidence"]) * (1.0 + AREA_GAIN * min(area, AREA_CAP))
    score = 100.0 * (1.0 - math.exp(-raw / SATURATION))
    severity = "none"
    if detections:
        severity = next(label for lo, label in BINS if score >= lo)
    return {"score": round(score, 2), "severity": severity, "n_defects": len(detections), "defect_counts": counts}
