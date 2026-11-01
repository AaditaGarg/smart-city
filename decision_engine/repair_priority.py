"""Road Damage Repair Prioritisation Engine.

Combines AI damage detection severity, defect types (prioritising high-hazard potholes),
and road segment traffic importance to compute actionable maintenance priorities.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def calculate_repair_priority(
    damage_score: float,
    traffic_importance: float,
    defect_counts: Dict[str, int],
    segment_name: str = "",
    segment_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Calculate composite repair priority score (0 - 100) for a road segment."""
    potholes = int(defect_counts.get("pothole", 0))
    alligator = int(defect_counts.get("alligator_crack", 0))
    longitudinal = int(defect_counts.get("longitudinal_crack", 0))
    transverse = int(defect_counts.get("transverse_crack", 0))

    # Weighting factors
    # Potholes pose severe structural & accident risks (high multiplier)
    hazard_multiplier = 1.0 + (0.18 * potholes) + (0.08 * alligator) + (0.03 * (longitudinal + transverse))
    importance_factor = max(0.2, min(2.5, traffic_importance / 2.0))

    # Composite formula
    raw_priority = damage_score * importance_factor * hazard_multiplier
    priority_score = round(min(100.0, max(0.0, raw_priority)), 2)

    # Classify priority level
    if priority_score >= 75.0 or (potholes >= 3 and damage_score >= 40.0):
        level = "CRITICAL"
        action = "Immediate emergency repair within 24 hours. High accident risk."
        auto_schedule_days = 1
    elif priority_score >= 50.0:
        level = "HIGH"
        action = "Schedule asphalt patching within 48-72 hours."
        auto_schedule_days = 3
    elif priority_score >= 25.0:
        level = "MEDIUM"
        action = "Include in weekly municipal maintenance batch."
        auto_schedule_days = 7
    else:
        level = "LOW"
        action = "Surface monitor during next routine inspection cycle."
        auto_schedule_days = 30

    return {
        "segment_id": segment_id,
        "segment_name": segment_name,
        "damage_score": damage_score,
        "traffic_importance": traffic_importance,
        "priority_score": priority_score,
        "priority_level": level,
        "recommended_action": action,
        "suggested_dispatch_days": auto_schedule_days,
        "defect_summary": {
            "potholes": potholes,
            "alligator_cracks": alligator,
            "surface_cracks": longitudinal + transverse,
        },
        "requires_alert": level in ("CRITICAL", "HIGH"),
    }


def rank_segments_for_repair(segments_with_conditions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank multiple road segments by urgency of repair."""
    ranked = []
    for item in segments_with_conditions:
        res = calculate_repair_priority(
            damage_score=item.get("damage_score", 0.0),
            traffic_importance=item.get("traffic_importance", 1.0),
            defect_counts=item.get("defect_counts", {}),
            segment_name=item.get("name", item.get("segment_name", "")),
            segment_id=item.get("segment_id", item.get("id")),
        )
        ranked.append(res)
    # Sort descending by priority score
    ranked.sort(key=lambda x: x["priority_score"], reverse=True)
    return ranked

