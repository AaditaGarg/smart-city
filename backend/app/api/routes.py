from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from decision_engine import (
    calculate_pcu,
    calculate_repair_priority,
    damage_score,
    rank_segments_for_repair,
    webster_optimum_cycle,
)

from ..core.config import settings
from ..core.security import validate_image_security
from ..db import models as M
from ..services.model_service import ModelRegistry, annotate
from . import schemas as S
from .deps import ROOT, get_db, get_registry

router = APIRouter()
MAX_UPLOAD = settings.max_upload_mb * 1024 * 1024


# ------------------------------------------------------------------ helpers
def _decode(file: UploadFile) -> np.ndarray:
    data = file.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, f"image larger than {settings.max_upload_mb} MB")
    
    # Fast byte sanity check
    if len(data) < 16:
        raise HTTPException(400, "file is too small to be a valid image")

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "file is not a readable image")
    return img


def _require(db: Session, model, id_: Optional[int], label: str):
    if id_ is not None and db.get(model, id_) is None:
        raise HTTPException(404, f"{label} {id_} not found")


def _save_prediction(db, task, registry, file, img, dets, ms, conf, segment_id, intersection_id) -> M.Prediction:
    H, W = img.shape[:2]
    pred = M.Prediction(task=task, model_name=registry.model_name(task), image_name=file.filename or "upload",
                        image_width=W, image_height=H, conf_threshold=conf, inference_ms=ms,
                        n_detections=len(dets), segment_id=segment_id, intersection_id=intersection_id)
    db.add(pred)
    db.flush()
    for d in dets:
        area = max(0.0, (d["x2"] - d["x1"]) * (d["y2"] - d["y1"])) / (W * H)
        db.add(M.DetectionRow(prediction_id=pred.id, area_frac=area, **d))
    return pred


def _counts(dets):
    out: dict[str, int] = {}
    for d in dets:
        out[d["class_name"]] = out.get(d["class_name"], 0) + 1
    return out


# ------------------------------------------------------------------ locations
@router.post("/segments", response_model=S.SegmentOut, status_code=201)
def create_segment(body: S.SegmentIn, db: Session = Depends(get_db)):
    if db.scalar(select(M.RoadSegment).where(M.RoadSegment.name == body.name)):
        raise HTTPException(409, "segment name already exists")
    row = M.RoadSegment(**body.model_dump())
    db.add(row)
    db.commit()
    return row


@router.get("/segments", response_model=list[S.SegmentOut])
def list_segments(db: Session = Depends(get_db)):
    return db.scalars(select(M.RoadSegment).order_by(M.RoadSegment.id)).all()


@router.post("/intersections", response_model=S.IntersectionOut, status_code=201)
def create_intersection(body: S.IntersectionIn, db: Session = Depends(get_db)):
    if db.scalar(select(M.Intersection).where(M.Intersection.name == body.name)):
        raise HTTPException(409, "intersection name already exists")
    row = M.Intersection(**body.model_dump())
    db.add(row)
    db.commit()
    return row


@router.get("/intersections", response_model=list[S.IntersectionOut])
def list_intersections(db: Session = Depends(get_db)):
    return db.scalars(select(M.Intersection).order_by(M.Intersection.id)).all()


# ------------------------------------------------------------------ inference
@router.post("/predict/vehicles", response_model=S.PredictOut)
def predict_vehicles(file: UploadFile = File(...), conf: float = Query(0.25, ge=0.01, le=0.99),
                     segment_id: Optional[int] = None, intersection_id: Optional[int] = None,
                     annotate_image: bool = False, db: Session = Depends(get_db),
                     registry: ModelRegistry = Depends(get_registry)):
    _require(db, M.RoadSegment, segment_id, "segment")
    _require(db, M.Intersection, intersection_id, "intersection")
    img = _decode(file)
    H, W = img.shape[:2]
    dets, ms = registry.detect("vehicle", img, conf)
    cong = registry.congestion(dets, W, H) if registry.available()["congestion"] else None
    pred = _save_prediction(db, "vehicle", registry, file, img, dets, ms, conf, segment_id, intersection_id)
    counts = _counts(dets)
    db.add(M.TrafficObservation(prediction_id=pred.id, segment_id=segment_id, intersection_id=intersection_id,
                                vehicle_count=len(dets), class_counts=counts,
                                congestion_index=cong["index"] if cong else None,
                                congestion_level=cong["level"] if cong else None))
    db.commit()
    return S.PredictOut(prediction_id=pred.id, task="vehicle", model=pred.model_name, image_width=W, image_height=H,
                        inference_ms=round(ms, 2), n_detections=len(dets), counts=counts, detections=dets,
                        congestion=cong, annotated_image_b64=annotate(img, dets, (0, 200, 0)) if annotate_image else None)


@router.post("/predict/road-damage", response_model=S.PredictOut)
def predict_road_damage(file: UploadFile = File(...), conf: float = Query(0.25, ge=0.01, le=0.99),
                        segment_id: Optional[int] = None, annotate_image: bool = False,
                        db: Session = Depends(get_db), registry: ModelRegistry = Depends(get_registry)):
    _require(db, M.RoadSegment, segment_id, "segment")
    img = _decode(file)
    H, W = img.shape[:2]
    dets, ms = registry.detect("road_damage", img, conf)
    cond = damage_score(dets, W, H)
    pred = _save_prediction(db, "road_damage", registry, file, img, dets, ms, conf, segment_id, None)
    if segment_id is not None:
        db.add(M.RoadCondition(prediction_id=pred.id, segment_id=segment_id, n_defects=cond["n_defects"],
                               defect_counts=cond["defect_counts"], damage_score=cond["score"],
                               severity=cond["severity"]))
        
        # Automatic alert generation for high road damage
        if cond["severity"] == "high" or cond["score"] >= 50.0:
            db.add(M.Alert(
                kind="severe_road_damage",
                severity="high",
                message=f"Critical road damage detected on segment {segment_id} (Score: {cond['score']})",
                segment_id=segment_id,
                payload={"damage_score": cond["score"], "defect_counts": cond["defect_counts"]},
                simulated=False,
            ))
    db.commit()
    return S.PredictOut(prediction_id=pred.id, task="road_damage", model=pred.model_name, image_width=W,
                        image_height=H, inference_ms=round(ms, 2), n_detections=len(dets), counts=_counts(dets),
                        detections=dets, road_condition=cond,
                        annotated_image_b64=annotate(img, dets) if annotate_image else None)


# ------------------------------------------------------------------ Dynamic Traffic Signal Optimization (SIH 2025)
@router.post("/intersections/{intersection_id}/optimize-signal", response_model=S.SignalOptimizationOut)
def optimize_intersection_signal(intersection_id: int, body: S.SignalOptimizationIn, db: Session = Depends(get_db)):
    """SIH 2025 Core Endpoint:
    Calculates Webster's Optimum Cycle & dynamic green splits from YOLO vehicle detections,
    handles Emergency Vehicle Green Corridor Override, and stores recommendations & alerts.
    """
    intersection = db.get(M.Intersection, intersection_id)
    if intersection is None:
        raise HTTPException(404, f"Intersection {intersection_id} not found")

    approach_dicts = [app.model_dump() for app in body.approaches]
    result = webster_optimum_cycle(approach_dicts)

    # Determine congestion level summary
    total_pcu = sum(a["pcu"] for a in result["phase_timings"])
    avg_pcu_per_lane = total_pcu / max(1, sum(a.get("lanes", 2) for a in result["phase_timings"]))
    cong_level = "heavy" if avg_pcu_per_lane > 10.0 else ("moderate" if avg_pcu_per_lane > 4.0 else "free")
    cong_idx = min(100.0, avg_pcu_per_lane * 7.5)

    # Save to database
    rec_text = f"Cycle: {result['cycle_length']}s. Mode: {result['mode']}. Sequence: {', '.join(result.get('recommended_phase_sequence', []))}"
    db_rec = M.SignalRecommendation(
        intersection_id=intersection_id,
        congestion_level=cong_level,
        congestion_index=round(cong_idx, 2),
        recommendation=rec_text,
        payload=result,
    )
    db.add(db_rec)

    # If Emergency Vehicle is detected, trigger high-priority alert!
    if result.get("mode") == "EMERGENCY_OVERRIDE":
        db.add(M.Alert(
            kind="emergency_vehicle_green_corridor",
            severity="critical",
            message=f"EMERGENCY VEHICLE DETECTED at {intersection.name}! Green corridor established on {result.get('emergency_approach')}.",
            intersection_id=intersection_id,
            payload={
                "emergency_approach": result.get("emergency_approach"),
                "clearing_time_seconds": result.get("cycle_length"),
            },
            simulated=False,
        ))

    db.commit()
    result["recommendation_id"] = db_rec.id
    return result


@router.post("/signals/optimize-dynamic", response_model=S.SignalOptimizationOut)
def optimize_dynamic_signals_generic(body: S.SignalOptimizationIn):
    """Stateless dynamic signal optimization (useful for frontend live simulator and camera feeds)."""
    approach_dicts = [app.model_dump() for app in body.approaches]
    result = webster_optimum_cycle(approach_dicts)
    return result


@router.get("/signals/latest", response_model=list[S.SignalRecommendationOut])
def list_latest_signal_recommendations(limit: int = Query(20, ge=1, le=100), db: Session = Depends(get_db)):
    """Fetch recent signal timing recommendations."""
    return db.scalars(select(M.SignalRecommendation).order_by(M.SignalRecommendation.id.desc()).limit(limit)).all()


# ------------------------------------------------------------------ Road Repair Prioritisation
@router.post("/segments/prioritize-repairs", response_model=list[S.RepairPriorityOut])
def prioritize_city_repairs(db: Session = Depends(get_db)):
    """Ranks all road segments across the city using composite damage score, traffic importance,
    and defect hazards (potholes).
    """
    segments = db.scalars(select(M.RoadSegment).order_by(M.RoadSegment.id)).all()
    segment_conditions = []

    for seg in segments:
        latest_cond = db.scalar(
            select(M.RoadCondition)
            .where(M.RoadCondition.segment_id == seg.id)
            .order_by(M.RoadCondition.id.desc())
            .limit(1)
        )
        if latest_cond:
            segment_conditions.append({
                "segment_id": seg.id,
                "name": seg.name,
                "traffic_importance": seg.traffic_importance,
                "damage_score": latest_cond.damage_score,
                "defect_counts": latest_cond.defect_counts,
            })
        else:
            # Segment has no scans yet; include with baseline
            segment_conditions.append({
                "segment_id": seg.id,
                "name": seg.name,
                "traffic_importance": seg.traffic_importance,
                "damage_score": 0.0,
                "defect_counts": {},
            })

    ranked = rank_segments_for_repair(segment_conditions)
    return ranked


# ------------------------------------------------------------------ read APIs
@router.get("/predictions", response_model=list[S.PredictionOut])
def list_predictions(task: Optional[str] = None, limit: int = Query(50, ge=1, le=500), db: Session = Depends(get_db)):
    q = select(M.Prediction).order_by(M.Prediction.id.desc()).limit(limit)
    if task:
        q = q.where(M.Prediction.task == task)
    return db.scalars(q).all()


@router.get("/traffic/latest", response_model=list[S.TrafficOut])
def traffic_latest(db: Session = Depends(get_db)):
    seen, out = set(), []
    for o in db.scalars(select(M.TrafficObservation).order_by(M.TrafficObservation.id.desc())):
        key = (o.segment_id, o.intersection_id)
        if key not in seen:
            seen.add(key)
            out.append(o)
    return out


@router.get("/road-conditions/latest", response_model=list[S.RoadConditionOut])
def road_conditions_latest(db: Session = Depends(get_db)):
    seen, out = set(), []
    for c in db.scalars(select(M.RoadCondition).order_by(M.RoadCondition.id.desc())):
        if c.segment_id not in seen:
            seen.add(c.segment_id)
            out.append(c)
    return out


@router.get("/alerts", response_model=list[S.AlertOut])
def list_alerts(limit: int = Query(50, ge=1, le=500), db: Session = Depends(get_db)):
    return db.scalars(select(M.Alert).order_by(M.Alert.id.desc()).limit(limit)).all()


@router.post("/alerts/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: int, db: Session = Depends(get_db)):
    alert = db.get(M.Alert, alert_id)
    if not alert:
        raise HTTPException(404, "Alert not found")
    # Mark acknowledged in payload
    payload = dict(alert.payload or {})
    payload["acknowledged"] = True
    payload["acknowledged_at"] = datetime.now(timezone.utc).isoformat()
    alert.payload = payload
    db.commit()
    return {"status": "success", "alert_id": alert_id, "acknowledged": True}


# ------------------------------------------------------------------ maintenance
@router.post("/maintenance", response_model=S.MaintenanceOut, status_code=201)
def create_maintenance(body: S.MaintenanceIn, db: Session = Depends(get_db)):
    _require(db, M.RoadSegment, body.segment_id, "segment")
    row = M.MaintenanceRecord(**body.model_dump())
    db.add(row)
    db.commit()
    return row


@router.get("/maintenance", response_model=list[S.MaintenanceOut])
def list_maintenance(status: Optional[str] = None, db: Session = Depends(get_db)):
    q = select(M.MaintenanceRecord).order_by(M.MaintenanceRecord.id.desc())
    if status:
        q = q.where(M.MaintenanceRecord.status == status)
    return db.scalars(q).all()


@router.patch("/maintenance/{record_id}", response_model=S.MaintenanceOut)
def update_maintenance(record_id: int, body: S.MaintenanceUpdate, db: Session = Depends(get_db)):
    row = db.get(M.MaintenanceRecord, record_id)
    if row is None:
        raise HTTPException(404, "maintenance record not found")
    for k, v in body.model_dump(exclude_unset=True).items():
        setattr(row, k, v)
    if row.status == "done" and row.completed_at is None:
        row.completed_at = datetime.now(timezone.utc)
    db.commit()
    return row


# ------------------------------------------------------------------ model metrics
@router.get("/model-metrics")
def model_metrics():
    t = ROOT / "reports" / "tables"
    out: dict = {}
    for task in ("vehicle", "road_damage"):
        s, c = t / f"{task}_test_summary.json", t / f"{task}_test_metrics.csv"
        if s.exists() and c.exists():
            out[task] = {"summary": json.loads(s.read_text()),
                         "per_class": json.loads(pd.read_csv(c).to_json(orient="records"))}
    cm = t / "congestion_metrics.json"
    if cm.exists():
        out["congestion"] = json.loads(cm.read_text())
    return out
