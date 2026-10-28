from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from decision_engine.scoring import damage_score

from ..db import models as M
from ..services.model_service import ModelRegistry, annotate
from . import schemas as S
from .deps import ROOT, get_db, get_registry

router = APIRouter()
MAX_UPLOAD = 15 * 1024 * 1024


# ------------------------------------------------------------------ helpers
def _decode(file: UploadFile) -> np.ndarray:
    data = file.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "image larger than 15 MB")
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
    db.commit()
    return S.PredictOut(prediction_id=pred.id, task="road_damage", model=pred.model_name, image_width=W,
                        image_height=H, inference_ms=round(ms, 2), n_detections=len(dets), counts=_counts(dets),
                        detections=dets, road_condition=cond,
                        annotated_image_b64=annotate(img, dets) if annotate_image else None)


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


# ------------------------------------------------------------------ model metrics (read from generated files)
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
