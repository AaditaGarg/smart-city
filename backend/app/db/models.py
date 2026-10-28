from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .session import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class RoadSegment(Base):
    __tablename__ = "road_segments"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    traffic_importance: Mapped[float] = mapped_column(Float, default=1.0)   # 0.1-5, weights repair priority
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Intersection(Base):
    __tablename__ = "intersections"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    approaches: Mapped[int] = mapped_column(Integer, default=4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    task: Mapped[str] = mapped_column(String(32), index=True)            # vehicle | road_damage
    model_name: Mapped[str] = mapped_column(String(120))
    image_name: Mapped[str] = mapped_column(String(255))
    image_width: Mapped[int] = mapped_column(Integer)
    image_height: Mapped[int] = mapped_column(Integer)
    conf_threshold: Mapped[float] = mapped_column(Float)
    inference_ms: Mapped[float] = mapped_column(Float)
    n_detections: Mapped[int] = mapped_column(Integer)
    segment_id: Mapped[Optional[int]] = mapped_column(ForeignKey("road_segments.id"), nullable=True)
    intersection_id: Mapped[Optional[int]] = mapped_column(ForeignKey("intersections.id"), nullable=True)


class DetectionRow(Base):
    __tablename__ = "detections"
    id: Mapped[int] = mapped_column(primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), index=True)
    class_name: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float] = mapped_column(Float)
    x1: Mapped[float] = mapped_column(Float)
    y1: Mapped[float] = mapped_column(Float)
    x2: Mapped[float] = mapped_column(Float)
    y2: Mapped[float] = mapped_column(Float)
    area_frac: Mapped[float] = mapped_column(Float)


class TrafficObservation(Base):
    __tablename__ = "traffic_observations"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"))
    segment_id: Mapped[Optional[int]] = mapped_column(ForeignKey("road_segments.id"), nullable=True)
    intersection_id: Mapped[Optional[int]] = mapped_column(ForeignKey("intersections.id"), nullable=True)
    vehicle_count: Mapped[int] = mapped_column(Integer)
    class_counts: Mapped[dict] = mapped_column(JSON)
    congestion_index: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    congestion_level: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)


class RoadCondition(Base):
    __tablename__ = "road_conditions"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"))
    segment_id: Mapped[int] = mapped_column(ForeignKey("road_segments.id"), index=True)
    n_defects: Mapped[int] = mapped_column(Integer)
    defect_counts: Mapped[dict] = mapped_column(JSON)
    damage_score: Mapped[float] = mapped_column(Float)
    severity: Mapped[str] = mapped_column(String(16))


class MaintenanceRecord(Base):
    __tablename__ = "maintenance_records"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    segment_id: Mapped[int] = mapped_column(ForeignKey("road_segments.id"), index=True)
    status: Mapped[str] = mapped_column(String(16), default="open")   # open|scheduled|in_progress|done|cancelled
    priority_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    description: Mapped[str] = mapped_column(Text, default="")
    notes: Mapped[str] = mapped_column(Text, default="")
    source: Mapped[str] = mapped_column(String(32), default="manual")  # manual | decision_engine
    scheduled_for: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class Alert(Base):
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    kind: Mapped[str] = mapped_column(String(32))
    severity: Mapped[str] = mapped_column(String(16))
    message: Mapped[str] = mapped_column(Text)
    segment_id: Mapped[Optional[int]] = mapped_column(ForeignKey("road_segments.id"), nullable=True)
    intersection_id: Mapped[Optional[int]] = mapped_column(ForeignKey("intersections.id"), nullable=True)
    simulated: Mapped[bool] = mapped_column(Boolean, default=True)   # no real integrations exist
    payload: Mapped[dict] = mapped_column(JSON, default=dict)


class SignalRecommendation(Base):
    __tablename__ = "signal_recommendations"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    intersection_id: Mapped[int] = mapped_column(ForeignKey("intersections.id"), index=True)
    congestion_level: Mapped[str] = mapped_column(String(16))
    congestion_index: Mapped[float] = mapped_column(Float)
    recommendation: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
