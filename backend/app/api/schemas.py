from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class SegmentIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    traffic_importance: float = Field(1.0, ge=0.1, le=5.0)


class SegmentOut(SegmentIn, ORM):
    id: int


class IntersectionIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    approaches: int = Field(4, ge=2, le=8)


class IntersectionOut(IntersectionIn, ORM):
    id: int


class DetectionOut(BaseModel):
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class PredictOut(BaseModel):
    prediction_id: int
    task: str
    model: str
    image_width: int
    image_height: int
    inference_ms: float
    n_detections: int
    counts: dict[str, int]
    detections: list[DetectionOut]
    congestion: Optional[dict] = None
    road_condition: Optional[dict] = None
    annotated_image_b64: Optional[str] = None


class PredictionOut(ORM):
    id: int
    created_at: datetime
    task: str
    model_name: str
    image_name: str
    n_detections: int
    inference_ms: float
    segment_id: Optional[int]
    intersection_id: Optional[int]


class TrafficOut(ORM):
    id: int
    created_at: datetime
    segment_id: Optional[int]
    intersection_id: Optional[int]
    vehicle_count: int
    class_counts: dict
    congestion_index: Optional[float]
    congestion_level: Optional[str]


class RoadConditionOut(ORM):
    id: int
    created_at: datetime
    segment_id: int
    n_defects: int
    defect_counts: dict
    damage_score: float
    severity: str


Status = Literal["open", "scheduled", "in_progress", "done", "cancelled"]


class MaintenanceIn(BaseModel):
    segment_id: int
    description: str = ""
    priority_score: Optional[float] = None
    scheduled_for: Optional[datetime] = None


class MaintenanceUpdate(BaseModel):
    status: Optional[Status] = None
    notes: Optional[str] = None
    priority_score: Optional[float] = None
    scheduled_for: Optional[datetime] = None


class MaintenanceOut(ORM):
    id: int
    created_at: datetime
    segment_id: int
    status: str
    priority_score: Optional[float]
    description: str
    notes: str
    source: str
    scheduled_for: Optional[datetime]
    completed_at: Optional[datetime]


class AlertOut(ORM):
    id: int
    created_at: datetime
    kind: str
    severity: str
    message: str
    segment_id: Optional[int]
    intersection_id: Optional[int]
    simulated: bool
    payload: dict
