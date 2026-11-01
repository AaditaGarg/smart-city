from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

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


# ------------------------------------------------------------------ Traffic Signal Optimization (SIH 2025)
class ApproachIn(BaseModel):
    name: str = Field(..., description="e.g. Northbound, Southbound, Eastbound, Westbound")
    lanes: int = Field(2, ge=1, le=8)
    counts: Optional[Dict[str, int]] = Field(default_factory=dict, description="Class-wise vehicle counts e.g. {'car': 12, 'truck': 2}")
    has_emergency: bool = Field(False, description="Emergency vehicle present on this approach (ambulance, fire truck)")


class SignalOptimizationIn(BaseModel):
    intersection_id: Optional[int] = None
    approaches: List[ApproachIn] = Field(..., min_length=2, max_length=8)


class SignalPhaseTimingOut(BaseModel):
    approach: str
    lanes: Optional[int] = 2
    pcu: float
    counts: Dict[str, int]
    green_time: float
    yellow_time: float
    all_red_time: Optional[float] = 1.5
    red_time: float
    green_ratio: Optional[float] = None
    is_active_green: Optional[bool] = None


class SignalBenefitsOut(BaseModel):
    baseline_fixed_cycle_seconds: float
    delay_reduction_pct: float
    fuel_saved_liters_per_hour: float
    co2_saved_kg_per_hour: float


class SignalOptimizationOut(BaseModel):
    mode: str
    cycle_length: float
    total_lost_time: float
    critical_flow_ratio_Y: Optional[float] = None
    saturation_status: Optional[str] = None
    phase_timings: List[SignalPhaseTimingOut]
    recommended_phase_sequence: Optional[List[str]] = None
    benefits: Optional[SignalBenefitsOut] = None
    emergency_approach: Optional[str] = None
    reason: Optional[str] = None
    recommendation_id: Optional[int] = None


class SignalRecommendationOut(ORM):
    id: int
    created_at: datetime
    intersection_id: int
    congestion_level: str
    congestion_index: float
    recommendation: str
    payload: dict


# ------------------------------------------------------------------ Repair Prioritisation
class RepairPriorityOut(BaseModel):
    segment_id: Optional[int]
    segment_name: str
    damage_score: float
    traffic_importance: float
    priority_score: float
    priority_level: str
    recommended_action: str
    suggested_dispatch_days: int
    defect_summary: dict
    requires_alert: bool
