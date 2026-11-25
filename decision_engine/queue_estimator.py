"""Turn camera detections into reliable queue estimates.

Fixes three real-world sensing problems:
  * the camera only sees the first ~80 m of a queue        -> carry-over estimate (QueueEstimator)
  * we don't know how many vehicles turn left/straight/right -> learned turn ratios (TurnRatioLearner)
  * a red car or signboard must never trigger a green corridor -> 2-frame ambulance confirmation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

AMBULANCE_MIN_CONF = 0.6
AMBULANCE_CONFIRM_FRAMES = 2
EMPTY_PCU = 0.5            # below this the stop-line zone counts as empty
LOW_CONFIDENCE = 0.5       # detector confidence under which we don't trust "empty"
STALE_AFTER_S = 10.0       # no frame for this long = camera failed


@dataclass
class QueueEstimator:
    """Per-approach queue estimate in PCU.

    estimate = visible count, unless the queue reaches the far edge of the camera view; then we
    keep a running balance (previous estimate + arrivals - departures) and take the larger value.
    """
    camera_range_pcu: float
    estimate: float = 0.0
    confidence: float = 1.0
    beyond_camera: bool = False
    empty_streak: int = 0
    last_update_t: Optional[float] = None

    def update(self, t: float, visible_pcu: float, confidence: float = 0.9,
               arrivals_pcu: float = 0.0, departures_pcu: float = 0.0) -> float:
        at_edge = visible_pcu >= 0.9 * self.camera_range_pcu
        # running balance; while the queue fills the whole view it can only be >= what we see
        carry = max(0.0, self.estimate + arrivals_pcu - departures_pcu)
        self.beyond_camera = at_edge
        if at_edge:
            est = max(visible_pcu, carry)
        elif confidence < LOW_CONFIDENCE:
            est = 0.5 * visible_pcu + 0.5 * carry      # don't trust a dark/rainy frame alone
        else:
            est = visible_pcu
        self.estimate = est
        self.confidence = confidence
        self.empty_streak = self.empty_streak + 1 if (visible_pcu < EMPTY_PCU and confidence >= LOW_CONFIDENCE) else 0
        self.last_update_t = t
        return est

    def advance_without_camera(self, arrivals_pcu: float, departures_pcu: float) -> float:
        """Camera down: keep the balance going from upstream/downstream counts."""
        self.estimate = max(0.0, self.estimate + arrivals_pcu - departures_pcu)
        self.empty_streak = 0
        return self.estimate

    def is_stale(self, t: float) -> bool:
        return self.last_update_t is None or (t - self.last_update_t) > STALE_AFTER_S

    def confirmed_empty(self) -> bool:
        """Early-switch rule: two checks in a row saw an empty stop line with a confident detector."""
        return self.empty_streak >= 2


@dataclass
class TurnRatioLearner:
    """Exponentially-smoothed turn ratios learned from counted departures per turn."""
    ratios: Dict[str, float]
    alpha: float = 0.02
    _seen: int = 0

    def observe(self, turn: str) -> None:
        if turn not in self.ratios:
            return
        for k in self.ratios:
            self.ratios[k] = (1 - self.alpha) * self.ratios[k] + self.alpha * (1.0 if k == turn else 0.0)
        self._seen += 1

    def get(self) -> Dict[str, float]:
        tot = sum(self.ratios.values()) or 1.0
        return {k: v / tot for k, v in self.ratios.items()}


@dataclass
class AmbulanceConfirmer:
    """Green corridor only after the ambulance class is seen with conf >= 0.6 in 2 frames in a row.

    A colour hint (red/blue lights) is recorded for the operator but can never trigger on its own.
    """
    streak: int = 0
    colour_hint: bool = False

    def update(self, ambulance_conf: Optional[float], colour_hint: bool = False) -> bool:
        self.colour_hint = colour_hint
        if ambulance_conf is not None and ambulance_conf >= AMBULANCE_MIN_CONF:
            self.streak += 1
        else:
            self.streak = 0
        return self.streak >= AMBULANCE_CONFIRM_FRAMES


def camera_range_pcu(lanes: int, view_m: float = 120.0, jam_spacing_m: float = 7.0) -> float:
    return view_m * lanes / jam_spacing_m


@dataclass
class ApproachSensor:
    """Everything we know about one approach, combining the three helpers above."""
    queue: QueueEstimator
    turns: TurnRatioLearner
    ambulance: AmbulanceConfirmer = field(default_factory=AmbulanceConfirmer)

    def movement_queues(self) -> Dict[str, float]:
        return {t: self.queue.estimate * r for t, r in self.turns.get().items()}
