"""In-memory live state for camera-driven (photo / video) junction control.

Keeps the last camera frame time per junction (watchdog), pending pedestrian requests and planned
green corridors. Single-process demo state; a deployment would keep this in Redis or the database.
"""
from __future__ import annotations

import threading
import time
from functools import lru_cache
from typing import Dict, List, Optional

from decision_engine.network_control import EmergencyManager, MAX_PED_WAIT_S
from decision_engine.network_model import Network, build_network
from decision_engine.network_sim import load_config
from decision_engine.queue_estimator import STALE_AFTER_S, AmbulanceConfirmer


@lru_cache(maxsize=4)
def get_network(name: str = "vikas_marg") -> Network:
    return build_network(load_config()["networks"][name])


class LiveNetworkState:
    def __init__(self, network: str = "vikas_marg"):
        self.network = network
        self.lock = threading.Lock()
        self.last_frame: Dict[str, float] = {}
        self.ped_requests: Dict[str, float] = {}
        self.corridors: List[dict] = []
        self.ambulance: Dict[tuple, AmbulanceConfirmer] = {}

    def ambulance_seen(self, junction_id: str, approach: str, conf: Optional[float], colour_hint: bool = False) -> dict:
        """Photo/video frames per approach; confirmed only after 2 frames in a row with conf >= 0.6."""
        with self.lock:
            c = self.ambulance.setdefault((junction_id, approach), AmbulanceConfirmer())
            confirmed = c.update(conf, colour_hint)
            return {"approach": approach, "confidence": conf, "streak": c.streak, "confirmed": confirmed,
                    "colour_hint_only": bool(colour_hint and (conf is None or conf < 0.6))}

    def camera_frame(self, junction_id: str) -> None:
        with self.lock:
            self.last_frame[junction_id] = time.time()

    def press_ped(self, junction_id: str) -> dict:
        with self.lock:
            t0 = self.ped_requests.setdefault(junction_id, time.time())
        return {"junction": junction_id, "requested_at": t0,
                "served_within_s": MAX_PED_WAIT_S,
                "note": "walk phase is inserted at the next phase change, at the latest after "
                        f"{MAX_PED_WAIT_S * 0.5:.0f}s of waiting"}

    def plan_corridor(self, route: List[str], from_link: Optional[str], source: str) -> dict:
        net = get_network(self.network)
        bad = [l for l in route if l not in net.links]
        if bad:
            raise ValueError(f"unknown links in route: {bad}")
        em = EmergencyManager(net)
        plans = em.trigger(0.0, len(self.corridors) + 1, route, from_link or route[0], source)
        out = {"source": source, "created_at": time.time(),
               "junctions": [{"junction": p.junction, "approach": p.approach, "turn": p.turn,
                              "eta_s": round(p.eta_t, 1), "green_from_s": round(max(0.0, p.eta_t - 12.0), 1)}
                             for p in plans]}
        with self.lock:
            self.corridors.append(out)
        return out

    def health(self) -> dict:
        now = time.time()
        net = get_network(self.network)
        rows = []
        for jid, j in net.junctions.items():
            last = self.last_frame.get(jid)
            stale = last is None or now - last > STALE_AFTER_S
            rows.append({"junction": jid, "name": j.name,
                         "last_frame_age_s": None if last is None else round(now - last, 1),
                         "mode": "fixed_time_fallback" if stale else "adaptive",
                         "pedestrian_waiting": jid in self.ped_requests})
        return {"stale_after_s": STALE_AFTER_S, "junctions": rows, "active_corridors": self.corridors[-5:]}


LIVE = LiveNetworkState()
