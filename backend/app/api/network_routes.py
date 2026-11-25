"""Multi-junction network control endpoints (simulation, live camera decisions, green corridor)."""
from __future__ import annotations

import json
import os
import tempfile
from typing import Literal, Optional

import cv2
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from decision_engine import calculate_pcu
from decision_engine.network_control import explain_from_counts
from decision_engine.network_sim import list_scenarios, run_comparison, run_scenario
from decision_engine.queue_estimator import QueueEstimator, camera_range_pcu

from ..core.config import settings
from ..db import models as M
from ..services.live_network import LIVE, get_network
from ..services.model_service import ModelRegistry, annotate
from . import schemas as S
from .deps import ROOT, get_db, get_registry
from .routes import MAX_UPLOAD, _counts, _decode

router = APIRouter(prefix="/network", tags=["network"])
DIRS4 = ("N", "E", "S", "W")
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")


@router.get("/scenarios")
def network_scenarios():
    return {"scenarios": list_scenarios(),
            "controllers": [{"id": "fixed", "label": "Fixed timer (today)"},
                            {"id": "webster", "label": "Webster (one junction)"},
                            {"id": "max_pressure", "label": "Max-Pressure (network)"},
                            {"id": "rl", "label": "AI (reinforcement learning)"}],
            "network": get_network("vikas_marg").to_dict()}


@router.post("/simulate")
def network_simulate(body: S.NetworkSimulateIn):
    try:
        return run_scenario(body.scenario, body.controller, body.seed, duration_s=body.duration_s,
                            corridor=body.corridor)
    except KeyError as e:
        raise HTTPException(404, str(e))


@router.post("/compare")
def network_compare(body: S.NetworkCompareIn):
    try:
        return run_comparison(body.scenario, tuple(body.controllers), range(body.seeds), duration_s=body.duration_s)
    except KeyError as e:
        raise HTTPException(404, str(e))


@router.get("/rl-status")
def rl_status():
    """Whether the RL agent beat Max-Pressure on held-out days (and so may be deployed)."""
    p = ROOT / "models" / "rl_eval.json"
    if not p.exists():
        raise HTTPException(404, "RL agent not evaluated yet: run `python ml/training/train_rl.py`")
    return json.loads(p.read_text(encoding="utf-8"))


@router.get("/report")
def network_report():
    p = ROOT / "reports" / "tables" / "network_comparison.json"
    if not p.exists():
        raise HTTPException(404, "run `python scripts/make_network_report.py` first")
    return json.loads(p.read_text(encoding="utf-8"))


def _junction(jid: str):
    net = get_network("vikas_marg")
    if jid not in net.junctions:
        raise HTTPException(404, f"junction {jid} not found; use one of {sorted(net.junctions)}")
    return net.junctions[jid]


@router.post("/decide")
def network_decide(body: S.NetworkDecideIn):
    """Camera counts per approach -> next phase and green time (one Max-Pressure step, explained)."""
    j = _junction(body.junction_id)
    pcu, lanes, down = {}, {}, {}
    for a, ap in body.approaches.items():
        pcu[a] = ap.pcu if ap.pcu is not None else calculate_pcu(ap.counts)[0]
        lanes[a] = ap.lanes
        down[a] = ap.down_queue_pcu
    LIVE.camera_frame(body.junction_id)
    return explain_from_counts(j, pcu, lanes, body.current_phase, body.green_elapsed_s, down)


def _vehicle_pcu(counts: dict) -> float:
    return calculate_pcu({k: v for k, v in counts.items() if k != "person"})[0]


@router.post("/decide-photos")
def network_decide_photos(junction_id: str = Query("J2"), conf: float = Query(0.25, ge=0.05, le=0.95),
                          north: Optional[UploadFile] = File(None), east: Optional[UploadFile] = File(None),
                          south: Optional[UploadFile] = File(None), west: Optional[UploadFile] = File(None),
                          registry: ModelRegistry = Depends(get_registry)):
    """One photo per approach -> YOLO counts -> PCU queue -> Max-Pressure decision for this junction."""
    j = _junction(junction_id)
    net = get_network("vikas_marg")
    files = dict(zip(DIRS4, (north, east, south, west)))
    approaches, pcu, lanes = {}, {}, {}
    for a, f in files.items():
        if a not in j.in_links:
            continue
        link = net.links[j.in_links[a]]
        lanes[a] = link.lanes
        if f is None:
            pcu[a] = 0.0
            approaches[a] = {"counts": {}, "pcu": 0.0, "n_vehicles": 0, "persons": 0, "image": None, "ambulance": None}
            continue
        img = _decode(f)
        dets, ms = registry.detect("vehicle", img, conf)
        counts = _counts(dets)
        p = _vehicle_pcu(counts)
        amb_conf = max((d["confidence"] for d in dets if d["class_name"] == "ambulance"), default=None)
        hint = any(d.get("emergency_color_hint") for d in dets)
        pcu[a] = p
        approaches[a] = {"counts": counts, "pcu": p,
                         "n_vehicles": sum(v for k, v in counts.items() if k != "person"),
                         "persons": counts.get("person", 0), "inference_ms": round(ms, 1),
                         "beyond_camera": p >= 0.9 * camera_range_pcu(link.lanes),
                         "ambulance": LIVE.ambulance_seen(junction_id, a, amb_conf, hint),
                         "image": annotate(img, dets, (0, 200, 0))}
    LIVE.camera_frame(junction_id)
    decision = explain_from_counts(j, pcu, lanes)
    confirmed = [a for a, v in approaches.items() if v.get("ambulance") and v["ambulance"]["confirmed"]]
    if confirmed:
        a = confirmed[0]
        idx = next(i for i in j.vehicle_phases() if (a, "S") in j.phases[i].movements)
        decision.update(recommended_phase=idx, phase_name=j.phases[idx].name,
                        explanation=f"GREEN CORRIDOR: ambulance confirmed on {a} in 2 frames -> "
                                    f"'{j.phases[idx].name}' green next (after a safe yellow + all-red)")
    elif any(v.get("ambulance") and v["ambulance"]["confidence"] for v in approaches.values()):
        decision["explanation"] += " | Ambulance seen once: upload a 2nd frame (or send a GPS request) to confirm."
    if any(v.get("persons") for v in approaches.values()):
        LIVE.press_ped(junction_id)
    return {"junction": {"id": j.id, "name": j.name}, "model": registry.model_name("vehicle"),
            "approaches": approaches, "decision": decision}


@router.post("/decide-video")
def network_decide_video(approach: Literal["N", "E", "S", "W"] = Query("N"), junction_id: str = Query("J2"),
                         max_frames: int = Query(20, ge=2, le=60), conf: float = Query(0.25, ge=0.05, le=0.95),
                         file: UploadFile = File(...), registry: ModelRegistry = Depends(get_registry)):
    """Short video of one approach, sampled at 1 frame/s -> queue over time + early-switch evidence."""
    j = _junction(junction_id)
    if approach not in j.in_links:
        raise HTTPException(400, f"{junction_id} has no {approach} approach")
    name = (file.filename or "").lower()
    if not name.endswith(VIDEO_EXT):
        raise HTTPException(400, f"video must be one of {VIDEO_EXT}")
    data = file.file.read(MAX_UPLOAD * 4 + 1)
    if len(data) > MAX_UPLOAD * 4:
        raise HTTPException(413, f"video larger than {settings.max_upload_mb * 4} MB")
    link = get_network("vikas_marg").links[j.in_links[approach]]
    est = QueueEstimator(camera_range_pcu(link.lanes))
    series = []
    fd, path = tempfile.mkstemp(suffix=os.path.splitext(name)[1])
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(fps)))
        k = idx = 0
        while len(series) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                dets, _ = registry.detect("vehicle", frame, conf)
                counts = _counts(dets)
                p = _vehicle_pcu(counts)
                mean_conf = (sum(d["confidence"] for d in dets) / len(dets)) if dets else 0.9
                est.update(float(k), p, mean_conf)
                amb_conf = max((d["confidence"] for d in dets if d["class_name"] == "ambulance"), default=None)
                amb = LIVE.ambulance_seen(junction_id, approach, amb_conf)
                series.append({"t": k, "pcu": round(p, 1), "estimate_pcu": round(est.estimate, 1),
                               "confidence": round(mean_conf, 2), "empty_confirmed": est.confirmed_empty(),
                               "beyond_camera": est.beyond_camera, "counts": counts, "ambulance": amb})
                k += 1
            idx += 1
        cap.release()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    if not series:
        raise HTTPException(400, "could not read any frames from the video")
    LIVE.camera_frame(junction_id)
    return {"junction": junction_id, "approach": approach, "frames": series,
            "final_queue_pcu": series[-1]["estimate_pcu"], "early_switch_ready": series[-1]["empty_confirmed"]}


@router.post("/emergency")
def network_emergency(body: S.EmergencyIn, db: Session = Depends(get_db)):
    """GPS/RFID ambulance request: plan green at every junction ahead of the ambulance."""
    try:
        plan = LIVE.plan_corridor(body.route, body.from_link, body.source)
    except ValueError as e:
        raise HTTPException(400, str(e))
    db.add(M.Alert(kind="emergency_vehicle_green_corridor", severity="critical",
                   message="Green corridor planned through " + " -> ".join(p["junction"] for p in plan["junctions"]),
                   payload=plan, simulated=True))
    db.commit()
    return plan


@router.post("/pedestrian-request")
def network_pedestrian(body: S.PedestrianIn):
    _junction(body.junction_id)
    return LIVE.press_ped(body.junction_id)


@router.get("/health")
def network_health():
    return LIVE.health()
