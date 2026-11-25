"""Dynamic Traffic Signal Optimization Engine (SIH 2025 Standard).

Implements Webster's Optimum Cycle Length Formula with PCU (Passenger Car Unit) conversion,
dynamic green split allocation, emergency vehicle priority override (Green Corridor),
and comparative delay / emissions reduction analytics.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

# PCU equivalents for Indian urban roads, IRC:106-1990 (column for low share of each type).
PCU_WEIGHTS: Dict[str, float] = {
    "car": 1.0,
    "van": 1.4,            # light commercial vehicle
    "pickup": 1.4,
    "truck": 2.2,
    "bus": 2.2,
    "tram": 3.5,
    "tractor": 4.0,
    "auto": 1.2,           # auto-rickshaw
    "autorickshaw": 1.2,
    "motorcycle": 0.5,
    "scooter": 0.5,
    "bike": 0.5,
    "bicycle": 0.4,
    "cyclist": 0.4,
    "person": 0.0,         # pedestrians are served by the walk phase, not counted as traffic
    "pedestrian": 0.0,
    "emergency": 999.0,     # flags immediate priority
    "ambulance": 999.0,
    "fire_truck": 999.0,
    "police": 999.0,
}

# Signal Timing Constraints (Seconds)
DEFAULT_SATURATION_FLOW_PER_LANE = 1800.0  # PCU / hr / lane
DEFAULT_LOST_TIME_PER_PHASE = 4.0          # start-up lost time + amber clearance
MIN_GREEN_TIME = 15.0                      # pedestrian minimum clearance
MAX_GREEN_TIME = 90.0                      # prevent starvation on opposite approaches
MIN_CYCLE_TIME = 45.0
MAX_CYCLE_TIME = 150.0
YELLOW_TIME = 3.0
ALL_RED_TIME = 1.5
EMERGENCY_GREEN_WINDOW = 40.0              # immediate green corridor duration


IDLE_FUEL_L_PER_H = 0.8     # idling fuel of a car-equivalent (assumption, stated in the report)
CO2_KG_PER_L = 2.31         # petrol


def _webster_avg_delay(approaches: List[Dict[str, Any]], greens: List[float], cycle: float,
                       period_h: float = 0.25) -> float:
    """Flow-weighted average control delay per vehicle (s), HCM 2000 signalised-intersection model.

    uniform     d1 = 0.5 C (1 - g/C)^2 / (1 - min(1, X) g/C)
    incremental d2 = 900 T [ (X - 1) + sqrt((X - 1)^2 + 4 X / (c T)) ]
    with capacity c = s g/C (PCU/h), X = v/c, analysis period T = 0.25 h. Unlike Webster's original
    formula this stays valid when an approach is overloaded (X >= 1).
    """
    tot_q, tot_d = 0.0, 0.0
    for app, g in zip(approaches, greens):
        v = app["flow_q"]
        lam = max(0.05, g / cycle)
        cap = app["sat_s"] * lam
        x = v / cap if cap > 0 else 2.0
        d1 = 0.5 * cycle * (1 - lam) ** 2 / (1 - min(1.0, x) * lam)
        d2 = 900 * period_h * ((x - 1) + math.sqrt((x - 1) ** 2 + 4 * x / (cap * period_h)))
        tot_q += v
        tot_d += (d1 + d2) * v
    return tot_d / tot_q if tot_q > 0 else 0.0


def calculate_pcu(detections_or_counts: List[Dict[str, Any]] | Dict[str, int]) -> tuple[float, Dict[str, int], bool]:
    """Calculate total PCU, class counts, and emergency vehicle flag from detections or counts dict.
    
    detections: list of dicts with 'class_name' OR dict of {class_name: count}
    returns: (total_pcu, class_counts, has_emergency)
    """
    counts: Dict[str, int] = {}
    has_emergency = False

    if isinstance(detections_or_counts, list):
        for d in detections_or_counts:
            c = str(d.get("class_name", "car")).lower()
            counts[c] = counts.get(c, 0) + 1
            if c in ("emergency", "ambulance", "fire_truck", "police") or d.get("is_emergency", False):
                has_emergency = True
    elif isinstance(detections_or_counts, dict):
        for k, v in detections_or_counts.items():
            c = str(k).lower()
            counts[c] = counts.get(c, 0) + int(v)
            if c in ("emergency", "ambulance", "fire_truck", "police") and int(v) > 0:
                has_emergency = True

    total_pcu = 0.0
    for c, count in counts.items():
        if c in ("emergency", "ambulance", "fire_truck", "police"):
            total_pcu += count * 2.0  # nominal flow impact for calculation
        else:
            w = PCU_WEIGHTS.get(c, 1.0)
            total_pcu += count * w

    return round(total_pcu, 2), counts, has_emergency


def webster_optimum_cycle(
    approaches: List[Dict[str, Any]],
    lost_time_per_phase: float = DEFAULT_LOST_TIME_PER_PHASE,
    saturation_flow_per_lane: float = DEFAULT_SATURATION_FLOW_PER_LANE,
) -> Dict[str, Any]:
    """Compute Webster's Optimum Cycle and Dynamic Green Times for an intersection.

    Each approach dict can have:
    - 'name': str (e.g. 'Northbound', 'Southbound', 'Phase A', 'East-West')
    - 'detections': list of detection dicts OR 'counts': dict of class counts
    - 'lanes': int (default 2)
    - 'has_emergency': bool (optional explicit flag)
    - 'is_emergency': bool (optional)
    """
    n_phases = len(approaches)
    if n_phases < 2:
        raise ValueError("At least 2 approaches/phases required for signal optimization.")

    total_lost_time = n_phases * lost_time_per_phase
    processed_approaches = []
    emergency_approach_idx: Optional[int] = None

    for idx, app in enumerate(approaches):
        name = app.get("name", f"Approach-{idx+1}")
        lanes = max(1, int(app.get("lanes", 2)))
        raw_det = app.get("detections") or app.get("counts") or {}
        pcu, counts, emergency_detected = calculate_pcu(raw_det)
        
        # Explicit emergency flag override
        if app.get("has_emergency") or app.get("is_emergency") or emergency_detected:
            emergency_approach_idx = idx

        # Estimate arrival flow q_i in PCU/hr (instantaneous density scaled up to hourly rate)
        # Using typical urban queue scaling factor: 1 vehicle in detection zone ~ 120 PCU/hr equivalent
        flow_rate_q = max(20.0, pcu * 120.0)
        saturation_flow_s = lanes * saturation_flow_per_lane
        flow_ratio_y = flow_rate_q / saturation_flow_s

        processed_approaches.append({
            "name": name,
            "lanes": lanes,
            "pcu": pcu,
            "counts": counts,
            "flow_q": round(flow_rate_q, 1),
            "sat_s": round(saturation_flow_s, 1),
            "flow_ratio_y": round(flow_ratio_y, 4),
            "is_emergency": (emergency_approach_idx == idx),
        })

    # Emergency Vehicle Override (Green Corridor)
    if emergency_approach_idx is not None:
        em_name = processed_approaches[emergency_approach_idx]["name"]
        green_splits = {}
        for idx, app in enumerate(processed_approaches):
            if idx == emergency_approach_idx:
                green_splits[app["name"]] = EMERGENCY_GREEN_WINDOW
            else:
                green_splits[app["name"]] = 0.0  # hold red

        return {
            "mode": "EMERGENCY_OVERRIDE",
            "reason": f"GREEN_CORRIDOR_ACTIVE: Priority vehicle detected on {em_name}",
            "emergency_approach": em_name,
            "cycle_length": EMERGENCY_GREEN_WINDOW + total_lost_time,
            "total_lost_time": total_lost_time,
            "yellow_time": YELLOW_TIME,
            "all_red_time": ALL_RED_TIME,
            "phase_timings": [
                {
                    "approach": app["name"],
                    "green_time": green_splits[app["name"]],
                    "yellow_time": YELLOW_TIME if app["is_emergency"] else 0.0,
                    "red_time": 0.0 if app["is_emergency"] else (EMERGENCY_GREEN_WINDOW + YELLOW_TIME),
                    "is_active_green": app["is_emergency"],
                    "pcu": app["pcu"],
                    "counts": app["counts"],
                }
                for app in processed_approaches
            ],
        }

    # Standard Webster Optimization
    total_y = sum(app["flow_ratio_y"] for app in processed_approaches)
    # Cap total_y at 0.85 to avoid mathematical singularity
    effective_y = min(0.85, max(0.10, total_y))

    # Webster's Formula: C_o = (1.5 * L + 5) / (1 - Y)
    calculated_c_o = (1.5 * total_lost_time + 5.0) / (1.0 - effective_y)
    optimum_cycle = round(min(MAX_CYCLE_TIME, max(MIN_CYCLE_TIME, calculated_c_o)), 1)
    available_green = optimum_cycle - total_lost_time

    # Allocate green times proportionally: G_i = (y_i / Y) * available_green
    raw_greens = []
    for app in processed_approaches:
        ratio = (app["flow_ratio_y"] / effective_y) if effective_y > 0 else (1.0 / n_phases)
        g = ratio * available_green
        g_clamped = min(MAX_GREEN_TIME, max(MIN_GREEN_TIME, g))
        raw_greens.append(g_clamped)

    # Normalize green times so their sum equals available_green
    sum_greens = sum(raw_greens)
    scale = available_green / sum_greens if sum_greens > 0 else 1.0
    normalized_greens = [round(max(MIN_GREEN_TIME, g * scale), 1) for g in raw_greens]
    final_cycle = round(sum(normalized_greens) + total_lost_time, 1)

    # Benefits vs a fixed timer with equal greens, using the HCM 2000 control-delay model
    # (uniform + random delay terms). These are model estimates for this junction only; the
    # network simulator (network_sim.py) gives the measured multi-junction comparison.
    static_cycle = 60.0 * (n_phases / 2.0)
    static_green = (static_cycle - total_lost_time) / n_phases
    greens = normalized_greens
    d_static = _webster_avg_delay(processed_approaches, [static_green] * n_phases, static_cycle)
    d_dynamic = _webster_avg_delay(processed_approaches, greens, final_cycle)
    delay_reduction_pct = round(max(0.0, (d_static - d_dynamic) / d_static * 100.0), 1) if d_static > 0 else 0.0
    total_flow = sum(a["flow_q"] for a in processed_approaches)              # PCU/h
    saved_veh_hours = max(0.0, d_static - d_dynamic) * total_flow / 3600.0   # per hour of operation
    fuel_saved_liters = round(saved_veh_hours * IDLE_FUEL_L_PER_H, 2)
    co2_saved_kg = round(fuel_saved_liters * CO2_KG_PER_L, 2)

    phase_timings = []
    for idx, app in enumerate(processed_approaches):
        g_time = normalized_greens[idx]
        r_time = round(final_cycle - g_time - YELLOW_TIME, 1)
        phase_timings.append({
            "approach": app["name"],
            "lanes": app["lanes"],
            "pcu": app["pcu"],
            "counts": app["counts"],
            "flow_q_pcu_hr": app["flow_q"],
            "flow_ratio_y": app["flow_ratio_y"],
            "green_time": g_time,
            "yellow_time": YELLOW_TIME,
            "all_red_time": ALL_RED_TIME,
            "red_time": max(0.0, r_time),
            "green_ratio": round(g_time / final_cycle, 3),
        })

    # Sort phase sequence: heaviest flow gets green first to clear congestion bottleneck
    phase_order = sorted(range(n_phases), key=lambda i: processed_approaches[i]["pcu"], reverse=True)
    sequence = [processed_approaches[i]["name"] for i in phase_order]

    return {
        "mode": "WEBSTER_DYNAMIC_OPTIMIZED",
        "cycle_length": final_cycle,
        "total_lost_time": total_lost_time,
        "critical_flow_ratio_Y": round(total_y, 4),
        "saturation_status": "HIGH" if total_y >= 0.75 else ("MODERATE" if total_y >= 0.45 else "LOW"),
        "phase_timings": phase_timings,
        "recommended_phase_sequence": sequence,
        "benefits": {
            "baseline_fixed_cycle_seconds": static_cycle,
            "delay_reduction_pct": delay_reduction_pct,
            "fuel_saved_liters_per_hour": fuel_saved_liters,
            "co2_saved_kg_per_hour": co2_saved_kg,
        },
    }

