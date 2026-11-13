"""Dynamic Traffic Signal Optimization Engine (SIH 2025 Standard).

Implements Webster's Optimum Cycle Length Formula with PCU (Passenger Car Unit) conversion,
dynamic green split allocation, emergency vehicle priority override (Green Corridor),
and comparative delay / emissions reduction analytics.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

# Indian Roads Congress (IRC: 106-1990) & International PCU Equivalents
PCU_WEIGHTS: Dict[str, float] = {
    "car": 1.0,
    "van": 1.5,
    "truck": 3.0,
    "tram": 3.5,
    "bus": 3.0,
    "motorcycle": 0.5,
    "bike": 0.5,
    "cyclist": 0.2,
    "pedestrian": 0.2,
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
FIXED_PLAN_CYCLE_PER_PHASE = 30.0          # baseline fixed-time plan: 30 s of cycle per phase
MAX_DEGREE_OF_SATURATION = 0.95            # x is capped here; the uniform-delay formula breaks down at x >= 1
IDLE_FUEL_L_PER_HR = 0.8
CO2_KG_PER_LITRE = 2.31


def _uniform_delay(cycle: float, green: float, flow: float, sat_flow: float) -> tuple[float, bool]:
    """Webster uniform delay per vehicle (s): d = c(1-λ)^2 / (2(1-λx)), λ = g/c, x = q/(sλ).

    Returns (delay, oversaturated). x is capped at MAX_DEGREE_OF_SATURATION so the result stays finite;
    ``oversaturated`` tells the caller that cap was hit, i.e. real delay would be higher than reported.
    """
    lam = green / cycle
    capacity = sat_flow * lam
    x = flow / capacity if capacity > 0 else MAX_DEGREE_OF_SATURATION
    over = x >= 1.0
    x = min(x, MAX_DEGREE_OF_SATURATION)
    return cycle * (1.0 - lam) ** 2 / (2.0 * (1.0 - lam * x)), over


def _plan_delays(cycle: float, greens: List[float], flows: List[float], sats: List[float]) -> Dict[str, Any]:
    results = [_uniform_delay(cycle, g, q, s) for g, q, s in zip(greens, flows, sats)]
    return {"delays": [d for d, _ in results], "oversaturated": any(o for _, o in results)}


def _flow_weighted(delays: List[float], flows: List[float]) -> float:
    total = sum(flows)
    return sum(d * q for d, q in zip(delays, flows)) / total if total > 0 else 0.0


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

    for idx, app in enumerate(approaches):
        name = app.get("name", f"Approach-{idx+1}")
        lanes = max(1, int(app.get("lanes", 2)))
        raw_det = app.get("detections") or app.get("counts") or {}
        pcu, counts, emergency_detected = calculate_pcu(raw_det)

        # This approach has an emergency vehicle (explicit flag or detected class)
        approach_has_emergency = bool(app.get("has_emergency") or app.get("is_emergency") or emergency_detected)

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
            "is_emergency": approach_has_emergency,
        })

    # Emergency Vehicle Override (Green Corridor)
    # If several approaches report an emergency vehicle at once they cannot all be green (they conflict),
    # so each is served in turn, in input order, for EMERGENCY_GREEN_WINDOW seconds. Everything else holds red.
    emergency_names = [a["name"] for a in processed_approaches if a["is_emergency"]]
    if emergency_names:
        n_em = len(emergency_names)
        if n_em == 1:
            reason = f"GREEN_CORRIDOR_ACTIVE: Priority vehicle detected on {emergency_names[0]}"
        else:
            reason = (f"GREEN_CORRIDOR_ACTIVE: Priority vehicles on {', '.join(emergency_names)}; "
                      f"served in sequence ({EMERGENCY_GREEN_WINDOW:g}s each)")

        return {
            "mode": "EMERGENCY_OVERRIDE",
            "reason": reason,
            "emergency_approach": emergency_names[0],          # first in service order (kept for compatibility)
            "emergency_approaches": emergency_names,           # all, in service order
            "cycle_length": EMERGENCY_GREEN_WINDOW * n_em + total_lost_time,
            "total_lost_time": total_lost_time,
            "yellow_time": YELLOW_TIME,
            "all_red_time": ALL_RED_TIME,
            "phase_timings": [
                {
                    "approach": app["name"],
                    "lanes": app["lanes"],
                    "green_time": EMERGENCY_GREEN_WINDOW if app["is_emergency"] else 0.0,
                    "yellow_time": YELLOW_TIME if app["is_emergency"] else 0.0,
                    # an emergency approach waits while the other emergency approaches are served
                    "red_time": ((n_em - 1) * EMERGENCY_GREEN_WINDOW if app["is_emergency"]
                                 else n_em * EMERGENCY_GREEN_WINDOW + YELLOW_TIME),
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

    # Comparative benefits vs a fixed-time plan (analytical, using Webster's uniform delay formula).
    # Baseline: 30 s of cycle per phase, split equally between approaches.
    static_cycle = FIXED_PLAN_CYCLE_PER_PHASE * n_phases
    static_green = (static_cycle - total_lost_time) / n_phases
    flows = [a["flow_q"] for a in processed_approaches]
    sats = [a["sat_s"] for a in processed_approaches]
    dyn_delay = _plan_delays(final_cycle, normalized_greens, flows, sats)
    fixed_delay = _plan_delays(static_cycle, [static_green] * n_phases, flows, sats)
    oversaturated = dyn_delay["oversaturated"] or fixed_delay["oversaturated"]

    avg_dyn = avg_fixed = delay_reduction_pct = fuel_saved_liters = co2_saved_kg = None
    if not oversaturated:
        # The uniform-delay formula is only valid below capacity (x < 1). Above it queues grow without bound, so
        # we report NO savings figure rather than a made-up one.
        avg_dyn = round(_flow_weighted(dyn_delay["delays"], flows), 1)
        avg_fixed = round(_flow_weighted(fixed_delay["delays"], flows), 1)
        delay_reduction_pct = round((avg_fixed - avg_dyn) / avg_fixed * 100.0, 1) if avg_fixed > 0 else 0.0

        # Fuel & CO2: vehicle-seconds of delay saved per hour, burned at an idling rate of ~0.8 L/hr
        # (2.31 kg CO2/L, petrol). Signed: negative means the optimized plan is worse than the fixed one.
        veh_s_saved = sum(q * (df - dd) for q, df, dd in zip(flows, fixed_delay["delays"], dyn_delay["delays"]))
        fuel_saved_liters = round(veh_s_saved / 3600.0 * IDLE_FUEL_L_PER_HR, 2)
        co2_saved_kg = round(fuel_saved_liters * CO2_KG_PER_LITRE, 2)

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
            "avg_delay_s_dynamic": avg_dyn,
            "avg_delay_s_fixed": avg_fixed,
            "oversaturated": oversaturated,   # True -> savings fields are None (model invalid above capacity)
            # Analytical (Webster uniform-delay) comparison on the *estimated* flows, NOT a simulation or field
            # measurement. Flows come from detected counts x a fixed scaling factor, so validate before quoting.
            "basis": "analytical_webster_uniform_delay",
        },
    }

