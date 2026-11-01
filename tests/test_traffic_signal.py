"""Tests for Dynamic Traffic Signal Optimization (Webster's method) and Repair Prioritisation."""
from decision_engine.traffic_signal import calculate_pcu, webster_optimum_cycle
from decision_engine.repair_priority import calculate_repair_priority, rank_segments_for_repair


def test_calculate_pcu_weights():
    counts = {"car": 10, "truck": 2, "van": 2, "motorcycle": 4}
    # 10*1.0 + 2*3.0 + 2*1.5 + 4*0.5 = 10 + 6 + 3 + 2 = 21.0 PCU
    pcu, class_counts, has_emergency = calculate_pcu(counts)
    assert pcu == 21.0
    assert class_counts["car"] == 10
    assert not has_emergency


def test_emergency_vehicle_flag():
    counts = {"car": 5, "ambulance": 1}
    pcu, class_counts, has_emergency = calculate_pcu(counts)
    assert has_emergency is True


def test_webster_optimum_cycle_calculation():
    approaches = [
        {"name": "North-South", "lanes": 2, "counts": {"car": 15, "truck": 2}},
        {"name": "East-West", "lanes": 2, "counts": {"car": 8, "motorcycle": 4}},
    ]
    res = webster_optimum_cycle(approaches)
    assert res["mode"] == "WEBSTER_DYNAMIC_OPTIMIZED"
    assert res["cycle_length"] >= 45.0
    assert len(res["phase_timings"]) == 2
    # North-South has higher traffic, should get more or equal green time
    ns_green = res["phase_timings"][0]["green_time"]
    ew_green = res["phase_timings"][1]["green_time"]
    assert ns_green >= ew_green
    assert res["benefits"]["delay_reduction_pct"] > 0


def test_emergency_green_corridor_override():
    approaches = [
        {"name": "Northbound", "lanes": 2, "counts": {"car": 25}},
        {"name": "Eastbound", "lanes": 2, "counts": {"car": 2, "ambulance": 1}, "has_emergency": True},
    ]
    res = webster_optimum_cycle(approaches)
    assert res["mode"] == "EMERGENCY_OVERRIDE"
    assert res["emergency_approach"] == "Eastbound"
    # Eastbound should have active green
    east_timing = next(p for p in res["phase_timings"] if p["approach"] == "Eastbound")
    north_timing = next(p for p in res["phase_timings"] if p["approach"] == "Northbound")
    assert east_timing["green_time"] > 0
    assert north_timing["green_time"] == 0


def test_repair_priority_scoring():
    res = calculate_repair_priority(
        damage_score=75.0,
        traffic_importance=3.5,
        defect_counts={"pothole": 4, "alligator_crack": 2},
        segment_name="MG Road",
        segment_id=1,
    )
    assert res["priority_score"] >= 75.0
    assert res["priority_level"] == "CRITICAL"
    assert res["requires_alert"] is True


def test_rank_segments_for_repair():
    segments = [
        {"segment_id": 1, "name": "Minor Alley", "damage_score": 80.0, "traffic_importance": 0.5, "defect_counts": {}},
        {"segment_id": 2, "name": "Highway Arterial", "damage_score": 70.0, "traffic_importance": 4.5, "defect_counts": {"pothole": 3}},
    ]
    ranked = rank_segments_for_repair(segments)
    assert len(ranked) == 2
    # Highway with potholes and high importance should be ranked #1
    assert ranked[0]["segment_id"] == 2
