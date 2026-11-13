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
    # Demand exceeds capacity here (Y > 1), so the uniform-delay model is invalid: no savings claimed.
    assert res["benefits"]["oversaturated"] is True
    assert res["benefits"]["delay_reduction_pct"] is None


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


def test_multiple_emergencies_are_served_in_sequence():
    """Two approaches reporting an emergency must both get green (in turn); none may be flagged green with 0 s."""
    approaches = [
        {"name": "N", "counts": {"car": 3}, "has_emergency": True},
        {"name": "E", "counts": {"car": 3}, "has_emergency": True},
        {"name": "S", "counts": {"car": 3}},
    ]
    res = webster_optimum_cycle(approaches)
    assert res["mode"] == "EMERGENCY_OVERRIDE"
    assert res["emergency_approaches"] == ["N", "E"]
    assert res["emergency_approach"] == "N"
    by_name = {p["approach"]: p for p in res["phase_timings"]}
    for name in ("N", "E"):
        assert by_name[name]["is_active_green"] is True and by_name[name]["green_time"] > 0
    assert by_name["S"]["is_active_green"] is False and by_name["S"]["green_time"] == 0
    # every approach flagged active green actually has green time
    assert all(p["green_time"] > 0 for p in res["phase_timings"] if p["is_active_green"])
    assert res["cycle_length"] == 2 * 40.0 + res["total_lost_time"]


def test_single_emergency_output_unchanged():
    res = webster_optimum_cycle([{"name": "N", "counts": {"car": 3}}, {"name": "E", "counts": {"ambulance": 1}}])
    assert res["emergency_approaches"] == ["E"]
    east = next(p for p in res["phase_timings"] if p["approach"] == "E")
    assert east["red_time"] == 0.0 and east["is_active_green"] is True


def test_delay_savings_computed_from_webster_delay_model():
    # Unbalanced, below capacity: Webster's plan should beat the equal-split fixed plan by a real margin.
    res = webster_optimum_cycle([{"name": "A", "counts": {"car": 9}}, {"name": "B", "counts": {"car": 2}}])
    b = res["benefits"]
    assert b["oversaturated"] is False
    assert b["avg_delay_s_dynamic"] < b["avg_delay_s_fixed"]
    expected = round((b["avg_delay_s_fixed"] - b["avg_delay_s_dynamic"]) / b["avg_delay_s_fixed"] * 100, 1)
    assert abs(b["delay_reduction_pct"] - expected) < 0.2          # rounding of the two averages
    assert b["fuel_saved_liters_per_hour"] > 0
    assert b["basis"] == "analytical_webster_uniform_delay"


def test_savings_not_a_constant_floor():
    """Regression: the old heuristic clamped savings to 15-55%, even for an almost empty junction."""
    light = webster_optimum_cycle([{"name": "A", "counts": {"car": 5}}, {"name": "B", "counts": {"car": 5}}])
    skewed = webster_optimum_cycle([{"name": "A", "counts": {"car": 9}}, {"name": "B", "counts": {"car": 2}}])
    assert light["benefits"]["delay_reduction_pct"] != skewed["benefits"]["delay_reduction_pct"]


def test_oversaturated_reports_no_savings():
    res = webster_optimum_cycle([{"name": "A", "counts": {"car": 200}}, {"name": "B", "counts": {"car": 200}}])
    b = res["benefits"]
    assert b["oversaturated"] is True
    assert b["delay_reduction_pct"] is None and b["fuel_saved_liters_per_hour"] is None
