"""Tests for multi-junction Max-Pressure control, the safety layer and the simulator."""
import pytest

from decision_engine.network_control import (
    MAX_RED_S,
    MIN_GREEN_S,
    EmergencyManager,
    JunctionView,
    MaxPressureController,
    MovementView,
    SafetySupervisor,
    SignalHead,
    explain_from_counts,
)
from decision_engine.network_model import (
    Junction,
    Phase,
    PhaseConflictError,
    build_network,
    exit_dir,
    movements_conflict,
    validate_phases,
)
from decision_engine.network_sim import Simulation, load_config, run_scenario
from decision_engine.queue_estimator import AmbulanceConfirmer, QueueEstimator, TurnRatioLearner


@pytest.fixture(scope="module")
def corridor():
    return build_network(load_config()["networks"]["vikas_marg"])


def _mv(appr, turn, q, down=0.0, storage=100.0, empty=False):
    return MovementView(appr, turn, f"in_{appr}", f"out_{appr}{turn}", q, 1.0, down, storage, empty)


def _view(j, cur, elapsed, queues, red=None, empty=(), down=None):
    mv = {}
    for p in j.phases:
        for (a, t) in p.movements:
            mv[(a, t)] = _mv(a, t, queues.get(a, 0.0) / 3, (down or {}).get(a, 0.0), 100.0, a in empty)
    return JunctionView(j, 100.0, cur, elapsed, mv, red or {i: 0.0 for i in range(len(j.phases))})


# ------------------------------------------------------------------ geometry & phases
def test_left_hand_traffic_exits():
    assert exit_dir("N", "L") == "E"      # coming from the north, a left turn goes east
    assert exit_dir("N", "S") == "S"
    assert exit_dir("N", "R") == "W"


def test_phase_plans_have_no_conflicts(corridor):
    for j in corridor.junctions.values():
        validate_phases(j)                 # raises if two crossing movements share a green


def test_conflicting_phase_is_rejected():
    assert movements_conflict(("N", "S"), ("E", "S"))
    j = Junction("X", "x", 0, 0, 0, 0, [Phase("bad", [("N", "S"), ("E", "S")])])
    with pytest.raises(PhaseConflictError):
        validate_phases(j)


# ------------------------------------------------------------------ max-pressure decisions
def test_hundred_vehicles_get_green_over_two(corridor):
    j = corridor.junctions["J2"]
    out = explain_from_counts(j, {"N": 80.0, "S": 5.0, "E": 1.0, "W": 1.0}, {"N": 2, "S": 2, "E": 3, "W": 3})
    assert out["phase_name"] == "N-S through"
    assert out["green_s"] > MIN_GREEN_S


def test_no_green_into_full_road():
    full = _mv("N", "S", q=40, down=90, storage=100)       # downstream 90 % full
    free = _mv("E", "S", q=10, down=0, storage=100)
    assert full.blocked and full.pressure == 0.0
    assert free.pressure > 0


def test_early_switch_only_after_confirmed_empty(corridor):
    j = corridor.junctions["J2"]
    ctrl = MaxPressureController()
    through_ew = 2
    # current E-W through is empty (confirmed twice), N-S has a queue -> switch
    v = _view(j, through_ew, 15.0, {"N": 30, "S": 20, "E": 0, "W": 0}, empty=("E", "W"))
    nxt, why = ctrl.choose(v)
    assert nxt != through_ew and "early switch" in why
    # same but not yet confirmed empty and a real queue -> keep clearing
    ctrl2 = MaxPressureController()
    v2 = _view(j, through_ew, 15.0, {"N": 5, "S": 5, "E": 30, "W": 30})
    assert ctrl2.choose(v2)[0] == through_ew


def test_queue_estimator_two_checks_and_low_confidence():
    q = QueueEstimator(camera_range_pcu=20)
    q.update(0, 0.0, confidence=0.9)
    assert not q.confirmed_empty()
    q.update(2, 0.0, confidence=0.9)
    assert q.confirmed_empty()
    q.update(4, 0.0, confidence=0.3)            # dark / rainy frame: never trust "empty"
    assert not q.confirmed_empty()


def test_queue_longer_than_camera_view():
    q = QueueEstimator(camera_range_pcu=20)
    q.update(0, 20.0, 0.9, arrivals_pcu=60.0)   # camera full to the far edge; 60 PCU arrived upstream
    assert q.beyond_camera and q.estimate >= 60.0


def test_turn_ratio_learning_moves_towards_truth():
    tl = TurnRatioLearner({"L": 0.6, "S": 0.2, "R": 0.2}, alpha=0.05)
    for _ in range(300):
        tl.observe("S")
    assert tl.get()["S"] > 0.9


# ------------------------------------------------------------------ safety layer
def _supervisor(corridor):
    return SafetySupervisor(corridor, MaxPressureController(), EmergencyManager(corridor))


def test_fairness_serves_small_road_within_max_red(corridor):
    j = corridor.junctions["J2"]
    sup = _supervisor(corridor)
    head = SignalHead(j, 4.0, phase=2, green_t0=100.0 - 30.0)
    red = {i: 0.0 for i in range(len(j.phases))}
    red[0] = MAX_RED_S + 5                    # N-S through has 2 vehicles, red too long
    v = _view(j, 2, 30.0, {"N": 2, "S": 0, "E": 60, "W": 60}, red=red)
    d = sup.decide(head, v)
    assert d.phase == 0 and d.mode == "fairness"


def test_min_green_is_never_cut(corridor):
    j = corridor.junctions["J2"]
    sup = _supervisor(corridor)
    head = SignalHead(j, 4.0, phase=2, green_t0=95.0)
    v = _view(j, 2, 5.0, {"N": 90, "S": 90, "E": 0, "W": 0}, empty=("E", "W"))
    assert sup.decide(head, v).phase == 2


def test_pedestrian_request_is_served(corridor):
    j = corridor.junctions["J2"]
    sup = _supervisor(corridor)
    sup.press_ped_button("J2", -10.0)          # waited 110 s: must walk before the 120 s guarantee
    head = SignalHead(j, 4.0, phase=0, green_t0=80.0)
    v = _view(j, 0, 20.0, {"N": 10, "S": 10, "E": 10, "W": 10})
    d = sup.decide(head, v)
    assert j.phases[d.phase].pedestrian


def test_signal_head_always_goes_through_yellow_and_all_red(corridor):
    j = corridor.junctions["J1"]
    h = SignalHead(j, 4.0, phase=0)
    assert h.request(20, 2, "test")
    assert h.state == "yellow"
    h.tick(23)
    assert h.state == "yellow"
    h.tick(24)
    assert h.state == "allred" and h.green_movements() == []
    h.tick(26)
    assert h.state == "green" and h.phase == 2


# ------------------------------------------------------------------ ambulance
def test_ambulance_needs_two_confident_frames():
    a = AmbulanceConfirmer()
    assert not a.update(None, colour_hint=True)     # red/blue colour alone never triggers
    assert not a.update(0.7)
    assert not a.update(0.5)                        # low confidence breaks the streak
    assert not a.update(0.8)
    assert a.update(0.9)


def test_corridor_preclears_downstream_junctions(corridor):
    em = EmergencyManager(corridor)
    route = ["src_W->J1", "J1->J2", "J2->J3", "J3->J4", "J4->sink_E"]
    plans = em.trigger(0.0, 1, route, "src_W->J1", "test")
    assert [p.junction for p in plans] == ["J1", "J2", "J3", "J4"]
    etas = [p.eta_t for p in plans]
    assert etas == sorted(etas)
    j3 = plans[2]
    assert em.demand_at("J3", j3.eta_t - 5) is not None     # J3 turns green before the ambulance arrives
    assert em.demand_at("J3", 0.0) is None                  # but not immediately


# ------------------------------------------------------------------ simulator
def test_camera_failure_falls_back_to_fixed_time():
    r = run_scenario("camera_failure", "max_pressure", seed=1, duration_s=700, record_frames=False)
    assert r["metrics"]["fallback_seconds"] > 0
    kinds = [a["kind"] for a in r["alerts"]]
    assert "camera_fault" in kinds and "camera_restored" in kinds


def test_simulation_is_repeatable():
    a = run_scenario("balanced", "max_pressure", seed=3, duration_s=300, record_frames=False)["metrics"]
    b = run_scenario("balanced", "max_pressure", seed=3, duration_s=300, record_frames=False)["metrics"]
    assert a == b


def test_greens_respect_minimum_in_simulation():
    sim = Simulation("morning_rush", "max_pressure", seed=2, duration_s=600, record_frames=False)
    starts, short = {}, []
    for t in range(600):
        sim.step(t)
        for jid, h in sim.heads.items():
            if h.state == "green" and starts.get(jid, (None,))[0] != (h.phase, h.green_t0):
                starts[jid] = ((h.phase, h.green_t0),)
            if h.state == "yellow" and h.state_t0 == t:
                g = t - h.green_t0
                ped = h.junction.phases[h.phase].pedestrian
                if g < (h.junction.ped_phase_s if ped else MIN_GREEN_S) - 1e-6:
                    short.append((jid, h.phase, g))
    assert short == []


def test_max_pressure_beats_fixed_time_on_waiting():
    fixed = [run_scenario("two_vs_hundred", "fixed", seed=s, duration_s=900, record_frames=False)["metrics"]
             for s in range(3)]
    mp = [run_scenario("two_vs_hundred", "max_pressure", seed=s, duration_s=900, record_frames=False)["metrics"]
          for s in range(3)]
    avg = lambda rows, k: sum(r[k] for r in rows) / len(rows)
    assert avg(mp, "avg_wait_s") < avg(fixed, "avg_wait_s")
    assert avg(mp, "worst_approach_avg_wait_s") < avg(fixed, "worst_approach_avg_wait_s")
