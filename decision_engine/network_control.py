"""Network-wide signal control: controllers + one shared safety layer.

Controllers only *propose* which phase should be green. The SafetySupervisor decides what the
lights actually do, and every controller (including the RL agent) goes through it:
  * min green 10 s, max green 60 s, yellow 3-4 s, all-red 2 s - never shortened
  * fairness: no approach with waiting vehicles stays red longer than MAX_RED_S
  * pedestrian walk phase served within MAX_PED_WAIT_S of a button press
  * ambulance green corridor (confirmed detection or GPS/RFID request) with pre-clearing downstream
  * watchdog: a junction whose camera is stale falls back to a fixed-time plan
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .network_model import Junction, Network
from .traffic_signal import webster_optimum_cycle

MIN_GREEN_S = 10.0
MAX_GREEN_S = 60.0
ALL_RED_S = 2.0            # >= 1.5 s, rounded up to the 1 s simulation step
MAX_RED_S = 90.0
MAX_PED_WAIT_S = 120.0
EMERGENCY_MIN_GREEN_S = 5.0
EMERGENCY_HOLD_MAX_S = 60.0
PREEMPT_LEAD_S = 12.0      # start clearing a downstream junction this long before the ambulance arrives
STARTUP_LOST_S = 2.0
ARRIVAL_STRETCH = 1.3      # extra green for vehicles arriving while the queue discharges
SWITCH_KEEP_RATIO = 0.6    # keep green while current pressure >= 60 % of the best other (a switch costs ~8 s)
FIXED_GREEN_S = {"through": 35.0, "all": 30.0, "right": 15.0}


def yellow_for_speed(speed_kmh: float) -> float:
    return 3.0 if speed_kmh <= 40 else 4.0


# ---------------------------------------------------------------------------------- views
@dataclass
class MovementView:
    approach: str
    turn: str
    in_link: str
    out_link: str
    queue_pcu: float            # estimated queue for this movement
    sat_pcu_s: float            # saturation flow of the approach (PCU/s)
    down_queue_pcu: float       # estimated queue on the downstream link (0 for exits)
    down_storage_pcu: float
    empty_confirmed: bool       # early-switch evidence from the camera
    beyond_camera: bool = False # queue fills the whole camera view: real queue is longer than we see

    @property
    def blocked(self) -> bool:
        return self.down_storage_pcu > 0 and self.down_queue_pcu >= 0.85 * self.down_storage_pcu

    @property
    def pressure(self) -> float:
        if self.blocked:
            return 0.0           # never push vehicles into a full road (spillback / gating)
        # queue in PCU (not weighted by road width: weighting favoured wide roads and was less fair
        # in our tests, with no gain in average wait)
        return max(0.0, self.queue_pcu - self.down_queue_pcu * 0.5)


@dataclass
class JunctionView:
    junction: Junction
    t: float
    current_phase: int
    green_elapsed: float
    movements: Dict[Tuple[str, str], MovementView]
    red_time: Dict[int, float]          # seconds each phase has been red
    sensors_ok: bool = True

    def phase_pressure(self, idx: int) -> float:
        p = self.junction.phases[idx]
        return sum(self.movements[m].pressure for m in p.movements if m in self.movements)

    def phase_queue(self, idx: int) -> float:
        p = self.junction.phases[idx]
        return sum(self.movements[m].queue_pcu for m in p.movements if m in self.movements)

    def phase_demand(self, idx: int) -> bool:
        return self.phase_queue(idx) >= 0.5

    def phase_empty(self, idx: int) -> bool:
        p = self.junction.phases[idx]
        mv = [self.movements[m] for m in p.movements if m in self.movements]
        return bool(mv) and all(m.empty_confirmed for m in mv)

    def clearing_green(self, idx: int) -> float:
        """Green needed to clear this phase's queue: 2 s start-up + queue / saturation flow,
        stretched by ~30 % for vehicles that keep arriving while the light is green."""
        p = self.junction.phases[idx]
        per_link: Dict[str, Tuple[float, float]] = {}
        for m in p.movements:
            mv = self.movements.get(m)
            if mv is None:
                continue
            if mv.beyond_camera and m[1] != "R":
                return MAX_GREEN_S          # we can't see the end of the queue: plan the full green
            q, s = per_link.get(mv.in_link, (0.0, mv.sat_pcu_s))
            per_link[mv.in_link] = (q + mv.queue_pcu, s)
        need = max((q / s for q, s in per_link.values() if s > 0), default=0.0)
        return min(MAX_GREEN_S, max(MIN_GREEN_S, STARTUP_LOST_S + ARRIVAL_STRETCH * need))


# ---------------------------------------------------------------------------- controllers
class Controller:
    name = "base"

    def reset(self, net: Network) -> None:
        pass

    def choose(self, view: JunctionView) -> Tuple[int, str]:
        """Return (desired vehicle phase index, short human-readable reason)."""
        raise NotImplementedError


class FixedTimeController(Controller):
    """Today's signals: every phase gets a fixed green in a fixed order, whatever the traffic."""
    name = "fixed"

    def green_for(self, j: Junction, idx: int) -> float:
        if j.phases[idx].pedestrian:
            return j.ped_phase_s
        n = j.phases[idx].name.lower()
        for key, g in FIXED_GREEN_S.items():
            if key in n:
                return g
        return 30.0

    def choose(self, view: JunctionView) -> Tuple[int, str]:
        # the walk stage is a fixed part of every cycle, as on today's pre-timed signals
        j, cur = view.junction, view.current_phase
        order = list(range(len(j.phases)))
        if view.green_elapsed >= self.green_for(j, cur):
            nxt = order[(order.index(cur) + 1) % len(order)]
            return nxt, f"fixed {self.green_for(j, cur):.0f}s timer ended"
        return cur, "fixed timer running"


class WebsterController(Controller):
    """One-junction-at-a-time adaptive timing (the project's original method), re-planned each cycle."""
    name = "webster"

    def __init__(self):
        self.plan: Dict[str, Dict[int, float]] = {}

    def reset(self, net: Network) -> None:
        self.plan = {}

    def _replan(self, view: JunctionView) -> Dict[int, float]:
        vp = view.junction.vehicle_phases()
        approaches = [{"name": str(i), "lanes": 2, "counts": {"car": int(round(view.phase_queue(i)))}} for i in vp]
        res = webster_optimum_cycle(approaches)
        return {int(pt["approach"]): float(pt["green_time"]) for pt in res["phase_timings"]}

    def choose(self, view: JunctionView) -> Tuple[int, str]:
        j, cur = view.junction, view.current_phase
        vp = j.vehicle_phases()
        if j.id not in self.plan or (cur == vp[0] and view.green_elapsed < 1.0):
            self.plan[j.id] = self._replan(view)
        if cur not in vp:
            return vp[0], "webster cycle start"
        g = self.plan[j.id].get(cur, 30.0)
        if view.green_elapsed >= g:
            return vp[(vp.index(cur) + 1) % len(vp)], f"webster green {g:.0f}s ended"
        return cur, "webster green running"


class MaxPressureController(Controller):
    """Max-Pressure (Varaiya 2013) with queue-clearing green and early switch.

    * pressure of a phase = sum over its movements of saturation flow x (queue - 0.5 x downstream queue);
      a movement into a nearly full road has zero pressure (no spillback, upstream gating).
    * the green planned at phase start only lasts long enough to clear the queue (a maximum);
    * it ends earlier if the stop line is confirmed empty twice in a row;
    * switching away early needs a clear margin, so lights don't flip-flop.
    """
    name = "max_pressure"
    SWITCH_FACTOR = 2.0
    SWITCH_ABS = 1.0

    def __init__(self):
        self.target: Dict[str, Tuple[int, float]] = {}

    def reset(self, net: Network) -> None:
        self.target = {}

    AGE_START_S = 30.0
    AGE_SCALE_S = 30.0

    def score(self, view: JunctionView, i: int) -> float:
        """Pressure, boosted the longer a road with waiting vehicles has been red (ageing), so small
        roads are served at the natural end of a green instead of hitting the hard max-red rule."""
        p = view.phase_pressure(i)
        if i == view.current_phase or not view.phase_demand(i):
            return p
        age = max(0.0, view.red_time.get(i, 0.0) - self.AGE_START_S) / self.AGE_SCALE_S
        return max(p, 1.0) * (1.0 + age)

    def best_other(self, view: JunctionView, cur: int) -> Tuple[Optional[int], float]:
        best, bp = None, 0.0
        for i in view.junction.vehicle_phases():
            if i == cur:
                continue
            p = self.score(view, i)
            if p > bp:
                best, bp = i, p
        return best, bp

    def choose(self, view: JunctionView) -> Tuple[int, str]:
        j, cur = view.junction, view.current_phase
        vp = j.vehicle_phases()
        if cur not in vp:
            best = max(vp, key=view.phase_pressure)
            return best, f"highest pressure {view.phase_pressure(best):.1f}"
        tgt = self.target.get(j.id)
        if tgt is None or tgt[0] != cur or view.green_elapsed < 1.0:
            self.target[j.id] = (cur, view.clearing_green(cur))
        planned = self.target[j.id][1]
        other, op = self.best_other(view, cur)
        cp = view.phase_pressure(cur)
        if other is None:
            return cur, "no other road is waiting"
        # over capacity: two or more roads each need a full max green -> equal turns (fair sharing)
        saturated = [i for i in vp if view.clearing_green(i) >= MAX_GREEN_S]
        if len(saturated) >= 2 and cur in saturated:
            if view.green_elapsed >= MAX_GREEN_S * 0.75:
                nxt = max((i for i in saturated if i != cur), key=lambda i: view.red_time.get(i, 0.0))
                return nxt, "over capacity: equal turns for the jammed roads"
            return cur, "over capacity: equal turns for the jammed roads"
        if view.phase_empty(cur):
            return other, "queue cleared (empty in 2 checks) - early switch"
        if view.green_elapsed >= planned:
            if cp >= SWITCH_KEEP_RATIO * op:
                # queue still there (e.g. longer than the camera view) and still the heaviest: extend
                self.target[j.id] = (cur, view.green_elapsed + MIN_GREEN_S / 2)
                return cur, f"queue not cleared yet, still heaviest ({cp:.1f} vs {op:.1f}) - extending"
            return other, f"planned {planned:.0f}s to clear queue reached"
        if op > self.SWITCH_FACTOR * cp + self.SWITCH_ABS * 10:
            return other, f"other road pressure {op:.1f} >> current {cp:.1f}"
        return cur, f"clearing queue ({view.green_elapsed:.0f}/{planned:.0f}s)"


class RLController(Controller):
    """Shared RL policy used at every junction. Falls back to Max-Pressure if no trained agent."""
    name = "rl"

    def __init__(self, model_path=None):
        from pathlib import Path
        self.fallback = MaxPressureController()
        self.model = None
        self.loaded_from = None
        path = Path(model_path) if model_path else Path(__file__).resolve().parents[1] / "models" / "rl_signal_agent.zip"
        if path.exists():
            try:
                from stable_baselines3 import DQN
                self.model = DQN.load(str(path), device="cpu")
                self.loaded_from = str(path)
            except Exception:
                self.model = None

    def reset(self, net: Network) -> None:
        self.fallback.reset(net)

    def choose(self, view: JunctionView) -> Tuple[int, str]:
        if self.model is None:
            idx, why = self.fallback.choose(view)
            return idx, "RL agent missing -> max-pressure: " + why
        from .rl_env import observe
        obs = observe(view)
        action, _ = self.model.predict(obs, deterministic=True)
        vp = view.junction.vehicle_phases()
        a = int(action)
        idx = vp[a] if a < len(vp) else view.current_phase
        return idx, f"RL chose phase {idx} (pressures: " + ", ".join(
            f"{i}:{view.phase_pressure(i):.1f}" for i in vp) + ")"


CONTROLLERS = {
    "fixed": FixedTimeController,
    "webster": WebsterController,
    "max_pressure": MaxPressureController,
    "rl": RLController,
}


def make_controller(name: str) -> Controller:
    if name not in CONTROLLERS:
        raise ValueError(f"unknown controller {name}; choose from {sorted(CONTROLLERS)}")
    return CONTROLLERS[name]()


# ------------------------------------------------------------------------- signal head
@dataclass
class SignalHead:
    """State machine of one junction's lights. Only `request` can change it, and it always
    passes through yellow and all-red."""
    junction: Junction
    yellow_s: float
    phase: int = 0
    state: str = "green"                 # green | yellow | allred
    state_t0: float = 0.0
    green_t0: float = 0.0
    next_phase: Optional[int] = None
    red_since: Dict[int, float] = field(default_factory=dict)
    green_reason: str = "start"

    def green_elapsed(self, t: float) -> float:
        return t - self.green_t0 if self.state == "green" else 0.0

    def request(self, t: float, nxt: int, reason: str) -> bool:
        if self.state != "green" or nxt == self.phase:
            return False
        self.state, self.state_t0, self.next_phase = "yellow", t, nxt
        self.green_reason = reason
        return True

    def tick(self, t: float) -> None:
        if self.state == "yellow" and t - self.state_t0 >= self.yellow_s:
            self.state, self.state_t0 = "allred", t
            self.red_since[self.phase] = t
        elif self.state == "allred" and t - self.state_t0 >= ALL_RED_S:
            self.phase, self.state, self.state_t0, self.green_t0 = self.next_phase, "green", t, t
            self.red_since.pop(self.phase, None)
            self.next_phase = None

    def green_movements(self) -> List[Tuple[str, str]]:
        if self.state != "green":
            return []
        return list(self.junction.phases[self.phase].movements)

    def is_ped_walk(self) -> bool:
        return self.state == "green" and self.junction.phases[self.phase].pedestrian

    def red_time(self, t: float) -> Dict[int, float]:
        out = {}
        for i in range(len(self.junction.phases)):
            if i == self.phase and self.state == "green":
                out[i] = 0.0
            else:
                out[i] = t - self.red_since.get(i, 0.0)
        return out


# --------------------------------------------------------------------- emergency corridor
@dataclass
class Preemption:
    junction: str
    approach: str
    turn: str
    eta_t: float                 # simulation time the ambulance is expected at the stop line
    vehicle_id: int
    released: bool = False


class EmergencyManager:
    """Schedules green-corridor pre-emption along an ambulance's route."""

    def __init__(self, net: Network, enabled: bool = True):
        self.net, self.enabled = net, enabled
        self.active: Dict[int, List[Preemption]] = {}

    def trigger(self, t: float, vehicle_id: int, route: List[str], from_link: str, source: str) -> List[Preemption]:
        """Ambulance confirmed (camera, 2 frames) or GPS/RFID request: plan green at every junction ahead."""
        if not self.enabled or vehicle_id in self.active:
            return []
        plans, eta = [], t
        started = False
        for i, lid in enumerate(route):
            if lid == from_link:
                started = True
            if not started:
                continue
            link = self.net.links[lid]
            eta += link.travel_time_s
            if link.to_node in self.net.junctions and i + 1 < len(route):
                nxt = route[i + 1]
                turn = next((tt for tt, ol in link.out_links.items() if ol == nxt), "S")
                plans.append(Preemption(link.to_node, link.approach, turn, eta, vehicle_id))
        self.active[vehicle_id] = plans
        return plans

    def release(self, vehicle_id: int, junction_id: str) -> None:
        for p in self.active.get(vehicle_id, []):
            if p.junction == junction_id:
                p.released = True

    def demand_at(self, junction_id: str, t: float) -> Optional[Preemption]:
        for plans in self.active.values():
            for p in plans:
                if p.junction == junction_id and not p.released and t >= p.eta_t - PREEMPT_LEAD_S \
                        and t <= p.eta_t + EMERGENCY_HOLD_MAX_S:
                    return p
        return None


# ------------------------------------------------------------------------ safety layer
@dataclass
class Decision:
    junction: str
    phase: int
    reason: str
    mode: str                     # normal | emergency | pedestrian | fairness | fallback


class SafetySupervisor:
    """Applies the fixed safety rules on top of whatever the controller proposes."""

    def __init__(self, net: Network, controller: Controller, emergency: EmergencyManager):
        self.net, self.controller, self.emergency = net, controller, emergency
        self.fallback = FixedTimeController()
        self.fallback_active: Dict[str, bool] = {j: False for j in net.junctions}
        self.ped_request_t: Dict[str, Optional[float]] = {j: None for j in net.junctions}
        self.last_walk_end: Dict[str, float] = {j: -1e9 for j in net.junctions}
        # today's fixed-time signals have no adaptive rules: keep that baseline honest
        self.adaptive = getattr(controller, "name", "") != "fixed"

    def press_ped_button(self, junction_id: str, t: float) -> None:
        if self.ped_request_t.get(junction_id) is None:
            self.ped_request_t[junction_id] = t

    def decide(self, head: SignalHead, view: JunctionView) -> Decision:
        j, t = head.junction, view.t
        elapsed = head.green_elapsed(t)
        cur = head.phase

        # 0. pedestrian walk phase runs for its fixed time, then hands back
        if j.phases[cur].pedestrian:
            if elapsed >= j.ped_phase_s:
                self.ped_request_t[j.id] = None
                self.last_walk_end[j.id] = t
                ctrl = self.fallback if self.fallback_active[j.id] else self.controller
                nxt, why = ctrl.choose(view)
                if nxt == cur or j.phases[nxt].pedestrian:
                    nxt = j.vehicle_phases()[0]
                return Decision(j.id, nxt, "walk time over; " + why, "pedestrian")
            return Decision(j.id, cur, "pedestrians crossing", "pedestrian")

        # 1. ambulance green corridor
        pre = self.emergency.demand_at(j.id, t)
        if pre is not None:
            target = next((i for i in j.vehicle_phases() if (pre.approach, pre.turn) in j.phases[i].movements), None)
            if target is not None:
                if target == cur:
                    return Decision(j.id, cur, f"GREEN CORRIDOR: holding green for ambulance from {pre.approach}", "emergency")
                if elapsed >= EMERGENCY_MIN_GREEN_S:
                    return Decision(j.id, target, f"GREEN CORRIDOR: ambulance from {pre.approach} in "
                                                  f"{max(0, pre.eta_t - t):.0f}s", "emergency")
                return Decision(j.id, cur, "ambulance coming - finishing minimum green safely", "emergency")

        if elapsed < MIN_GREEN_S:
            return Decision(j.id, cur, f"minimum green ({elapsed:.0f}/{MIN_GREEN_S:.0f}s)", "normal")

        if not self.adaptive:
            nxt, why = self.controller.choose(view)
            return Decision(j.id, nxt, why, "normal")

        # 2. pedestrian button: guaranteed walk before MAX_PED_WAIT_S
        req = self.ped_request_t.get(j.id)
        ped = j.ped_phase_index()
        if req is not None and ped is not None and t - req >= MAX_PED_WAIT_S - 15:
            return Decision(j.id, ped, f"pedestrians waiting {t - req:.0f}s", "pedestrian")

        # 3. fairness: someone waiting too long at red
        starving = [(view.red_time.get(i, 0.0), i) for i in j.vehicle_phases()
                    if i != cur and view.phase_demand(i) and view.red_time.get(i, 0.0) >= MAX_RED_S]
        if starving:
            rt, i = max(starving)
            return Decision(j.id, i, f"fairness: phase {i} red for {rt:.0f}s", "fairness")

        # 4. watchdog: stale camera -> fixed plan
        if not view.sensors_ok:
            self.fallback_active[j.id] = True
            nxt, why = self.fallback.choose(view)
            return Decision(j.id, nxt, "CAMERA FAULT -> fixed-time fallback; " + why, "fallback")
        self.fallback_active[j.id] = False

        nxt, why = self.controller.choose(view)
        if j.phases[nxt].pedestrian:
            nxt = cur

        # 5. max green: must give way if anybody else waits
        if elapsed >= MAX_GREEN_S and nxt == cur:
            others = [i for i in j.vehicle_phases() if i != cur and view.phase_demand(i)]
            if others:
                nxt = max(others, key=lambda i: view.red_time.get(i, 0.0))
                why = f"maximum green {MAX_GREEN_S:.0f}s reached"
        # serve a pending pedestrian request at the natural end of a green (not every change)
        if nxt != cur and req is not None and ped is not None and t - req >= 60 \
                and t - self.last_walk_end[j.id] >= 90:
            return Decision(j.id, ped, "pedestrian request served at phase change", "pedestrian")
        return Decision(j.id, nxt, why, "normal")


# ------------------------------------------------------------------ hardware boundary
class SignalControllerAdapter:
    """Boundary to real signal controller hardware (future work: vendor protocol + police approval)."""

    def apply(self, junction_id: str, phase: int, state: str) -> None:
        raise NotImplementedError


class SimulatedAdapter(SignalControllerAdapter):
    def __init__(self):
        self.log: List[Tuple[str, int, str]] = []

    def apply(self, junction_id: str, phase: int, state: str) -> None:
        self.log.append((junction_id, phase, state))


# ---------------------------------------------------------------- live (photo) decision
def explain_from_counts(junction: Junction, approach_pcu: Dict[str, float], lanes: Dict[str, int],
                        current_phase: int = 0, green_elapsed: float = MIN_GREEN_S,
                        down_queue: Optional[Dict[str, float]] = None) -> dict:
    """One Max-Pressure decision for a single junction from per-approach PCU (e.g. from YOLO photos)."""
    from .network_model import SAT_FLOW_PER_M, exit_dir
    down_queue = down_queue or {}
    mv: Dict[Tuple[str, str], MovementView] = {}
    ratios = {"L": 0.25, "S": 0.5, "R": 0.25}
    for p in junction.phases:
        for (a, tn) in p.movements:
            q = approach_pcu.get(a, 0.0)
            ln = lanes.get(a, 2)
            mv[(a, tn)] = MovementView(a, tn, a, exit_dir(a, tn), q * ratios[tn], SAT_FLOW_PER_M * ln * 3.5 / 3600,
                                       down_queue.get(exit_dir(a, tn), 0.0), 100.0, q < 0.5)
    view = JunctionView(junction, 0.0, current_phase, green_elapsed, mv, {i: 0.0 for i in range(len(junction.phases))})
    ctrl = MaxPressureController()
    vp = junction.vehicle_phases()
    best = max(vp, key=view.phase_pressure)
    green = view.clearing_green(best)
    ranked = sorted(vp, key=view.phase_pressure, reverse=True)
    heavy = max(approach_pcu, key=approach_pcu.get) if approach_pcu else None
    light = min(approach_pcu, key=approach_pcu.get) if approach_pcu else None
    text = (f"{heavy} has {approach_pcu.get(heavy, 0):.0f} PCU vs {light} {approach_pcu.get(light, 0):.0f} PCU "
            f"-> '{junction.phases[best].name}' green for up to {green:.0f}s (ends early if the queue clears)")
    return {
        "recommended_phase": best,
        "phase_name": junction.phases[best].name,
        "green_s": round(green, 1),
        "yellow_s": 3.0,
        "all_red_s": ALL_RED_S,
        "explanation": text,
        "phases": [{"index": i, "name": junction.phases[i].name, "pressure": round(view.phase_pressure(i), 2),
                    "queue_pcu": round(view.phase_queue(i), 1), "clearing_green_s": round(view.clearing_green(i), 1)}
                   for i in ranked],
        "_controller": ctrl.name,
    }
