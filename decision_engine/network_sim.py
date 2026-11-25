"""Second-by-second traffic simulator for comparing signal controllers on a real-road network.

Model (deliberately simple, transparent and seeded so every run can be repeated):
  * vehicles of Indian mixed types (bike, car, auto, bus, truck, bicycle) arrive at the network edge
    as random (Poisson) arrivals following a peak-hour profile;
  * on each road (link) they drive at the speed limit, then join the queue of their turn;
  * a green movement discharges at the Indian mixed-traffic saturation flow S = 525 x width PCU/h,
    with +/-15 % random variation, 2 s start-up loss, and two-wheelers filtering forward;
  * a vehicle can only enter the next road if it has space (spillback blocks junctions upstream);
  * controllers do NOT see the true queues - they see camera estimates (limited view, missed
    detections, camera failure), exactly like the real system would.
"""
from __future__ import annotations

import math
import random
import statistics
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from .network_control import (
    STARTUP_LOST_S,
    Decision,
    EmergencyManager,
    JunctionView,
    MovementView,
    SafetySupervisor,
    SignalHead,
    make_controller,
    yellow_for_speed,
)
from .network_model import DEFAULT_CONFIG, TURNS, Network, build_network
from .queue_estimator import (
    EMPTY_PCU,
    LOW_CONFIDENCE,
    AmbulanceConfirmer,
    ApproachSensor,
    QueueEstimator,
    TurnRatioLearner,
    camera_range_pcu,
)
from .traffic_signal import PCU_WEIGHTS

SENSE_EVERY_S = 2
FRAME_EVERY_S = 2
OVERCAP_SHARE = 0.85
OVERCAP_HOLD_S = 60


class Vehicle:
    __slots__ = ("id", "vtype", "pcu", "t_enter", "wait", "q_t", "route", "turn", "arrive_t", "amb",
                 "t_buf", "link")

    def __init__(self, vid, vtype, pcu, t, route=None, amb=False):
        self.id, self.vtype, self.pcu, self.t_enter = vid, vtype, pcu, t
        self.wait, self.q_t, self.route, self.turn, self.arrive_t = 0.0, None, route, None, 0.0
        self.amb, self.t_buf, self.link = amb, None, None


@dataclass
class LinkState:
    transit: deque = field(default_factory=deque)
    queues: Dict[str, deque] = field(default_factory=lambda: {t: deque() for t in TURNS})
    entry: deque = field(default_factory=deque)     # vehicles waiting to enter a full edge road
    occ_pcu: float = 0.0                              # transit + queued PCU on the road
    cap_acc: float = 0.0
    arr_pcu: float = 0.0                              # since last camera frame
    dep_pcu: float = 0.0
    full_since: Optional[float] = None
    alerted: bool = False

    def queued_pcu(self) -> float:
        return sum(v.pcu for q in self.queues.values() for v in q) + sum(v.pcu for v in self.entry)

    def queued_veh(self) -> int:
        return sum(len(q) for q in self.queues.values()) + len(self.entry)


def _poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


def _profile(points: Optional[List[List[float]]], t: float) -> float:
    if not points:
        return 1.0
    if t <= points[0][0]:
        return points[0][1]
    for (t0, v0), (t1, v1) in zip(points, points[1:]):
        if t0 <= t <= t1:
            return v0 + (v1 - v0) * (t - t0) / max(1e-9, t1 - t0)
    return points[-1][1]


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def list_scenarios(path: Path = DEFAULT_CONFIG) -> List[dict]:
    cfg = load_config(path)
    return [{"id": k, "title": v.get("title", k), "network": v["network"], "duration_s": v.get("duration_s", 1200)}
            for k, v in cfg["scenarios"].items()]


class Simulation:
    def __init__(self, scenario: str, controller: str = "max_pressure", seed: int = 0,
                 corridor: Optional[bool] = None, miss_rate: float = 0.05, ratio_error: float = 0.0,
                 learn_ratios: bool = True, duration_s: Optional[int] = None, record_frames: bool = True,
                 config_path: Path = DEFAULT_CONFIG, controller_obj=None, lane_discipline: float = 1.0):
        cfg = load_config(config_path)
        if scenario not in cfg["scenarios"]:
            raise KeyError(f"unknown scenario {scenario}")
        self.scn = cfg["scenarios"][scenario]
        self.scenario_id = scenario
        self.net: Network = build_network(cfg["networks"][self.scn["network"]])
        self.mix = cfg.get("vehicle_mix", {"car": 1.0})
        self.rng = random.Random(seed)
        self.seed = seed
        self.duration = int(duration_s or self.scn.get("duration_s", 1200))
        self.miss_rate, self.record_frames = miss_rate, record_frames
        self.lane_discipline = lane_discipline     # <1 = more chaotic discharge (sensitivity runs)
        for lid, d in (self.scn.get("demand") or {}).items():
            if lid in self.net.sources:
                self.net.sources[lid] = float(d)

        self.controller = controller_obj or make_controller(controller)
        self.controller_name = getattr(self.controller, "name", controller)
        self.controller.reset(self.net)
        if corridor is None:
            corridor = self.controller_name != "fixed"      # today's fixed signals have no green corridor
        self.emergency = EmergencyManager(self.net, enabled=corridor)
        self.supervisor = SafetySupervisor(self.net, self.controller, self.emergency)
        self.heads = {jid: SignalHead(j, yellow_for_speed(max(self.net.links[l].speed_kmh for l in j.in_links.values())))
                      for jid, j in self.net.junctions.items()}
        for h in self.heads.values():
            h.phase = h.junction.vehicle_phases()[0]

        self.ls: Dict[str, LinkState] = {lid: LinkState() for lid in self.net.links}
        self.sensors: Dict[str, ApproachSensor] = {}
        for lid, link in self.net.links.items():
            if link.to_node in self.net.junctions:
                ratios = dict(link.turn_ratios)
                if ratio_error:
                    keys = list(ratios)
                    ratios = {k: max(0.01, v * (1 + ratio_error * (1 if i % 2 == 0 else -1)))
                              for i, (k, v) in enumerate(ratios.items())}
                tot = sum(ratios.values())
                ratios = {k: v / tot for k, v in ratios.items()}
                self.sensors[lid] = ApproachSensor(QueueEstimator(camera_range_pcu(link.lanes)),
                                                   TurnRatioLearner(ratios, alpha=0.02 if learn_ratios else 0.0))
        self.mv_empty_streak: Dict[Tuple[str, str], int] = {}

        self.t = 0
        self.vid = 0
        self.vehicles: List[Vehicle] = []
        self.done: List[Vehicle] = []
        self.appr_wait: Dict[Tuple[str, str], List[float]] = {}
        self.max_queue: Dict[str, int] = {}
        self.ped_wait_list: Dict[str, List[float]] = {j: [] for j in self.net.junctions}
        self.ped_waits: List[float] = []
        self.alerts: List[dict] = []
        self.frames: List[dict] = []
        self.decisions: Dict[str, Decision] = {}
        self.fallback_seconds: Dict[str, int] = {j: 0 for j in self.net.junctions}
        self.amb: Optional[Vehicle] = None
        self.amb_done_t: Optional[float] = None
        self.amb_confirmed = False
        self.fault = self.scn.get("camera_failure")
        self._init_queues()

    # ------------------------------------------------------------------ vehicles
    def _new_vehicle(self, t, route=None, amb=False) -> Vehicle:
        if amb:
            vt = "ambulance"
        else:
            r, acc, vt = self.rng.random(), 0.0, "car"
            for k, share in self.mix.items():
                acc += share
                if r <= acc:
                    vt = k
                    break
        v = Vehicle(self.vid, vt, 1.0 if amb else PCU_WEIGHTS.get(vt, 1.0), t, route, amb)
        self.vid += 1
        self.vehicles.append(v)
        return v

    def _choose_turn(self, v: Vehicle, lid: str) -> Optional[str]:
        link = self.net.links[lid]
        if not link.out_links:
            return None
        if v.route and lid in v.route:
            i = v.route.index(lid)
            if i + 1 < len(v.route):
                nxt = v.route[i + 1]
                for tn, ol in link.out_links.items():
                    if ol == nxt:
                        return tn
        r, acc = self.rng.random(), 0.0
        for tn, ratio in link.turn_ratios.items():
            acc += ratio
            if r <= acc:
                return tn
        return next(iter(link.out_links))

    def _enter_link(self, v: Vehicle, lid: str, t: float) -> None:
        link = self.net.links[lid]
        v.link, v.turn = lid, self._choose_turn(v, lid)
        v.arrive_t = t + link.travel_time_s
        st = self.ls[lid]
        st.transit.append(v)
        st.occ_pcu += v.pcu

    def _init_queues(self) -> None:
        for lid, n in (self.scn.get("initial_queues") or {}).items():
            st, link = self.ls[lid], self.net.links[lid]
            for _ in range(int(n)):
                v = self._new_vehicle(0.0)
                if st.occ_pcu + v.pcu <= link.storage_pcu:
                    v.link, v.turn, v.q_t = lid, self._choose_turn(v, lid), 0.0
                    st.queues[v.turn].append(v)
                    st.occ_pcu += v.pcu
                else:
                    v.t_buf = 0.0
                    st.entry.append(v)

    # ------------------------------------------------------------------ main loop
    def run(self) -> dict:
        for t in range(self.duration):
            self.step(t)
        return self.results()

    def step(self, t: int) -> None:
        self.t = t
        self._arrivals(t)
        self._advance(t)
        if t % SENSE_EVERY_S == 0:
            self._sense(t)
            self._decide(t)
        for h in self.heads.values():
            h.tick(t)
        self._discharge(t)
        self._pedestrians(t)
        if t % 5 == 0:
            for lid, st in self.ls.items():
                if self.net.links[lid].to_node in self.net.junctions:
                    self.max_queue[lid] = max(self.max_queue.get(lid, 0), st.queued_veh())
            self._capacity_alerts(t)
        if self.record_frames and t % FRAME_EVERY_S == 0:
            self.frames.append(self._frame(t))

    def _arrivals(self, t: int) -> None:
        mult = _profile(self.scn.get("profile"), t)
        for lid, rate in self.net.sources.items():
            for _ in range(_poisson(self.rng, rate * mult / 3600.0)):
                v = self._new_vehicle(t)
                v.t_buf = t
                self.ls[lid].entry.append(v)
        amb = self.scn.get("ambulance")
        if amb and t == int(amb["t"]):
            v = self._new_vehicle(t, route=list(amb["route"]), amb=True)
            v.t_buf = t
            self.amb = v
            self.ls[amb["enter"]].entry.appendleft(v)
            if amb.get("gps"):
                self._trigger_corridor(t, v, amb["enter"], "GPS/RFID request")

    def _advance(self, t: int) -> None:
        for lid, st in self.ls.items():
            link = self.net.links[lid]
            while st.entry and st.occ_pcu + st.entry[0].pcu <= link.storage_pcu:
                v = st.entry.popleft()
                v.wait += t - v.t_buf
                self._enter_link(v, lid, t)
            while st.transit and st.transit[0].arrive_t <= t:
                v = st.transit.popleft()
                if link.to_node == "sink":
                    st.occ_pcu -= v.pcu
                    self.done.append(v)
                    if v.amb:
                        self.amb_done_t = t
                    continue
                v.q_t = t
                st.queues[v.turn].append(v)
                st.arr_pcu += v.pcu

    # ------------------------------------------------------------------ sensing (cameras)
    def _camera_down(self, jid: str, t: int) -> bool:
        f = self.fault
        return bool(f) and f["junction"] == jid and f["t_start"] <= t < f["t_end"]

    def _sense(self, t: int) -> None:
        for lid, sen in self.sensors.items():
            link, st = self.net.links[lid], self.ls[lid]
            jid = link.to_node
            if self._camera_down(jid, t):
                sen.queue.advance_without_camera(st.arr_pcu, st.dep_pcu)
            else:
                true_q = st.queued_pcu()
                visible = min(true_q, sen.queue.camera_range_pcu)
                visible *= (1.0 - self.miss_rate * self.rng.random())
                conf = min(0.99, max(0.2, self.rng.gauss(0.88, 0.05)))
                sen.queue.update(t, visible, conf, st.arr_pcu, st.dep_pcu)
                for tn in link.out_links:
                    key = (lid, tn)
                    # the camera watches the stop-line zone of each lane group: empty = nobody there
                    # (a missed detection can make it look empty - that's why 2 checks are required)
                    front = sum(v.pcu for v in list(st.queues[tn])[:3])
                    empty = front * (1.0 - self.miss_rate * self.rng.random()) < EMPTY_PCU
                    self.mv_empty_streak[key] = self.mv_empty_streak.get(key, 0) + 1 if (empty and conf >= LOW_CONFIDENCE) else 0
                self._detect_ambulance(lid, sen, t)
            st.arr_pcu = st.dep_pcu = 0.0

    def _detect_ambulance(self, lid: str, sen: ApproachSensor, t: int) -> None:
        a = self.amb
        st, link = self.ls[lid], self.net.links[lid]
        visible = False
        if a is not None and a.link == lid and self.amb_done_t is None:
            in_queue = any(a in q for q in st.queues.values())
            near = a in st.transit and (a.arrive_t - t) * (link.speed_kmh / 3.6) <= 80.0
            visible = in_queue or near
        colour_false_alarm = self.rng.random() < 0.03          # red car / signboard: must be ignored
        conf = min(0.99, max(0.0, self.rng.gauss(0.8, 0.12))) if visible else None
        if sen.ambulance.update(conf, colour_hint=colour_false_alarm) and visible and not self.amb_confirmed:
            self.amb_confirmed = True
            self._trigger_corridor(t, a, lid, "camera (2 frames, conf>=0.6)")

    def _trigger_corridor(self, t: int, v: Vehicle, from_link: str, source: str) -> None:
        plans = self.emergency.trigger(t, v.id, v.route or [], from_link, source)
        if plans:
            self.alerts.append({"t": t, "kind": "green_corridor", "severity": "critical",
                                "message": f"Ambulance confirmed by {source}: green corridor through "
                                           + " -> ".join(p.junction for p in plans)})

    # ------------------------------------------------------------------ decisions
    def view(self, jid: str, t: int) -> JunctionView:
        j, head = self.net.junctions[jid], self.heads[jid]
        mv: Dict[Tuple[str, str], MovementView] = {}
        sensors_ok = True
        for appr, lid in j.in_links.items():
            sen = self.sensors[lid]
            if sen.queue.is_stale(t):
                sensors_ok = False
            link = self.net.links[lid]
            mq = sen.movement_queues()
            for tn, out in link.out_links.items():
                dl = self.net.links[out]
                down_q = self.sensors[out].queue.estimate if out in self.sensors else 0.0
                mv[(appr, tn)] = MovementView(appr, tn, lid, out, mq.get(tn, 0.0), link.sat_flow_pcu_s,
                                              down_q, dl.storage_pcu if out in self.sensors else 0.0,
                                              self.mv_empty_streak.get((lid, tn), 0) >= 2,
                                              sen.queue.beyond_camera)
        return JunctionView(j, t, head.phase, head.green_elapsed(t), mv, head.red_time(t), sensors_ok)

    def _decide(self, t: int) -> None:
        for jid, head in self.heads.items():
            if head.state != "green":
                continue
            d = self.supervisor.decide(head, self.view(jid, t))
            prev = self.decisions.get(jid)
            if d.mode == "fallback" and (prev is None or prev.mode != "fallback"):
                self.alerts.append({"t": t, "kind": "camera_fault", "severity": "warning",
                                    "message": f"{jid}: camera data stale -> safe fixed-time fallback"})
            if prev is not None and prev.mode == "fallback" and d.mode != "fallback":
                self.alerts.append({"t": t, "kind": "camera_restored", "severity": "info",
                                    "message": f"{jid}: camera restored -> adaptive control resumed"})
            self.decisions[jid] = d
            if d.phase != head.phase:
                head.request(t, d.phase, d.reason)
        for jid in self.heads:
            if self.decisions.get(jid) and self.decisions[jid].mode == "fallback":
                self.fallback_seconds[jid] += SENSE_EVERY_S

    # ------------------------------------------------------------------ discharge
    def _discharge(self, t: int) -> None:
        for jid, head in self.heads.items():
            green = head.green_movements()
            if not green or t - head.green_t0 < STARTUP_LOST_S:
                continue
            j = head.junction
            by_link: Dict[str, List[str]] = {}
            for appr, tn in green:
                lid = j.in_links.get(appr)
                if lid and tn in self.net.links[lid].out_links:
                    by_link.setdefault(lid, []).append(tn)
            for lid, turns in by_link.items():
                self._discharge_link(jid, lid, turns, t)

    def _discharge_link(self, jid: str, lid: str, turns: List[str], t: int) -> None:
        link, st = self.net.links[lid], self.ls[lid]
        jitter = self.rng.uniform(0.85, 1.15) * (0.85 + 0.15 * self.lane_discipline)
        st.cap_acc = min(st.cap_acc + link.sat_flow_pcu_s * jitter, 2 * link.sat_flow_pcu_s + 3.0)
        active = [tn for tn in turns if st.queues[tn]]
        while active:
            # ambulances first, then the earliest queued vehicle; bikes sometimes filter forward
            def pick(tn):
                q = st.queues[tn]
                for i, v in enumerate(q):
                    if v.amb:
                        return i
                    if i >= 4:
                        break
                if self.rng.random() < 0.25:
                    for i in range(min(4, len(q))):
                        if q[i].vtype in ("bike", "bicycle"):
                            return i
                return 0
            tn = min(active, key=lambda x: (not any(v.amb for v in st.queues[x]), st.queues[x][0].q_t))
            q = st.queues[tn]
            i = pick(tn)
            v = q[i]
            out = link.out_links[tn]
            dl = self.net.links[out]
            if dl.to_node != "sink" and self.ls[out].occ_pcu + v.pcu > dl.storage_pcu:
                active.remove(tn)              # spillback: next road full, this movement waits
                continue
            if st.cap_acc < v.pcu:
                break
            del q[i]
            st.cap_acc -= v.pcu
            st.occ_pcu -= v.pcu
            st.dep_pcu += v.pcu
            waited = t - v.q_t
            v.wait += waited
            self.appr_wait.setdefault((jid, link.approach), []).append(waited)
            self.sensors[lid].turns.observe(tn)
            if v.amb:
                self.emergency.release(v.id, jid)
            self._enter_link(v, out, t)
            if not q:
                active.remove(tn)
        if not any(st.queues[tn] for tn in turns):
            st.cap_acc = 0.0

    # ------------------------------------------------------------------ pedestrians
    def _pedestrians(self, t: int) -> None:
        for jid, j in self.net.junctions.items():
            for _ in range(_poisson(self.rng, j.ped_demand_per_h / 3600.0)):
                self.ped_wait_list[jid].append(t)
                self.supervisor.press_ped_button(jid, t)
            if self.heads[jid].is_ped_walk() and self.ped_wait_list[jid]:
                self.ped_waits.extend(t - a for a in self.ped_wait_list[jid])
                self.ped_wait_list[jid] = []

    # ------------------------------------------------------------------ alerts
    def _capacity_alerts(self, t: int) -> None:
        for lid, st in self.ls.items():
            link = self.net.links[lid]
            if link.to_node not in self.net.junctions:
                continue
            full = st.queued_pcu() >= OVERCAP_SHARE * link.storage_pcu
            if full:
                st.full_since = t if st.full_since is None else st.full_since
                if not st.alerted and t - st.full_since >= OVERCAP_HOLD_S:
                    st.alerted = True
                    self.alerts.append({"t": t, "kind": "over_capacity", "severity": "high",
                                        "link": lid, "message": self._diversion_text(lid)})
            else:
                st.full_since, st.alerted = None, False

    def _diversion_text(self, lid: str) -> str:
        link = self.net.links[lid]
        jid = link.to_node
        base = f"{lid}: queue over {int(OVERCAP_SHARE * 100)}% of road space for {OVERCAP_HOLD_S}s at {jid}. "
        if link.from_node == "src":
            return base + ("More traffic than the road can carry. Upstream: advise diversion on display boards / "
                           "traffic police; junction is sharing green fairly and not feeding the jam downstream.")
        cost = {l: self.net.links[l].travel_time_s + 3.0 * self.ls[l].queued_pcu() / max(0.1, self.net.links[l].sat_flow_pcu_s)
                for l in self.net.links}
        cost[lid] = 1e9
        starts = [l for l in self.net.junctions[link.from_node].in_links.values()]
        for s in starts:
            path = self.net.shortest_path(s, jid, cost)
            if path and lid not in path and sum(cost[p] for p in path) < 1e8:
                return base + "Suggested diversion: " + " -> ".join(path) + ". Upstream junctions are holding traffic (gating)."
        return base + "No parallel route in the model; upstream junctions are holding traffic back (gating)."

    # ------------------------------------------------------------------ outputs
    def _frame(self, t: int) -> dict:
        sig = {}
        for jid, h in self.heads.items():
            d = self.decisions.get(jid)
            sig[jid] = {"phase": h.phase, "state": h.state, "next": h.next_phase,
                        "green_elapsed": round(h.green_elapsed(t), 1),
                        "mode": d.mode if d else "normal", "reason": d.reason if d else ""}
        q = {lid: st.queued_veh() for lid, st in self.ls.items() if self.net.links[lid].to_node in self.net.junctions}
        # queued PCU per link: with lanes and JAM_SPACING_M this gives the physical queue length (spillback check)
        qp = {lid: round(self.ls[lid].queued_pcu(), 1) for lid in q}
        mov = {lid: round(sum(v.pcu for v in st.transit), 1) for lid, st in self.ls.items()}
        amb = None
        if self.amb is not None and self.amb_done_t is None and self.amb.link:
            a = self.amb
            link = self.net.links[a.link]
            prog = 1.0 if any(a in qq for qq in self.ls[a.link].queues.values()) else \
                max(0.0, min(1.0, 1 - (a.arrive_t - t) / max(1.0, link.travel_time_s)))
            amb = {"link": a.link, "progress": round(prog, 2)}
        return {"t": t, "signals": sig, "queues": q, "queue_pcu": qp, "moving_pcu": mov, "ambulance": amb,
                "exited": len(self.done)}

    def results(self) -> dict:
        t_end = self.duration
        waits = []
        queued_ids = {v.id for st in self.ls.values() for q in st.queues.values() for v in q}
        for v in self.vehicles:
            if v.link is None:                       # never got onto the network (edge queue)
                waits.append(v.wait + t_end - v.t_buf)
                continue
            w = v.wait
            if v.id in queued_ids:                   # still waiting at a stop line at the end
                w += t_end - v.q_t
                lk = self.net.links[v.link]
                self.appr_wait.setdefault((lk.to_node, lk.approach), []).append(t_end - v.q_t)
            waits.append(w)
        for lid, st in self.ls.items():               # edge queues count against their approach too
            lk = self.net.links[lid]
            if st.entry and lk.to_node in self.net.junctions:
                self.appr_wait.setdefault((lk.to_node, lk.approach), []).extend(t_end - v.t_buf for v in st.entry)
        for jid, lst in self.ped_wait_list.items():
            self.ped_waits.extend(t_end - a for a in lst)
        waits.sort()
        p95 = waits[int(0.95 * (len(waits) - 1))] if waits else 0.0
        worst_appr = max(((k, statistics.mean(v)) for k, v in self.appr_wait.items() if v), key=lambda x: x[1],
                         default=(("-", "-"), 0.0))
        amb_tt = None
        if self.amb is not None:
            amb_tt = (self.amb_done_t - self.amb.t_enter) if self.amb_done_t is not None else None
        metrics = {
            "vehicles_entered": len(self.vehicles),
            "vehicles_exited": len(self.done),
            "avg_wait_s": round(statistics.mean(waits), 1) if waits else 0.0,
            "p95_wait_s": round(p95, 1),
            "worst_approach": f"{worst_appr[0][0]} {worst_appr[0][1]}",
            "worst_approach_avg_wait_s": round(worst_appr[1], 1),
            "max_queue_veh": max(self.max_queue.values(), default=0),
            "stopped_vehicle_hours": round(sum(waits) / 3600.0, 1),
            "ped_avg_wait_s": round(statistics.mean(self.ped_waits), 1) if self.ped_waits else 0.0,
            "ambulance_travel_s": amb_tt,
            "fallback_seconds": sum(self.fallback_seconds.values()),
        }
        return {
            "scenario": self.scenario_id,
            "title": self.scn.get("title", self.scenario_id),
            "controller": self.controller_name,
            "seed": self.seed,
            "duration_s": self.duration,
            "metrics": metrics,
            "alerts": self.alerts,
            "frames": self.frames,
            "network": self.net.to_dict(),
        }


def run_scenario(scenario: str, controller: str = "max_pressure", seed: int = 0, **kw) -> dict:
    return Simulation(scenario, controller, seed, **kw).run()


NUMERIC = ["avg_wait_s", "p95_wait_s", "worst_approach_avg_wait_s", "max_queue_veh", "vehicles_exited",
           "stopped_vehicle_hours", "ped_avg_wait_s", "ambulance_travel_s"]


def run_comparison(scenario: str, controllers=("fixed", "webster", "max_pressure"), seeds=range(5), **kw) -> dict:
    """Mean and standard deviation of each metric over several random seeds, per controller."""
    out = {}
    for c in controllers:
        runs = [run_scenario(scenario, c, s, record_frames=False, **kw)["metrics"] for s in seeds]
        row = {}
        for k in NUMERIC:
            vals = [r[k] for r in runs if r.get(k) is not None]
            if vals:
                row[k] = {"mean": round(statistics.mean(vals), 1),
                          "std": round(statistics.pstdev(vals), 1) if len(vals) > 1 else 0.0}
        row["worst_approach"] = runs[0]["worst_approach"]
        out[c] = row
    return out
