"""Road-network model for multi-junction signal control.

Directions are indexed clockwise: N=0, E=1, S=2, W=3. An *approach* is named after the side the
vehicles come FROM. India drives on the LEFT, so from approach d:
    straight exit = (d + 2) % 4,  left exit = (d + 1) % 4,  right exit = (d + 3) % 4
(e.g. coming from the North, heading south, a left turn goes East).
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

DIRS = ["N", "E", "S", "W"]
DIR_IDX = {d: i for i, d in enumerate(DIRS)}
TURNS = ("L", "S", "R")

# Indian mixed-traffic saturation flow (Road Research Lab / IRC:93 style): S = 525 * W PCU/h, W in metres.
SAT_FLOW_PER_M = 525.0
JAM_SPACING_M = 7.0          # metres of lane occupied by one queued PCU


def exit_dir(approach: str, turn: str) -> str:
    d = DIR_IDX[approach]
    return DIRS[{"S": (d + 2) % 4, "L": (d + 1) % 4, "R": (d + 3) % 4}[turn]]


def movements_conflict(a: Tuple[str, str], b: Tuple[str, str]) -> bool:
    """True if movements a=(approach, turn) and b cannot be green together (left-hand traffic)."""
    (da, ta), (db, tb) = a, b
    if da == db:
        return False
    if exit_dir(da, ta) == exit_dir(db, tb):
        return True                         # two streams merging into the same exit
    if ta == "L" or tb == "L":
        return False                        # kerb-side left turns cross nobody else
    opposite = (DIR_IDX[da] + 2) % 4 == DIR_IDX[db]
    if opposite:
        return {ta, tb} == {"S", "R"}       # S+S ok, R+R ok (they pass each other), S vs R crosses
    return True                             # adjacent approaches, both straight/right: crossing paths


@dataclass
class Link:
    id: str
    from_node: str                # junction id or "src"
    to_node: str                  # junction id or "sink"
    approach: Optional[str]       # side of to_node it enters from (None for exit links)
    length_m: float
    lanes: int
    width_m: float
    speed_kmh: float = 40.0
    turn_ratios: Dict[str, float] = field(default_factory=lambda: {"L": 0.2, "S": 0.6, "R": 0.2})
    out_links: Dict[str, str] = field(default_factory=dict)   # turn -> outgoing link id

    @property
    def storage_pcu(self) -> float:
        return self.length_m * self.lanes / JAM_SPACING_M

    @property
    def sat_flow_pcu_s(self) -> float:
        return SAT_FLOW_PER_M * self.width_m / 3600.0

    @property
    def travel_time_s(self) -> float:
        return self.length_m / (self.speed_kmh / 3.6)


@dataclass
class Phase:
    name: str
    movements: List[Tuple[str, str]]       # (approach, turn)
    pedestrian: bool = False               # all-vehicle-red walk phase


@dataclass
class Junction:
    id: str
    name: str
    lat: float
    lon: float
    x: float                                # map coordinates for the dashboard
    y: float
    phases: List[Phase]
    in_links: Dict[str, str] = field(default_factory=dict)    # approach -> link id
    out_links: Dict[str, str] = field(default_factory=dict)   # exit direction -> link id
    crossing_width_m: float = 14.0
    ped_demand_per_h: float = 120.0

    @property
    def ped_phase_s(self) -> float:
        # walk time from crossing width at 1.2 m/s, never below 7 s
        return max(7.0, round(self.crossing_width_m / 1.2, 1))

    def vehicle_phases(self) -> List[int]:
        return [i for i, p in enumerate(self.phases) if not p.pedestrian]

    def ped_phase_index(self) -> Optional[int]:
        for i, p in enumerate(self.phases):
            if p.pedestrian:
                return i
        return None


class PhaseConflictError(ValueError):
    pass


PHASE_PLANS = {
    # one approach at a time (common Indian "split" phasing); always conflict-free
    "split4": lambda appr: [Phase(f"{d} all", [(d, t) for t in TURNS]) for d in appr],
    # through+left for opposing pair, then protected rights, for both axes
    "protected_right": lambda appr: [
        Phase("N-S through", [(d, t) for d in ("N", "S") if d in appr for t in ("S", "L")]),
        Phase("N-S right", [(d, "R") for d in ("N", "S") if d in appr]),
        Phase("E-W through", [(d, t) for d in ("E", "W") if d in appr for t in ("S", "L")]),
        Phase("E-W right", [(d, "R") for d in ("E", "W") if d in appr]),
    ],
}


def validate_phases(j: Junction) -> None:
    for p in j.phases:
        mv = p.movements
        for i in range(len(mv)):
            for k in range(i + 1, len(mv)):
                if movements_conflict(mv[i], mv[k]):
                    raise PhaseConflictError(f"{j.id} phase '{p.name}': {mv[i]} conflicts with {mv[k]}")


class Network:
    def __init__(self, name: str, junctions: Dict[str, Junction], links: Dict[str, Link],
                 sources: Dict[str, float], meta: Optional[dict] = None):
        self.name, self.junctions, self.links, self.sources = name, junctions, links, sources
        self.meta = meta or {}
        for j in junctions.values():
            validate_phases(j)

    # ---------------------------------------------------------------- routing helpers
    def downstream_junctions(self, link_id: str) -> Optional[str]:
        to = self.links[link_id].to_node
        return to if to in self.junctions else None

    def shortest_path(self, start_link: str, dest_node: str, cost: Dict[str, float]) -> Optional[List[str]]:
        """Dijkstra over links; cost[link] = traversal cost in seconds. Returns list of link ids."""
        pq = [(cost.get(start_link, 0.0), start_link, [start_link])]
        seen = set()
        while pq:
            c, lid, path = heapq.heappop(pq)
            if lid in seen:
                continue
            seen.add(lid)
            link = self.links[lid]
            if link.to_node == dest_node:
                return path
            for nxt in link.out_links.values():
                if nxt not in seen:
                    heapq.heappush(pq, (c + cost.get(nxt, self.links[nxt].travel_time_s), nxt, path + [nxt]))
        return None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "meta": self.meta,
            "junctions": [
                {"id": j.id, "name": j.name, "lat": j.lat, "lon": j.lon, "x": j.x, "y": j.y,
                 "phases": [{"name": p.name, "movements": [list(m) for m in p.movements],
                             "pedestrian": p.pedestrian} for p in j.phases],
                 "in_links": j.in_links, "out_links": j.out_links}
                for j in self.junctions.values()
            ],
            "links": [
                {"id": l.id, "from": l.from_node, "to": l.to_node, "approach": l.approach,
                 "length_m": l.length_m, "lanes": l.lanes, "width_m": l.width_m,
                 "storage_pcu": round(l.storage_pcu, 1)}
                for l in self.links.values()
            ],
        }


# ---------------------------------------------------------------------------- builder
def build_network(cfg: dict) -> Network:
    """Build a network from a config dict (see configs/network_corridor.yaml).

    Config gives junctions with grid positions and, for each junction, which neighbour sits on
    which side. Boundary sides get a source link (inflow) and a sink link (outflow).
    """
    defaults = cfg.get("defaults", {})
    junctions: Dict[str, Junction] = {}
    links: Dict[str, Link] = {}
    sources: Dict[str, float] = {}

    for jc in cfg["junctions"]:
        appr = jc.get("approaches", DIRS)
        plan = PHASE_PLANS[jc.get("phase_plan", defaults.get("phase_plan", "split4"))](appr)
        plan = [p for p in plan if p.movements]
        if jc.get("pedestrian_phase", defaults.get("pedestrian_phase", True)):
            plan.append(Phase("Pedestrian walk", [], pedestrian=True))
        junctions[jc["id"]] = Junction(
            id=jc["id"], name=jc.get("name", jc["id"]), lat=jc.get("lat", 0.0), lon=jc.get("lon", 0.0),
            x=jc["x"], y=jc["y"], phases=plan,
            crossing_width_m=jc.get("crossing_width_m", defaults.get("crossing_width_m", 14.0)),
            ped_demand_per_h=jc.get("ped_demand_per_h", defaults.get("ped_demand_per_h", 120.0)),
        )

    def road_props(spec: dict) -> dict:
        lanes = spec.get("lanes", defaults.get("lanes", 2))
        return dict(lanes=lanes, width_m=spec.get("width_m", lanes * defaults.get("lane_width_m", 3.5)),
                    speed_kmh=spec.get("speed_kmh", defaults.get("speed_kmh", 40.0)))

    # internal links between neighbours
    for jc in cfg["junctions"]:
        for side, nb in jc.get("neighbours", {}).items():
            # link from neighbour nb into this junction, entering from `side`
            length = nb.get("length_m")
            spec = {**nb, **road_props(nb)}
            lid = f"{nb['id']}->{jc['id']}"
            links[lid] = Link(id=lid, from_node=nb["id"], to_node=jc["id"], approach=side,
                              length_m=length, lanes=spec["lanes"], width_m=spec["width_m"],
                              speed_kmh=spec["speed_kmh"],
                              turn_ratios=dict(nb.get("turn_ratios", defaults.get("turn_ratios",
                                                                                   {"L": 0.2, "S": 0.6, "R": 0.2}))))
            junctions[jc["id"]].in_links[side] = lid
            opp_side = DIRS[(DIR_IDX[side] + 2) % 4]
            # in the neighbour, this link leaves towards us (exit direction = opposite of the side we enter)
            junctions[nb["id"]].out_links[opp_side] = lid

    # boundary source/sink links
    for jc in cfg["junctions"]:
        j = junctions[jc["id"]]
        for side in jc.get("approaches", DIRS):
            bspec = jc.get("boundary", {}).get(side, {})
            props = road_props(bspec)
            if side not in j.in_links:
                lid = f"src_{side}->{j.id}"
                links[lid] = Link(id=lid, from_node="src", to_node=j.id, approach=side,
                                  length_m=bspec.get("length_m", defaults.get("boundary_length_m", 300.0)),
                                  lanes=props["lanes"], width_m=props["width_m"], speed_kmh=props["speed_kmh"],
                                  turn_ratios=dict(bspec.get("turn_ratios", defaults.get("turn_ratios",
                                                                                         {"L": 0.2, "S": 0.6, "R": 0.2}))))
                j.in_links[side] = lid
                sources[lid] = float(bspec.get("demand_veh_h", defaults.get("demand_veh_h", 400.0)))
            if side not in j.out_links:
                lid = f"{j.id}->sink_{side}"
                links[lid] = Link(id=lid, from_node=j.id, to_node="sink", approach=None,
                                  length_m=defaults.get("boundary_length_m", 300.0), lanes=props["lanes"],
                                  width_m=props["width_m"], speed_kmh=props["speed_kmh"])
                j.out_links[side] = lid

    # wire turns: incoming link at junction j from side d, turn t -> j.out_links[exit_dir(d, t)]
    for j in junctions.values():
        for side, lid in j.in_links.items():
            link = links[lid]
            for t in TURNS:
                ex = exit_dir(side, t)
                if ex in j.out_links:
                    link.out_links[t] = j.out_links[ex]
            # renormalise ratios over turns that exist
            tot = sum(link.turn_ratios.get(t, 0.0) for t in link.out_links) or 1.0
            link.turn_ratios = {t: link.turn_ratios.get(t, 0.0) / tot for t in link.out_links}

    return Network(cfg.get("name", "network"), junctions, links, sources, cfg.get("meta"))


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "network_corridor.yaml"


def load_networks(path: Path = DEFAULT_CONFIG) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """Return (network configs by name, scenario configs by name)."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return cfg["networks"], cfg["scenarios"]
