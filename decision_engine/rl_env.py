"""Reinforcement-learning environment for signal control (one shared policy for every junction).

Each episode picks a random scenario, seed and "focal" junction. The agent controls the focal
junction; the other junctions run Max-Pressure. At deployment the same policy runs at every junction.

Safety is NOT learned: every action goes through SafetySupervisor (min/max green, yellow + all-red,
max red, pedestrian walk, ambulance corridor, camera watchdog), so the agent can only choose *which*
phase should be green next, inside those limits.

Training variety (domain randomisation) so the policy doesn't overfit a perfect world:
random scenario/demand, camera miss rate 0-15 %, wrong starting turn ratios (up to 30 %), and
more chaotic lane discipline.
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np

from .network_control import MAX_GREEN_S, MAX_RED_S, Controller, JunctionView, MaxPressureController

MAX_PHASES = 4
OBS_SIZE = MAX_PHASES * 3 + MAX_PHASES + 2
DECISION_S = 5
TRAIN_SCENARIOS = ["morning_rush", "two_vs_hundred", "over_capacity", "balanced", "camera_failure", "grid_rush"]


def observe(view: JunctionView) -> np.ndarray:
    """Fixed-size observation for any junction (padded to 4 vehicle phases)."""
    vp = view.junction.vehicle_phases()[:MAX_PHASES]
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    for k, i in enumerate(vp):
        obs[3 * k] = min(view.phase_queue(i) / 50.0, 3.0)
        obs[3 * k + 1] = min(view.phase_pressure(i) / 100.0, 3.0)
        obs[3 * k + 2] = min(view.red_time.get(i, 0.0) / MAX_RED_S, 2.0)
    if view.current_phase in vp:
        obs[MAX_PHASES * 3 + vp.index(view.current_phase)] = 1.0
    obs[-2] = min(view.green_elapsed / MAX_GREEN_S, 2.0)
    obs[-1] = 1.0 if view.sensors_ok else 0.0
    return obs


class _AgentSlot(Controller):
    """Focal junction follows the agent's latest action; the rest run Max-Pressure."""
    name = "rl_training"

    def __init__(self):
        self.mp = MaxPressureController()
        self.focal: Optional[str] = None
        self.action: Optional[int] = None

    def reset(self, net) -> None:
        self.mp.reset(net)

    def choose(self, view: JunctionView):
        if view.junction.id != self.focal or self.action is None:
            return self.mp.choose(view)
        vp = view.junction.vehicle_phases()
        if self.action < len(vp):
            return vp[self.action], "agent"
        return (view.current_phase if view.current_phase in vp else vp[0]), "agent keeps"


try:
    import gymnasium as gym
    from gymnasium import spaces

    class SignalEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, scenarios=None, duration_s: int = 600, randomise: bool = True):
            super().__init__()
            self.scenarios = scenarios or TRAIN_SCENARIOS
            self.duration_s, self.randomise = duration_s, randomise
            self.action_space = spaces.Discrete(MAX_PHASES)
            self.observation_space = spaces.Box(0.0, 3.0, shape=(OBS_SIZE,), dtype=np.float32)
            self.sim = None
            self.slot = _AgentSlot()
            self._rng = random.Random(0)

        def _focal_obs(self):
            return observe(self.sim.view(self.slot.focal, self.sim.t))

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            if seed is not None:
                self._rng.seed(seed)
            from .network_sim import Simulation
            scn = self._rng.choice(self.scenarios)
            kw = {}
            if self.randomise:
                kw = dict(miss_rate=self._rng.uniform(0.0, 0.15), ratio_error=self._rng.uniform(0.0, 0.3),
                          lane_discipline=self._rng.uniform(0.6, 1.0))
            self.slot = _AgentSlot()
            self.sim = Simulation(scn, seed=self._rng.randrange(10_000), duration_s=self.duration_s,
                                  record_frames=False, controller_obj=self.slot, **kw)
            self.slot.focal = self._rng.choice(sorted(self.sim.net.junctions))
            self.sim.step(0)
            return self._focal_obs(), {"scenario": scn, "focal": self.slot.focal}

        def _focal_waiting_pcu(self) -> float:
            j = self.sim.net.junctions[self.slot.focal]
            return sum(self.sim.ls[l].queued_pcu() for l in j.in_links.values())

        def step(self, action):
            self.slot.action = int(action)
            before = self._focal_waiting_pcu()
            fair, amb_wait = 0, 0
            for _ in range(DECISION_S):
                t = self.sim.t + 1
                if t >= self.duration_s:
                    break
                self.sim.step(t)
                d = self.sim.decisions.get(self.slot.focal)
                if d is not None and d.mode == "fairness":
                    fair += 1
                a = self.sim.amb
                if a is not None and a.link in self.sim.net.junctions[self.slot.focal].in_links.values() \
                        and any(a in q for q in self.sim.ls[a.link].queues.values()):
                    amb_wait += 1
            after = self._focal_waiting_pcu()
            # reward: queues going down is good, a long queue is bad, starving / blocking an ambulance is very bad
            reward = -(after - before) / 10.0 - after / 100.0 - 0.5 * fair - 2.0 * amb_wait
            terminated = self.sim.t + 1 >= self.duration_s
            return self._focal_obs(), float(reward), terminated, False, {}

except ImportError:          # gymnasium is optional at runtime (only needed for training)
    SignalEnv = None
