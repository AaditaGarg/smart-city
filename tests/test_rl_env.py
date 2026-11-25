"""The RL agent can only pick phases; the safety layer still enforces every rule."""
import numpy as np
import pytest

from decision_engine.network_control import MIN_GREEN_S, RLController
from decision_engine.network_sim import Simulation
from decision_engine.rl_env import OBS_SIZE, SignalEnv, observe

gym = pytest.importorskip("gymnasium")


def test_env_steps_and_shapes():
    env = SignalEnv(duration_s=120)
    obs, info = env.reset(seed=1)
    assert obs.shape == (OBS_SIZE,) and obs.dtype == np.float32
    total = 0.0
    for _ in range(10):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        total += r
        assert env.observation_space.contains(obs)
        if term:
            break
    assert np.isfinite(total)


def test_random_agent_never_breaks_minimum_green():
    env = SignalEnv(scenarios=["morning_rush"], duration_s=400, randomise=False)
    env.reset(seed=3)
    rng = np.random.default_rng(0)
    focal = env.slot.focal
    head = env.sim.heads[focal]
    shortest = 1e9
    last_state = head.state
    while True:
        _, _, term, _, _ = env.step(int(rng.integers(0, 4)))     # adversarial: flips every 5 s
        if head.state == "yellow" and last_state == "green":
            g = head.state_t0 - head.green_t0
            if not head.junction.phases[head.phase].pedestrian:
                shortest = min(shortest, g)
        last_state = head.state
        if term:
            break
    assert shortest >= MIN_GREEN_S - 2    # decisions are taken every 2 s


def test_missing_agent_falls_back_to_max_pressure(tmp_path):
    ctrl = RLController(model_path=tmp_path / "nope.zip")
    sim = Simulation("balanced", controller_obj=ctrl, duration_s=60, record_frames=False)
    v = sim.view("J1", 0)
    idx, why = ctrl.choose(v)
    assert "max-pressure" in why
    assert observe(v).shape == (OBS_SIZE,)
