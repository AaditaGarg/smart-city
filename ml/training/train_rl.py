"""Train the shared RL signal policy (DQN) and decide whether it may be deployed.

    python ml/training/train_rl.py --steps 60000          # ~30-60 min on a laptop CPU
    python ml/training/train_rl.py --eval-only            # re-run the held-out comparison

Deployment rule: the RL agent is only marked deployable if it beats Max-Pressure on average waiting
time over held-out seeds the agent never trained on. Otherwise the system stays on Max-Pressure.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from decision_engine.network_sim import run_comparison  # noqa: E402
from decision_engine.rl_env import SignalEnv  # noqa: E402

MODEL = ROOT / "models" / "rl_signal_agent.zip"
EVAL = ROOT / "models" / "rl_eval.json"
EVAL_SCENARIOS = ["morning_rush", "two_vs_hundred", "balanced", "grid_rush"]
HELD_OUT_SEEDS = range(1000, 1004)


def train(steps: int, seed: int) -> None:
    from stable_baselines3 import DQN
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(SignalEnv(duration_s=600))
    model = DQN("MlpPolicy", env, learning_rate=5e-4, buffer_size=50_000, learning_starts=2_000, batch_size=64,
                gamma=0.95, train_freq=4, target_update_interval=1_000, exploration_fraction=0.3,
                exploration_final_eps=0.05, policy_kwargs=dict(net_arch=[64, 64]), seed=seed, verbose=1,
                device="cpu")
    t0 = time.time()
    model.learn(total_timesteps=steps, log_interval=20)
    MODEL.parent.mkdir(exist_ok=True)
    model.save(str(MODEL))
    print(f"saved {MODEL} after {steps} steps in {(time.time() - t0) / 60:.1f} min")


def evaluate() -> dict:
    rows, rl_better = {}, []
    for scn in EVAL_SCENARIOS:
        res = run_comparison(scn, ("max_pressure", "rl"), HELD_OUT_SEEDS, duration_s=900)
        mp, rl = res["max_pressure"]["avg_wait_s"]["mean"], res["rl"]["avg_wait_s"]["mean"]
        rows[scn] = {"max_pressure_avg_wait_s": mp, "rl_avg_wait_s": rl,
                     "max_pressure_worst_s": res["max_pressure"]["worst_approach_avg_wait_s"]["mean"],
                     "rl_worst_s": res["rl"]["worst_approach_avg_wait_s"]["mean"]}
        rl_better.append(rl < mp)
        print(f"{scn:16s} max-pressure {mp:6.1f}s   RL {rl:6.1f}s")
    mp_mean = sum(r["max_pressure_avg_wait_s"] for r in rows.values()) / len(rows)
    rl_mean = sum(r["rl_avg_wait_s"] for r in rows.values()) / len(rows)
    out = {"held_out_seeds": list(HELD_OUT_SEEDS), "scenarios": rows,
           "mean_avg_wait_s": {"max_pressure": round(mp_mean, 1), "rl": round(rl_mean, 1)},
           "deploy_rl": rl_mean < mp_mean,
           "rule": "RL is deployed only if its mean waiting time over held-out seeds beats Max-Pressure"}
    EVAL.write_text(json.dumps(out, indent=2))
    print(("DEPLOY RL" if out["deploy_rl"] else "KEEP MAX-PRESSURE") + f"  (MP {mp_mean:.1f}s vs RL {rl_mean:.1f}s)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args()
    if not a.eval_only:
        train(a.steps, a.seed)
    evaluate()
