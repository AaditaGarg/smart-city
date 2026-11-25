"""Measure every controller on every scenario and write tables + charts for the report and the deck.

    python scripts/make_network_report.py              # 5 seeds per run
    python scripts/make_network_report.py --seeds 3 --quick

Writes reports/tables/network_comparison.json, network_metrics.csv, reports/figures/network_*.png
and reports/NETWORK_RESULTS.md. All numbers are simulation results (see decision_engine/network_sim.py).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from decision_engine.network_sim import list_scenarios, run_comparison, run_scenario  # noqa: E402

TABLES, FIGS = ROOT / "reports" / "tables", ROOT / "reports" / "figures"
LABEL = {"fixed": "Fixed timer", "webster": "Webster (1 junction)", "max_pressure": "Max-Pressure (network)",
         "rl": "AI / RL"}
COLOR = {"fixed": "#9aa0a6", "webster": "#e8a33d", "max_pressure": "#1f8f5f", "rl": "#4f6bd8"}


def bar_chart(results: dict, metric: str, title: str, path: Path) -> None:
    scns = list(results)
    ctrls = list(next(iter(results.values())))
    w = 0.8 / len(ctrls)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for k, c in enumerate(ctrls):
        means = [results[s][c].get(metric, {}).get("mean", 0) for s in scns]
        stds = [results[s][c].get(metric, {}).get("std", 0) for s in scns]
        ax.bar([i + k * w for i in range(len(scns))], means, w, yerr=stds, capsize=3, label=LABEL[c], color=COLOR[c])
    ax.set_xticks([i + w * (len(ctrls) - 1) / 2 for i in range(len(scns))])
    ax.set_xticklabels([s.replace("_", " ") for s in scns], rotation=15)
    ax.set_title(title)
    ax.set_ylabel("seconds")
    ax.legend(frameon=False, ncol=len(ctrls))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def queue_timeline(path: Path) -> dict:
    """Queue on the busy side road (J2 North) over time: fixed timer vs Max-Pressure."""
    out = {}
    fig, ax = plt.subplots(figsize=(10, 4))
    for c in ("fixed", "max_pressure"):
        r = run_scenario("two_vs_hundred", c, seed=0)
        ts = [f["t"] / 60 for f in r["frames"]]
        north = [f["queues"].get("src_N->J2", 0) for f in r["frames"]]
        main = [f["queues"].get("J1->J2", 0) for f in r["frames"]]
        ax.plot(ts, north, color=COLOR[c], label=f"{LABEL[c]} - busy side road (N)")
        ax.plot(ts, main, color=COLOR[c], ls="--", alpha=0.7, label=f"{LABEL[c]} - near-empty main road")
        out[c] = {"peak_north_queue": max(north), "end_north_queue": north[-1]}
    ax.set_xlabel("minutes")
    ax.set_ylabel("vehicles waiting")
    ax.set_title("'2 vs 100' at Vikas Marg signal 2: who is left waiting?")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return out


def ambulance_test(seeds) -> dict:
    on = [run_scenario("ambulance", "max_pressure", s, corridor=True, record_frames=False)["metrics"]["ambulance_travel_s"]
          for s in seeds]
    off = [run_scenario("ambulance", "max_pressure", s, corridor=False, record_frames=False)["metrics"]["ambulance_travel_s"]
           for s in seeds]
    fixed = [run_scenario("ambulance", "fixed", s, record_frames=False)["metrics"]["ambulance_travel_s"] for s in seeds]
    avg = lambda xs: round(sum(x for x in xs if x is not None) / max(1, sum(x is not None for x in xs)), 1)
    return {"fixed_no_corridor_s": avg(fixed), "max_pressure_no_corridor_s": avg(off),
            "max_pressure_with_corridor_s": avg(on), "route_length_m": 410 + 650 + 970 + 600}


def sensitivity(seeds) -> dict:
    cases = {
        "ideal cameras": {},
        "15% missed detections": {"miss_rate": 0.15},
        "turn ratios 30% wrong (learning on)": {"ratio_error": 0.3},
        "turn ratios 30% wrong (learning off)": {"ratio_error": 0.3, "learn_ratios": False},
        "chaotic lane discipline": {"lane_discipline": 0.6},
    }
    out = {}
    for name, kw in cases.items():
        res = run_comparison("morning_rush", ("fixed", "max_pressure"), seeds, duration_s=1200, **kw)
        out[name] = {c: res[c]["avg_wait_s"]["mean"] for c in res}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="shorter runs for a fast check")
    a = ap.parse_args()
    seeds = range(a.seeds)
    ctrls = ["fixed", "webster", "max_pressure"]
    if (ROOT / "models" / "rl_signal_agent.zip").exists():
        ctrls.append("rl")
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = {}
    for s in list_scenarios():
        if s["id"] == "ambulance":
            continue
        dur = min(s["duration_s"], 600) if a.quick else None
        results[s["id"]] = run_comparison(s["id"], ctrls, seeds, duration_s=dur)
        print(f"{s['id']:16s} " + "  ".join(f"{c}={results[s['id']][c]['avg_wait_s']['mean']:.0f}s" for c in ctrls))
    amb = ambulance_test(seeds)
    sens = sensitivity(range(min(3, a.seeds)))
    timeline = queue_timeline(FIGS / "network_two_vs_hundred_queue.png")
    bar_chart(results, "avg_wait_s", "Average waiting time per vehicle (mean ± sd over seeds)", FIGS / "network_avg_wait.png")
    bar_chart(results, "worst_approach_avg_wait_s", "Worst road's average wait (fairness)", FIGS / "network_worst_wait.png")
    bar_chart(results, "p95_wait_s", "95th-percentile waiting time", FIGS / "network_p95_wait.png")

    fig, ax = plt.subplots(figsize=(7, 3.4))
    names = ["Fixed timer", "Max-Pressure,\nno corridor", "Max-Pressure +\ngreen corridor"]
    vals = [amb["fixed_no_corridor_s"], amb["max_pressure_no_corridor_s"], amb["max_pressure_with_corridor_s"]]
    ax.barh(names, vals, color=[COLOR["fixed"], COLOR["webster"], COLOR["max_pressure"]])
    for i, v in enumerate(vals):
        ax.text(v + 2, i, f"{v:.0f}s", va="center")
    ax.set_xlabel("ambulance travel time through the 4 Vikas Marg signals (s)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "network_ambulance.png", dpi=150)
    plt.close(fig)

    out = {"generated_s": round(time.time() - t0, 1), "seeds": a.seeds, "quick": a.quick, "controllers": ctrls,
           "scenarios": {s["id"]: s for s in list_scenarios()}, "results": results, "ambulance": amb,
           "sensitivity_morning_rush_avg_wait_s": sens, "two_vs_hundred_queue": timeline}
    (TABLES / "network_comparison.json").write_text(json.dumps(out, indent=2))
    with open(TABLES / "network_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "controller", "avg_wait_s", "sd", "p95_wait_s", "worst_approach_avg_wait_s",
                    "max_queue_veh", "vehicles_exited", "ped_avg_wait_s"])
        for s, row in results.items():
            for c, m in row.items():
                w.writerow([s, c, m["avg_wait_s"]["mean"], m["avg_wait_s"]["std"], m["p95_wait_s"]["mean"],
                            m["worst_approach_avg_wait_s"]["mean"], m["max_queue_veh"]["mean"],
                            m["vehicles_exited"]["mean"], m.get("ped_avg_wait_s", {}).get("mean")])

    md = ["# Network signal-control results", "",
          f"_Generated by `scripts/make_network_report.py` ({a.seeds} seeds per cell, mean ± sd). "
          "These are **simulation** results on the Vikas Marg corridor (4 real signals from OpenStreetMap) and a "
          "2x2 grid with typical Indian peak-hour volumes, not field measurements._", "",
          "| scenario | " + " | ".join(LABEL[c] for c in ctrls) + " |", "|---|" + "---|" * len(ctrls)]
    for s, row in results.items():
        md.append(f"| {s.replace('_', ' ')} | " + " | ".join(
            f"{row[c]['avg_wait_s']['mean']:.0f} ± {row[c]['avg_wait_s']['std']:.0f} s" for c in ctrls) + " |")
    md += ["", "Average waiting time per vehicle. Worst-road wait (fairness):", "",
           "| scenario | " + " | ".join(LABEL[c] for c in ctrls) + " |", "|---|" + "---|" * len(ctrls)]
    for s, row in results.items():
        md.append(f"| {s.replace('_', ' ')} | " + " | ".join(
            f"{row[c]['worst_approach_avg_wait_s']['mean']:.0f} s" for c in ctrls) + " |")
    md += ["", "## Ambulance green corridor", "",
           f"- Fixed timer: {amb['fixed_no_corridor_s']} s",
           f"- Max-Pressure without corridor: {amb['max_pressure_no_corridor_s']} s",
           f"- Max-Pressure with green corridor: {amb['max_pressure_with_corridor_s']} s", "",
           "## Robustness (morning rush, average wait)", "", "| condition | Fixed | Max-Pressure |", "|---|---|---|"]
    for k, v in sens.items():
        md.append(f"| {k} | {v['fixed']:.0f} s | {v['max_pressure']:.0f} s |")
    md += ["", "![avg wait](figures/network_avg_wait.png)", "![worst](figures/network_worst_wait.png)",
           "![2 vs 100](figures/network_two_vs_hundred_queue.png)", "![ambulance](figures/network_ambulance.png)"]
    (ROOT / "reports" / "NETWORK_RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s -> reports/NETWORK_RESULTS.md")


if __name__ == "__main__":
    main()
