"""Evaluate the congestion model from reports/tables/congestion_test_predictions.csv (written by
ml.training.congestion). Produces a 3-panel figure, a level confusion-matrix CSV and a metrics table.

    python -m ml.evaluation.eval_congestion
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from ml.config import ROOT, load_config


def evaluate(pred_csv: Path, metrics_json: Path, levels, out_fig: Path, out_tables: Path) -> pd.DataFrame:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(pred_csv)
    m = json.loads(Path(metrics_json).read_text())
    cm = confusion_matrix(df["level_true"], df["level_pred"], labels=list(range(len(levels))))
    out_tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cm, index=[f"true_{l}" for l in levels], columns=[f"pred_{l}" for l in levels]).to_csv(
        out_tables / "congestion_level_confusion_matrix.csv")
    rep = classification_report(df["level_true"], df["level_pred"], labels=list(range(len(levels))),
                                target_names=levels, output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(out_tables / "congestion_level_report.csv")

    table = pd.DataFrame([
        {"metric": "index MAE (0-100)", "learned model": m["index_mae"], "detector-only baseline": m["baseline_detector_index_mae"]},
        {"metric": "index RMSE (0-100)", "learned model": m["index_rmse"], "detector-only baseline": m["baseline_detector_index_rmse"]},
        {"metric": "level accuracy", "learned model": m["level_accuracy"], "detector-only baseline": m["baseline_level_accuracy"]},
        {"metric": "level macro F1", "learned model": m["level_macro_f1"], "detector-only baseline": m["baseline_level_macro_f1"]},
        {"metric": "vehicle count MAE (raw detector)", "learned model": np.nan, "detector-only baseline": m["vehicle_count_mae_detector"]},
        {"metric": "vehicle count RMSE (raw detector)", "learned model": np.nan, "detector-only baseline": m["vehicle_count_rmse_detector"]},
    ])
    table.to_csv(out_tables / "congestion_metrics_table.csv", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].scatter(df["gt_index"], df["pred_index"], s=8, alpha=0.5)
    ax[0].plot([0, 100], [0, 100], "r--", lw=1)
    ax[0].set(xlabel="proxy index from annotations", ylabel="predicted index",
              title=f"Index (MAE {m['index_mae']:.1f}, RMSE {m['index_rmse']:.1f})")
    ax[1].imshow(cm, cmap="Blues")
    ax[1].set(xticks=range(len(levels)), yticks=range(len(levels)), xticklabels=levels, yticklabels=levels,
              xlabel="predicted level", ylabel="true level", title=f"Levels (acc {m['level_accuracy']:.2f})")
    for i in range(len(levels)):
        for j in range(len(levels)):
            ax[1].text(j, i, int(cm[i, j]), ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
    jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(df))
    ax[2].scatter(df["gt_count"] + jitter, df["n_total"] + jitter, s=8, alpha=0.5)
    top = max(df["gt_count"].max(), df["n_total"].max())
    ax[2].plot([0, top], [0, top], "r--", lw=1)
    ax[2].set(xlabel="annotated vehicles", ylabel="detected vehicles",
              title=f"Count (detector MAE {m['vehicle_count_mae_detector']:.2f})")
    fig.tight_layout()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=140)
    plt.close(fig)
    return table


def main():
    cfg = load_config()
    t = ROOT / "reports" / "tables"
    p, mj = t / "congestion_test_predictions.csv", t / "congestion_metrics.json"
    if not p.exists():
        raise SystemExit("Run `python -m ml.training.congestion` first.")
    print(evaluate(p, mj, cfg["congestion"]["levels"], ROOT / "reports" / "figures" / "congestion_evaluation.png", t)
          .to_string(index=False))


if __name__ == "__main__":
    main()
