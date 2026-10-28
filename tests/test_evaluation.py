import json

import numpy as np
import pandas as pd

from ml.evaluation.common import (add_overall, count_instances, df_to_md, f1, latency_stats, per_class_table,
                                  plot_training_curves)
from ml.evaluation.eval_congestion import evaluate


def test_f1_and_latency():
    assert f1(0.5, 0.5) == 0.5 and f1(0, 0) == 0.0
    s = latency_stats([10, 20, 30, 40])
    assert s["mean_ms"] == 25 and s["median_ms"] == 25 and abs(s["fps_from_mean"] - 40) < 1e-9


def test_count_instances_skips_aug(tmp_path):
    (tmp_path / "a.txt").write_text("0 .5 .5 .1 .1\n0 .2 .2 .1 .1\n1 .5 .5 .1 .1\n")
    (tmp_path / "b.txt").write_text("")
    (tmp_path / "a_aug0.txt").write_text("0 .5 .5 .1 .1\n")
    inst, imgs, n = count_instances(tmp_path, 3)
    assert list(inst) == [2, 1, 0] and list(imgs) == [1, 1, 0] and n == 2


def test_per_class_table_handles_missing_class_and_low_support():
    names = {0: "car", 1: "tram", 2: "ghost"}
    df = per_class_table(names, [0, 1], np.array([0.9, 0.5]), np.array([0.8, 0.5]), np.array([0.85, 0.4]),
                         np.array([0.6, 0.3]), np.array([100, 10, 0]), np.array([50, 8, 0]))
    assert np.isnan(df.loc[2, "precision"]) and df.loc[1, "low_support"] and not df.loc[0, "low_support"]
    full = add_overall(df, 60, 0.7, 0.65, 0.62, 0.45)
    assert full.iloc[-1]["class"] == "ALL (macro)" and abs(full.iloc[-1]["f1"] - np.mean([f1(.9, .8), f1(.5, .5)])) < 1e-9
    md = df_to_md(full)
    assert md.count("\n") == len(full) + 1 and "–" in md


def test_training_curves_plot(tmp_path):
    cols = ["epoch", "train/box_loss", "val/box_loss", "train/cls_loss", "val/cls_loss",
            "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
    pd.DataFrame([[i, 1 / (i + 1), 1.1 / (i + 1), 2 / (i + 1), 2.1 / (i + 1), .5, .5, .6, .4] for i in range(1, 6)],
                 columns=[" " + c for c in cols]).to_csv(tmp_path / "results.csv", index=False)
    assert plot_training_curves(tmp_path / "results.csv", tmp_path / "c.png", "t") == 5 and (tmp_path / "c.png").exists()


def test_congestion_eval_on_synthetic_predictions(tmp_path):
    rng = np.random.default_rng(1)
    n = 60
    gt = rng.uniform(0, 100, n)
    df = pd.DataFrame({"id": range(n), "gt_count": rng.integers(0, 20, n), "n_total": rng.integers(0, 20, n),
                       "gt_index": gt, "pred_index": gt + rng.normal(0, 5, n)})
    df["level_true"] = np.digitize(df["gt_index"], [33, 66])
    df["level_pred"] = np.digitize(df["pred_index"], [33, 66])
    df.to_csv(tmp_path / "p.csv", index=False)
    keys = ["index_mae", "index_rmse", "baseline_detector_index_mae", "baseline_detector_index_rmse", "level_accuracy",
            "level_macro_f1", "baseline_level_accuracy", "baseline_level_macro_f1", "vehicle_count_mae_detector",
            "vehicle_count_rmse_detector"]
    (tmp_path / "m.json").write_text(json.dumps({k: 1.0 for k in keys}))
    t = evaluate(tmp_path / "p.csv", tmp_path / "m.json", ["free", "moderate", "heavy"], tmp_path / "f.png", tmp_path)
    assert len(t) == 6 and (tmp_path / "f.png").exists() and (tmp_path / "congestion_level_confusion_matrix.csv").exists()
