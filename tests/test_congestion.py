import numpy as np
import pandas as pd

from ml.training.congestion import (compute_refs, congestion_index, feature_names, features_from_boxes,
                                    fit_and_eval, to_level)

CLASSES = ["car", "van", "truck", "tram"]


def test_features_empty_and_nonempty():
    f0 = features_from_boxes(np.empty((0, 4)), np.empty(0), np.empty(0), CLASSES)
    assert f0["n_total"] == 0 and f0["area_sum"] == 0 and set(f0) == set(feature_names(CLASSES))
    f = features_from_boxes([[0.5, 0.5, 0.2, 0.4], [0.2, 0.3, 0.05, 0.05]], [0, 2], [0.9, 0.5], CLASSES)
    assert f["n_total"] == 2 and f["n_car"] == 1 and f["n_truck"] == 1
    assert f["n_near"] == 1 and f["n_far"] == 1 and abs(f["area_sum"] - (0.08 + 0.0025)) < 1e-9


def test_index_bounds_and_monotonic():
    a = congestion_index([0, 5, 50], [0, 0.1, 5], 10, 0.3)
    assert a[0] == 0 and a[2] == 100 and a[0] < a[1] < a[2]


def test_levels_and_refs(tmp_path):
    for i, n in enumerate([1, 2, 3, 5, 8, 12]):
        (tmp_path / f"{i}.txt").write_text("".join("0 0.5 0.5 0.1 0.1\n" for _ in range(n)))
    refs = compute_refs(tmp_path)
    assert refs["c_ref"] >= 1 and refs["thresholds"][0] < refs["thresholds"][1]
    assert list(to_level([0, 50, 100], [33, 66])) == [0, 1, 2]


def test_fit_and_eval_runs_on_synthetic_data():
    rng = np.random.default_rng(0)

    def make(n):
        cnt = rng.integers(0, 25, n)
        d = pd.DataFrame({c: 0.0 for c in feature_names(CLASSES)}, index=range(n))
        d["n_total"] = np.clip(cnt + rng.integers(-2, 3, n), 0, None)
        d["area_sum"] = cnt * 0.01 + rng.normal(0, 0.01, n).clip(0)
        d["gt_count"], d["gt_area_sum"] = cnt.astype(float), cnt * 0.01
        d["gt_index"] = congestion_index(d["gt_count"], d["gt_area_sum"], 15, 0.15)
        d["id"] = [str(i) for i in range(n)]
        return d

    refs = {"c_ref": 15, "a_ref": 0.15, "thresholds": [33.0, 66.0]}
    model, m, out = fit_and_eval(make(200), make(100), CLASSES, refs, folds=3)
    assert m["index_mae"] < m["baseline_detector_index_mae"] * 2 and 0 <= m["level_accuracy"] <= 1
    assert len(out) == 100 and {"pred_index", "level_pred"} <= set(out.columns)
