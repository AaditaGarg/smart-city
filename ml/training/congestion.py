"""Congestion estimation with scikit-learn, driven by vehicle-detector outputs.

IMPORTANT - what the target is: KITTI has no congestion labels. The training target is therefore a
*proxy index* computed from the ANNOTATED vehicles in each image (50% vehicle count, 50% fraction of the image
covered by vehicle boxes, each scaled by the train-split 95th percentile and mapped to 0-100). Low/moderate/heavy
levels are the train-split tertiles of that index. The regressor learns to recover this index from what the
DETECTOR sees (counts per class, box areas, near/far split, ...), so its error measures how well detector output
reproduces annotated density - not agreement with human-rated traffic. Reported as such.

Protocol: the detector was fitted on the train split, so its train-split outputs are optimistically clean.
The regressor is therefore fitted on VAL-split detector outputs (model chosen by K-fold CV on val) and scored on the
untouched TEST split.

    python -m ml.training.congestion --weights models/vehicle_best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict

from ml.config import ROOT, load_config


def feature_names(classes):
    return (["n_total"] + [f"n_{c}" for c in classes] +
            ["area_sum", "area_mean", "area_max", "mean_conf", "n_near", "n_far", "mean_cy"])


def features_from_boxes(xywhn, cls, conf, classes) -> dict:
    xywhn = np.asarray(xywhn, dtype=float).reshape(-1, 4)
    cls, conf = np.asarray(cls).reshape(-1), np.asarray(conf, dtype=float).reshape(-1)
    n = len(cls)
    areas = xywhn[:, 2] * xywhn[:, 3]
    f = {"n_total": float(n)}
    for i, c in enumerate(classes):
        f[f"n_{c}"] = float((cls == i).sum())
    f.update(area_sum=float(areas.sum()), area_mean=float(areas.mean()) if n else 0.0,
             area_max=float(areas.max()) if n else 0.0, mean_conf=float(conf.mean()) if n else 0.0,
             n_near=float((xywhn[:, 3] > 0.25).sum()), n_far=float((xywhn[:, 3] < 0.10).sum()),
             mean_cy=float(xywhn[:, 1].mean()) if n else 0.0)
    return f


def read_gt(label_path: Path) -> np.ndarray:
    rows = [ln.split()[:5] for ln in Path(label_path).read_text().splitlines() if ln.strip()]
    return np.array(rows, dtype=float).reshape(-1, 5)


def congestion_index(count, area_sum, c_ref, a_ref):
    count, area_sum = np.asarray(count, dtype=float), np.asarray(area_sum, dtype=float)
    return 100.0 * (0.5 * np.minimum(count / c_ref, 1.0) + 0.5 * np.minimum(area_sum / a_ref, 1.0))


def to_level(index, thresholds):
    return np.digitize(np.asarray(index, dtype=float), thresholds)


def compute_refs(label_dir: Path) -> dict:
    counts, areas = [], []
    for lp in sorted(Path(label_dir).glob("*.txt")):
        if "_aug" in lp.stem:
            continue
        gt = read_gt(lp)
        counts.append(len(gt))
        areas.append(float((gt[:, 3] * gt[:, 4]).sum()))
    c_ref, a_ref = max(1.0, float(np.percentile(counts, 95))), max(1e-6, float(np.percentile(areas, 95)))
    idx = congestion_index(counts, areas, c_ref, a_ref)
    t1, t2 = (float(v) for v in np.percentile(idx, [100 / 3, 200 / 3]))
    return {"c_ref": c_ref, "a_ref": a_ref, "thresholds": [t1, t2]}


def extract_features(weights, split_dir: Path, split: str, classes, refs, conf, imgsz, device,
                     model=None, chunk: int = 16) -> pd.DataFrame:
    """Run the detector over a split in small chunks (a single huge batch exhausts GPU memory)."""
    if model is None:
        from ultralytics import YOLO
        model = YOLO(str(weights))
    imgs = sorted((split_dir / "images" / split).glob("*.*"))
    rows = []
    for i in range(0, len(imgs), chunk):
        batch_paths = [str(p) for p in imgs[i:i + chunk]]
        for r in model.predict(source=batch_paths, conf=conf, imgsz=imgsz, device=device,
                               batch=chunk, verbose=False):
            stem = Path(r.path).stem
            b = r.boxes
            f = features_from_boxes(b.xywhn.cpu().numpy(), b.cls.cpu().numpy(), b.conf.cpu().numpy(), classes)
            gt = read_gt(split_dir / "labels" / split / f"{stem}.txt")
            f.update(id=stem, gt_count=float(len(gt)), gt_area_sum=float((gt[:, 3] * gt[:, 4]).sum()),
                     infer_ms=float(r.speed.get("inference", 0.0)))
            rows.append(f)
    df = pd.DataFrame(rows)
    df["gt_index"] = congestion_index(df["gt_count"], df["gt_area_sum"], refs["c_ref"], refs["a_ref"])
    return df


def _rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def fit_and_eval(df_val, df_test, classes, refs, seed=42, folds=5):
    X_cols = feature_names(classes)
    Xv, yv, Xt, yt = df_val[X_cols].values, df_val["gt_index"].values, df_test[X_cols].values, df_test["gt_index"].values
    candidates = {"random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1),
                  "gradient_boosting": GradientBoostingRegressor(random_state=seed)}
    cv = KFold(folds, shuffle=True, random_state=seed)
    cv_table = {n: float(mean_absolute_error(yv, cross_val_predict(m, Xv, yv, cv=cv))) for n, m in candidates.items()}
    best = min(cv_table, key=cv_table.get)
    model = candidates[best].fit(Xv, yv)
    pred = np.clip(model.predict(Xt), 0, 100)

    det_idx = congestion_index(df_test["n_total"], df_test["area_sum"], refs["c_ref"], refs["a_ref"])
    lv_true, lv_pred, lv_base = (to_level(v, refs["thresholds"]) for v in (yt, pred, det_idx))
    metrics = {"selected_model": best, "cv_mae_on_val": cv_table, "n_val": len(df_val), "n_test": len(df_test),
               "index_mae": float(mean_absolute_error(yt, pred)), "index_rmse": _rmse(yt, pred),
               "baseline_detector_index_mae": float(mean_absolute_error(yt, det_idx)),
               "baseline_detector_index_rmse": _rmse(yt, det_idx),
               "vehicle_count_mae_detector": float(mean_absolute_error(df_test["gt_count"], df_test["n_total"])),
               "vehicle_count_rmse_detector": _rmse(df_test["gt_count"], df_test["n_total"]),
               "level_accuracy": float(accuracy_score(lv_true, lv_pred)),
               "level_macro_f1": float(f1_score(lv_true, lv_pred, average="macro")),
               "baseline_level_accuracy": float(accuracy_score(lv_true, lv_base)),
               "baseline_level_macro_f1": float(f1_score(lv_true, lv_base, average="macro"))}
    out = df_test[["id", "gt_count", "n_total", "gt_index"]].copy()
    out["pred_index"], out["level_true"], out["level_pred"] = pred, lv_true, lv_pred
    return model, metrics, out


def main(argv=None):
    cfg = load_config()
    cc, classes = cfg["congestion"], cfg["vehicle"]["classes"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models" / "vehicle_best.pt"))
    ap.add_argument("--device", default=cfg.get("device_hint", "auto"))
    ap.add_argument("--reuse-features", action="store_true")
    a = ap.parse_args(argv)

    vdir = ROOT / cfg["vehicle"]["out_dir"]
    fdir = ROOT / cc["out_dir"]
    fdir.mkdir(parents=True, exist_ok=True)
    refs = compute_refs(vdir / "labels" / "train")
    dfs = {}
    for split in ("val", "test"):
        csv = fdir / f"{split}_features.csv"
        if a.reuse_features and csv.exists():
            dfs[split] = pd.read_csv(csv, dtype={"id": str})
        else:
            from ml.training.train_yolo import pick_device
            dfs[split] = extract_features(a.weights, vdir, split, classes, refs, cc["conf"],
                                          cfg["image_size"], pick_device(a.device))
            dfs[split].to_csv(fdir / f"{split}_features.csv", index=False)
    model, metrics, preds = fit_and_eval(dfs["val"], dfs["test"], classes, refs, cfg["seed"], cc["cv_folds"])

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": feature_names(classes), "refs": refs, "levels": cc["levels"]},
                ROOT / "models" / "congestion_model.joblib")
    tables = ROOT / "reports" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    preds.to_csv(tables / "congestion_test_predictions.csv", index=False)
    (tables / "congestion_metrics.json").write_text(json.dumps({**metrics, "refs": refs}, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
