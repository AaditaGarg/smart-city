"""Evaluate a trained YOLO model on the held-out TEST split (never used for training or model selection).

    python -m ml.evaluation.evaluate_yolo --task vehicle
    python -m ml.evaluation.evaluate_yolo --task road_damage --also-cpu

Writes (all computed from your weights and data):
  reports/tables/<task>_<split>_metrics.csv          per-class + macro precision/recall/F1/mAP, instance counts
  reports/tables/<task>_<split>_confusion_matrix.csv
  reports/tables/<task>_<split>_summary.json         overall metrics, latency, weights, device, epochs trained
  reports/figures/<task>_<split>_{confusion_matrix,PR_curve,F1_curve,...}.png   (Ultralytics plots)
  reports/figures/<task>_training_curves.png         from runs/<task>/results.csv
  reports/figures/<task>_<split>_predictions.png     ground truth vs prediction on sample test images

Notes: mAP is computed at conf=0.001 (standard). Precision/recall/F1 are the values Ultralytics reports at its
best-F1 confidence; the confusion matrix and prediction images use conf=0.25.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ml.config import ROOT, load_config

from .common import (add_overall, clean, count_instances, latency_stats, per_class_table, plot_training_curves)

UL_PLOTS = ["confusion_matrix.png", "confusion_matrix_normalized.png", "PR_curve.png", "F1_curve.png",
            "P_curve.png", "R_curve.png"]


def benchmark(model, images, imgsz, device, warmup=10, n=100) -> dict:
    paths = [str(p) for p in images[: warmup + n]]
    for p in paths[:warmup]:
        model.predict(p, imgsz=imgsz, device=device, verbose=False)
    ms = []
    for p in paths[warmup:]:
        t = time.perf_counter()
        model.predict(p, imgsz=imgsz, device=device, verbose=False)
        ms.append((time.perf_counter() - t) * 1000)
    return latency_stats(ms)


def _draw(img, boxes, color, names, scores=None):
    import cv2
    H, W = img.shape[:2]
    th = max(1, round(min(H, W) / 300))
    for k, (cls, x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, th)
        txt = names[int(cls)] + (f" {scores[k]:.2f}" if scores is not None else "")
        cv2.putText(img, txt, (int(x1), max(12, int(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45 * th ** 0.3, color, 1, cv2.LINE_AA)


def prediction_grid(model, split_dir: Path, split: str, names: dict, out_png: Path, n=5, conf=0.25,
                    imgsz=640, device="cpu", seed=42):
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    imgs = sorted((split_dir / "images" / split).glob("*.*"))

    def has_gt(p):
        lp = split_dir / "labels" / split / f"{p.stem}.txt"
        return lp.exists() and lp.read_text().strip() != ""

    pos = [p for p in imgs if has_gt(p)]
    neg = [p for p in imgs if not has_gt(p)]
    rng = random.Random(seed)
    pick = rng.sample(pos, min(n - 1, len(pos))) + rng.sample(neg, min(1, len(neg)))
    fig, axes = plt.subplots(len(pick), 2, figsize=(11, 3.4 * len(pick)), squeeze=False)
    for row, p in enumerate(pick):
        img = cv2.imread(str(p))
        H, W = img.shape[:2]
        gt = []
        for ln in (split_dir / "labels" / split / f"{p.stem}.txt").read_text().splitlines():
            if ln.strip():
                c, xc, yc, w, h = ln.split()[:5]
                xc, yc, w, h = float(xc) * W, float(yc) * H, float(w) * W, float(h) * H
                gt.append((int(c), xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2))
        left = img.copy()
        _draw(left, gt, (0, 200, 0), names)
        r = model.predict(str(p), imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        right = img.copy()
        b = r.boxes
        pred = [(int(c), *xy) for c, xy in zip(b.cls.cpu().numpy(), b.xyxy.cpu().numpy())]
        _draw(right, pred, (0, 0, 255), names, b.conf.cpu().numpy())
        for col, im, ttl in [(0, left, f"ground truth ({len(gt)})"), (1, right, f"prediction ({len(pred)}) conf>={conf}")]:
            axes[row][col].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            axes[row][col].set_title(f"{p.stem}: {ttl}", fontsize=9)
            axes[row][col].axis("off")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main(argv=None):
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["vehicle", "road_damage"], required=True)
    ap.add_argument("--weights")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=cfg.get("device_hint", "auto"))
    ap.add_argument("--also-cpu", action="store_true", help="also benchmark latency on CPU")
    ap.add_argument("--no-latency", action="store_true")
    a = ap.parse_args(argv)

    from ultralytics import YOLO

    from ml.training.train_yolo import pick_device

    dc = cfg[a.task]
    out_dir = ROOT / dc["out_dir"]
    weights = Path(a.weights) if a.weights else ROOT / "models" / f"{a.task}_best.pt"
    if not weights.exists():
        raise SystemExit(f"{weights} not found. Train first (python -m ml.training.train_yolo --task {a.task}).")
    meta_p = ROOT / "models" / f"{a.task}_train_meta.json"
    imgsz = a.imgsz or (json.loads(meta_p.read_text())["train_args"]["imgsz"] if meta_p.exists() else dc["train"]["imgsz"])
    device = pick_device(a.device)
    tag = f"{a.task}_{a.split}"
    tables, figs = ROOT / "reports" / "tables", ROOT / "reports" / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    m = model.val(data=str(out_dir / "data.yaml"), split=a.split, imgsz=imgsz, batch=a.batch, conf=0.001, iou=0.6,
                  device=device, plots=True, project=str(ROOT / "runs" / "eval"), name=tag, exist_ok=True)
    names = {int(k): v for k, v in m.names.items()}
    inst, imgs_per_class, n_img = count_instances(out_dir / "labels" / a.split, len(names))
    box = m.box
    df = per_class_table(names, box.ap_class_index, box.p, box.r, box.ap50, box.ap, inst, imgs_per_class)
    df = add_overall(df, n_img, box.mp, box.mr, box.map50, box.map)
    df.to_csv(tables / f"{tag}_metrics.csv", index=False)

    cm = np.asarray(m.confusion_matrix.matrix)
    labels = [names[i] for i in sorted(names)] + ["background"]
    pd.DataFrame(cm, index=[f"pred_{l}" for l in labels], columns=[f"true_{l}" for l in labels]).to_csv(
        tables / f"{tag}_confusion_matrix.csv")
    for fn in UL_PLOTS:
        src = Path(m.save_dir) / fn
        if src.exists():
            shutil.copy2(src, figs / f"{tag}_{fn}")

    results_csv = ROOT / "runs" / a.task / "results.csv"
    epochs = plot_training_curves(results_csv, figs / f"{a.task}_training_curves.png",
                                  f"{a.task}: training curves (validation split)") if results_csv.exists() else None

    split_imgs = sorted((out_dir / "images" / a.split).glob("*.*"))
    latency = {}
    if not a.no_latency:
        latency[str(device)] = benchmark(model, split_imgs, imgsz, device)
        if a.also_cpu and str(device) != "cpu":
            latency["cpu"] = benchmark(YOLO(str(weights)), split_imgs, imgsz, "cpu")
    prediction_grid(model, out_dir, a.split, names, figs / f"{tag}_predictions.png", imgsz=imgsz, device=device)

    overall = df.iloc[-1]
    summary = {"task": a.task, "split": a.split, "weights": str(weights),
               "weights_mb": round(weights.stat().st_size / 1e6, 1), "epochs_trained": epochs, "imgsz": imgsz,
               "device": str(device), "images_evaluated": n_img,
               "overall": {k: clean(overall[k]) for k in ["precision", "recall", "f1", "mAP50", "mAP50-95"]},
               "latency_ms_per_image": latency,
               "ultralytics_speed_ms_per_image": {k: float(v) for k, v in dict(m.speed).items()},
               "low_support_classes": df.loc[df["low_support"], "class"].tolist()}
    (tables / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print(df.to_string(index=False))
    print(json.dumps({k: summary[k] for k in ("overall", "latency_ms_per_image", "low_support_classes")}, indent=2))


if __name__ == "__main__":
    main()
