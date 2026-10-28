"""Fine-tune a COCO-pretrained YOLO model on a prepared dataset.

    python -m ml.training.train_yolo --task vehicle --quick        # smoke test: tiny subset, 2 epochs
    python -m ml.training.train_yolo --task vehicle
    python -m ml.training.train_yolo --task road_damage
    python -m ml.training.train_yolo --task road_damage --batch 8  # if you run out of memory

Outputs: runs/<name>/ (results.csv, curves, weights), models/<name>_best.pt and models/<name>_train_meta.json
(real wall-clock time, versions, final validation metrics reported by Ultralytics).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # let unsupported ops fall back to CPU on Apple GPUs

from ml.config import ROOT, load_config  # noqa: E402


def pick_device(pref: str | None = None):
    if pref and pref != "auto":
        return int(pref) if pref.isdigit() else pref
    import torch
    if torch.cuda.is_available():
        return 0
    mps = getattr(torch.backends, "mps", None)
    return "mps" if mps and mps.is_available() else "cpu"


def build_train_args(tc: dict, cli: argparse.Namespace, seed: int) -> dict:
    epochs = cli.epochs or tc["epochs"]
    imgsz = cli.imgsz or tc["imgsz"]
    batch = cli.batch or tc["batch"]
    fraction = cli.fraction or 1.0
    if cli.quick:
        epochs, imgsz, fraction = 2, 320, 0.05
    hyp = dict(tc.get("hyp", {}))
    hyp["close_mosaic"] = min(hyp.get("close_mosaic", 10), max(epochs - 1, 0))
    return dict(epochs=epochs, imgsz=imgsz, batch=batch, fraction=fraction, patience=tc.get("patience", 10),
                workers=tc.get("workers", 2), seed=seed, plots=True, **hyp)


def main(argv=None):
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["vehicle", "road_damage"], required=True)
    ap.add_argument("--model", help="override pretrained weights, e.g. yolo11s.pt")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--fraction", type=float, help="train on this fraction of the train split")
    ap.add_argument("--device", default=cfg.get("device_hint", "auto"))
    ap.add_argument("--quick", action="store_true", help="smoke test: 5%% of data, 2 epochs, imgsz 320")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)

    tc = cfg[a.task]["train"]
    data = ROOT / cfg[a.task]["out_dir"] / "data.yaml"
    if not data.exists():
        sys.exit(f"{data} not found. Run the preparation step first "
                 f"(python -m ml.preprocessing.prepare_{'kitti' if a.task == 'vehicle' else 'rdd'}).")

    from ultralytics import YOLO
    import torch
    import ultralytics

    name = f"{a.task}{'_quick' if a.quick else ''}"
    device = pick_device(a.device)
    args = build_train_args(tc, a, cfg["seed"])
    print(f"task={a.task} device={device} data={data}\nargs={args}")

    last = ROOT / "runs" / name / "weights" / "last.pt"
    model = YOLO(str(last) if a.resume and last.exists() else (a.model or tc["model"]))
    t0 = time.time()
    if a.resume and last.exists():
        results = model.train(resume=True)
    else:
        results = model.train(data=str(data), device=device, project=str(ROOT / "runs"), name=name,
                              exist_ok=True, **args)
    wall = time.time() - t0

    best = Path(getattr(model.trainer, "best", "")) if getattr(model, "trainer", None) else None
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    if best and best.exists():
        shutil.copy2(best, models_dir / f"{name}_best.pt")
    final = {k: float(v) for k, v in (getattr(results, "results_dict", {}) or {}).items()}
    meta = {"task": a.task, "name": name, "pretrained": a.model or tc["model"], "device": str(device),
            "train_args": args, "wall_clock_seconds": round(wall, 1), "ultralytics": ultralytics.__version__,
            "torch": torch.__version__, "python": platform.python_version(), "machine": platform.platform(),
            "final_val_metrics": final, "best_weights": str(models_dir / f"{name}_best.pt")}
    (models_dir / f"{name}_train_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDone in {wall / 60:.1f} min. Weights: models/{name}_best.pt | curves: runs/{name}/")


if __name__ == "__main__":
    main()
