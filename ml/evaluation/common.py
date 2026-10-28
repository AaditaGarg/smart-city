"""Pure helpers for evaluation (no Ultralytics import, unit-tested)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

LOW_SUPPORT = 30   # classes with fewer test instances than this are flagged as low-confidence


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def clean(x):
    """float or None (JSON/NaN safe)."""
    x = float(x)
    return None if math.isnan(x) else x


def count_instances(label_dir: Path, n_classes: int):
    inst, imgs, n_img = np.zeros(n_classes, int), np.zeros(n_classes, int), 0
    for lp in Path(label_dir).glob("*.txt"):
        if "_aug" in lp.stem:
            continue
        n_img += 1
        ids = [int(ln.split()[0]) for ln in lp.read_text().splitlines() if ln.strip()]
        for i in ids:
            inst[i] += 1
        for i in set(ids):
            imgs[i] += 1
    return inst, imgs, n_img


def per_class_table(names: dict, class_index, p, r, ap50, ap, inst, imgs) -> pd.DataFrame:
    pos = {int(c): k for k, c in enumerate(class_index)}
    rows = []
    for i in sorted(int(k) for k in names):
        if i in pos:
            k = pos[i]
            pi, ri = float(p[k]), float(r[k])
            rows.append({"class": names[i], "images": int(imgs[i]), "instances": int(inst[i]), "precision": pi,
                         "recall": ri, "f1": f1(pi, ri), "mAP50": float(ap50[k]), "mAP50-95": float(ap[k])})
        else:
            rows.append({"class": names[i], "images": int(imgs[i]), "instances": int(inst[i]), "precision": np.nan,
                         "recall": np.nan, "f1": np.nan, "mAP50": np.nan, "mAP50-95": np.nan})
    df = pd.DataFrame(rows)
    df["low_support"] = df["instances"] < LOW_SUPPORT
    return df


def add_overall(df: pd.DataFrame, n_images: int, mp, mr, map50, map5095) -> pd.DataFrame:
    valid = df.dropna(subset=["f1"])
    row = {"class": "ALL (macro)", "images": n_images, "instances": int(df["instances"].sum()),
           "precision": float(mp), "recall": float(mr), "f1": float(valid["f1"].mean()) if len(valid) else np.nan,
           "mAP50": float(map50), "mAP50-95": float(map5095), "low_support": False}
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)


def latency_stats(ms) -> dict:
    a = np.asarray(ms, dtype=float)
    return {"n_images": int(len(a)), "mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
            "p95_ms": float(np.percentile(a, 95)), "fps_from_mean": float(1000.0 / a.mean())}


def df_to_md(df: pd.DataFrame, nd: int = 3) -> str:
    def fmt(v):
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else ""
        if isinstance(v, (float, np.floating)):
            return "–" if math.isnan(v) else f"{v:.{nd}f}"
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def plot_training_curves(results_csv: Path, out_png: Path, title: str) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    x = df["epoch"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for a, cols, ttl in [(ax[0][0], ["train/box_loss", "val/box_loss"], "Box loss"),
                         (ax[0][1], ["train/cls_loss", "val/cls_loss"], "Class loss"),
                         (ax[1][0], ["metrics/precision(B)", "metrics/recall(B)"], "Precision / recall (val)"),
                         (ax[1][1], ["metrics/mAP50(B)", "metrics/mAP50-95(B)"], "mAP (val)")]:
        for c in cols:
            if c in df:
                a.plot(x, df[c], marker="o", ms=3, label=c.split("/")[-1])
        a.set_title(ttl)
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle(title)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return len(df)
