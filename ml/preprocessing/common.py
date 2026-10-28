"""Shared helpers: box cleaning, stratified splitting, dataset writing, statistics."""
from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------- boxes
def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def clean_boxes(boxes, W, H, tol=2.0, min_px=2.0, dup_iou=0.95):
    """boxes: [(label, x1, y1, x2, y2)] in pixels. Returns (kept, issues).
    Fixes swapped corners, clips to the image, drops degenerate and duplicate boxes."""
    kept, issues = [], []
    for label, x1, y1, x2, y2 in boxes:
        if x2 < x1 or y2 < y1:
            issues.append(("swapped_coords", f"{label}"))
            x1, x2, y1, y2 = min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)
        over = max(0 - x1, 0 - y1, x2 - W, y2 - H)
        cx1, cy1, cx2, cy2 = max(0.0, x1), max(0.0, y1), min(float(W), x2), min(float(H), y2)
        if over > tol:
            issues.append(("box_clipped", f"{label} overshoot={over:.1f}px"))
        if (cx2 - cx1) < min_px or (cy2 - cy1) < min_px:
            issues.append(("degenerate_box", f"{label} {cx2 - cx1:.1f}x{cy2 - cy1:.1f}px"))
            continue
        box = (cx1, cy1, cx2, cy2)
        if any(k[0] == label and iou(box, k[1:]) > dup_iou for k in kept):
            issues.append(("duplicate_box", f"{label}"))
            continue
        kept.append((label, *box))
    return kept, issues


def to_yolo(x1, y1, x2, y2, W, H):
    return ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)


# ---------------------------------------------------------------- splitting
def rarest_class_key(class_sets):
    """Per image: the class that is rarest (by image count) among those it contains."""
    freq = Counter(c for s in class_sets for c in set(s))
    return [min(set(s), key=lambda c: (freq[c], c)) if s else "__neg__" for s in class_sets]


def stratified_split(keys, fractions, seed=42, min_stratum=10):
    n = len(keys)
    idx = list(range(n))
    cnt = Counter(keys)
    keys = [k if cnt[k] >= min_stratum else "__rare__" for k in keys]
    if "__rare__" in keys and keys.count("__rare__") < min_stratum:
        top = Counter(k for k in keys if k != "__rare__").most_common(1)[0][0]
        keys = [top if k == "__rare__" else k for k in keys]
    f_val, f_test = fractions["val"], fractions["test"]
    tr, tmp, _, k_tmp = train_test_split(idx, keys, test_size=f_val + f_test, stratify=keys, random_state=seed)
    va, te = train_test_split(tmp, test_size=f_test / (f_val + f_test), stratify=k_tmp, random_state=seed)
    return {"train": sorted(tr), "val": sorted(va), "test": sorted(te)}


# ---------------------------------------------------------------- writing
def place_file(src: Path, dst: Path, mode: str = "copy"):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(Path(src).resolve(), dst)
    else:
        shutil.copy2(src, dst)


def write_dataset(records, split_idx, out_dir, mode="copy"):
    """records: [{'id', 'image': Path, 'boxes': [(cls_idx, xc, yc, w, h)]}]"""
    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids)), "duplicate image ids"
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    rows = []
    for split, idxs in split_idx.items():
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        for i in idxs:
            r = records[i]
            src = Path(r["image"])
            place_file(src, out / "images" / split / f"{r['id']}{src.suffix.lower()}", mode)
            lines = "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in r["boxes"])
            (out / "labels" / split / f"{r['id']}.txt").write_text(lines)
            rows.append({"id": r["id"], "split": split, "n_boxes": len(r["boxes"]), "source": str(src)})
    pd.DataFrame(rows).to_csv(out / "split_manifest.csv", index=False)


def write_data_yaml(out_dir, names):
    out = Path(out_dir).resolve()
    body = [f"path: {out}", "train: images/train", "val: images/val", "test: images/test", "names:"]
    body += [f"  {i}: {n}" for i, n in enumerate(names)]
    (out / "data.yaml").write_text("\n".join(body) + "\n")


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


# ---------------------------------------------------------------- statistics
def compute_stats(out_dir, class_names) -> pd.DataFrame:
    out = Path(out_dir)
    rows = []
    for split in ("train", "val", "test"):
        cnt, empty, aug, n = Counter(), 0, 0, 0
        for p in sorted((out / "labels" / split).glob("*.txt")):
            n += 1
            aug += "_aug" in p.stem
            lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
            empty += not lines
            cnt.update(int(ln.split()[0]) for ln in lines)
        row = {"split": split, "images": n, "aug_images": aug, "background_images": empty}
        row.update({f"boxes_{nm}": cnt.get(i, 0) for i, nm in enumerate(class_names)})
        rows.append(row)
    return pd.DataFrame(rows)


def plot_stats(df: pd.DataFrame, class_names, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.8 / len(df)
    for j, (_, row) in enumerate(df.iterrows()):
        ax.bar([i + j * w for i in range(len(class_names))], [row[f"boxes_{n}"] for n in class_names],
               width=w, label=f"{row['split']} ({row['images']} img)")
    ax.set_xticks([i + 0.4 - w / 2 for i in range(len(class_names))])
    ax.set_xticklabels(class_names, rotation=15)
    ax.set_ylabel("boxes")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report(out_dir, class_names, name, root: Path):
    df = compute_stats(out_dir, class_names)
    tables, figs = root / "reports" / "tables", root / "reports" / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables / f"{name}_split_stats.csv", index=False)
    plot_stats(df, class_names, figs / f"{name}_class_distribution.png", f"{name}: boxes per class and split")
    return df
