"""Offline augmentation of rare classes (TRAIN split only) + a preview figure.
    python -m ml.preprocessing.augment --dataset road_damage            # oversample per config
    python -m ml.preprocessing.augment --dataset road_damage --preview-only
YOLO's own online augmentation (mosaic, HSV, flips) is configured in Step 4; this adds targeted
copies of images containing under-represented classes so they are seen more often."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import albumentations as A
import cv2

from ml.config import ROOT, load_config

from .common import report


def build_pipeline() -> A.Compose:
    return A.Compose(
        [A.HorizontalFlip(p=0.5),
         A.Affine(scale=(0.85, 1.15), translate_percent=(-0.05, 0.05), rotate=(-5, 5), p=0.7),
         A.RandomBrightnessContrast(p=0.6),
         A.HueSaturationValue(p=0.4),
         A.MotionBlur(blur_limit=(3, 7), p=0.25),
         A.GaussNoise(std_range=(0.02, 0.08), p=0.2)],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_ids"], min_visibility=0.4))


def _read_labels(p: Path):
    rows = [ln.split() for ln in p.read_text().splitlines() if ln.strip()]
    return [int(r[0]) for r in rows], [[float(v) for v in r[1:5]] for r in rows]


def augment_one(img_bgr, class_ids, bboxes, pipe):
    bboxes = [[min(max(v, 0.0), 1.0) for v in b] for b in bboxes]
    out = pipe(image=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), bboxes=bboxes, class_ids=class_ids)
    return (cv2.cvtColor(out["image"], cv2.COLOR_RGB2BGR), [int(c) for c in out["class_ids"]],
            [[float(v) for v in b] for b in out["bboxes"]])


def oversample(out_dir: Path, target_ids: set[int], copies: int, seed: int = 42, split: str = "train") -> int:
    img_dir, lbl_dir = out_dir / "images" / split, out_dir / "labels" / split
    for p in list(img_dir.glob("*_aug*")) + list(lbl_dir.glob("*_aug*")):   # idempotent
        p.unlink()
    random.seed(seed)
    pipe = build_pipeline()
    pipe.set_random_seed(seed) if hasattr(pipe, "set_random_seed") else None
    made = 0
    for lp in sorted(lbl_dir.glob("*.txt")):
        ids, boxes = _read_labels(lp)
        if not target_ids.intersection(ids):
            continue
        ip = next((q for q in img_dir.glob(f"{lp.stem}.*")), None)
        img = cv2.imread(str(ip)) if ip else None
        if img is None:
            continue
        for k in range(copies):
            im2, ids2, bx2 = augment_one(img, ids, boxes, pipe)
            if not ids2 or not target_ids.intersection(ids2):
                continue
            cv2.imwrite(str(img_dir / f"{lp.stem}_aug{k}{ip.suffix}"), im2)
            (lbl_dir / f"{lp.stem}_aug{k}.txt").write_text(
                "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, (x, y, w, h) in zip(ids2, bx2)))
            made += 1
    return made


def preview(out_dir: Path, names, target_ids, path: Path, n: int = 4, seed: int = 42):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img_dir, lbl_dir = out_dir / "images" / "train", out_dir / "labels" / "train"
    cands = [lp for lp in sorted(lbl_dir.glob("*.txt")) if "_aug" not in lp.stem
             and target_ids.intersection(_read_labels(lp)[0])]
    random.Random(seed).shuffle(cands)
    pipe = build_pipeline()
    fig, axes = plt.subplots(2, min(n, len(cands)) or 1, figsize=(4 * max(1, min(n, len(cands))), 7), squeeze=False)

    def draw(ax, img, ids, boxes, title):
        im = img.copy()
        H, W = im.shape[:2]
        for c, (x, y, w, h) in zip(ids, boxes):
            p1, p2 = (int((x - w / 2) * W), int((y - h / 2) * H)), (int((x + w / 2) * W), int((y + h / 2) * H))
            cv2.rectangle(im, p1, p2, (0, 255, 0), 2)
            cv2.putText(im, names[c], (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    for j, lp in enumerate(cands[:n]):
        ids, boxes = _read_labels(lp)
        img = cv2.imread(str(next(img_dir.glob(f"{lp.stem}.*"))))
        draw(axes[0][j], img, ids, boxes, f"original {lp.stem}")
        im2, ids2, bx2 = augment_one(img, ids, boxes, pipe)
        draw(axes[1][j], im2, ids2, bx2, "augmented")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None):
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["road_damage", "vehicle"], default="road_damage")
    ap.add_argument("--classes", nargs="+")
    ap.add_argument("--copies", type=int)
    ap.add_argument("--preview-only", action="store_true")
    a = ap.parse_args(argv)

    dc = cfg[a.dataset]
    out = ROOT / dc["out_dir"]
    names = [c["name"] for c in dc["classes"]] if a.dataset == "road_damage" else dc["classes"]
    over = dc.get("oversample", {"classes": [], "copies": 0})
    wanted = a.classes or over["classes"]
    copies = a.copies if a.copies is not None else over["copies"]
    codes = {c["code"]: c["name"] for c in dc["classes"]} if a.dataset == "road_damage" else {}
    target = {names.index(codes.get(w, w)) for w in wanted}
    if not target:
        raise SystemExit("No classes to oversample: set oversample.classes in configs/project.yaml or use --classes")

    preview(out, names, target, ROOT / "reports" / "figures" / f"{a.dataset}_augmentation_examples.png")
    if a.preview_only:
        return
    made = oversample(out, target, copies, cfg["seed"])
    print(f"created {made} augmented training images for classes {[names[i] for i in sorted(target)]}")
    print(report(out, names, a.dataset, ROOT).to_string(index=False))


if __name__ == "__main__":
    main()
