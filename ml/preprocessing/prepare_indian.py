"""Indian vehicles: convert the DataCluster Labs VOC set to YOLO, merge the Kaggle ambulance set (YOLO),
map everything to 8 traffic classes, shrink images to a 640 px long side and make a seeded 70/15/15 split.

    python -m ml.preprocessing.prepare_indian                    (run from the project root)
    python -m ml.preprocessing.prepare_indian --delete-raw-if-low-disk 3   # free raw 4K images if < 3 GB free

Outputs: data/processed/indian_vehicle/{images,labels}/{train,val,test}, data.yaml,
reports/tables/indian_vehicle_{split_stats.csv,prep_summary.json,validation_issues.csv},
reports/figures/indian_vehicle_class_distribution.png
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

from ml.config import ROOT, load_config

from .common import rarest_class_key, report, save_json, stratified_split, write_data_yaml, write_dataset
from .validate import validate_voc, validate_yolo

CLASSES = ["car", "bike", "bicycle", "auto", "bus", "truck", "ambulance", "person"]

# raw label (normalised by norm_label) -> kept class. Anything not listed is dropped and reported.
DEFAULT_CLASS_MAP = {
    "car": "car", "taxi": "car", "suv": "car", "jeep": "car",
    "bike": "bike", "motorbike": "bike", "motorcycle": "bike", "motor_cycle": "bike", "scooter": "bike",
    "scooty": "bike", "two_wheeler": "bike", "two_wheelers": "bike",
    "bicycle": "bicycle", "cycle": "bicycle",
    "auto": "auto", "autorickshaw": "auto", "auto_rickshaw": "auto", "e_rickshaw": "auto", "erickshaw": "auto",
    "three_wheeler": "auto",
    "bus": "bus", "minibus": "bus", "mini_bus": "bus", "school_bus": "bus",
    "truck": "truck", "lorry": "truck", "truck_tanker": "truck", "tanker": "truck", "pickup": "truck",
    "pickup_truck": "truck", "pick_up": "truck", "van": "truck", "tractor": "truck", "mini_truck": "truck",
    "tempo": "truck", "vehicle_truck": "truck", "concrete_mixer": "truck",
    "ambulance": "ambulance",
    "person": "person", "pedestrian": "person", "people": "person",
}

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def norm_label(name: str) -> str:
    """'Auto-Rickshaw ' -> 'auto_rickshaw'"""
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def map_class(raw: str, class_map: dict, classes: list) -> int | None:
    """Index into `classes` for a raw label, or None if the label is dropped."""
    kept = class_map.get(norm_label(raw))
    return classes.index(kept) if kept in classes else None


def voc_class_map(raw_names, class_map: dict, classes: list) -> dict:
    """Raw VOC names (as written in the XML) -> class index, for validate_voc."""
    out = {}
    for n in raw_names:
        i = map_class(n, class_map, classes)
        if i is not None:
            out[n] = i
    return out


def safe_id(prefix: str, rel: Path) -> str:
    return prefix + "_" + re.sub(r"[^A-Za-z0-9]+", "_", str(rel.with_suffix(""))).strip("_")


def resize_long_side(src: Path, dst: Path, long_side: int = 640, quality: int = 90, transpose: bool = False) -> tuple:
    """Downscale so the longer side is <= long_side; saved as JPEG without EXIF so every loader sees the same
    pixels. transpose=True applies the EXIF rotation first (for annotations made on the rotated view).
    Returns the new (W, H)."""
    from PIL import ImageOps
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        if transpose:
            im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        W, H = im.size
        s = long_side / max(W, H)
        if s < 1:
            im = im.resize((max(1, round(W * s)), max(1, round(H * s))), Image.BILINEAR)
        im.save(dst, "JPEG", quality=quality)
        return im.size


def annotated_rotated(xml: Path, img: Path) -> bool:
    """True if the XML size equals the EXIF-rotated (displayed) size but not the raw pixel size, i.e. the
    annotator worked on the rotated view (orientation tags 5-8 swap width and height)."""
    import xml.etree.ElementTree as ET
    try:
        sz = ET.parse(xml).getroot().find("size")
        xw, xh = int(float(sz.findtext("width"))), int(float(sz.findtext("height")))
        with Image.open(img) as im:
            W, H = im.size
            orient = im.getexif().get(274, 1)
    except Exception:
        return False
    return orient in (5, 6, 7, 8) and (xw, xh) == (H, W) and W != H


def find_image(xml: Path, filename: str | None, by_stem: dict) -> Path | None:
    cands = []
    if filename:
        cands += [xml.parent / filename, xml.parent.parent / filename]
        cands += [d / filename for d in xml.parent.parent.iterdir() if d.is_dir()]
    for c in cands:
        if c.is_file() and c.suffix.lower() in IMG_EXT:
            return c
    hits = by_stem.get(xml.stem, [])
    return hits[0] if len(hits) == 1 else None


def voc_records(raw: Path, class_map: dict, classes: list, size_policy: str, tmp_dir: Path | None = None,
                long_side: int = 640):
    import xml.etree.ElementTree as ET
    xmls = sorted(raw.rglob("*.xml"))
    by_stem = {}
    for p in raw.rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            by_stem.setdefault(p.stem, []).append(p)
    raw_names = set()
    parsed = []
    for x in xmls:
        try:
            root = ET.parse(x).getroot()
        except Exception:
            parsed.append((x, None))
            continue
        raw_names.update(o.findtext("name", "").strip() for o in root.iter("object"))
        parsed.append((x, root.findtext("filename")))
    cmap = voc_class_map(raw_names, class_map, classes)
    for x, fname in tqdm(parsed, desc="indian_vehicle (VOC)"):
        img = find_image(x, fname, by_stem)
        if img is None:
            yield x, None, {"ok": False, "fatal": "missing_image", "issues": [("missing_image", x.name)],
                            "raw_codes": Counter(), "dropped_codes": Counter()}, None
            continue
        if tmp_dir is not None and annotated_rotated(x, img):
            tmp_dir.mkdir(parents=True, exist_ok=True)
            # validate against an EXIF-rotated, downscaled copy; same aspect as the XML, so 'rescale' is exact
            tmp = tmp_dir / f"_rot_{len(list(tmp_dir.glob('_rot_*')))}.jpg"
            resize_long_side(img, tmp, long_side, transpose=True)
            res = validate_voc(x, tmp, cmap, size_policy="rescale")
            res["issues"] = [i for i in res["issues"] if i[0] != "size_mismatch"] + [("exif_rotated", img.name)]
            yield x, img, res, tmp
            continue
        yield x, img, validate_voc(x, img, cmap, size_policy=size_policy), None


def yolo_names(raw: Path) -> list:
    for y in sorted(raw.rglob("*.yaml")):
        try:
            d = yaml.safe_load(y.read_text())
        except Exception:
            continue
        if isinstance(d, dict) and "names" in d:
            n = d["names"]
            return [n[k] for k in sorted(n)] if isinstance(n, dict) else list(n)
    c = next(iter(sorted(raw.rglob("classes.txt"))), None)
    if c:
        return [ln.strip() for ln in c.read_text().splitlines() if ln.strip()]
    sys.exit(f"No data.yaml / classes.txt with class names under {raw}")


def yolo_records(raw: Path, class_map: dict, classes: list):
    names = yolo_names(raw)
    keep = {i: map_class(n, class_map, classes) for i, n in enumerate(names)}
    keep = {i: c for i, c in keep.items() if c is not None}
    imgs = sorted(p for p in raw.rglob("*") if p.suffix.lower() in IMG_EXT and "images" in p.parts)
    for img in tqdm(imgs, desc="ambulance (YOLO)"):
        parts = list(img.parts)
        j = len(parts) - 1 - parts[::-1].index("images")
        lbl = Path(*parts[:j], "labels", *parts[j + 1:]).with_suffix(".txt")
        res = validate_yolo(lbl, img, keep)
        res["raw_codes"] = Counter({names[c] if c < len(names) else str(c): n for c, n in res["raw_codes"].items()})
        res["dropped_codes"] = Counter({names[c] if c < len(names) else str(c): n
                                        for c, n in res["dropped_codes"].items()})
        yield lbl, img, res, None


def group_key(prefix: str, img: Path) -> str:
    """Roboflow exports name augmented copies '<orig>_jpg.rf.<hash>.jpg'; all copies of one photo share a group
    so they land in the same split (otherwise the test set leaks into training)."""
    return prefix + ":" + re.sub(r"(_(jpe?g|png|bmp|webp))?\.rf\.[0-9a-fA-F]+$", "", img.stem).lower()


def group_split(groups: list, class_sets: list, fractions: dict, seed: int) -> dict:
    """stratified_split over groups (by each group's rarest class), expanded back to record indices."""
    order = sorted(set(groups))
    gi = {g: i for i, g in enumerate(order)}
    g_classes = [set() for _ in order]
    for g, cs in zip(groups, class_sets):
        g_classes[gi[g]] |= set(cs)
    g_split = stratified_split(rarest_class_key(g_classes), fractions, seed)
    owner = {g: split for split, idxs in g_split.items() for g in idxs}
    return {split: [i for i, g in enumerate(groups) if owner[gi[g]] == split] for split in ("train", "val", "test")}


def box_overlap(a, b) -> float:
    """max(IoU, intersection / area of the smaller box) for normalised (xc, yc, w, h) boxes."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    aa, ba = a[2] * a[3], b[2] * b[3]
    if inter <= 0 or aa <= 0 or ba <= 0:
        return 0.0
    return max(inter / (aa + ba - inter), inter / min(aa, ba))


def merge_pseudo(human: list, preds: list, max_overlap: float = 0.3) -> list:
    """Add predicted boxes (cls, xc, yc, w, h) that don't overlap any human box or an already-added prediction.
    Human labels always win, so an auto that COCO calls 'car' or an ambulance it calls 'truck' is not duplicated."""
    out = list(human)
    added = []
    for p in preds:
        if all(box_overlap(p[1:], k[1:]) < max_overlap for k in out):
            out.append(tuple(p))
            added.append(tuple(p))
    return added


def pseudo_label(records: list, classes: list, pc: dict) -> Counter:
    """Fill in objects the source datasets never annotated (people everywhere, ordinary traffic in the ambulance
    photos) with confident COCO predictions, so the detector is not taught that they are background."""
    from ultralytics import YOLO
    w = Path(pc.get("weights", "yolov8n.pt"))
    w = w if w.is_absolute() else ROOT / w
    model = YOLO(str(w))
    cmap = {k: classes.index(v) for k, v in pc["coco_map"].items() if v in classes}
    base_conf = pc.get("conf", 0.4)
    per_class = pc.get("conf_per_class") or {}       # COCO name -> stricter threshold (e.g. bus fires on vans)
    added = Counter()
    for r in tqdm(records, desc="pseudo-labelling"):
        res = model.predict(str(r["image"]), conf=base_conf, imgsz=640, device="cpu", verbose=False)[0]
        preds = []
        for c, s, xywhn in zip(res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                               res.boxes.xywhn.cpu().numpy()):
            name = model.names[int(c)]
            if name in cmap and float(s) >= per_class.get(name, base_conf):
                preds.append((cmap[name], *map(float, xywhn)))
        new = merge_pseudo(r["boxes"], preds, pc.get("max_overlap", 0.3))
        r["boxes"] += new
        r["n_pseudo"] = len(new)
        added.update(classes[b[0]] for b in new)
    return added


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def main(argv=None):
    cfg = load_config()
    ic = cfg["indian_vehicle"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc-raw", default=str(ROOT / ic["raw_dirs"]["voc"]))
    ap.add_argument("--amb-raw", default=str(ROOT / ic["raw_dirs"]["ambulance"]))
    ap.add_argument("--out", default=str(ROOT / ic["out_dir"]))
    ap.add_argument("--resized", default=str(ROOT / ic["resized_dir"]))
    ap.add_argument("--long-side", type=int, default=ic.get("long_side", 640))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-pseudo", action="store_true", help="skip COCO pseudo-labelling of unannotated objects")
    ap.add_argument("--delete-raw-if-low-disk", type=float, default=0.0, metavar="GB",
                    help="after preparing, delete the raw 4K images if free space is below this many GB")
    a = ap.parse_args(argv)

    classes = ic["classes"]
    class_map = {norm_label(k): v for k, v in (ic.get("class_map") or DEFAULT_CLASS_MAP).items()}
    voc_raw, amb_raw, out, resized = Path(a.voc_raw), Path(a.amb_raw), Path(a.out), Path(a.resized)
    for d in (voc_raw, amb_raw):
        if not d.exists():
            sys.exit(f"{d} missing - run: python scripts/download_datasets.py indian")
    if resized.exists():
        shutil.rmtree(resized)

    records, issues, excluded, raw_codes, dropped, per_source = [], [], Counter(), Counter(), Counter(), Counter()
    seen, per_group = set(), Counter()
    max_copies = int(ic.get("max_copies_per_photo", 0) or 0)
    resized.mkdir(parents=True)
    sources = [("iv", voc_raw, voc_records(voc_raw, class_map, classes, ic.get("size_mismatch", "drop"),
                                           resized, a.long_side)),
               ("amb", amb_raw, yolo_records(amb_raw, class_map, classes))]
    for prefix, base, gen in sources:
        n = 0
        for ann, img, res, prepared in gen:
            if a.limit and n >= a.limit:
                break
            n += 1
            rid = safe_id(prefix, (img or ann).relative_to(base))
            if rid in seen:
                excluded["duplicate_image"] += 1   # a second annotation file for an image already used
                continue
            issues += [{"id": rid, "issue": i, "detail": d} for i, d in res["issues"]]
            raw_codes.update(res["raw_codes"])
            dropped.update(res["dropped_codes"])
            if not res["ok"]:
                excluded[res["fatal"]] += 1
                continue
            if not res["boxes"]:
                excluded["no_kept_boxes"] += 1   # image only had dropped classes; not a reliable negative
                continue
            g = group_key(prefix, img)
            if max_copies and per_group[g] >= max_copies:
                excluded["extra_augmented_copy"] += 1   # Roboflow pre-augmented duplicate of a photo already kept
                continue
            per_group[g] += 1
            seen.add(rid)
            dst = resized / f"{rid}.jpg"
            if prepared is not None:
                prepared.replace(dst)            # already rotated + resized during validation
            else:
                resize_long_side(img, dst, a.long_side)
            records.append({"id": rid, "image": dst, "boxes": list(res["boxes"]), "group": g})
            per_source[prefix] += 1
    if not records:
        sys.exit("No usable images found.")

    pc = ic.get("pseudo_label") or {}
    pseudo_added = Counter()
    if pc.get("enabled") and not a.no_pseudo:
        pseudo_added = pseudo_label(records, classes, pc)
        print("pseudo-labels added:", dict(pseudo_added))
    split_idx = group_split([r["group"] for r in records], [{classes[b[0]] for b in r["boxes"]} for r in records],
                            cfg["split"], cfg["seed"])
    write_dataset(records, split_idx, out, "copy")
    write_data_yaml(out, classes)
    shutil.rmtree(resized)   # write_dataset copied the resized files; keep one copy on disk

    tables = ROOT / "reports" / "tables"
    df = report(out, classes, "indian_vehicle", ROOT)
    pd.DataFrame(issues, columns=["id", "issue", "detail"]).to_csv(tables / "indian_vehicle_validation_issues.csv",
                                                                  index=False)
    deleted = []
    if a.delete_raw_if_low_disk and free_gb(out) < a.delete_raw_if_low_disk:
        for d in (voc_raw, amb_raw):
            for p in list(d.rglob("*")):
                if p.suffix.lower() in IMG_EXT:
                    p.unlink()
            deleted.append(str(d))
        print(f"[disk] free space below {a.delete_raw_if_low_disk} GB - deleted raw images in {deleted}")
    save_json({"images_used": len(records), "images_per_source": dict(per_source),
               "source_photo_groups": len({r["group"] for r in records}), "excluded": dict(excluded),
               "raw_box_classes": dict(raw_codes), "dropped_classes": dict(dropped),
               "class_map": class_map, "long_side": a.long_side,
               "issue_counts": dict(Counter(i["issue"] for i in issues)),
               "split_sizes": {k: len(v) for k, v in split_idx.items()},
               "pseudo_labels_added": dict(pseudo_added),
               "pseudo_labels_per_split": {k: sum(records[i].get("n_pseudo", 0) for i in v)
                                           for k, v in split_idx.items()},
               "max_copies_per_photo": max_copies,
               "raw_images_deleted": deleted, "free_gb_after": round(free_gb(out), 2)},
              tables / "indian_vehicle_prep_summary.json")
    print(df.to_string(index=False))
    print("excluded:", dict(excluded), "| dropped classes:", dict(dropped))


if __name__ == "__main__":
    main()
