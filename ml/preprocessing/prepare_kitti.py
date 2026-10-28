"""KITTI: validate labels, keep vehicle classes, re-split train+val into train/val/test.
    python -m ml.preprocessing.prepare_kitti          (run from the project root)"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ml.config import ROOT, load_config

from .common import report, rarest_class_key, save_json, stratified_split, write_data_yaml, write_dataset
from .validate import validate_yolo


def main(argv=None):
    cfg = load_config()
    vc = cfg["vehicle"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / vc["raw_dir"]))
    ap.add_argument("--out", default=str(ROOT / vc["out_dir"]))
    ap.add_argument("--mode", choices=["copy", "symlink"], default="copy")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)

    raw, out = Path(a.raw), Path(a.out)
    names = vc["classes"]
    keep_map = {i: names.index(n) for i, n in enumerate(vc["kitti_names"]) if n in names}
    pool = [(s, p) for s in ("train", "val") for p in sorted((raw / "images" / s).glob("*.png"))]
    if a.limit:
        pool = pool[: a.limit]
    if not pool:
        sys.exit(f"No images in {raw}/images - run scripts/download_datasets.py kitti first.")

    records, issues, excluded, raw_codes, dropped = [], [], Counter(), Counter(), Counter()
    for s, img in tqdm(pool, desc="validating"):
        res = validate_yolo(raw / "labels" / s / f"{img.stem}.txt", img, keep_map)
        issues += [{"id": img.stem, "issue": i, "detail": d} for i, d in res["issues"]]
        raw_codes.update(vc["kitti_names"][c] for c in res["raw_codes"].elements())
        dropped.update(vc["kitti_names"][c] for c in res["dropped_codes"].elements())
        if not res["ok"]:
            excluded[res["fatal"]] += 1
            continue
        records.append({"id": img.stem, "image": img, "boxes": list(res["boxes"])})

    keys = rarest_class_key([{names[b[0]] for b in r["boxes"]} for r in records])
    split_idx = stratified_split(keys, cfg["split"], cfg["seed"])
    write_dataset(records, split_idx, out, a.mode)
    write_data_yaml(out, names)

    df = report(out, names, "vehicle", ROOT)
    pd.DataFrame(issues).to_csv(ROOT / "reports" / "tables" / "vehicle_validation_issues.csv", index=False)
    save_json({"images_found": len(pool), "images_used": len(records), "excluded": dict(excluded),
               "raw_box_classes": dict(raw_codes), "dropped_classes": dict(dropped),
               "issue_counts": dict(Counter(i["issue"] for i in issues)),
               "split_sizes": {k: len(v) for k, v in split_idx.items()}},
              ROOT / "reports" / "tables" / "vehicle_prep_summary.json")
    print(df.to_string(index=False))
    print("excluded:", dict(excluded), "| dropped classes:", dict(dropped))


if __name__ == "__main__":
    main()
