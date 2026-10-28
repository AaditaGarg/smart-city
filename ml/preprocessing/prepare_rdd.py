"""RDD2022 India: validate -> clean -> VOC->YOLO -> stratified train/val/test.
    python -m ml.preprocessing.prepare_rdd            (run from the project root)"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ml.config import ROOT, load_config

from .common import report, rarest_class_key, save_json, stratified_split, write_data_yaml, write_dataset
from .validate import validate_voc


def main(argv=None):
    cfg = load_config()
    rc = cfg["road_damage"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / rc["raw_dir"]))
    ap.add_argument("--out", default=str(ROOT / rc["out_dir"]))
    ap.add_argument("--mode", choices=["copy", "symlink"], default="copy")
    ap.add_argument("--limit", type=int, default=0, help="debug: only use the first N annotations")
    ap.add_argument("--force", action="store_true", help="continue even if >20%% of images are excluded for size mismatch")
    a = ap.parse_args(argv)

    raw, out = Path(a.raw), Path(a.out)
    codes = [c["code"] for c in rc["classes"]]
    names = [c["name"] for c in rc["classes"]]
    code_idx = {c: i for i, c in enumerate(codes)}
    xmls = sorted((raw / "annotations" / "xmls").glob("*.xml"))
    if a.limit:
        xmls = xmls[: a.limit]
    if not xmls:
        sys.exit(f"No XML files in {raw}/annotations/xmls - run scripts/fetch_rdd2022.py first.")

    positives, negatives, issues = [], [], []
    excluded, raw_codes, dropped = Counter(), Counter(), Counter()
    for xp in tqdm(xmls, desc="validating"):
        res = validate_voc(xp, raw / "images" / f"{xp.stem}.jpg", rc["class_map"], rc["size_mismatch"])
        issues += [{"id": xp.stem, "issue": i, "detail": d} for i, d in res["issues"]]
        raw_codes.update(res["raw_codes"])
        dropped.update(res["dropped_codes"])
        if not res["ok"]:
            excluded[res["fatal"]] += 1
            continue
        rec = {"id": xp.stem, "image": raw / "images" / f"{xp.stem}.jpg",
               "boxes": [(code_idx[lab], *rest) for lab, *rest in res["boxes"]]}
        if rec["boxes"]:
            positives.append(rec)
        elif res["n_raw"] == 0:
            negatives.append(rec)
        else:
            excluded["only_dropped_or_invalid_boxes"] += 1   # ambiguous: may hold unlabeled damage

    if excluded["size_mismatch"] / len(xmls) > 0.2 and not a.force:
        sys.exit(f"{excluded['size_mismatch']} of {len(xmls)} images have XML/image size mismatches. "
                 "Inspect the issues CSV, then set size_mismatch: rescale in configs/project.yaml or use --force.")

    rng = random.Random(cfg["seed"])
    n_neg = min(len(negatives), int(rc["negatives_ratio"] * len(positives)))
    records = positives + rng.sample(negatives, n_neg)
    keys = rarest_class_key([{names[b[0]] for b in r["boxes"]} for r in records])
    split_idx = stratified_split(keys, cfg["split"], cfg["seed"])
    write_dataset(records, split_idx, out, a.mode)
    write_data_yaml(out, names)

    df = report(out, names, "road_damage", ROOT)
    pd.DataFrame(issues).to_csv(ROOT / "reports" / "tables" / "road_damage_validation_issues.csv", index=False)
    summary = {"annotations_found": len(xmls), "positives": len(positives), "negatives_available": len(negatives),
               "negatives_used": n_neg, "excluded": dict(excluded), "raw_box_codes": dict(raw_codes),
               "dropped_codes": dict(dropped), "issue_counts": dict(Counter(i["issue"] for i in issues)),
               "split_sizes": {k: len(v) for k, v in split_idx.items()}}
    save_json(summary, ROOT / "reports" / "tables" / "road_damage_prep_summary.json")

    print(df.to_string(index=False))
    print("excluded:", dict(excluded), "| dropped box codes:", dict(dropped))
    for nm in names:
        for sp in ("val", "test"):
            if int(df.loc[df.split == sp, f"boxes_{nm}"].iloc[0]) < 10:
                print(f"WARNING: class '{nm}' has <10 boxes in {sp}; its metrics will be unreliable")


if __name__ == "__main__":
    main()
