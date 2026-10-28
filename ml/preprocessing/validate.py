"""Annotation validation. Each validator returns a dict:
ok, fatal (reason if not ok), issues [(code, detail)], boxes [(label, xc, yc, w, h)] normalised,
n_raw (objects in the file), raw_codes / dropped_codes (Counters)."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image

from .common import clean_boxes, to_yolo


def image_size(path: Path):
    """(W, H) if the image decodes cleanly, else None."""
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def _result(**kw):
    base = {"ok": True, "fatal": None, "issues": [], "boxes": [], "n_raw": 0,
            "raw_codes": Counter(), "dropped_codes": Counter()}
    base.update(kw)
    return base


def validate_voc(xml_path: Path, img_path: Path, class_map: dict, size_policy="drop"):
    res = _result()
    if not Path(img_path).exists():
        return _result(ok=False, fatal="missing_image", issues=[("missing_image", str(img_path))])
    try:
        root = ET.parse(xml_path).getroot()
        sz = root.find("size")
        xw = int(float(sz.findtext("width", "0"))) if sz is not None else 0
        xh = int(float(sz.findtext("height", "0"))) if sz is not None else 0
        objs = []
        for o in root.iter("object"):
            bb = o.find("bndbox")
            objs.append((o.findtext("name", "").strip(), float(bb.findtext("xmin")), float(bb.findtext("ymin")),
                         float(bb.findtext("xmax")), float(bb.findtext("ymax"))))
    except Exception as e:
        return _result(ok=False, fatal="xml_parse_error", issues=[("xml_parse_error", str(e)[:80])])

    size = image_size(img_path)
    if size is None:
        return _result(ok=False, fatal="unreadable_image", issues=[("unreadable_image", str(img_path.name))])
    W, H = size
    res["n_raw"] = len(objs)
    res["raw_codes"].update(o[0] for o in objs)

    sx = sy = 1.0
    if xw and xh and (xw, xh) != (W, H):
        res["issues"].append(("size_mismatch", f"xml={xw}x{xh} image={W}x{H}"))
        if size_policy == "drop":
            res.update(ok=False, fatal="size_mismatch")
            return res
        if size_policy == "rescale":
            sx, sy = W / xw, H / xh

    mapped = []
    for code, x1, y1, x2, y2 in objs:
        if code in class_map:
            mapped.append((class_map[code], x1 * sx, y1 * sy, x2 * sx, y2 * sy))
        else:
            res["dropped_codes"][code] += 1
    kept, issues = clean_boxes(mapped, W, H)
    res["issues"] += issues
    res["boxes"] = [(lab, *to_yolo(x1, y1, x2, y2, W, H)) for lab, x1, y1, x2, y2 in kept]
    return res


def validate_yolo(label_path: Path, img_path: Path, keep_map: dict):
    """keep_map: raw class index -> kept label; other classes are dropped (background)."""
    if not Path(img_path).exists():
        return _result(ok=False, fatal="missing_image", issues=[("missing_image", str(img_path))])
    if not Path(label_path).exists():
        return _result(ok=False, fatal="missing_label", issues=[("missing_label", str(label_path))])
    size = image_size(img_path)
    if size is None:
        return _result(ok=False, fatal="unreadable_image", issues=[("unreadable_image", img_path.name)])
    W, H = size
    res = _result()
    boxes = []
    for ln in Path(label_path).read_text().splitlines():
        if not ln.strip():
            continue
        try:
            c, xc, yc, w, h = ln.split()[:5]
            c, xc, yc, w, h = int(c), float(xc), float(yc), float(w), float(h)
        except ValueError:
            res["issues"].append(("label_parse_error", ln[:40]))
            continue
        res["n_raw"] += 1
        res["raw_codes"][c] += 1
        if c not in keep_map:
            res["dropped_codes"][c] += 1
            continue
        boxes.append((keep_map[c], (xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H))
    kept, issues = clean_boxes(boxes, W, H)
    res["issues"] += issues
    res["boxes"] = [(lab, *to_yolo(x1, y1, x2, y2, W, H)) for lab, x1, y1, x2, y2 in kept]
    return res
