from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from ml.preprocessing.common import clean_boxes, rarest_class_key, stratified_split, to_yolo
from ml.preprocessing.validate import validate_voc, validate_yolo

XML = """<annotation><size><width>{w}</width><height>{h}</height><depth>3</depth></size>{objs}</annotation>"""
OBJ = "<object><name>{n}</name><bndbox><xmin>{a}</xmin><ymin>{b}</ymin><xmax>{c}</xmax><ymax>{d}</ymax></bndbox></object>"


def _img(p, w=100, h=80):
    Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)).save(p)


def test_clean_boxes_clips_swaps_and_drops():
    kept, issues = clean_boxes([("a", -5, -5, 50, 40), ("a", 60, 70, 30, 20), ("a", 10, 10, 10.5, 10.5),
                                ("a", 0, 0, 50, 40)], 100, 80)
    codes = [i[0] for i in issues]
    assert "swapped_coords" in codes and "degenerate_box" in codes and "duplicate_box" in codes
    assert all(0 <= b[1] and b[3] <= 100 and 0 <= b[2] and b[4] <= 80 for b in kept)


def test_to_yolo():
    assert to_yolo(0, 0, 50, 40, 100, 80) == (0.25, 0.25, 0.5, 0.5)


def test_validate_voc_mapping_and_size_policy(tmp_path):
    _img(tmp_path / "a.jpg")
    (tmp_path / "a.xml").write_text(XML.format(w=100, h=80, objs=OBJ.format(n="D01", a=10, b=10, c=40, d=40)
                                               + OBJ.format(n="D44", a=1, b=1, c=30, d=30)))
    r = validate_voc(tmp_path / "a.xml", tmp_path / "a.jpg", {"D01": "D00"})
    assert r["ok"] and r["boxes"][0][0] == "D00" and r["dropped_codes"] == Counter({"D44": 1}) and r["n_raw"] == 2
    (tmp_path / "b.xml").write_text(XML.format(w=200, h=80, objs=""))
    _img(tmp_path / "b.jpg")
    assert validate_voc(tmp_path / "b.xml", tmp_path / "b.jpg", {}, "drop")["fatal"] == "size_mismatch"
    assert validate_voc(tmp_path / "b.xml", tmp_path / "b.jpg", {}, "ignore")["ok"]


def test_validate_voc_bad_inputs(tmp_path):
    (tmp_path / "c.xml").write_text("<not xml")
    _img(tmp_path / "c.jpg")
    assert validate_voc(tmp_path / "c.xml", tmp_path / "c.jpg", {})["fatal"] == "xml_parse_error"
    (tmp_path / "d.jpg").write_bytes(b"garbage")
    (tmp_path / "d.xml").write_text(XML.format(w=1, h=1, objs=""))
    assert validate_voc(tmp_path / "d.xml", tmp_path / "d.jpg", {})["fatal"] == "unreadable_image"


def test_validate_yolo_filters_classes(tmp_path):
    _img(tmp_path / "k.png")
    (tmp_path / "k.txt").write_text("0 0.5 0.5 0.2 0.2\n3 0.5 0.5 0.2 0.2\n")
    r = validate_yolo(tmp_path / "k.txt", tmp_path / "k.png", {0: 0})
    assert len(r["boxes"]) == 1 and r["dropped_codes"] == Counter({3: 1})


def test_stratified_split_is_disjoint_complete_and_keeps_classes():
    keys = ["a"] * 200 + ["b"] * 60 + ["__neg__"] * 40
    s = stratified_split(keys, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=1)
    allidx = s["train"] + s["val"] + s["test"]
    assert sorted(allidx) == list(range(len(keys)))
    for sp in ("val", "test"):
        assert {"a", "b"} <= {keys[i] for i in s[sp]}


def test_rarest_class_key():
    assert rarest_class_key([{"x"}, {"x", "y"}, set(), {"x"}]) == ["x", "y", "__neg__", "x"]


def test_augmentation_writes_integer_class_ids(tmp_path):
    import cv2
    from ml.preprocessing.augment import oversample
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train").mkdir(parents=True)
    cv2.imwrite(str(tmp_path / "images" / "train" / "x.jpg"), np.full((200, 200, 3), 120, np.uint8))
    (tmp_path / "labels" / "train" / "x.txt").write_text("1 0.5 0.5 0.4 0.4\n")
    made = oversample(tmp_path, {1}, 3)
    assert made >= 1
    for lp in (tmp_path / "labels" / "train").glob("x_aug*.txt"):
        for ln in lp.read_text().splitlines():
            assert ln.split()[0].isdigit()
