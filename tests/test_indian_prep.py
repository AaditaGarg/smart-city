"""VOC -> YOLO conversion, class mapping and the end-to-end Indian-vehicle prep on tiny synthetic fixtures."""
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ml.config import load_config
from ml.preprocessing import prepare_indian as pi
from ml.preprocessing.validate import validate_voc

XML = ("<annotation><filename>{f}</filename><size><width>{w}</width><height>{h}</height><depth>3</depth></size>"
       "{objs}</annotation>")
OBJ = "<object><name>{n}</name><bndbox><xmin>{a}</xmin><ymin>{b}</ymin><xmax>{c}</xmax><ymax>{d}</ymax></bndbox></object>"
CLASSES = pi.CLASSES


def _img(p: Path, w=200, h=100):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((h, w, 3), 90, dtype=np.uint8)).save(p)


def _cmap():
    return {pi.norm_label(k): v for k, v in pi.DEFAULT_CLASS_MAP.items()}


@pytest.mark.parametrize("raw,expected", [
    ("car", "car"), ("Car ", "car"),
    ("motorcycle", "bike"), ("Scooter", "bike"), ("bike", "bike"), ("two_wheelers", "bike"),
    ("vehicle_truck", "truck"), ("concrete_mixer", "truck"),
    ("bicycle", "bicycle"),
    ("auto", "auto"), ("Auto-Rickshaw", "auto"),
    ("bus", "bus"),
    ("truck", "truck"), ("truck_tanker", "truck"), ("tanker", "truck"), ("pickup", "truck"), ("van", "truck"),
    ("Tractor", "truck"),
    ("ambulance", "ambulance"), ("person", "person"),
    ("dog", None), ("traffic_light", None), ("", None),
])
def test_class_mapping(raw, expected):
    i = pi.map_class(raw, _cmap(), CLASSES)
    assert (CLASSES[i] if i is not None else None) == expected


def test_config_class_map_matches_spec():
    ic = load_config()["indian_vehicle"]
    assert ic["classes"] == ["car", "bike", "bicycle", "auto", "bus", "truck", "ambulance", "person"]
    cm = {pi.norm_label(k): v for k, v in ic["class_map"].items()}
    assert set(cm.values()) <= set(ic["classes"])
    for raw in ("motorcycle", "scooter"):
        assert cm[raw] == "bike"
    for raw in ("truck", "truck_tanker", "pickup", "van", "tractor"):
        assert cm[raw] == "truck"


def test_voc_to_yolo_conversion(tmp_path):
    _img(tmp_path / "a.jpg", 200, 100)
    objs = OBJ.format(n="Scooter", a=20, b=10, c=60, d=90) + OBJ.format(n="tractor", a=100, b=0, c=200, d=50) \
        + OBJ.format(n="signboard", a=0, b=0, c=10, d=10)
    (tmp_path / "a.xml").write_text(XML.format(f="a.jpg", w=200, h=100, objs=objs))
    cmap = pi.voc_class_map(["Scooter", "tractor", "signboard"], _cmap(), CLASSES)
    r = validate_voc(tmp_path / "a.xml", tmp_path / "a.jpg", cmap)
    assert r["ok"] and r["dropped_codes"] == Counter({"signboard": 1})
    (c1, *b1), (c2, *b2) = r["boxes"]
    assert CLASSES[c1] == "bike" and b1 == pytest.approx([0.2, 0.5, 0.2, 0.8])
    assert CLASSES[c2] == "truck" and b2 == pytest.approx([0.75, 0.25, 0.5, 0.5])


def test_augmented_copies_share_a_split():
    assert pi.group_key("amb", Path("01_jpg.rf.53767469.jpg")) == pi.group_key("amb", Path("01_jpg.rf.59f307fb.jpg"))
    assert pi.group_key("amb", Path("01_jpg.rf.53767469.jpg")) != pi.group_key("amb", Path("02_jpg.rf.53767469.jpg"))
    groups = [f"g{i // 3}" for i in range(90)]                 # 30 photos x 3 augmented copies
    split = pi.group_split(groups, [{"ambulance"}] * 90, {"train": 0.7, "val": 0.15, "test": 0.15}, 42)
    owner = {groups[i]: s for s, idxs in split.items() for i in idxs}
    assert all(owner[groups[i]] == s for s, idxs in split.items() for i in idxs)
    assert sorted(i for v in split.values() for i in v) == list(range(90))
    assert len(split["test"]) % 3 == 0 and split["test"]


def test_resize_long_side_keeps_aspect(tmp_path):
    _img(tmp_path / "big.png", 3840, 2160)
    assert pi.resize_long_side(tmp_path / "big.png", tmp_path / "o.jpg", 640) == (640, 360)
    _img(tmp_path / "small.png", 300, 200)
    assert pi.resize_long_side(tmp_path / "small.png", tmp_path / "s.jpg", 640) == (300, 200)  # never upscales


def test_prepare_end_to_end(tmp_path, monkeypatch):
    voc, amb, out = tmp_path / "raw" / "iv", tmp_path / "raw" / "amb", tmp_path / "out"
    names = ["car", "Auto", "Bus", "motorcycle"]
    for i in range(40):
        sub = voc / names[i % 4]
        _img(sub / f"img{i}.jpg", 1280, 720)
        objs = OBJ.format(n=names[i % 4], a=100, b=100, c=500, d=400)
        if i % 5 == 0:
            objs += OBJ.format(n="dog", a=0, b=0, c=50, d=50)
        (sub / f"img{i}.xml").write_text(XML.format(f=f"img{i}.jpg", w=1280, h=720, objs=objs))
    (voc / "car" / "broken.xml").write_text(XML.format(f="img0.jpg", w=999, h=720, objs=""))  # size mismatch
    (amb / "data.yaml").parent.mkdir(parents=True)
    (amb / "data.yaml").write_text("names: ['Ambulance']\nnc: 1\n")
    for i in range(12):
        _img(amb / "train" / "images" / f"a{i}.jpg", 640, 480)
        lp = amb / "train" / "labels" / f"a{i}.txt"
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text("0 0.5 0.5 0.4 0.4\n")

    monkeypatch.setattr(pi, "ROOT", tmp_path)
    pi.main(["--no-pseudo", "--voc-raw", str(voc), "--amb-raw", str(amb), "--out", str(out),
             "--resized", str(tmp_path / "interim"), "--long-side", "320"])

    yaml_txt = (out / "data.yaml").read_text()
    for i, n in enumerate(CLASSES):
        assert f"  {i}: {n}" in yaml_txt
    imgs = {s: sorted((out / "images" / s).glob("*.jpg")) for s in ("train", "val", "test")}
    assert sum(len(v) for v in imgs.values()) == 52            # broken.xml excluded
    assert len(imgs["train"]) > len(imgs["val"]) > 0 and len(imgs["test"]) > 0
    ids = [p.stem for v in imgs.values() for p in v]
    assert len(ids) == len(set(ids))
    with Image.open(imgs["train"][0]) as im:
        assert max(im.size) == 320
    seen = Counter()
    for lf in (out / "labels").rglob("*.txt"):
        for ln in lf.read_text().splitlines():
            c, *xywh = ln.split()
            assert all(0 <= float(v) <= 1 for v in xywh)
            seen[CLASSES[int(c)]] += 1
    assert seen == Counter({"car": 10, "auto": 10, "bus": 10, "bike": 10, "ambulance": 12})
    assert (tmp_path / "reports" / "tables" / "indian_vehicle_split_stats.csv").exists()
    assert (tmp_path / "reports" / "tables" / "indian_vehicle_prep_summary.json").exists()
    assert not (tmp_path / "interim").exists()


def test_backend_weight_lookup_and_no_fake_detections(tmp_path):
    from backend.app.services.model_service import COCO_TO_VEHICLE, ModelRegistry, ModelUnavailable
    assert COCO_TO_VEHICLE["motorcycle"] == "bike" and "dog" not in COCO_TO_VEHICLE
    models = tmp_path / "models"
    models.mkdir()
    reg = ModelRegistry(models)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[200:400, 100:400] = (0, 0, 255)        # big red block: the old colour heuristic called this an ambulance
    for task in ("vehicle", "road_damage"):    # no weights anywhere -> error, never invented boxes
        with pytest.raises(ModelUnavailable):
            reg.detect(task, img)
    assert reg.available() == {"vehicle": False, "road_damage": False, "congestion": False}
    (tmp_path / "yolov8n.pt").write_bytes(b"x")                # COCO in the project root
    assert reg.weights_path("vehicle") == tmp_path / "yolov8n.pt" and reg.is_coco("vehicle")
    assert not reg.weights_path("road_damage").exists()        # road damage never falls back to a COCO car model
    (models / "indian_vehicle_best.pt").write_bytes(b"x")      # trained Indian model wins
    assert reg.weights_path("vehicle") == models / "indian_vehicle_best.pt" and not reg.is_coco("vehicle")


def test_exif_rotated_annotation_is_kept(tmp_path):
    # raw pixels 200x100 tagged "rotate 90 CW" (orientation 6); annotator saw the 100x200 rotated view
    raw = tmp_path / "rot.jpg"
    im = Image.fromarray(np.full((100, 200, 3), 90, dtype=np.uint8))
    ex = Image.Exif()
    ex[274] = 6
    im.save(raw, exif=ex)
    xml = tmp_path / "rot.xml"
    xml.write_text(XML.format(f="rot.jpg", w=100, h=200, objs=OBJ.format(n="bus", a=10, b=20, c=60, d=120)))
    assert pi.annotated_rotated(xml, raw)
    out = list(pi.voc_records(tmp_path, _cmap(), CLASSES, "drop", tmp_path / "tmp", 64))
    assert len(out) == 1
    _, _, res, prepared = out[0]
    assert res["ok"] and prepared is not None
    with Image.open(prepared) as p:
        assert p.size == (32, 64)                       # rotated to portrait, long side 64
    c, *box = res["boxes"][0]
    assert CLASSES[c] == "bus" and box == pytest.approx([0.35, 0.35, 0.5, 0.5], abs=0.02)


def test_pseudo_labels_never_override_human_boxes():
    human = [(6, 0.5, 0.5, 0.4, 0.4)]                      # ambulance, labelled by a person
    preds = [(5, 0.51, 0.5, 0.42, 0.38),                   # COCO calls the ambulance a truck -> skipped
             (0, 0.1, 0.1, 0.1, 0.1),                      # unlabelled car in the corner -> added
             (0, 0.11, 0.1, 0.1, 0.1),                     # duplicate of that car -> skipped
             (7, 0.5, 0.5, 0.05, 0.1)]                     # small box fully inside the ambulance -> skipped
    added = pi.merge_pseudo(human, preds, 0.3)
    assert added == [(0, 0.1, 0.1, 0.1, 0.1)]
    assert human == [(6, 0.5, 0.5, 0.4, 0.4)]              # input untouched


def test_max_copies_per_photo(tmp_path, monkeypatch):
    voc, amb, out = tmp_path / "iv", tmp_path / "amb", tmp_path / "out"
    for i in range(20):
        _img(voc / f"v{i}.jpg", 400, 300)
        (voc / f"v{i}.xml").write_text(XML.format(f=f"v{i}.jpg", w=400, h=300,
                                                  objs=OBJ.format(n="car", a=10, b=10, c=200, d=200)))
    (amb / "data.yaml").parent.mkdir(parents=True)
    (amb / "data.yaml").write_text("names: {0: AMBULANCE}\n")
    for i in range(15):
        for h in ("aa11", "bb22", "cc33"):                  # 3 Roboflow copies of each photo
            _img(amb / "train" / "images" / f"p{i}_jpg.rf.{h}.jpg", 320, 320)
            lp = amb / "train" / "labels" / f"p{i}_jpg.rf.{h}.txt"
            lp.parent.mkdir(parents=True, exist_ok=True)
            lp.write_text("0 0.5 0.5 0.3 0.3\n")
    monkeypatch.setattr(pi, "ROOT", tmp_path)
    pi.main(["--no-pseudo", "--voc-raw", str(voc), "--amb-raw", str(amb), "--out", str(out),
             "--resized", str(tmp_path / "interim")])
    amb_imgs = [p for p in (out / "images").rglob("amb_*.jpg")]
    assert len(amb_imgs) == 15                               # one copy per photo (config max_copies_per_photo: 1)
