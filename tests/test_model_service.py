"""ModelRegistry behaviour: no fabricated detections, explicit demo mode, correct ONNX decoding."""
import os

os.environ["DATABASE_URL"] = "sqlite://"

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from backend.app.services.model_service import ModelRegistry, ModelUnavailable  # noqa: E402

BLANK = np.full((240, 320, 3), 100, np.uint8)


def test_missing_weights_raise_unavailable_by_default(tmp_path):
    reg = ModelRegistry(tmp_path)
    for task in ("vehicle", "road_damage"):
        with pytest.raises(ModelUnavailable, match="not found"):
            reg.detect(task, BLANK, 0.25)
    assert reg.available()["vehicle"] is False and reg.available()["heuristic_demo_mode"] is False


def test_demo_mode_is_labelled_and_never_invents_detections(tmp_path):
    reg = ModelRegistry(tmp_path, allow_fallback=True)
    assert reg.model_name("vehicle").endswith("heuristic_DEMO")
    assert reg.available()["heuristic_demo_mode"] is True
    for task in ("vehicle", "road_damage"):
        dets, _ = reg.detect(task, BLANK, 0.25)
        assert dets == []                         # featureless frame -> nothing detected, nothing made up


def test_available_reflects_weight_files(tmp_path):
    reg = ModelRegistry(tmp_path)
    (tmp_path / "vehicle_best.pt").write_bytes(b"x")
    assert reg.available()["vehicle"] is True and reg.available()["road_damage"] is False
    assert reg.model_name("vehicle") == "vehicle_best.pt"


def test_corrupt_weights_report_a_clear_error_instead_of_faking(tmp_path):
    (tmp_path / "road_damage_best.pt").write_bytes(b"not a real checkpoint")
    reg = ModelRegistry(tmp_path)
    with pytest.raises(ModelUnavailable):
        reg.detect("road_damage", BLANK, 0.25)


class FakeOnnxSession:
    class _In:
        name = "images"

    def __init__(self, out):
        self.out = out

    def get_inputs(self):
        return [self._In()]

    def run(self, _names, _feed):
        return [self.out]


def test_onnx_decoding_maps_boxes_and_class_names_back_to_original_image():
    # Original 200x100 (WxH) -> imgsz 640: scale 3.2, resized 640x320, vertical padding 160.
    n_anchor = 20
    out = np.zeros((8, n_anchor), np.float32)                     # (4 + 4 classes, anchors)
    out[:4, 0] = [320, 320, 100, 64]                              # cx, cy, w, h in letterbox pixels
    out[4 + 2, 0] = 0.9                                           # class index 2 -> "truck"
    out[:4, 1] = [100, 200, 40, 40]
    out[4 + 0, 1] = 0.10                                          # below threshold -> dropped
    img = np.zeros((100, 200, 3), np.uint8)

    dets, ms = ModelRegistry._infer_onnx(FakeOnnxSession(out[None]), img, 0.25, 640,
                                         ["car", "van", "truck", "tram"])
    assert len(dets) == 1 and ms >= 0
    d = dets[0]
    assert d["class_name"] == "truck" and d["confidence"] == pytest.approx(0.9, abs=1e-3)
    assert d["x1"] == pytest.approx(84.375, abs=0.6) and d["y1"] == pytest.approx(40.0, abs=0.6)
    assert d["x2"] == pytest.approx(115.625, abs=0.6) and d["y2"] == pytest.approx(60.0, abs=0.6)


def test_onnx_falls_back_to_generic_names_when_unknown():
    out = np.zeros((1, 6, 20), np.float32)
    out[0, :4, 0] = [320, 320, 100, 64]
    out[0, 5, 0] = 0.8
    dets, _ = ModelRegistry._infer_onnx(FakeOnnxSession(out), np.zeros((100, 200, 3), np.uint8), 0.25, 640, [])
    assert dets and dets[0]["class_name"] == "class_1"
