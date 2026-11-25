"""Loads YOLO models lazily. If a task has no weights on disk, detect() raises ModelUnavailable (the API turns it
into HTTP 503) instead of returning made-up detections.
"""
from __future__ import annotations

import base64
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


# COCO class name -> project vehicle class. Anything else from a COCO model (dogs, chairs, ...) is ignored.
COCO_TO_VEHICLE = {"car": "car", "motorcycle": "bike", "bicycle": "bicycle", "bus": "bus", "truck": "truck",
                   "person": "person"}


class ModelUnavailable(RuntimeError):
    pass


class ModelRegistry:
    def __init__(self, model_dir: Path, device=None):
        self.model_dir, self.device = Path(model_dir), device
        self._models, self._cong, self._lock = {}, None, threading.Lock()

    def coco_weights(self) -> Path | None:
        """COCO yolov8n.pt next to the models folder: models/, the project root, or ml_models/."""
        for d in (self.model_dir, self.model_dir.parent, self.model_dir.parent / "ml_models"):
            p = Path(d) / "yolov8n.pt"
            if p.exists():
                return p
        return None

    def weights_path(self, task: str) -> Path:
        """Vehicle task: Indian-vehicle detector if trained, else COCO yolov8n. Other tasks: <task>_best.pt."""
        if task == "vehicle":
            indian = self.model_dir / "indian_vehicle_best.pt"
            if indian.exists():
                return indian
            return self.coco_weights() or indian
        return self.model_dir / f"{task}_best.pt"

    def is_coco(self, task: str) -> bool:
        return self.weights_path(task).name == "yolov8n.pt"

    def model_name(self, task: str) -> str:
        p = self.weights_path(task)
        return p.name if p.exists() else "unavailable"

    def available(self) -> dict:
        return {
            "vehicle": self.weights_path("vehicle").exists(),
            "road_damage": self.weights_path("road_damage").exists(),
            "congestion": (self.model_dir / "congestion_model.joblib").exists(),
        }

    def _imgsz(self, task: str) -> int:
        if self.is_coco(task):
            return 640
        stem = self.weights_path(task).name.replace("_best.pt", "")
        meta = self.model_dir / f"{stem}_train_meta.json"
        if meta.exists():
            try:
                return int(json.loads(meta.read_text())["train_args"]["imgsz"])
            except Exception:
                pass
        return 640

    def _load(self, task: str):
        if task not in self._models:
            p = self.weights_path(task)
            if not p.exists():
                hint = ("put models/indian_vehicle_best.pt or yolov8n.pt in the project root" if task == "vehicle"
                        else f"train it with: python -m ml.training.train_yolo --task {task}")
                raise ModelUnavailable(f"{task} model weights not found ({p.name}); {hint}")
            try:
                from ultralytics import YOLO
                from ml.training.train_yolo import pick_device
                self.device = self.device if self.device is not None else pick_device("auto")
                self._models[task] = YOLO(str(p))
            except Exception as e:
                raise ModelUnavailable(f"could not load {task} model {p.name}: {e}") from e
        return self._models[task]

    def detect(self, task: str, img_bgr: np.ndarray, conf: float = 0.25) -> Tuple[List[Dict[str, Any]], float]:
        with self._lock:
            model = self._load(task)
            t0 = time.perf_counter()
            r = model.predict(img_bgr, conf=conf, imgsz=self._imgsz(task), device=self.device, verbose=False)[0]
            ms = (time.perf_counter() - t0) * 1000
            b = r.boxes
            names = model.names
            coco = task == "vehicle" and self.is_coco(task)
            dets = []
            for c, s, (x1, y1, x2, y2) in zip(b.cls.cpu().numpy(), b.conf.cpu().numpy(), b.xyxy.cpu().numpy()):
                name = names[int(c)]
                if coco:
                    name = COCO_TO_VEHICLE.get(name)
                    if name is None:
                        continue
                dets.append({"class_name": name, "confidence": float(s),
                             "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)})
            return dets, ms

    def congestion(self, dets: list[dict], width: int, height: int) -> dict:
        import joblib
        p = self.model_dir / "congestion_model.joblib"
        if p.exists() and self._cong is None:
            try:
                self._cong = joblib.load(p)
            except Exception:
                self._cong = None

        if self._cong is not None:
            try:
                from ml.training.congestion import features_from_boxes, to_level
                bundle = self._cong
                classes = [f[2:] for f in bundle["features"] if f.startswith("n_") and f != "n_total" and f not in ("n_near", "n_far")]
                idx = {c: i for i, c in enumerate(classes)}
                d = [x for x in dets if x["class_name"] in idx]
                xywhn = [[(x["x1"] + x["x2"]) / 2 / width, (x["y1"] + x["y2"]) / 2 / height,
                          (x["x2"] - x["x1"]) / width, (x["y2"] - x["y1"]) / height] for x in d]
                f = features_from_boxes(np.array(xywhn).reshape(-1, 4), [idx[x["class_name"]] for x in d],
                                        [x["confidence"] for x in d], classes)
                index = float(np.clip(bundle["model"].predict([[f[k] for k in bundle["features"]]])[0], 0, 100))
                level = int(to_level(index, bundle["refs"]["thresholds"]))
                return {"index": round(index, 2), "level": bundle["levels"][level]}
            except Exception:
                pass

        pcu = sum(3.0 if d["class_name"] in ("truck", "bus", "tram") else 1.0 for d in dets)
        idx = min(100.0, max(5.0, pcu * 6.5))
        lvl = "heavy" if idx >= 65 else ("moderate" if idx >= 30 else "free")
        return {"index": round(idx, 1), "level": lvl}


def annotate(img_bgr: np.ndarray, dets: list[dict], color=(0, 255, 0)) -> str:
    im = img_bgr.copy()
    th = max(2, round(min(im.shape[:2]) / 250))
    for d in dets:
        c_name = d["class_name"]
        box_color = (0, 0, 255) if c_name in ("ambulance", "emergency", "fire_truck") else color
        p1, p2 = (int(d["x1"]), int(d["y1"])), (int(d["x2"]), int(d["y2"]))
        cv2.rectangle(im, p1, p2, box_color, th)
        label = f"{c_name.upper()} {d['confidence']:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(im, (p1[0], max(0, p1[1] - 22)), (p1[0] + w + 6, max(22, p1[1])), box_color, -1)
        cv2.putText(im, label, (p1[0] + 3, max(16, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()
