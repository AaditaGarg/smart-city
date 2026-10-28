"""Loads the REAL trained models lazily. There is deliberately no mock fallback: if weights are missing the API
answers 503 instead of inventing predictions."""
from __future__ import annotations

import base64
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np


class ModelUnavailable(RuntimeError):
    pass


class ModelRegistry:
    def __init__(self, model_dir: Path, device=None):
        self.model_dir, self.device = Path(model_dir), device
        self._models, self._cong, self._lock = {}, None, threading.Lock()

    def weights_path(self, task: str) -> Path:
        return self.model_dir / f"{task}_best.pt"

    def model_name(self, task: str) -> str:
        return self.weights_path(task).name

    def available(self) -> dict:
        return {"vehicle": self.weights_path("vehicle").exists(),
                "road_damage": self.weights_path("road_damage").exists(),
                "congestion": (self.model_dir / "congestion_model.joblib").exists()}

    def _imgsz(self, task: str) -> int:
        meta = self.model_dir / f"{task}_train_meta.json"
        return int(json.loads(meta.read_text())["train_args"]["imgsz"]) if meta.exists() else 640

    def _load(self, task: str):
        if task not in self._models:
            p = self.weights_path(task)
            if not p.exists():
                raise ModelUnavailable(f"{p} not found - train the model first (python -m ml.training.train_yolo --task {task})")
            from ultralytics import YOLO
            from ml.training.train_yolo import pick_device
            self.device = self.device if self.device is not None else pick_device("auto")
            self._models[task] = YOLO(str(p))
        return self._models[task]

    def detect(self, task: str, img_bgr: np.ndarray, conf: float = 0.25):
        with self._lock:
            model = self._load(task)
            t0 = time.perf_counter()
            r = model.predict(img_bgr, conf=conf, imgsz=self._imgsz(task), device=self.device, verbose=False)[0]
            ms = (time.perf_counter() - t0) * 1000
        b = r.boxes
        names = model.names
        dets = [{"class_name": names[int(c)], "confidence": float(s), "x1": float(x1), "y1": float(y1),
                 "x2": float(x2), "y2": float(y2)}
                for c, s, (x1, y1, x2, y2) in zip(b.cls.cpu().numpy(), b.conf.cpu().numpy(), b.xyxy.cpu().numpy())]
        return dets, ms

    def congestion(self, dets: list[dict], width: int, height: int) -> dict:
        import joblib
        from ml.training.congestion import features_from_boxes, to_level
        p = self.model_dir / "congestion_model.joblib"
        if not p.exists():
            raise ModelUnavailable(f"{p} not found - run python -m ml.training.congestion")
        if self._cong is None:
            self._cong = joblib.load(p)
        bundle = self._cong
        classes = [f[2:] for f in bundle["features"] if f.startswith("n_") and f != "n_total"
                   and f not in ("n_near", "n_far")]
        idx = {c: i for i, c in enumerate(classes)}
        d = [x for x in dets if x["class_name"] in idx]
        xywhn = [[(x["x1"] + x["x2"]) / 2 / width, (x["y1"] + x["y2"]) / 2 / height,
                  (x["x2"] - x["x1"]) / width, (x["y2"] - x["y1"]) / height] for x in d]
        f = features_from_boxes(np.array(xywhn).reshape(-1, 4), [idx[x["class_name"]] for x in d],
                                [x["confidence"] for x in d], classes)
        index = float(np.clip(bundle["model"].predict([[f[k] for k in bundle["features"]]])[0], 0, 100))
        level = int(to_level(index, bundle["refs"]["thresholds"]))
        return {"index": round(index, 2), "level": bundle["levels"][level]}


def annotate(img_bgr: np.ndarray, dets: list[dict], color=(0, 0, 255)) -> str:
    im = img_bgr.copy()
    th = max(1, round(min(im.shape[:2]) / 300))
    for d in dets:
        p1, p2 = (int(d["x1"]), int(d["y1"])), (int(d["x2"]), int(d["y2"]))
        cv2.rectangle(im, p1, p2, color, th)
        cv2.putText(im, f"{d['class_name']} {d['confidence']:.2f}", (p1[0], max(12, p1[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()
