"""Model registry: loads trained YOLO weights (ONNX Runtime preferred, PyTorch otherwise).

Behaviour when weights are missing
----------------------------------
By default the registry raises ``ModelUnavailable`` (HTTP 503). It never invents detections.
For demos without weights you can opt in with ``ALLOW_HEURISTIC_FALLBACK=true``; results are then produced by a
classical-CV heuristic and the model name is reported as ``<task>_heuristic_DEMO`` so it is visible in API
responses, the database and the UI.
"""
from __future__ import annotations

import base64
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..core.config import settings

ROOT = Path(__file__).resolve().parents[3]


class ModelUnavailable(RuntimeError):
    pass


class ModelRegistry:
    def __init__(self, model_dir: Path, device=None, allow_fallback: Optional[bool] = None):
        self.model_dir, self.device = Path(model_dir), device
        self.allow_fallback = settings.allow_heuristic_fallback if allow_fallback is None else allow_fallback
        self._models: Dict[str, Tuple[str, Any]] = {}
        self._load_errors: Dict[str, str] = {}
        self._cong_bundle = None
        self._cong_meta: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------ paths / status
    def weights_path(self, task: str) -> Path:
        return self.model_dir / f"{task}_best.pt"

    def onnx_path(self, task: str) -> Path:
        return self.model_dir / f"{task}_best.onnx"

    def has_weights(self, task: str) -> bool:
        return self.onnx_path(task).exists() or self.weights_path(task).exists()

    def model_name(self, task: str) -> str:
        if self.onnx_path(task).exists():
            return self.onnx_path(task).name
        if self.weights_path(task).exists():
            return self.weights_path(task).name
        if self.allow_fallback:
            return f"{task}_heuristic_DEMO"
        return f"{task}_unavailable"

    def available(self) -> dict:
        return {
            "vehicle": self.has_weights("vehicle"),
            "road_damage": self.has_weights("road_damage"),
            "congestion": (self.model_dir / "congestion_model.joblib").exists(),
            "onnx_accelerated": any(self.onnx_path(t).exists() for t in ("vehicle", "road_damage")),
            "heuristic_demo_mode": bool(self.allow_fallback),
        }

    def _imgsz(self, task: str) -> int:
        meta = self.model_dir / f"{task}_train_meta.json"
        if meta.exists():
            try:
                return int(json.loads(meta.read_text())["train_args"]["imgsz"])
            except Exception:
                pass
        return 640

    def _class_names(self, task: str) -> List[str]:
        """Class order used at training time (needed because ONNX files carry no names)."""
        try:
            import yaml
            cfg = yaml.safe_load((ROOT / "configs" / "project.yaml").read_text())
            if task == "vehicle":
                return list(cfg["vehicle"]["classes"])
            return [c["name"] for c in cfg["road_damage"]["classes"]]
        except Exception:
            return []

    # ------------------------------------------------------------ loading
    def _load(self, task: str) -> Optional[Tuple[str, Any]]:
        """Lazy loader: ONNX Runtime first, then PyTorch weights. Returns None if no weights/loader."""
        if task in self._models:
            return self._models[task]

        onnx_file = self.onnx_path(task)
        if onnx_file.exists():
            try:
                import onnxruntime as ort
                session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
                self._models[task] = ("onnx", session)
                return self._models[task]
            except Exception as e:
                self._load_errors[task] = f"could not load {onnx_file.name}: {e}"

        pt_file = self.weights_path(task)
        if pt_file.exists():
            try:
                from ultralytics import YOLO
                from ml.training.train_yolo import pick_device
                self.device = self.device if self.device is not None else pick_device("auto")
                self._models[task] = ("yolo", YOLO(str(pt_file)))
                return self._models[task]
            except Exception as e:
                self._load_errors[task] = f"could not load {pt_file.name}: {e}"
        return None

    # ------------------------------------------------------------ inference
    def detect(self, task: str, img_bgr: np.ndarray, conf: float = 0.25) -> Tuple[List[Dict[str, Any]], float]:
        with self._lock:
            bundle = self._load(task)
            if bundle is not None:
                m_type, model = bundle
                if m_type == "yolo":
                    t0 = time.perf_counter()
                    use_half = self.device in ("mps", 0, "cuda:0")
                    r = model.predict(img_bgr, conf=conf, imgsz=self._imgsz(task), device=self.device,
                                      half=use_half, verbose=False)[0]
                    ms = (time.perf_counter() - t0) * 1000
                    b, names = r.boxes, model.names
                    dets = [
                        {"class_name": names[int(c)], "confidence": float(s),
                         "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
                        for c, s, (x1, y1, x2, y2) in zip(b.cls.cpu().numpy(), b.conf.cpu().numpy(),
                                                          b.xyxy.cpu().numpy())
                    ]
                    return dets, round(ms, 2)
                return self._infer_onnx(model, img_bgr, conf, self._imgsz(task), self._class_names(task))

            if self.allow_fallback:
                return self._cv_edge_detect(task, img_bgr, conf)

            reason = self._load_errors.get(task) or (
                f"{task} weights not found: expected {self.onnx_path(task).name} or "
                f"{self.weights_path(task).name} in {self.model_dir}")
            raise ModelUnavailable(
                f"{reason}. Download/train the weights (see models/README.md), or set "
                f"ALLOW_HEURISTIC_FALLBACK=true for a clearly-labelled demo mode.")

    @staticmethod
    def _infer_onnx(session, img_bgr: np.ndarray, conf: float, imgsz: int = 640,
                    names: Optional[List[str]] = None) -> Tuple[List[Dict[str, Any]], float]:
        """ONNX Runtime inference for YOLOv8/11 exports (output (1, 4+nc, N)); centred letterbox like Ultralytics."""
        t0 = time.perf_counter()
        H, W = img_bgr.shape[:2]
        scale = min(imgsz / H, imgsz / W)
        new_w, new_h = int(round(W * scale)), int(round(H * scale))
        pad_x, pad_y = (imgsz - new_w) // 2, (imgsz - new_h) // 2
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255.0, (imgsz, imgsz), swapRB=True, crop=False)
        out = session.run(None, {session.get_inputs()[0].name: blob})[0]
        preds = out[0] if out.ndim == 3 else out
        if preds.shape[0] < preds.shape[1]:          # (4+nc, N) -> (N, 4+nc)
            preds = preds.T

        scores = preds[:, 4:]
        cls_ids = scores.argmax(axis=1)
        confs = scores.max(axis=1)
        keep = confs >= conf
        preds, cls_ids, confs = preds[keep], cls_ids[keep], confs[keep]

        dets: List[Dict[str, Any]] = []
        if len(confs):
            cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
            x1 = (cx - w / 2 - pad_x) / scale
            y1 = (cy - h / 2 - pad_y) / scale
            boxes = [[float(a), float(b), float(c / scale), float(d / scale)] for a, b, c, d in zip(x1, y1, w, h)]
            idxs = cv2.dnn.NMSBoxes(boxes, [float(c) for c in confs], conf, 0.45)
            for i in (np.array(idxs).flatten() if len(idxs) else []):
                bx, by, bw, bh = boxes[i]
                cid = int(cls_ids[i])
                dets.append({
                    "class_name": names[cid] if names and cid < len(names) else f"class_{cid}",
                    "confidence": round(float(confs[i]), 4),
                    "x1": float(max(0.0, bx)), "y1": float(max(0.0, by)),
                    "x2": float(min(W - 1, bx + bw)), "y2": float(min(H - 1, by + bh)),
                })
        return dets, round((time.perf_counter() - t0) * 1000, 2)

    def _cv_edge_detect(self, task: str, img_bgr: np.ndarray, conf: float) -> Tuple[List[Dict[str, Any]], float]:
        """DEMO-ONLY classical-CV heuristic. NOT a trained model; accuracy is unmeasured.

        Only reachable when ALLOW_HEURISTIC_FALLBACK=true. The red/blue "ambulance" rule below is a colour
        heuristic and must never be relied on for real emergency-vehicle preemption.

        Original description:
        
        Optimizations:
        1. Resolution Normalization: standardizes inference frame to max 640px, speeding up
           CLAHE, Canny, and morphology by 8x-15x on high-resolution camera frames.
        2. Dual-Strobe (Red + Blue) Emergency Beacon Analysis: checks strobe masks strictly
           within individual vehicle ROIs to avoid false emergency alerts.
        3. C++ Accelerated NMS: replaces python loops with OpenCV's SIMD-compiled cv2.dnn.NMSBoxes.
        4. Geometric Pothole Filtering: uses convex hull solidity and circularity to discriminate
           potholes from linear cracks.
        """
        t0 = time.perf_counter()
        orig_H, orig_W = img_bgr.shape[:2]

        # Resolution normalization (max dimension 640px)
        max_dim = 640
        scale = min(max_dim / max(orig_W, orig_H), 1.0)
        if scale < 1.0:
            proc_W, proc_H = int(orig_W * scale), int(orig_H * scale)
            proc_img = cv2.resize(img_bgr, (proc_W, proc_H), interpolation=cv2.INTER_AREA)
        else:
            proc_W, proc_H = orig_W, orig_H
            proc_img = img_bgr
            scale = 1.0

        proc_area = proc_W * proc_H
        dets = []

        if task == "vehicle":
            gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
            # Analyze roadway perspective (bottom 85% of traffic frame)
            roi_y = int(proc_H * 0.15)
            roi_gray = gray[roi_y:proc_H, 0:proc_W]

            # Fast adaptive contrast enhancement via CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
            enhanced = clahe.apply(roi_gray)

            # Dual-threshold Edge Extraction & Morphological closing
            edges = cv2.Canny(enhanced, 45, 135)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Dual-Strobe Emergency Beacon Detection (Red + Blue Indian Emergency Lighting)
            hsv = cv2.cvtColor(proc_img, cv2.COLOR_BGR2HSV)
            red_mask = cv2.bitwise_or(
                cv2.inRange(hsv, np.array([0, 120, 120]), np.array([10, 255, 255])),
                cv2.inRange(hsv, np.array([170, 120, 120]), np.array([180, 255, 255]))
            )
            blue_mask = cv2.inRange(hsv, np.array([100, 130, 120]), np.array([130, 255, 255]))
            strobe_mask = cv2.bitwise_or(red_mask, blue_mask)
            has_emergency_lights = cv2.countNonZero(strobe_mask) > (proc_area * 0.0025)

            candidate_boxes = []
            candidate_scores = []

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                aspect = float(w) / max(1, h)

                # Filter realistic vehicle bounds
                if (proc_area * 0.006) <= area <= (proc_area * 0.42) and 0.35 <= aspect <= 3.8:
                    real_y = y + roi_y
                    # Split wide multi-vehicle contours
                    if w > (proc_W * 0.42) and aspect > 2.0:
                        half_w = w // 2
                        candidate_boxes.append([x, real_y, half_w, h])
                        candidate_scores.append(0.85)
                        candidate_boxes.append([x + half_w, real_y, half_w, h])
                        candidate_scores.append(0.85)
                    else:
                        candidate_boxes.append([x, real_y, w, h])
                        score = min(0.95, 0.68 + (area / proc_area) * 0.4)
                        candidate_scores.append(float(score))

            # Fast C++ Non-Maximum Suppression
            if candidate_boxes:
                indices = cv2.dnn.NMSBoxes(candidate_boxes, candidate_scores, score_threshold=conf, nms_threshold=0.45)
                if len(indices) > 0:
                    indices = indices.flatten() if hasattr(indices, "flatten") else [i[0] for i in indices]
                    for idx in indices[:15]:
                        bx, by, bw, bh = candidate_boxes[idx]
                        aspect = float(bw) / max(1, bh)
                        area = bw * bh

                        # Check strobe ratio strictly inside THIS bounding box ROI
                        roi_strobe = strobe_mask[by:by + bh, bx:bx + bw]
                        box_strobe_ratio = cv2.countNonZero(roi_strobe) / max(1.0, float(area))
                        is_amb = has_emergency_lights and (box_strobe_ratio > 0.02)

                        if is_amb:
                            c_name = "ambulance"
                            conf_val = 0.95
                        elif area > (proc_area * 0.10) or bw > (proc_W * 0.28):
                            c_name = "truck" if aspect > 1.2 else "tram"
                            conf_val = round(min(0.95, 0.76 + (area / proc_area) * 0.4), 2)
                        elif area > (proc_area * 0.04):
                            c_name = "van"
                            conf_val = round(min(0.92, 0.72 + (area / proc_area) * 0.4), 2)
                        elif aspect < 0.7 and area < (proc_area * 0.03):
                            c_name = "bike"
                            conf_val = 0.80
                        else:
                            c_name = "car"
                            conf_val = round(min(0.93, 0.70 + (area / proc_area) * 0.4), 2)

                        dets.append({
                            "class_name": c_name,
                            "confidence": conf_val,
                            "x1": float(bx / scale),
                            "y1": float(by / scale),
                            "x2": float(min(orig_W - 1, (bx + bw) / scale)),
                            "y2": float(min(orig_H - 1, (by + bh) / scale)),
                        })


        else:
            # Potholes and pavement distress
            gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            edges = cv2.Canny(enhanced, 35, 110)

            # Elliptical dilation to identify depression cavities
            kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated = cv2.dilate(edges, kernel_ellipse, iterations=1)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            candidate_boxes = []
            candidate_scores = []
            candidate_classes = []

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                aspect = float(w) / max(1, h)
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(cv2.contourArea(cnt)) / max(1.0, hull_area)

                if (proc_area * 0.005) <= area <= (proc_area * 0.22):
                    # Potholes exhibit compact circularity and moderate-to-high solidity
                    if 0.65 <= aspect <= 1.55 and (solidity > 0.40 or area > proc_area * 0.02):
                        d_class = "pothole"
                        conf_val = round(min(0.94, 0.70 + (area / proc_area) * 0.8), 2)
                    elif area > (proc_area * 0.035):
                        d_class = "alligator_crack"
                        conf_val = round(min(0.90, 0.68 + (area / proc_area) * 0.6), 2)
                    elif aspect > 2.0:
                        d_class = "transverse_crack"
                        conf_val = 0.72
                    else:
                        d_class = "longitudinal_crack"
                        conf_val = round(min(0.88, 0.65 + (area / proc_area) * 0.7), 2)

                    candidate_boxes.append([x, y, w, h])
                    candidate_scores.append(conf_val)
                    candidate_classes.append(d_class)

            if candidate_boxes:
                indices = cv2.dnn.NMSBoxes(candidate_boxes, candidate_scores, score_threshold=conf, nms_threshold=0.40)
                if len(indices) > 0:
                    indices = indices.flatten() if hasattr(indices, "flatten") else [i[0] for i in indices]
                    for idx in indices[:15]:
                        bx, by, bw, bh = candidate_boxes[idx]
                        dets.append({
                            "class_name": candidate_classes[idx],
                            "confidence": candidate_scores[idx],
                            "x1": float(bx / scale),
                            "y1": float(by / scale),
                            "x2": float(min(orig_W - 1, (bx + bw) / scale)),
                            "y2": float(min(orig_H - 1, (by + bh) / scale)),
                        })


        ms = (time.perf_counter() - t0) * 1000
        return dets[:15], round(ms, 2)

    def congestion(self, dets: list[dict], width: int, height: int) -> dict:
        """Estimates congestion index (0-100) and traffic state (free, moderate, heavy)
        with cached feature extractors and vectorized coordinate normalization.
        """
        p = self.model_dir / "congestion_model.joblib"
        if p.exists() and self._cong_meta is None:
            try:
                import joblib
                bundle = joblib.load(p)
                features = bundle["features"]
                classes = [f[2:] for f in features if f.startswith("n_") and f != "n_total" and f not in ("n_near", "n_far")]
                class_to_idx = {c: i for i, c in enumerate(classes)}
                self._cong_meta = {
                    "model": bundle["model"],
                    "features": features,
                    "classes": classes,
                    "class_to_idx": class_to_idx,
                    "refs": bundle["refs"],
                    "levels": bundle["levels"],
                }
            except Exception:
                self._cong_meta = False

        if self._cong_meta:
            try:
                from ml.training.congestion import features_from_boxes, to_level
                meta = self._cong_meta
                classes = meta["classes"]
                idx_map = meta["class_to_idx"]
                d = [x for x in dets if x["class_name"] in idx_map]

                if d:
                    xywhn = [[(x["x1"] + x["x2"]) * 0.5 / width, (x["y1"] + x["y2"]) * 0.5 / height,
                              (x["x2"] - x["x1"]) / width, (x["y2"] - x["y1"]) / height] for x in d]
                    cls_indices = [idx_map[x["class_name"]] for x in d]
                    confidences = [x["confidence"] for x in d]
                else:
                    xywhn = np.empty((0, 4))
                    cls_indices = []
                    confidences = []

                f = features_from_boxes(xywhn, cls_indices, confidences, classes)
                feat_row = [f[k] for k in meta["features"]]
                index = float(np.clip(meta["model"].predict([feat_row])[0], 0, 100))
                level = int(to_level(index, meta["refs"]["thresholds"]))
                return {"index": round(index, 2), "level": meta["levels"][level], "source": "random_forest"}
            except Exception:
                pass

        pcu = sum(3.0 if d["class_name"] in ("truck", "tram") else 1.0 for d in dets)
        idx = min(100.0, max(5.0, pcu * 6.5))
        lvl = "heavy" if idx >= 65 else ("moderate" if idx >= 30 else "free")
        return {"index": round(idx, 1), "level": lvl, "source": "rule_based_fallback"}


def annotate(img_bgr: np.ndarray, dets: list[dict], default_color=(0, 255, 0)) -> str:
    """Renders high-visibility bounding boxes with color-coded classification tags."""
    im = img_bgr.copy()
    th = max(2, round(min(im.shape[:2]) / 280))

    # Distinct high-contrast palette
    CLASS_COLORS = {
        "ambulance": (0, 0, 240),          # Bold Red
        "emergency": (0, 0, 240),
        "police": (240, 100, 0),           # Bright Blue
        "fire_truck": (0, 30, 240),
        "truck": (0, 165, 255),            # Orange
        "tram": (0, 165, 255),
        "van": (180, 180, 0),              # Teal
        "pothole": (0, 0, 230),            # Red
        "alligator_crack": (0, 140, 255),  # Amber
        "transverse_crack": (220, 80, 180),# Purple
        "longitudinal_crack": (220, 80, 180),
        "car": (0, 220, 120),              # Emerald Green
        "bike": (240, 200, 0),             # Cyan
    }

    for d in dets:
        c_name = d["class_name"]
        box_color = CLASS_COLORS.get(c_name, default_color)
        p1 = (int(d["x1"]), int(d["y1"]))
        p2 = (int(d["x2"]), int(d["y2"]))
        cv2.rectangle(im, p1, p2, box_color, th)

        label = f"{c_name.upper()} {d['confidence']:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        tag_top = max(0, p1[1] - h - 8)
        cv2.rectangle(im, (p1[0], tag_top), (p1[0] + w + 8, p1[1]), box_color, -1)
        cv2.putText(
            im,
            label,
            (p1[0] + 4, max(h + 2, p1[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()
