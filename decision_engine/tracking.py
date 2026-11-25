"""Multi-object tracking helpers: unique-vehicle counting and virtual-line crossing counts.

A single-frame detector counts every vehicle in the frame, so the same car is counted again in every frame of a
video. A tracker (ByteTrack) gives each vehicle a persistent ``track_id``; this module turns those tracked
detections into counts that are safe to feed into ``webster_optimum_cycle``.

``TrackCounter`` is pure Python (no torch/ultralytics) so it can be unit-tested. ``track_video`` is the thin
Ultralytics/ByteTrack wrapper that produces its input.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

Point = Tuple[float, float]


def _side(p: Point, a: Point, b: Point) -> float:
    """Sign of the cross product: which side of line a->b the point p is on."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


class TrackCounter:
    """Accumulates tracked detections frame by frame.

    line: optional virtual counting line ((x1, y1), (x2, y2)) in pixels. A vehicle is counted once when the centre
    of its box moves from one side of the line to the other.
    min_frames: a track must be seen this many frames before it counts as a vehicle, which suppresses one-frame
    false positives and ID flicker.
    """

    def __init__(self, line: Optional[Tuple[Point, Point]] = None, min_frames: int = 3):
        self.line = line
        self.min_frames = max(1, int(min_frames))
        self.frames = 0
        self._classes: Dict[int, Counter] = {}      # track_id -> votes for each class
        self._seen: Dict[int, int] = {}              # track_id -> frames seen
        self._last_side: Dict[int, float] = {}
        self._crossed: Dict[int, int] = {}           # track_id -> +1 / -1 (direction of the crossing)

    def update(self, detections: Iterable[Dict[str, Any]]) -> None:
        """detections: dicts with track_id, class_name, x1, y1, x2, y2 (detections without a track_id are ignored)."""
        self.frames += 1
        for d in detections:
            tid = d.get("track_id")
            if tid is None:
                continue
            tid = int(tid)
            self._seen[tid] = self._seen.get(tid, 0) + 1
            self._classes.setdefault(tid, Counter())[str(d["class_name"])] += 1
            if self.line is not None and tid not in self._crossed:
                c = ((d["x1"] + d["x2"]) / 2.0, (d["y1"] + d["y2"]) / 2.0)
                side = _side(c, *self.line)
                prev = self._last_side.get(tid)
                if prev is not None and prev * side < 0:
                    self._crossed[tid] = 1 if side > 0 else -1
                if side != 0:
                    self._last_side[tid] = side

    # ---------------------------------------------------------------- results
    def _final_class(self, tid: int) -> str:
        return self._classes[tid].most_common(1)[0][0]   # majority vote across the track's life

    def _valid_tracks(self) -> List[int]:
        return [t for t, n in self._seen.items() if n >= self.min_frames]

    def unique_counts(self) -> Dict[str, int]:
        """Distinct vehicles seen per class (each track counted once, using its majority class)."""
        out: Dict[str, int] = {}
        for t in self._valid_tracks():
            c = self._final_class(t)
            out[c] = out.get(c, 0) + 1
        return out

    def line_counts(self) -> Dict[str, int]:
        """Vehicles that crossed the virtual line, per class."""
        out: Dict[str, int] = {}
        for t in self._crossed:
            if self._seen.get(t, 0) >= self.min_frames:
                c = self._final_class(t)
                out[c] = out.get(c, 0) + 1
        return out

    def summary(self) -> Dict[str, Any]:
        uc = self.unique_counts()
        return {"frames": self.frames, "unique_vehicles": sum(uc.values()), "unique_counts": uc,
                "line_crossings": sum(self.line_counts().values()), "line_counts": self.line_counts()}


def track_video(video_path: str, weights: str, conf: float = 0.25, imgsz: int = 640,
                line: Optional[Tuple[Point, Point]] = None, min_frames: int = 3,
                tracker: str = "bytetrack.yaml", device: Any = None, max_frames: int = 0) -> Dict[str, Any]:
    """Run YOLO + ByteTrack over a video and return TrackCounter.summary().

    Needs ``ultralytics`` and a trained vehicle weights file. ``max_frames`` > 0 stops early (for quick tests).
    """
    from ultralytics import YOLO

    model = YOLO(weights)
    counter = TrackCounter(line=line, min_frames=min_frames)
    stream = model.track(source=video_path, stream=True, persist=True, tracker=tracker, conf=conf,
                         imgsz=imgsz, device=device, verbose=False)
    for i, r in enumerate(stream):
        if max_frames and i >= max_frames:
            break
        b = r.boxes
        if b is None or b.id is None:
            counter.update([])
            continue
        names = model.names
        dets = [{"track_id": int(t), "class_name": names[int(c)], "x1": float(x1), "y1": float(y1),
                 "x2": float(x2), "y2": float(y2)}
                for t, c, (x1, y1, x2, y2) in zip(b.id.cpu().numpy(), b.cls.cpu().numpy(), b.xyxy.cpu().numpy())]
        counter.update(dets)
    return counter.summary()


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Count unique vehicles in a video with YOLO + ByteTrack.")
    ap.add_argument("video")
    ap.add_argument("--weights", default="models/vehicle_best.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--min-frames", type=int, default=3)
    ap.add_argument("--line", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                    help="virtual counting line in pixels")
    a = ap.parse_args()
    ln = ((a.line[0], a.line[1]), (a.line[2], a.line[3])) if a.line else None
    print(json.dumps(track_video(a.video, a.weights, a.conf, line=ln, min_frames=a.min_frames), indent=2))
