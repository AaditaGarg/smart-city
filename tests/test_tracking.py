"""Tests for unique-vehicle counting and line crossing (pure logic; the YOLO/ByteTrack wrapper needs weights)."""
from decision_engine.tracking import TrackCounter
from decision_engine.traffic_signal import webster_optimum_cycle


def det(tid, cls, cx, cy, size=20.0):
    return {"track_id": tid, "class_name": cls, "x1": cx - size, "y1": cy - size, "x2": cx + size, "y2": cy + size}


def test_same_vehicle_over_many_frames_counts_once():
    tc = TrackCounter(min_frames=3)
    for f in range(30):
        tc.update([det(1, "car", 100 + f, 200)])
    assert tc.unique_counts() == {"car": 1}      # a per-frame detector would have reported 30


def test_short_lived_tracks_are_ignored():
    tc = TrackCounter(min_frames=3)
    tc.update([det(1, "car", 10, 10), det(2, "truck", 50, 50)])
    tc.update([det(1, "car", 12, 10)])
    tc.update([det(1, "car", 14, 10)])
    assert tc.unique_counts() == {"car": 1}      # track 2 was a one-frame flicker


def test_class_is_majority_vote_not_first_frame():
    tc = TrackCounter(min_frames=1)
    for cls in ["van", "car", "car", "car", "van"]:
        tc.update([det(7, cls, 10, 10)])
    assert tc.unique_counts() == {"car": 1}


def test_detections_without_ids_are_ignored():
    tc = TrackCounter(min_frames=1)
    tc.update([{"class_name": "car", "x1": 0, "y1": 0, "x2": 10, "y2": 10}])
    assert tc.summary()["unique_vehicles"] == 0


def test_line_crossing_counted_once_and_only_when_crossing():
    tc = TrackCounter(line=((0, 100), (400, 100)), min_frames=1)       # horizontal line at y=100
    for y in range(60, 141, 10):                                       # car 1 drives down across the line
        tc.update([det(1, "car", 200, y), det(2, "truck", 300, 40)])   # truck 2 stays above it
    assert tc.line_counts() == {"car": 1}
    # jittering around the line must not count twice
    for y in (95, 105, 95, 105):
        tc.update([det(1, "car", 200, y)])
    assert tc.line_counts() == {"car": 1}


def test_counts_feed_the_signal_engine():
    tc = TrackCounter(min_frames=1)
    for f in range(5):
        tc.update([det(1, "car", f, 0), det(2, "truck", f, 50)])
    counts = tc.unique_counts()
    res = webster_optimum_cycle([{"name": "A", "counts": counts}, {"name": "B", "counts": {"car": 1}}])
    assert res["phase_timings"][0]["pcu"] == 4.0          # 1 car + 1 truck = 1.0 + 3.0
