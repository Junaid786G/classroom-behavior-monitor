"""Regression tests for the per-student adaptive yaw/pitch baseline (Bug 8).

These lock in the fix where DISTRACTED is decided by deviation from each track's
OWN median resting pose rather than one fixed global yaw threshold. Students in
side columns must angle their head toward the center board to be attentive, so a
global threshold flagged their resting pose as distraction.

The baseline mirrors the _ear_history pattern exactly: a fixed frame-count window
(_EAR_HISTORY_LEN), no baseline below _EAR_HISTORY_MIN frames, and every measured
frame entering the history regardless of how it was classified.
"""
import types

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from backend.pipeline import behavior as bmod
from backend.pipeline.behavior import BehaviorAnalyzer, BehaviorType


# A side-column seat rests ABOVE the global threshold — this is the whole bug.
_SIDE_COLUMN_YAW = bmod._YAW_THRESHOLD + 5.0
_FRAME_MS = 1000.0   # 1 s/frame, so the dwell timer can elapse inside the window


# ── Fake MediaPipe landmark construction ──────────────────────────────────────

class _Pt:
    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


def _open_eye_landmarks():
    """Symmetric, wide-open eye landmarks: EAR plays no part in these tests
    (mode is "cnn" and the CNN gate is stubbed), but the geometry must be
    well-formed so _ear() stays finite."""
    n = max(max(bmod._LEFT_EYE), max(bmod._RIGHT_EYE)) + 1
    lms = [_Pt(0.5, 0.5) for _ in range(n)]
    for indices, cx in ((bmod._LEFT_EYE, 0.3), (bmod._RIGHT_EYE, 0.6)):
        i0, i1, i2, i3, i4, i5 = indices
        lms[i0] = _Pt(cx - 0.05, 0.4)
        lms[i3] = _Pt(cx + 0.05, 0.4)
        lms[i1] = _Pt(cx - 0.025, 0.4 + 0.015)
        lms[i5] = _Pt(cx - 0.025, 0.4 - 0.015)
        lms[i2] = _Pt(cx + 0.025, 0.4 + 0.015)
        lms[i4] = _Pt(cx + 0.025, 0.4 - 0.015)
    return lms


def _pose_matrix(pitch=0.0, yaw=0.0, roll=0.0):
    """A 4×4 transform matrix that _rotation_to_euler() decodes back to the
    given (pitch, yaw, roll) in degrees."""
    m = np.eye(4)
    m[:3, :3] = Rotation.from_euler("xyz", [pitch, yaw, roll], degrees=True).as_matrix()
    return m


@pytest.fixture
def frame_and_bbox():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = np.array([100, 100, 200, 200], dtype=float)
    return frame, bbox


@pytest.fixture
def feeder(monkeypatch, frame_and_bbox):
    """(analyzer, feed) where feed(yaw, pitch, tid, frame_no) runs one frame at
    that head pose through the full _analyze_one path.

    Eyes are stubbed open so the pose branch is what decides the label, and
    frame_no drives the video timestamp at _FRAME_MS per frame.
    """
    frame, bbox = frame_and_bbox
    monkeypatch.setattr(bmod, "_EYE_STATE_METHOD", "cnn")

    analyzer = BehaviorAnalyzer()
    analyzer._ready = True
    det = types.SimpleNamespace(
        face_landmarks=[_open_eye_landmarks()],
        facial_transformation_matrixes=[_pose_matrix()],
    )
    analyzer._lm = types.SimpleNamespace(detect=lambda _img: det)
    analyzer._eyes_closed_cnn = lambda _f, _b, _l: False   # eyes open: pose decides

    def feed(yaw, *, tid, frame_no, pitch=0.0, pose=True):
        det.facial_transformation_matrixes = (
            [_pose_matrix(pitch=pitch, yaw=yaw)] if pose else []
        )
        return analyzer._analyze_one(
            frame, bbox, track_id=tid, timestamp_ms=frame_no * _FRAME_MS
        )

    return analyzer, feed


# ── The side-column case ──────────────────────────────────────────────────────

def test_side_column_student_never_distracted_while_attentive(feeder):
    """THE BUG: a student whose resting yaw sits above the global threshold is
    attentive the whole time and must never be flagged DISTRACTED.

    Their resting pose alone used to clear _YAW_THRESHOLD on every frame, so the
    dwell timer ran to completion and reported DISTRACTED. With the adaptive
    baseline the timer is cleared once their own median is known — and during the
    warm-up frames before that, the dwell timer has not yet elapsed, so no frame
    in the sequence is ever DISTRACTED.
    """
    analyzer, feed = feeder
    tid = 20

    labels = [
        feed(_SIDE_COLUMN_YAW + (2.0 if i % 2 else -2.0), tid=tid, frame_no=i).behavior
        for i in range(bmod._EAR_HISTORY_LEN)
    ]

    assert BehaviorType.DISTRACTED not in labels
    assert labels[-1] is BehaviorType.ATTENTIVE
    # Baseline settled on their true resting pose, not on 0.
    assert analyzer._yaw_history[tid]
    assert float(np.median(analyzer._yaw_history[tid])) == pytest.approx(
        _SIDE_COLUMN_YAW, abs=2.5
    )
    # Timer cleared, not merely still accruing.
    assert analyzer._distracted_since[tid] is None


def test_side_column_student_still_distracted_when_genuinely_turning_away(feeder):
    """The fix must not make side-column students unflaggable: once the baseline
    is learned, a sustained turn well past it is still DISTRACTED."""
    analyzer, feed = feeder
    tid = 21

    for i in range(bmod._EAR_HISTORY_LEN):
        feed(_SIDE_COLUMN_YAW, tid=tid, frame_no=i)

    turned = _SIDE_COLUMN_YAW + bmod._YAW_DEVIATION + 10.0
    n_turned = int(bmod._DISTRACTED_SECONDS) + 2
    labels = [
        feed(turned, tid=tid, frame_no=bmod._EAR_HISTORY_LEN + i).behavior
        for i in range(n_turned)
    ]

    assert labels[-1] is BehaviorType.DISTRACTED
    # The median is robust to the turned frames still entering the window, so the
    # baseline does not chase the student's own distraction within this window.
    assert float(np.median(analyzer._yaw_history[tid])) == pytest.approx(
        _SIDE_COLUMN_YAW, abs=2.5
    )


def test_center_student_turning_away_is_still_distracted(feeder):
    """A center-seated student (resting yaw ~0) turning to a yaw the old global
    threshold would have caught is still DISTRACTED under the deviation test."""
    analyzer, feed = feeder
    tid = 22

    for i in range(bmod._EAR_HISTORY_LEN):
        feed(1.0 if i % 2 else -1.0, tid=tid, frame_no=i)

    turned = bmod._YAW_DEVIATION + 10.0
    labels = [
        feed(turned, tid=tid, frame_no=bmod._EAR_HISTORY_LEN + i).behavior
        for i in range(int(bmod._DISTRACTED_SECONDS) + 2)
    ]

    assert labels[-1] is BehaviorType.DISTRACTED


def test_head_down_uses_pitch_deviation_from_own_baseline(feeder):
    """Pitch is adaptive too, and stays one-sided. The front-row analogue of the
    side-column case: a student who naturally sits head-tipped past the global
    head-down bar (taking notes, short desk) is attentive at their own resting
    pitch, but dropping further below their own median is DISTRACTED."""
    analyzer, feed = feeder
    tid = 23
    resting_pitch = bmod._POSE_PITCH_HEAD_DOWN + 5.0

    labels = [
        feed(0.0, pitch=resting_pitch, tid=tid, frame_no=i).behavior
        for i in range(bmod._EAR_HISTORY_LEN)
    ]
    assert BehaviorType.DISTRACTED not in labels

    dropped = resting_pitch + bmod._PITCH_DEVIATION + 5.0
    labels = [
        feed(0.0, pitch=dropped, tid=tid, frame_no=bmod._EAR_HISTORY_LEN + i).behavior
        for i in range(int(bmod._DISTRACTED_SECONDS) + 2)
    ]
    assert labels[-1] is BehaviorType.DISTRACTED


# ── Baseline mechanics (mirrors the _ear_history pattern) ─────────────────────

def test_baseline_is_median_over_frame_count_window():
    """No baseline below _EAR_HISTORY_MIN frames, median thereafter, and the
    window is capped by FRAME COUNT at _EAR_HISTORY_LEN."""
    analyzer = BehaviorAnalyzer()
    tid = 24

    for i in range(bmod._EAR_HISTORY_MIN - 1):
        assert analyzer._pose_baseline(tid, 10.0, 4.0) == (None, None), f"frame {i}"

    baseline_yaw, baseline_pitch = analyzer._pose_baseline(tid, 10.0, 4.0)
    assert baseline_yaw == pytest.approx(10.0)
    assert baseline_pitch == pytest.approx(4.0)

    # A single extreme frame must not move the median the way a mean would.
    baseline_yaw, _ = analyzer._pose_baseline(tid, 900.0, 4.0)
    assert baseline_yaw == pytest.approx(10.0)

    for i in range(100):
        analyzer._pose_baseline(tid, float(i), float(i))
    assert len(analyzer._yaw_history[tid]) == bmod._EAR_HISTORY_LEN
    assert len(analyzer._pitch_history[tid]) == bmod._EAR_HISTORY_LEN


def test_warmup_falls_back_to_global_thresholds():
    """Before a baseline exists the fixed global thresholds still apply — the
    pose analogue of _EAR_FALLBACK."""
    analyzer = BehaviorAnalyzer()

    assert analyzer._pose_off_axis(bmod._YAW_THRESHOLD + 5.0, 0.0, None, None) is True
    assert analyzer._pose_off_axis(bmod._YAW_THRESHOLD - 5.0, 0.0, None, None) is False
    assert analyzer._pose_off_axis(
        0.0, bmod._POSE_PITCH_HEAD_DOWN + 5.0, None, None
    ) is True


def test_pitch_deviation_is_one_sided():
    """Looking DOWN relative to one's own median counts; looking up does not,
    preserving the original pitch > _POSE_PITCH_HEAD_DOWN semantics."""
    analyzer = BehaviorAnalyzer()
    beyond = bmod._PITCH_DEVIATION + 5.0

    assert analyzer._pose_off_axis(0.0, beyond, 0.0, 0.0) is True
    assert analyzer._pose_off_axis(0.0, -beyond, 0.0, 0.0) is False


def test_frames_without_pose_do_not_enter_the_baseline(feeder):
    """A frame with no transform matrix is an absent measurement, not a measured
    zero: feeding placeholder 0.0 in would drag the baseline toward center and
    re-create the side-column bias."""
    analyzer, feed = feeder
    tid = 25

    for i in range(bmod._EAR_HISTORY_LEN):
        feed(_SIDE_COLUMN_YAW, tid=tid, frame_no=i)
    for i in range(bmod._EAR_HISTORY_LEN):
        feed(0.0, tid=tid, frame_no=bmod._EAR_HISTORY_LEN + i, pose=False)

    assert len(analyzer._yaw_history[tid]) == bmod._EAR_HISTORY_LEN
    assert float(np.median(analyzer._yaw_history[tid])) == pytest.approx(
        _SIDE_COLUMN_YAW, abs=0.5
    )


def test_reset_clears_pose_histories(feeder):
    """reset(track_id) must drop the new yaw/pitch dicts alongside _ear_history,
    so a recycled track id never inherits another student's baseline."""
    analyzer, feed = feeder
    tid = 26

    for i in range(bmod._EAR_HISTORY_MIN + 1):
        feed(_SIDE_COLUMN_YAW, tid=tid, frame_no=i)
    assert tid in analyzer._yaw_history and tid in analyzer._pitch_history

    analyzer.reset(tid)

    assert tid not in analyzer._yaw_history
    assert tid not in analyzer._pitch_history
    assert tid not in analyzer._ear_history
