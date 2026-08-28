"""Regression tests for CNN-mode SLEEPING classification in BehaviorAnalyzer.

These lock in the fix where `eye_reliable` (an EAR-landmark-symmetry proxy) must
NOT gate the CNN eye-state path. The CNN reads pixel intensities and does not
depend on EAR landmark geometry, so an untrustworthy-EAR frame must never
suppress a CNN-driven SLEEPING call. In "ear" mode the gate still applies.
"""
import types

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from backend.pipeline import behavior as bmod
from backend.pipeline.behavior import BehaviorAnalyzer, BehaviorType


# ── Fake MediaPipe landmark construction ──────────────────────────────────────

class _Pt:
    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


def _set_eye(lms, indices, cx, cy, half_height):
    """Place 6 eye landmarks so that _ear() == 20 * half_height.

    Layout (width fixed at 0.10, vertical gap 2*half_height):
        p0/p3 = horizontal corners, p1/p5 and p2/p4 = upper/lower lids.
    """
    i0, i1, i2, i3, i4, i5 = indices
    lms[i0] = _Pt(cx - 0.05, cy)              # p0  left corner
    lms[i3] = _Pt(cx + 0.05, cy)              # p3  right corner  -> d(p0,p3)=0.10
    lms[i1] = _Pt(cx - 0.025, cy + half_height)
    lms[i5] = _Pt(cx - 0.025, cy - half_height)   # d(p1,p5)=2*half_height
    lms[i2] = _Pt(cx + 0.025, cy + half_height)
    lms[i4] = _Pt(cx + 0.025, cy - half_height)   # d(p2,p4)=2*half_height


def _asymmetric_landmarks():
    """Landmarks whose left/right EAR differ well beyond _EYE_ASYM_MAX (0.15),
    i.e. eye_reliable == False for this frame."""
    n = max(max(bmod._LEFT_EYE), max(bmod._RIGHT_EYE)) + 1
    lms = [_Pt(0.5, 0.5) for _ in range(n)]
    _set_eye(lms, bmod._LEFT_EYE,  cx=0.3, cy=0.4, half_height=0.020)  # EAR ~ 0.40
    _set_eye(lms, bmod._RIGHT_EYE, cx=0.6, cy=0.4, half_height=0.005)  # EAR ~ 0.10
    return lms


def _pose_matrix(pitch=0.0, yaw=0.0, roll=0.0):
    """A 4×4 transform matrix that _rotation_to_euler() decodes back to the
    given (pitch, yaw, roll) in degrees."""
    m = np.eye(4)
    m[:3, :3] = Rotation.from_euler("xyz", [pitch, yaw, roll], degrees=True).as_matrix()
    return m


def _make_analyzer(monkeypatch, mode, eyes_closed, pose=None):
    """Build a BehaviorAnalyzer wired to fake detection and a stubbed CNN result.

    pose=None => no transform matrix => neutral head pose (pitch=yaw=roll=0);
    otherwise pass a 4×4 matrix (see _pose_matrix)."""
    monkeypatch.setattr(bmod, "_EYE_STATE_METHOD", mode)

    analyzer = BehaviorAnalyzer()
    analyzer._ready = True

    det = types.SimpleNamespace(
        face_landmarks=[_asymmetric_landmarks()],
        facial_transformation_matrixes=[] if pose is None else [pose],
    )
    analyzer._lm = types.SimpleNamespace(detect=lambda _img: det)
    # CNN gate is stubbed so no ONNX model is loaded during the test.
    analyzer._eyes_closed_cnn = lambda _frame, _bbox, _lms: eyes_closed
    return analyzer


@pytest.fixture
def frame_and_bbox():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = np.array([100, 100, 200, 200], dtype=float)
    return frame, bbox


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_cnn_mode_sleeps_despite_unreliable_eye_landmarks(monkeypatch, frame_and_bbox):
    """CNN says eyes closed + sustained -> SLEEPING, even though the EAR
    landmarks are asymmetric (eye_reliable would be False)."""
    frame, bbox = frame_and_bbox
    analyzer = _make_analyzer(monkeypatch, mode="cnn", eyes_closed=True)
    tid = 7
    # Sustain already exceeded (video-time) so the dwell timer immediately qualifies.
    analyzer._sleeping_since[tid] = 0.0

    bf = analyzer._analyze_one(frame, bbox, track_id=tid, timestamp_ms=10_000.0)

    assert bf.behavior is BehaviorType.SLEEPING
    assert bf.confidence == pytest.approx(0.97)


def test_cnn_mode_accrues_before_sleeping(monkeypatch, frame_and_bbox):
    """Without sustained closure the CNN result is trusted but SLEEPING is held
    off (ATTENTIVE while the dwell timer accrues) — the gate is duration, not
    eye_reliable."""
    frame, bbox = frame_and_bbox
    analyzer = _make_analyzer(monkeypatch, mode="cnn", eyes_closed=True)
    tid = 8

    bf = analyzer._analyze_one(frame, bbox, track_id=tid, timestamp_ms=0.0)

    assert bf.behavior is BehaviorType.ATTENTIVE
    assert analyzer._sleeping_since[tid] is not None   # timer started, not reset


def test_ear_mode_still_gates_on_unreliable_landmarks(monkeypatch, frame_and_bbox):
    """In EAR mode the eye_reliable gate is preserved: asymmetric landmarks with
    a neutral pose bypass the EAR sleeping check and yield UNKNOWN, clearing the
    sleeping timer."""
    frame, bbox = frame_and_bbox
    analyzer = _make_analyzer(monkeypatch, mode="ear", eyes_closed=True)
    tid = 9
    analyzer._sleeping_since[tid] = 0.0   # would sleep if not gated (video-time)

    bf = analyzer._analyze_one(frame, bbox, track_id=tid, timestamp_ms=10_000.0)

    assert bf.behavior is BehaviorType.UNKNOWN
    assert analyzer._sleeping_since[tid] is None


# ── HEAD_DOWN merged into DISTRACTED (3-state model) ──────────────────────────

def test_head_down_pitch_is_distracted_not_head_down(monkeypatch, frame_and_bbox):
    """A head-down pitch (eyes open) is now classified DISTRACTED, never the
    removed HEAD_DOWN. Dwell is pre-elapsed so it resolves immediately."""
    frame, bbox = frame_and_bbox
    pose = _pose_matrix(pitch=bmod._POSE_PITCH_HEAD_DOWN + 10.0)
    analyzer = _make_analyzer(monkeypatch, mode="cnn", eyes_closed=False, pose=pose)
    tid = 10
    analyzer._distracted_since[tid] = 0.0

    bf = analyzer._analyze_one(frame, bbox, track_id=tid, timestamp_ms=(bmod._DISTRACTED_SECONDS + 1.0) * 1000.0)

    assert bf.behavior is BehaviorType.DISTRACTED
    assert bf.behavior is not BehaviorType.HEAD_DOWN


def test_head_down_accrues_distracted_via_dwell(monkeypatch, frame_and_bbox):
    """Head-down distraction now flows through the dwell timer: a fresh track is
    ATTENTIVE while the timer accrues (previously HEAD_DOWN was emitted at once)."""
    frame, bbox = frame_and_bbox
    pose = _pose_matrix(pitch=bmod._POSE_PITCH_HEAD_DOWN + 10.0)
    analyzer = _make_analyzer(monkeypatch, mode="cnn", eyes_closed=False, pose=pose)
    tid = 11

    bf = analyzer._analyze_one(frame, bbox, track_id=tid, timestamp_ms=0.0)

    assert bf.behavior is BehaviorType.ATTENTIVE
    assert analyzer._distracted_since[tid] is not None   # timer started


def test_sleeping_wins_over_head_down_pitch(monkeypatch, frame_and_bbox):
    """SLEEPING is checked first and is independent of head pose: eyes closed +
    sustained yields SLEEPING even with a strong head-down pitch."""
    frame, bbox = frame_and_bbox
    pose = _pose_matrix(pitch=bmod._POSE_PITCH_HEAD_DOWN + 20.0)
    analyzer = _make_analyzer(monkeypatch, mode="cnn", eyes_closed=True, pose=pose)
    tid = 12
    analyzer._sleeping_since[tid] = 0.0   # sustain already exceeded (video-time)

    bf = analyzer._analyze_one(frame, bbox, track_id=tid, timestamp_ms=10_000.0)

    assert bf.behavior is BehaviorType.SLEEPING


# ── Eye-crop coordinate mapping (single-resize fix) ───────────────────────────

def test_eyes_closed_cnn_maps_landmarks_to_original_frame(monkeypatch, frame_and_bbox):
    """_eyes_closed_cnn crops from the ORIGINAL frame using landmarks mapped from
    normalized 256-crop space back to original-frame pixels — no 256×256 upscale."""
    frame, bbox = frame_and_bbox

    n = max(max(bmod._LEFT_EYE), max(bmod._RIGHT_EYE)) + 1
    lms = [_Pt(0.10 + 0.01 * i, 0.20 + 0.01 * i) for i in range(n)]

    captured = {}

    def fake_both(frame_bgr, left, right):
        captured["frame"] = frame_bgr
        captured["left"] = left
        captured["right"] = right
        return True

    analyzer = BehaviorAnalyzer()
    analyzer._eye_clf = types.SimpleNamespace(both_eyes_closed=fake_both)

    result = analyzer._eyes_closed_cnn(frame, bbox, lms)

    # Expected mapping, recomputed from the same SQUARE-box helper the code uses.
    # Was _padded_face_box with separate rw/rh scales; _crop_face now produces an
    # aspect-preserving square crop, so the inverse mapping is one scale for both
    # axes. See _square_face_box for why the crop changed.
    sx1, sy1, side = bmod._square_face_box(frame, bbox)
    exp_left = [(sx1 + lms[i].x * side, sy1 + lms[i].y * side) for i in bmod._LEFT_EYE]
    exp_right = [(sx1 + lms[i].x * side, sy1 + lms[i].y * side) for i in bmod._RIGHT_EYE]

    assert result is True
    assert captured["frame"] is frame                     # original frame, not a 256×256 crop
    assert captured["frame"].shape == (480, 640, 3)
    assert captured["left"] == pytest.approx(exp_left)
    assert captured["right"] == pytest.approx(exp_right)


def test_eyes_closed_cnn_falls_back_when_bbox_too_small(monkeypatch):
    """A degenerate bbox yields no padded box; the CNN gate falls back to True so
    a mapping failure never suppresses an otherwise-valid SLEEPING call."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = np.array([100, 100, 103, 103], dtype=float)   # < 5px -> _square_face_box None

    called = {"n": 0}

    def fake_both(*_a, **_k):
        called["n"] += 1
        return False

    analyzer = BehaviorAnalyzer()
    analyzer._eye_clf = types.SimpleNamespace(both_eyes_closed=fake_both)

    assert analyzer._eyes_closed_cnn(frame, bbox, []) is True
    assert called["n"] == 0                                # classifier never invoked
