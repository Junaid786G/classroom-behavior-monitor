"""Regression tests for extreme head yaw, and for the SLEEPING-over-pose priority.

=============================================================================
WHAT WENT WRONG
=============================================================================
A student who turned their head to an extreme yaw was reported UNKNOWN even
though RetinaFace had boxed their face -- so the UI drew a box around a clearly
visible student and labelled them "unknown". Replaying real footage through the
real analyzer (a student's enrollment clip, reframed to close-up) showed TWO
independent routes to that label, both present in a single 40-sample sequence:

  frames 218-258  MediaPipe FaceLandmarker returned NO landmarks at all.
                  _carry_or_unknown replayed the last behaviour for
                  _CARRY_FORWARD_MAX frames and then reported UNKNOWN.

  frames 166-198  MediaPipe DID return landmarks (yaw ~-38 deg), but the
                  eye-asymmetry gate tripped -- a turned head foreshortens the
                  far eye and inflates its EAR -- and _pose_off_axis said
                  "on-axis" because the turn had been held long enough to become
                  this track's OWN median. That branch's fallback is UNKNOWN.

Measured failure rates that size the fix (17,044 classroom detections from
test_clip_8min.mp4, 1,972 close-up detections from 17 enrollment clips):

  * among high-confidence detections the landmark-failure rate is flat at ~2%
    out to ~30 deg of yaw, then 9% at ~36-48 deg and 44% beyond;
  * the eye-asymmetry gate trips on 0.0% of frames below 10 deg of yaw and on
    80-91% above 40 deg, i.e. it is in practice a yaw detector;
  * at close-up scale 24 of 24 landmark failures were extreme turns, while at
    classroom scale only 1.7% were -- the rest are small marginal detections
    that fail for want of pixels. A blanket "no landmarks => distracted" rule
    would have mislabelled that entire classroom remainder, which is why the
    fix is gated on RetinaFace's own keypoint geometry (_kps_yaw_index).

=============================================================================
FIXTURES
=============================================================================
The eye-state tests reuse the real-image fixtures of
test_behavior_eye_geometry.py -- same gitignored directory, same generator
(scripts/06_make_eye_fixtures.py), same clean skip when absent. Eye state is
never stubbed in those tests: the closed/open decision is made from the real
image, and only the head POSE is imposed, because pose is the one thing a still
frame cannot vary.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from backend.pipeline import behavior as B
from backend.pipeline.behavior import BehaviorAnalyzer, BehaviorType

FIXTURE_DIR = Path(__file__).parent / "fixtures_local" / "behavior_eye"
MANIFEST = FIXTURE_DIR / "manifest.json"
SCALES = ("classroom", "closeup")

needs_fixtures = pytest.mark.skipif(
    not MANIFEST.exists(),
    reason=f"fixtures absent ({MANIFEST}); run scripts/06_make_eye_fixtures.py",
)
needs_model = pytest.mark.skipif(
    not Path(B._MODEL_PATH).exists(),
    reason=f"FaceLandmarker model absent ({B._MODEL_PATH}); run scripts/00_fetch_models.py",
)


# ── keypoint helpers ─────────────────────────────────────────────────────────

def _kps(yaw_index: float, iod: float = 40.0, face_v: float = 50.0) -> np.ndarray:
    """RetinaFace-style 5 keypoints whose _kps_yaw_index() is *yaw_index*.

    Eyes on a horizontal line, mouth corners face_v below them, nose displaced
    along the eye axis by yaw_index * face_v -- the exact quantity the helper
    measures.
    """
    left_eye = (100.0, 100.0)
    right_eye = (100.0 + iod, 100.0)
    eye_mid_x = 100.0 + iod / 2.0
    nose = (eye_mid_x + yaw_index * face_v, 100.0 + face_v * 0.5)
    mouth_l = (eye_mid_x - 15.0, 100.0 + face_v)
    mouth_r = (eye_mid_x + 15.0, 100.0 + face_v)
    return np.array([left_eye, right_eye, nose, mouth_l, mouth_r], dtype=np.float32)


FRONTAL_KPS = _kps(0.0)
EXTREME_KPS = _kps(B._KPS_YAW_EXTREME + 0.15)


def test_kps_yaw_index_measures_the_constructed_offset():
    for want in (0.0, 0.1, 0.25, 0.5, 1.0):
        got = B._kps_yaw_index(_kps(want))
        assert got == pytest.approx(want, abs=1e-6), f"wanted {want}, got {got}"


def test_kps_yaw_index_is_scale_free():
    """The same pose at two face sizes must give the same index -- this is what
    lets one threshold hold at classroom distance and at close-up."""
    small = B._kps_yaw_index(_kps(0.4, iod=12.0, face_v=15.0))
    large = B._kps_yaw_index(_kps(0.4, iod=180.0, face_v=225.0))
    assert small == pytest.approx(large, abs=1e-6)


def test_kps_yaw_index_is_sign_agnostic():
    assert B._kps_yaw_index(_kps(0.4)) == pytest.approx(B._kps_yaw_index(_kps(-0.4)))


@pytest.mark.parametrize("bad", [
    None,
    np.zeros((5, 2), dtype=np.float32),          # _match_tracks_to_dets placeholder
    np.full((5, 2), np.nan, dtype=np.float32),
    np.zeros((3, 2), dtype=np.float32),          # wrong shape
])
def test_kps_yaw_index_rejects_degenerate_input(bad):
    """A track with no matching raw detection carries an all-zero placeholder.
    Reading a pose out of it would invent an extreme yaw from nothing."""
    assert B._kps_yaw_index(bad) is None


# ── the absolute yaw bar ─────────────────────────────────────────────────────

def test_absolute_bar_catches_a_turn_the_baseline_has_absorbed():
    """_pose_baseline's median window is fed by every measured frame, so a turn
    held long enough becomes the track's own median and the deviation test goes
    quiet. The absolute bar is what stops that."""
    an = BehaviorAnalyzer()
    held = B._YAW_ABSOLUTE_EXTREME + 5.0
    # baseline sits exactly on the turn: deviation is zero
    assert an._pose_off_axis(held, 0.0, held, 0.0) is True


def test_absolute_pitch_bar_catches_a_head_down_the_baseline_has_absorbed():
    """The pitch axis had the identical blind spot, and it is the one Issue 2
    turns on: a student looking down at a phone has that pitch absorbed into
    their own median within ~8 samples and silently reverts to ATTENTIVE."""
    an = BehaviorAnalyzer()
    held = B._PITCH_ABSOLUTE_EXTREME + 5.0
    assert an._pose_off_axis(0.0, held, 0.0, held) is True


def test_absolute_pitch_bar_stays_one_sided():
    """Looking UP off one's own baseline still does not count, preserving the
    head-down semantics of the original pitch test."""
    an = BehaviorAnalyzer()
    up = -(B._PITCH_ABSOLUTE_EXTREME + 5.0)
    assert an._pose_off_axis(0.0, up, 0.0, up) is False


def test_absolute_pitch_bar_leaves_the_front_row_case_intact():
    """The camera is mounted top-centre and looks DOWN, so every measured pitch
    is biased positive: the most head-tipped of 24 real seats rests at 23.0 deg
    and frame-level pitch p99.9 is 34.8. The bar must clear that whole envelope
    or a naturally head-tipped student is permanently distracted."""
    an = BehaviorAnalyzer()
    assert B._PITCH_ABSOLUTE_EXTREME > 34.8
    for resting in (14.5, 23.0, 30.0, 34.0):
        assert an._pose_off_axis(0.0, resting, 0.0, resting) is False, resting


def test_absolute_bar_leaves_the_side_column_fix_intact():
    """A real side-column seat rests at 30.6 deg on test_clip_8min.mp4 and
    frame-level |yaw| p99 is 34.1. The bar must sit above that whole envelope,
    or it re-creates the bias the adaptive baseline was added to remove."""
    an = BehaviorAnalyzer()
    assert B._YAW_ABSOLUTE_EXTREME > 34.1
    for resting in (25.0, 30.6, 34.0):
        assert an._pose_off_axis(resting, 0.0, resting, 0.0) is False, resting
        assert an._pose_off_axis(-resting, 0.0, -resting, 0.0) is False, resting


# ── the no-landmark path ─────────────────────────────────────────────────────

@pytest.fixture
def blind_analyzer(monkeypatch):
    """An analyzer whose FaceLandmarker always returns nothing, i.e. every frame
    takes the no-landmark path."""
    an = BehaviorAnalyzer()
    an._ready = True
    an._lm = types.SimpleNamespace(
        detect=lambda _img: types.SimpleNamespace(
            face_landmarks=[], facial_transformation_matrixes=[]))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = np.array([100, 100, 200, 200], dtype=float)
    return an, frame, bbox


def _run(an, frame, bbox, kps, n, tid=1, dt=1000.0, t0=0.0):
    return [an._analyze_one(frame, bbox, track_id=tid, timestamp_ms=t0 + i * dt,
                            kps=kps).behavior for i in range(n)]


def test_no_landmarks_with_extreme_kps_yaw_becomes_distracted(blind_analyzer):
    """THE BUG. RetinaFace boxed the face, so the student is visibly present;
    reporting UNKNOWN was never the honest answer."""
    an, frame, bbox = blind_analyzer
    labels = _run(an, frame, bbox, EXTREME_KPS, int(B._DISTRACTED_SECONDS) + 3)
    assert BehaviorType.UNKNOWN not in labels
    assert labels[-1] is BehaviorType.DISTRACTED


def test_no_landmarks_without_extreme_yaw_still_falls_back_to_unknown(blind_analyzer):
    """The other 98.3% of classroom landmark failures: small, marginal
    detections that fail for want of pixels. Calling those DISTRACTED would be a
    far worse bug than the one being fixed, so the old path must be preserved."""
    an, frame, bbox = blind_analyzer
    labels = _run(an, frame, bbox, FRONTAL_KPS, B._CARRY_FORWARD_MAX + 3)
    assert labels[-1] is BehaviorType.UNKNOWN
    assert BehaviorType.DISTRACTED not in labels


def test_no_landmarks_with_no_kps_at_all_still_falls_back_to_unknown(blind_analyzer):
    an, frame, bbox = blind_analyzer
    labels = _run(an, frame, bbox, None, B._CARRY_FORWARD_MAX + 3)
    assert labels[-1] is BehaviorType.UNKNOWN


def test_extreme_yaw_does_not_demote_a_sleeping_track(blind_analyzer):
    """Eye state is unmeasurable without landmarks. A student confirmed asleep
    moments ago keeps that label through the carry window rather than being
    relabelled by the turn -- a student can be asleep AND facing away."""
    an, frame, bbox = blind_analyzer
    an._record(7, BehaviorType.SLEEPING)
    labels = _run(an, frame, bbox, EXTREME_KPS, B._CARRY_FORWARD_MAX, tid=7)
    assert all(b is BehaviorType.SLEEPING for b in labels), labels


# ── SLEEPING outranks pose, on real images ───────────────────────────────────

def _manifest():
    return json.loads(MANIFEST.read_text())


def _load(entry):
    img = cv2.imread(str(FIXTURE_DIR / entry["file"]))
    assert img is not None, f"unreadable fixture {entry['file']}"
    return img, np.asarray(entry["bbox"], dtype=float)


def _group(scale, eyes):
    return [_load(e) for e in _manifest()
            if e["scale"] == scale and e["eyes"] == eyes]


@pytest.fixture(scope="module")
def real_analyzer():
    an = BehaviorAnalyzer()
    an.warmup()
    return an


def _drive_at_pose(an, frames, *, pitch, yaw, tid, secs, dt=1000.0, warm=None):
    """Feed real images through the real landmarker while forcing the head pose.

    Eye state stays REAL -- it is computed from the image. Only _rotation_to_euler
    is overridden, because a still frame cannot be re-posed.

    The warm-up runs at a NEUTRAL pose and the measured stretch at the requested
    one, because both baselines are adaptive: holding a head-down pitch from the
    very first frame makes head-down that student's own resting pose, which
    _pose_off_axis is deliberately built to read as attentive (this is the
    front-row case pinned by test_behavior_pose_baseline.py). Warming up neutral
    is what makes the later pitch a genuine departure from their baseline.
    """
    an.reset_all()
    labels = []
    t = 0.0
    orig = B._rotation_to_euler
    pose = [0.0, 0.0]
    B._rotation_to_euler = lambda _m: (pose[0], pose[1], 0.0)
    try:
        stages = ((warm or frames, 12, (0.0, 0.0)),
                  (frames, max(1, int(secs * 1000 / dt)), (pitch, yaw)))
        for imgs, n, (p, y) in stages:
            pose[0], pose[1] = p, y
            for i in range(n):
                img, bbox = imgs[i % len(imgs)]
                bf = an._analyze_one(img, bbox, track_id=tid, timestamp_ms=t, kps=None)
                labels.append(bf.behavior)
                t += dt
        return labels[12:]
    finally:
        B._rotation_to_euler = orig


@needs_fixtures
@needs_model
@pytest.mark.parametrize("scale", SCALES)
def test_head_down_with_eyes_open_is_distracted(real_analyzer, scale):
    """Head down + eyes OPEN: the student is looking at a phone, notes or the
    desk. That is DISTRACTED."""
    opened = _group(scale, "open")
    if len(opened) < 3:
        pytest.skip(f"no {scale}/open fixtures")
    pitch = B._POSE_PITCH_HEAD_DOWN + B._PITCH_DEVIATION + 20.0
    labels = _drive_at_pose(real_analyzer, opened, pitch=pitch, yaw=0.0,
                            tid=101, secs=int(B._DISTRACTED_SECONDS) + 4)
    assert labels[-1] is BehaviorType.DISTRACTED, [b.value for b in labels]
    assert BehaviorType.SLEEPING not in labels


@needs_fixtures
@needs_model
@pytest.mark.parametrize("scale", SCALES)
def test_head_down_with_eyes_open_stays_distracted_while_it_lasts(real_analyzer, scale):
    """A phone is looked at for minutes, not seconds. Held long enough, the pitch
    used to be absorbed into the track's own median and the student reverted to
    ATTENTIVE mid-episode -- the same absorption seen on the yaw axis."""
    opened = _group(scale, "open")
    if len(opened) < 3:
        pytest.skip(f"no {scale}/open fixtures")
    pitch = B._POSE_PITCH_HEAD_DOWN + B._PITCH_DEVIATION + 20.0
    labels = _drive_at_pose(real_analyzer, opened, pitch=pitch, yaw=0.0,
                            tid=104, secs=40)
    tail = labels[-10:]
    assert all(b is BehaviorType.DISTRACTED for b in tail), (
        f"{scale}: head-down lapsed before the episode ended -- last 10 were "
        f"{[b.value for b in tail]}. The pitch baseline absorbed the posture.")


@needs_fixtures
@needs_model
@pytest.mark.parametrize("scale", SCALES)
def test_head_down_with_eyes_closed_is_sleeping_not_distracted(real_analyzer, scale):
    """Head down + eyes CLOSED is the commonest real sleeping posture. SLEEPING
    must win over the head-down pitch, at the SAME pitch that produces
    DISTRACTED with the eyes open."""
    closed = _group(scale, "closed")
    opened = _group(scale, "open")
    if len(closed) < 3 or len(opened) < 3:
        pytest.skip(f"no {scale} fixtures")
    pitch = B._POSE_PITCH_HEAD_DOWN + B._PITCH_DEVIATION + 20.0
    labels = _drive_at_pose(real_analyzer, closed, pitch=pitch, yaw=0.0,
                            tid=102, secs=20, warm=opened)
    assert BehaviorType.SLEEPING in labels, [b.value for b in labels]
    assert labels[-1] is BehaviorType.SLEEPING
    assert BehaviorType.DISTRACTED not in labels


@needs_fixtures
@needs_model
@pytest.mark.parametrize("scale", SCALES)
def test_extreme_yaw_and_pitch_with_eyes_closed_still_sleeps(real_analyzer, scale):
    """The interaction case: a turn extreme enough to clear the absolute yaw bar,
    combined with a head-down pitch, must STILL resolve to SLEEPING when the eyes
    are really closed. Neither DISTRACTED nor UNKNOWN."""
    closed = _group(scale, "closed")
    opened = _group(scale, "open")
    if len(closed) < 3 or len(opened) < 3:
        pytest.skip(f"no {scale} fixtures")
    labels = _drive_at_pose(
        real_analyzer, closed,
        pitch=B._POSE_PITCH_HEAD_DOWN + B._PITCH_DEVIATION + 20.0,
        yaw=B._YAW_ABSOLUTE_EXTREME + 10.0,
        tid=103, secs=20, warm=opened)
    assert BehaviorType.SLEEPING in labels, [b.value for b in labels]
    assert labels[-1] is BehaviorType.SLEEPING
    assert BehaviorType.UNKNOWN not in labels
