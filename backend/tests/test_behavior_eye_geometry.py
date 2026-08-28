"""Real-image regression tests for eye geometry and sustained-closure detection.

=============================================================================
WHY THIS FILE EXISTS
=============================================================================
Every other behaviour test stubs out the layer where the bug actually lived.
test_behavior_cnn_sleeping.py monkeypatches `_eyes_closed_cnn` to a constant;
test_behavior_pose_baseline.py feeds synthetic yaw/pitch. Nothing put a real
image through `_crop_face` -> MediaPipe -> `_ear`, so nothing could see that

    `_crop_face` resized a RECTANGULAR padded box to 256x256 without
    preserving aspect. At classroom distance that box is ~0.86 wide/high and
    the distortion is mild; when a face fills the frame the box clamps to the
    frame and the ratio reaches 1.78. On the squashed close-up crop MediaPipe
    returned NO landmarks for a CLOSED eye 83-92% of the time (0% for open
    eyes at identical framing, 0% with aspect preserved). behavior.py's
    no-landmark path is `_carry_or_unknown`, which replays the last confident
    behaviour -- ATTENTIVE -- so a student with their eyes shut read as
    attentive indefinitely and the sleeping dwell timer never even started.

These tests therefore use REAL frames at BOTH scales and never stub the eye
state. A test that mocks the eye decision cannot catch this class of bug.

=============================================================================
THE FIXTURES ARE DELIBERATELY NOT IN GIT
=============================================================================
They are frames of real students' faces. Committing them would put
identifiable faces in git history permanently, which is the one thing a
classroom-monitoring project should not do. They live in a gitignored
directory and are regenerated on demand:

    ./venv/bin/python scripts/06_make_eye_fixtures.py

Every test here SKIPS cleanly when that directory is absent, so a fresh clone
and CI both pass without them. That is a real coverage gap on a fresh clone,
accepted knowingly in exchange for not storing the faces: run the generator on
a machine that has the source video before trusting a green run.

The FaceLandmarker model (models/, also gitignored) is likewise required; see
scripts/00_fetch_models.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.pipeline import behavior as B

FIXTURE_DIR = Path(__file__).parent / "fixtures_local" / "behavior_eye"
MANIFEST = FIXTURE_DIR / "manifest.json"

pytestmark = [
    pytest.mark.skipif(
        not MANIFEST.exists(),
        reason=f"fixtures absent ({MANIFEST}); run scripts/06_make_eye_fixtures.py",
    ),
    pytest.mark.skipif(
        not Path(B._MODEL_PATH).exists(),
        reason=f"FaceLandmarker model absent ({B._MODEL_PATH}); "
               f"run scripts/00_fetch_models.py",
    ),
]

SCALES = ("classroom", "closeup")


def _manifest():
    return json.loads(MANIFEST.read_text())


def _load(entry):
    img = cv2.imread(str(FIXTURE_DIR / entry["file"]))
    assert img is not None, f"unreadable fixture {entry['file']}"
    return img, np.asarray(entry["bbox"], dtype=float)


def _group(scale, eyes):
    return [e for e in _manifest() if e["scale"] == scale and e["eyes"] == eyes]


@pytest.fixture(scope="module")
def analyzer():
    a = B.BehaviorAnalyzer()
    a.warmup()
    return a


def _ear_or_none(analyzer, img, bbox):
    """Measured EAR through the real crop + landmarker, or None if the
    landmarker returned nothing for this frame."""
    crop = B._crop_face(img, bbox)
    if crop is None:
        return None
    import mediapipe as mp
    det = analyzer._lm.detect(
        mp.Image(image_format=mp.ImageFormat.SRGB,
                 data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    )
    if not det.face_landmarks:
        return None
    lms = det.face_landmarks[0]
    return (B._ear(lms, B._LEFT_EYE) + B._ear(lms, B._RIGHT_EYE)) / 2.0


# ── 1. landmarks must survive closed eyes at every face size ─────────────────

@pytest.mark.parametrize("scale", SCALES)
def test_closed_eyes_still_produce_landmarks(analyzer, scale):
    """The core geometry regression.

    A closed eye must not make the landmarker fail. This failed at `closeup`
    before the aspect-preserving crop: 83-92% of closed-eye frames returned no
    landmarks, while open eyes at identical framing returned 100%.
    """
    group = _group(scale, "closed")
    if not group:
        pytest.skip(f"no {scale}/closed fixtures")
    ok = sum(_ear_or_none(analyzer, *_load(e)) is not None for e in group)
    assert ok == len(group), (
        f"{scale}: landmarker returned nothing for {len(group)-ok}/{len(group)} "
        f"CLOSED-eye frames. Open eyes at the same framing do not fail; this is "
        f"the crop-geometry defect, and downstream it reads as ATTENTIVE."
    )


@pytest.mark.parametrize("scale", SCALES)
def test_open_eyes_still_produce_landmarks(analyzer, scale):
    """Guards the other direction: the fix must not cost open-eye detection."""
    group = _group(scale, "open")
    if not group:
        pytest.skip(f"no {scale}/open fixtures")
    ok = sum(_ear_or_none(analyzer, *_load(e)) is not None for e in group)
    assert ok == len(group), f"{scale}: {len(group)-ok}/{len(group)} open-eye frames lost"


# ── 2. closed must separate from open, by the margin the gate needs ──────────

@pytest.mark.parametrize("scale", SCALES)
def test_closed_eyes_clear_the_sleep_ratio_against_open_baseline(analyzer, scale):
    """EAR is a ratio test against the track's own baseline, so what matters is
    separation, not absolute value. Median closed EAR must sit below
    _SLEEP_RATIO x (75th percentile of open EAR) -- the same comparison
    _analyze_one makes."""
    closed = [_ear_or_none(analyzer, *_load(e)) for e in _group(scale, "closed")]
    opened = [_ear_or_none(analyzer, *_load(e)) for e in _group(scale, "open")]
    closed = [v for v in closed if v is not None]
    opened = [v for v in opened if v is not None]
    if len(closed) < 3 or len(opened) < 3:
        pytest.skip(f"not enough measurable {scale} fixtures")
    med = float(np.median(closed))
    baseline = float(np.percentile(opened, 75))
    assert med < baseline * B._SLEEP_RATIO, (
        f"{scale}: closed median {med:.4f} is not below "
        f"{baseline:.4f} x {B._SLEEP_RATIO} = {baseline*B._SLEEP_RATIO:.4f}; "
        f"SLEEPING can never fire at this face size"
    )


# ── 3. a sustained closure must be reported, and STAY reported ───────────────

def _drive(analyzer, track_id, open_frames, closed_frames, dt_ms, closed_secs):
    """Feed open frames to establish a baseline, then closed frames for
    `closed_secs`, returning the behaviour emitted for each closed sample."""
    analyzer.reset_all()
    t = 0.0
    for i in range(12):                       # establish the open-eye baseline
        img, bbox = open_frames[i % len(open_frames)]
        analyzer._analyze_one(img, bbox, track_id=track_id, timestamp_ms=t)
        t += dt_ms
    out = []
    n = max(1, int(round(closed_secs * 1000.0 / dt_ms)))
    for i in range(n):
        img, bbox = closed_frames[i % len(closed_frames)]
        bf = analyzer._analyze_one(img, bbox, track_id=track_id, timestamp_ms=t)
        out.append(((t / 1000.0), bf.behavior))
        t += dt_ms
    return out


@pytest.mark.parametrize("scale", SCALES)
def test_sustained_closure_is_reported_and_does_not_lapse(analyzer, scale):
    """Defect B: the EAR baseline is a rolling window that used to include the
    closure itself, so after ~10 consecutive closed samples the baseline
    collapsed onto the closed value and the student silently reverted to
    ATTENTIVE ~9s into a genuine closure.

    A closure held for 20s must be reported SLEEPING and must STILL be
    SLEEPING at the end.
    """
    closed = [_load(e) for e in _group(scale, "closed")]
    opened = [_load(e) for e in _group(scale, "open")]
    if not closed or not opened:
        pytest.skip(f"no {scale} fixtures")
    trace = _drive(analyzer, 1, opened, closed, dt_ms=1000.0, closed_secs=20.0)
    behaviours = [b for _, b in trace]
    assert B.BehaviorType.SLEEPING in behaviours, (
        f"{scale}: 20s of closed eyes never produced SLEEPING; got "
        f"{sorted({b.value for b in behaviours})}"
    )
    tail = [b for _, b in trace[-5:]]
    assert all(b is B.BehaviorType.SLEEPING for b in tail), (
        f"{scale}: SLEEPING lapsed before the closure ended -- last 5 samples "
        f"were {[b.value for b in tail]}. The baseline absorbed the closure."
    )


# ── 4. the result must not depend on how fast frames arrive ──────────────────

@pytest.mark.parametrize("dt_ms", [1000.0, 100.0], ids=["1fps", "10fps"])
def test_sustained_closure_detected_at_any_sampling_rate(analyzer, dt_ms):
    """The baseline window must be defined in TIME, not frame count.

    A 15-frame count-based window spans 15s at 1fps but 1.5s at 10fps, so on a
    fast live feed the baseline collapsed before the dwell timer matured and
    SLEEPING became unreachable. Same closure, two rates, same verdict.
    """
    closed = [_load(e) for e in _group("classroom", "closed")]
    opened = [_load(e) for e in _group("classroom", "open")]
    if not closed or not opened:
        pytest.skip("no classroom fixtures")
    trace = _drive(analyzer, 2, opened, closed, dt_ms=dt_ms, closed_secs=20.0)
    behaviours = [b for _, b in trace]
    assert B.BehaviorType.SLEEPING in behaviours, (
        f"at {1000/dt_ms:.0f}fps sampling a 20s closure produced no SLEEPING: "
        f"{sorted({b.value for b in behaviours})}"
    )
