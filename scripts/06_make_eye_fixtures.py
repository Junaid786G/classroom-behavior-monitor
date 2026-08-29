#!/usr/bin/env python3
"""Extract real-image fixtures for backend/tests/test_behavior_eye_geometry.py.

WHY THIS SCRIPT EXISTS INSTEAD OF COMMITTED IMAGES
--------------------------------------------------
The fixtures are frames of real students' faces. Committing them would put
identifiable faces in git history permanently, which is the one thing this
project should not do. So the images live in a gitignored directory and this
script regenerates them from the source video on any machine that has it. The
tests skipif themselves away when the directory is absent, so a fresh clone
still passes -- it simply does not run these checks.

Sources (both gitignored, see .gitignore):
    data/videos/test_clip_8min.mp4              classroom distance, 28-77px faces
    data/backups/aneeq_enrollment_original.mp4  reframed to close-up

The classroom closed-eye frames come from the sustained closure at f6775-6975
that backend/tests/test_eye_gate_selection.py documents (EAR 0.023-0.056, "0 of
32 intervening frames open -- unambiguous sleep"). The sleeping seat is
re-derived here rather than hard-coded, so a different clip cannot silently
select the wrong student.

Usage:  ./venv/bin/python scripts/06_make_eye_fixtures.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.pipeline.behavior import (  # noqa: E402
    _LEFT_EYE, _MODEL_PATH, _RIGHT_EYE, _ear, _padded_face_box,
    _rotation_to_euler,
)

import mediapipe as mp  # noqa: E402
from mediapipe.tasks.python import vision as mp_vision  # noqa: E402
from mediapipe.tasks.python.core.base_options import BaseOptions  # noqa: E402
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (  # noqa: E402
    VisionTaskRunningMode,
)
from insightface.app import FaceAnalysis  # noqa: E402

CLASSROOM_SRC = Path("data/videos/test_clip_8min.mp4")
CLOSEUP_SRC = Path("data/backups/aneeq_enrollment_original.mp4")
OUT_DIR = Path("backend/tests/fixtures_local/behavior_eye")

# The documented sustained closure, and an open window for the same seat.
SLEEP_WINDOW = range(6775, 6976, 10)
OPEN_WINDOW = range(5000, 5201, 10)
# Close-up output size: native pixels, no upscale, face fills the frame -- the
# geometry of the webcam test that exposed the bug.
CLOSEUP_SIZE = (420, 236)
SEAT_CELL = 70          # px grid used to group detections into seats
PER_CLASS = 6           # fixtures per (scale, eye-state) combination
# Candidates are ranked by EAR, so without a pose filter "highest EAR" quietly
# selects "most turned head" rather than "most open eye": yaw foreshortens the
# eye's horizontal extent, which is EAR's denominator. Measured on the close-up
# set this produced open-eye fixtures at |yaw| 60-68 deg with EAR up to 2.04
# against a normal 0.2-0.4, and the inflated open baseline made
# test_closed_eyes_clear_the_sleep_ratio_against_open_baseline pass trivially.
# Both eye states are therefore selected from near-frontal frames only.
FRONTAL_MAX_YAW = 20.0


def _landmarker():
    return mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
    )


def _ear_of(lm, frame, bbox):
    """(EAR, |yaw|) through the CURRENT pipeline crop -- used only to sort frames
    into open/closed here, never as an assertion. |yaw| accompanies it so turned
    heads can be kept out of the ranking; see FRONTAL_MAX_YAW."""
    box = _padded_face_box(frame, np.asarray(bbox, dtype=float))
    if box is None:
        return None
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (256, 256))
    det = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
    if not det.face_landmarks:
        return None
    pts = det.face_landmarks[0]
    yaw = 0.0
    if det.facial_transformation_matrixes:
        yaw = abs(_rotation_to_euler(det.facial_transformation_matrixes[0])[1])
    return (_ear(pts, _LEFT_EYE) + _ear(pts, _RIGHT_EYE)) / 2.0, yaw


def _read(src: Path, idxs):
    cap = cv2.VideoCapture(str(src))
    out = {}
    for i in sorted(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            out[int(i)] = f
    cap.release()
    return out


def _biggest(app, frame):
    faces = app.get(frame)
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def _reframe(frame, bbox, margin=0.5, out_size=CLOSEUP_SIZE):
    """Crop tight around the face at NATIVE resolution so it fills the frame."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = (x2 - x1) * margin, (y2 - y1) * margin
    ar = out_size[0] / out_size[1]
    if hw / hh < ar:
        hw = hh * ar
    else:
        hh = hw / ar
    X1, Y1 = int(max(0, cx - hw)), int(max(0, cy - hh))
    X2, Y2 = int(min(w, cx + hw)), int(min(h, cy + hh))
    sub = frame[Y1:Y2, X1:X2]
    if sub.size == 0:
        return None
    return cv2.resize(sub, out_size, interpolation=cv2.INTER_AREA)


def classroom_fixtures(app, lm, manifest):
    if not CLASSROOM_SRC.exists():
        print(f"  SKIP classroom: {CLASSROOM_SRC} not present")
        return
    frames = _read(CLASSROOM_SRC, list(SLEEP_WINDOW) + list(OPEN_WINDOW))

    # Group detections into seats, then pick the seat with the lowest median EAR
    # across the sleep window -- that is the student who is actually asleep.
    seats = defaultdict(lambda: {"sleep": [], "open": []})
    for window, idxs in (("sleep", SLEEP_WINDOW), ("open", OPEN_WINDOW)):
        for i in idxs:
            frame = frames.get(int(i))
            if frame is None:
                continue
            for f in app.get(frame):
                bbox = [float(v) for v in f.bbox]
                key = (int((bbox[0] + bbox[2]) / 2) // SEAT_CELL,
                       int((bbox[1] + bbox[3]) / 2) // SEAT_CELL)
                measured = _ear_of(lm, frame, bbox)
                if measured is not None and measured[1] <= FRONTAL_MAX_YAW:
                    seats[key][window].append((int(i), bbox, measured[0]))

    ranked = [
        (float(np.median([r[2] for r in v["sleep"]])), k)
        for k, v in seats.items()
        if len(v["sleep"]) >= 10 and len(v["open"]) >= 10
    ]
    if not ranked:
        print("  SKIP classroom: no seat had enough samples in both windows")
        return
    ranked.sort()
    med, seat = ranked[0]
    print(f"  classroom sleeping seat {seat} (median EAR {med:.4f} in closure window)")

    for state, window in (("closed", "sleep"), ("open", "open")):
        rows = sorted(seats[seat][window], key=lambda r: r[2])
        picks = rows[:PER_CLASS] if state == "closed" else rows[-PER_CLASS:]
        for n, (idx, bbox, e) in enumerate(picks):
            name = f"classroom_{state}_{n:02d}.jpg"
            cv2.imwrite(str(OUT_DIR / name), frames[idx],
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            manifest.append(dict(file=name, scale="classroom", eyes=state,
                                 bbox=[round(v, 2) for v in bbox],
                                 source_frame=idx, ear_current_crop=round(e, 4)))


def closeup_fixtures(app, lm, manifest):
    if not CLOSEUP_SRC.exists():
        print(f"  SKIP close-up: {CLOSEUP_SRC} not present")
        return
    cap = cv2.VideoCapture(str(CLOSEUP_SRC))
    scored, fi = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi % 2 == 0:
            f = _biggest(app, frame)
            if f is not None:
                measured = _ear_of(lm, frame, [float(v) for v in f.bbox])
                if measured is not None and measured[1] <= FRONTAL_MAX_YAW:
                    scored.append((measured[0], fi, [float(v) for v in f.bbox]))
        fi += 1
    cap.release()
    if len(scored) < 2 * PER_CLASS:
        print("  SKIP close-up: too few usable frames")
        return
    scored.sort()
    groups = (("closed", scored[:PER_CLASS]), ("open", scored[-PER_CLASS:]))
    wanted = {i for _, g in groups for _, i, _ in g}
    frames = _read(CLOSEUP_SRC, wanted)

    for state, group in groups:
        for n, (e, idx, bbox) in enumerate(group):
            frame = frames.get(idx)
            if frame is None:
                continue
            img = _reframe(frame, bbox)
            if img is None:
                continue
            # Re-detect in the reframed image: the test needs a bbox valid for
            # the image it actually loads, not for the 1920x1080 original.
            f = _biggest(app, img)
            if f is None:
                print(f"    note: no detection after reframing frame {idx}, skipped")
                continue
            name = f"closeup_{state}_{n:02d}.jpg"
            cv2.imwrite(str(OUT_DIR / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            manifest.append(dict(file=name, scale="closeup", eyes=state,
                                 bbox=[round(float(v), 2) for v in f.bbox],
                                 source_frame=idx, ear_current_crop=round(e, 4)))


def main() -> int:
    if not Path(_MODEL_PATH).exists():
        print(f"FaceLandmarker model missing at {_MODEL_PATH}; "
              f"run scripts/00_fetch_models.py first")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    lm = _landmarker()

    manifest: list[dict] = []
    print("extracting fixtures:")
    classroom_fixtures(app, lm, manifest)
    closeup_fixtures(app, lm, manifest)

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    by = defaultdict(int)
    for m in manifest:
        by[(m["scale"], m["eyes"])] += 1
    print(f"\nwrote {len(manifest)} fixtures to {OUT_DIR}")
    for k, v in sorted(by.items()):
        print(f"  {k[0]:>10} {k[1]:>6}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
