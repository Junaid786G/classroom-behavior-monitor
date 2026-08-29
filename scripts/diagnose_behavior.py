#!/usr/bin/env python3
"""
scripts/diagnose_behavior.py

Process NUM_FRAMES evenly-spaced frames from data/videos/test_clip_8min.mp4.
For each detected face:
  - Reproduce _crop_face exactly (60% pad, 256x256) and re-run MediaPipe
  - Run BehaviorAnalyzer._analyze_one to get EAR / yaw / pitch / label
  - Compare the EAR-based eyes-closed decision against the CNN classifier
    (_eyes_closed_cnn) side by side
  - Draw the 6 EAR eye-landmark points (green = left eye, orange = right eye)
  - Draw EAR, yaw, pitch, label, both eyes-closed decisions as text overlay
  - Save the 256x256 face debug image to data/behavior_debug/
  - Save the actual 32x32 CNN eye crops (upscaled to 128x128) to
    data/behavior_debug/eye_crops/
Print a summary table at the end.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from backend.config import get_settings
from backend.gallery.manager import GalleryManager
from backend.pipeline.behavior import (
    BehaviorAnalyzer, BehaviorFrame,
    _LEFT_EYE, _RIGHT_EYE, _crop_face, _padded_face_box,
    _EAR_FALLBACK, _EAR_HISTORY_MIN, _EAR_STD_UNSTABLE,
    _SLEEP_RATIO, _SLEEP_RATIO_UNSTABLE,
)
from backend.pipeline.detector import FaceDetector
from backend.pipeline.embedder import ArcFaceEmbedder
from backend.pipeline.recognizer import FaceRecognizer
from backend.pipeline.tracker import ByteTracker

settings   = get_settings()
VIDEO_PATH   = ROOT / "data" / "videos" / "test_clip_8min.mp4"
OUTPUT_DIR   = ROOT / "data" / "behavior_debug"
EYE_CROP_DIR = OUTPUT_DIR / "eye_crops"
NUM_FRAMES   = 20


# ── DB helper (best-effort; falls back to empty map) ─────────────────────────

async def _fetch_student_map() -> Dict[int, str]:
    try:
        from backend.database import AsyncSessionLocal
        from sqlalchemy import text
        async with AsyncSessionLocal() as db:
            rows = (await db.execute(text("SELECT id, full_name FROM students"))).fetchall()
            return {row.id: row.full_name for row in rows}
    except Exception as exc:
        print(f"  [warn] DB unreachable ({exc}); names will fall back to student_<id>")
        return {}


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_text_block(img_bgr: np.ndarray, lines: List[str], start_y: int = 14) -> None:
    for i, line in enumerate(lines):
        y = start_y + i * 16
        # thin black shadow for legibility on any background
        for dx, dy in [(-1, -1), (1, 1), (-1, 1), (1, -1)]:
            cv2.putText(img_bgr, line, (4 + dx, y + dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img_bgr, line, (4, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_eye_landmarks(img_bgr: np.ndarray, landmarks, size: int = 256) -> None:
    """
    Draw the 6-point EAR ring for each eye.
    Green  = left eye  (_LEFT_EYE  indices 362,385,387,263,373,380)
    Orange = right eye (_RIGHT_EYE indices  33,160,158,133,153,144)
    """
    for indices, color in [(_LEFT_EYE,  (0, 220,   0)),
                            (_RIGHT_EYE, (0, 128, 255))]:
        pts = []
        for idx in indices:
            lm = landmarks[idx]
            x, y = int(lm.x * size), int(lm.y * size)
            x = max(0, min(size - 1, x))
            y = max(0, min(size - 1, y))
            pts.append((x, y))
            cv2.circle(img_bgr, (x, y), 2, color, -1, cv2.LINE_AA)
        # connect the ring: 0-1-2-3-4-5-0
        for i in range(len(pts)):
            cv2.line(img_bgr, pts[i], pts[(i + 1) % len(pts)], color, 1, cv2.LINE_AA)


def _safe_name(name: Optional[str], student_id: Optional[int]) -> str:
    if name:
        return name.replace(" ", "_")
    if student_id is not None:
        return f"student_{student_id}"
    return "unknown"


def _ear_eyes_closed(analyzer, track_id: int, ear: float) -> bool:
    """Reproduce BehaviorAnalyzer's EAR-only eyes-closed test for this frame.

    Uses the same adaptive 75th-percentile baseline and unstable-track ratio the
    analyzer applies in "ear" mode. _analyze_one has already appended this frame's
    EAR to the per-track history, so we read that same history back.
    """
    history = analyzer._ear_history.get(track_id, [])
    if len(history) < _EAR_HISTORY_MIN:
        baseline = _EAR_FALLBACK
    else:
        baseline = float(np.percentile(history, 75))
    sleep_ratio = _SLEEP_RATIO
    if len(history) >= _EAR_HISTORY_MIN and float(np.std(history)) > _EAR_STD_UNSTABLE:
        sleep_ratio = _SLEEP_RATIO_UNSTABLE
    return ear < baseline * sleep_ratio


def _cnn_eye_crops(analyzer, frame, bbox, lms):
    """Reproduce the exact 32x32 BGR eye crops the CNN receives.

    Mirrors _eyes_closed_cnn's landmark->original-frame mapping and the
    classifier's own _crop_eye, so the tiles we save are byte-for-byte what the
    model sees. Returns {"L": (crop32|None, (open,closed)|None), "R": (...)} or
    None if there is no box/landmarks.
    """
    box = _padded_face_box(frame, bbox)
    if box is None or lms is None:
        return None
    x1c, y1c, x2c, y2c = box
    rw, rh = x2c - x1c, y2c - y1c
    clf = analyzer._eye_clf
    clf._ensure_loaded()
    out = {}
    for tag, indices in (("L", _LEFT_EYE), ("R", _RIGHT_EYE)):
        pts = [(x1c + lms[i].x * rw, y1c + lms[i].y * rh) for i in indices]
        crop = clf._crop_eye(frame, pts)               # 32x32 BGR or None
        probs = clf._infer(crop) if crop is not None else None
        out[tag] = (crop, probs)
    return out


def _render_eye_debug(eye_data, view: int = 128):
    """Upscale both eye crops for viewing and annotate the per-eye CNN result.

    Returns (side_by_side_bgr, cnn_both_closed). An eye whose crop is None is
    treated as open (matching EyeStateClassifier.both_eyes_closed).
    """
    tiles, closed_flags = [], {}
    for tag in ("L", "R"):
        crop, probs = eye_data[tag]
        tile = np.zeros((view, view, 3), np.uint8)
        if crop is None:
            closed_flags[tag] = False
            label = f"{tag}: none"
        else:
            tile = cv2.resize(crop, (view, view), interpolation=cv2.INTER_NEAREST)
            open_p, closed_p = probs
            closed = closed_p > open_p
            closed_flags[tag] = closed
            label = f"{tag} {'CLOSED' if closed else 'open'} c={closed_p:.2f}"
        strip = np.zeros((18, view, 3), np.uint8)
        cv2.putText(strip, label, (2, 13), cv2.FONT_HERSHEY_SIMPLEX,
                    0.34, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.vstack([strip, tile]))
    gap = np.zeros((tiles[0].shape[0], 4, 3), np.uint8)
    combined = np.hstack([tiles[0], gap, tiles[1]])
    return combined, (closed_flags["L"] and closed_flags["R"])


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EYE_CROP_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Student name map from DB (optional)
    student_map = asyncio.run(_fetch_student_map())
    print(f"Student map loaded: {len(student_map)} entries")

    # 2. Pipeline components (all synchronous after init)
    print("Loading FaceDetector ...")
    detector = FaceDetector(model_name=settings.insightface_model_name, det_thresh=0.5)
    detector.warmup()

    print("Loading ArcFaceEmbedder ...")
    embedder = ArcFaceEmbedder(model_name=settings.insightface_model_name)
    embedder.warmup()

    print("Loading GalleryManager ...")
    gallery = GalleryManager()
    gallery.load()

    recognizer = FaceRecognizer(
        gallery=gallery,
        detector=detector,
        embedder=embedder,
        tracker=ByteTracker(),
        vote_window=1,          # single-frame votes – no temporal smoothing in diagnostic
    )

    print("Loading BehaviorAnalyzer (MediaPipe FaceLandmarker) ...")
    analyzer = BehaviorAnalyzer()
    analyzer.warmup()

    # 3. Open video
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open {VIDEO_PATH}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {VIDEO_PATH.name}  |  {total} frames  |  {fps:.1f} fps\n")

    # 8 evenly-spaced indices spanning first..last frame
    indices = [
        int(round(i * (total - 1) / (NUM_FRAMES - 1)))
        for i in range(NUM_FRAMES)
    ]
    print(f"Sampling frame indices: {indices}\n")

    summary: List[dict] = []

    for sample_n, fi in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            print(f"[warn] Frame {fi}: read failed - skipping")
            continue

        print(f"--- Frame {fi} (sample {sample_n + 1}/{NUM_FRAMES}) ---")

        results = recognizer.process_frame(
            frame=frame,
            frame_number=fi,
            timestamp_ms=int(cap.get(cv2.CAP_PROP_POS_MSEC)),
            student_map=student_map,
        )

        if not results:
            print("  No faces detected.\n")
            continue

        for r in results:
            bbox = r.bbox_xyxy
            x1, y1, x2, y2 = (int(v) for v in bbox[:4])
            orig_face_w = x2 - x1

            # Behavior analysis (EAR, yaw, pitch, label). RetinaFace's own 5
            # keypoints go in too: on frames where MediaPipe returns no landmarks
            # they are the only pose signal left, and _analyze_one uses them to
            # tell an extreme head turn from a marginal detection. Without them
            # this tool would report UNKNOWN where the pipeline reports
            # DISTRACTED — see _kps_yaw_index in backend/pipeline/behavior.py.
            bf: BehaviorFrame = analyzer._analyze_one(
                frame, bbox, track_id=r.track_id, kps=r.landmarks)

            # Reproduce _crop_face exactly (BGR, 256x256), then convert to RGB for
            # MediaPipe so the landmarks — and the CNN eye crops derived from them
            # below — match what _analyze_one computed for the pipeline.
            crop_bgr = _crop_face(frame, bbox)   # pad=0.60, resize to 256x256, BGR
            if crop_bgr is None:
                print(f"  track={r.track_id}: _crop_face returned None - skipping")
                continue
            crop_bgr = crop_bgr.copy()           # own buffer for drawing
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

            lms = None
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
            lm_result = analyzer._lm.detect(mp_img)
            if lm_result.face_landmarks:
                lms = lm_result.face_landmarks[0]

            if lms is not None:
                _draw_eye_landmarks(crop_bgr, lms)
            else:
                print(f"  track={r.track_id}: MediaPipe found no landmarks on crop")

            # (a) EAR-based decision (reproduces the analyzer's adaptive-baseline test)
            ear_closed = _ear_eyes_closed(analyzer, r.track_id, bf.ear)
            # (b) CNN decision — exactly the call the pipeline makes
            cnn_closed = bool(analyzer._eyes_closed_cnn(frame, bbox, lms))
            disagree = "  <-- DISAGREE" if ear_closed != cnn_closed else ""
            print(f"  track={r.track_id}: eyes_closed  "
                  f"EAR={ear_closed}  CNN={cnn_closed}{disagree}")

            label_str = bf.behavior.value
            _draw_text_block(crop_bgr, [
                f"EAR={bf.ear:.3f}",
                f"Yaw={bf.yaw:.1f}",
                f"Pitch={bf.pitch:.1f}",
                f"Label={label_str}",
                f"FaceW={orig_face_w}px",
                f"EAR-closed={ear_closed}",
                f"CNN-closed={cnn_closed}",
            ])

            student_name = _safe_name(r.student_name, r.student_id)
            filename = (
                f"frame{fi}_{student_name}_{label_str}"
                f"_ear{bf.ear:.3f}"
                f"_yaw{bf.yaw:.1f}"
                f"_pitch{bf.pitch:.1f}"
                f"_px{orig_face_w}.png"
            )
            cv2.imwrite(str(OUTPUT_DIR / filename), crop_bgr)
            print(f"  Saved: {filename}")

            # Second debug image: the actual 32x32 eye crops the CNN receives,
            # upscaled to 128x128 (nearest-neighbour) for visibility.
            eye_data = _cnn_eye_crops(analyzer, frame, bbox, lms)
            if eye_data is not None:
                eye_img, _ = _render_eye_debug(eye_data)
                eye_fn = (
                    f"frame{fi}_{student_name}"
                    f"_cnn{'closed' if cnn_closed else 'open'}"
                    f"_ear{'closed' if ear_closed else 'open'}.png"
                )
                cv2.imwrite(str(EYE_CROP_DIR / eye_fn), eye_img)
                print(f"  Saved eye crops: {eye_fn}")
            else:
                print(f"  track={r.track_id}: no landmarks/box for CNN eye crops")

            summary.append({
                "frame":      fi,
                "student":    student_name,
                "face_px":    orig_face_w,
                "ear":        bf.ear,
                "label":      label_str,
                "ear_closed": ear_closed,
                "cnn_closed": cnn_closed,
            })

        print()

    cap.release()

    # Summary table
    print("=" * 88)
    print(f"{'Frame':>6}  {'Student':<22}  {'FaceW':>6}  {'EAR':>6}  "
          f"{'EARcl':>6}  {'CNNcl':>6}  Label")
    print("-" * 88)
    for row in summary:
        print(
            f"{row['frame']:>6}  {row['student']:<22}  {row['face_px']:>6}  "
            f"{row['ear']:>6.3f}  {str(row['ear_closed']):>6}  "
            f"{str(row['cnn_closed']):>6}  {row['label']}"
        )
    print("=" * 88)
    print(f"\nCrops saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
