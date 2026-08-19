"""BehaviorAnalyzer — crop-then-analyze via MediaPipe FaceLandmarker."""
from __future__ import annotations
import logging
import math
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from scipy.spatial.transform import Rotation

from backend.config import get_settings
from backend.pipeline.eye_classifier import get_eye_classifier

logger = logging.getLogger(__name__)

_MODEL_PATH      = "/home/muhammadjunaidmalik/classroom_monitor/models/face_landmarker.task"
_LEFT_EYE        = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE       = [33,  160, 158, 133, 153, 144]
_YAW_THRESHOLD        = get_settings().behavior_yaw_threshold
_DISTRACTED_SECONDS = 5.0
_SLEEPING_SECONDS     = get_settings().behavior_sleeping_seconds
_SLEEP_RATIO          = get_settings().behavior_sleep_ratio
_EAR_FALLBACK       = 0.20
_EAR_HISTORY_LEN    = 15
_EAR_HISTORY_MIN    = 5
_SLEEP_RATIO_UNSTABLE = 0.35   # stricter sleep ratio for high-EAR-variance (glasses/occluded) tracks
_SLEEPING_SECONDS_UNSTABLE = 3.0   # longer sustain required before SLEEPING on unstable tracks
_EAR_STD_UNSTABLE     = 0.08   # per-track EAR std-dev above this ⇒ landmarks unreliable
_EYE_ASYM_MAX         = 0.15   # |EAR_left − EAR_right| above this ⇒ eye landmarks unreliable this frame
_POSE_PITCH_HEAD_DOWN = get_settings().behavior_pitch_head_down   # pitch-only head-down bar when EAR is untrustworthy
_YAW_DEVIATION        = get_settings().behavior_yaw_deviation     # |yaw − own median| above this ⇒ looking away
_PITCH_DEVIATION      = get_settings().behavior_pitch_deviation   # pitch above own median by this ⇒ head down
_CARRY_FORWARD_MAX    = 3      # max consecutive no-landmark frames to carry last behavior forward
_EYE_STATE_METHOD     = get_settings().eye_state_method   # "cnn" = ONNX eye-state gate, "ear" = EAR-only


class BehaviorType(str, Enum):
    ATTENTIVE  = "attentive"
    DISTRACTED = "distracted"
    DROWSY     = "drowsy"
    SLEEPING   = "sleeping"
    HEAD_DOWN  = "head_down"
    UNKNOWN    = "unknown"


@dataclass
class BehaviorFrame:
    track_id:     int             = -1
    behavior:     BehaviorType    = BehaviorType.ATTENTIVE
    confidence:   float           = 1.0
    ear:          float           = 0.3
    yaw:          float           = 0.0
    pitch:        float           = 0.0
    roll:         float           = 0.0
    pose_metrics: Optional[dict]  = None
    head_pitch:   Optional[float] = None
    head_yaw:     Optional[float] = None


def _ear(landmarks, indices: List[int]) -> float:
    """Eye Aspect Ratio from 6 normalized landmarks (no image size needed)."""
    pts = [(landmarks[i].x, landmarks[i].y) for i in indices]
    def d(a, b): return math.hypot(a[0] - b[0], a[1] - b[1])
    return (d(pts[1], pts[5]) + d(pts[2], pts[4])) / (2.0 * d(pts[0], pts[3]) + 1e-6)


def _rotation_to_euler(mat4x4) -> Tuple[float, float, float]:
    """Return (pitch, yaw, roll) in degrees from a 4×4 MediaPipe transform matrix."""
    R = np.asarray(mat4x4)[:3, :3]
    pitch, yaw, roll = Rotation.from_matrix(R).as_euler('xyz', degrees=True)
    return float(pitch), float(yaw), float(roll)



def _padded_face_box(frame: np.ndarray, bbox: np.ndarray,
                     pad: float = 0.60) -> Optional[Tuple[int, int, int, int]]:
    """Clamped (x1, y1, x2, y2) padded face box in original-frame pixels, or None."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox[:4])
    bw, bh = x2 - x1, y2 - y1
    if bw < 5 or bh < 5:
        return None
    px, py = int(bw * pad), int(bh * pad)
    x1c, y1c = max(0, x1 - px), max(0, y1 - py)
    x2c, y2c = min(w, x2 + px), min(h, y2 + py)
    return x1c, y1c, x2c, y2c


def _crop_face(frame: np.ndarray, bbox: np.ndarray, pad: float = 0.60) -> Optional[np.ndarray]:
    """Crop padded face region from full frame, resize to 256×256, return BGR."""
    box = _padded_face_box(frame, bbox, pad)
    if box is None:
        return None
    x1c, y1c, x2c, y2c = box
    crop = frame[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (256, 256))


class BehaviorAnalyzer:
    def __init__(self):
        self._ready = False
        self._lm: Optional[mp_vision.FaceLandmarker] = None
        self._eye_clf = get_eye_classifier()
        self._distracted_since: Dict[int, Optional[float]] = {}
        self._sleeping_since: Dict[int, Optional[float]] = {}
        self._ear_history: Dict[int, list] = {}
        self._yaw_history:   Dict[int, list] = {}
        self._pitch_history: Dict[int, list] = {}
        self._last_behavior: Dict[int, BehaviorType] = {}   # last confident behavior per track
        self._carry_count:   Dict[int, int] = {}            # consecutive carried-forward frames

    def _record(self, track_id: int, behavior: BehaviorType) -> None:
        """Remember a confident classification and reset the carry-forward counter."""
        self._last_behavior[track_id] = behavior
        self._carry_count[track_id]   = 0

    def _pose_baseline(self, track_id: int, yaw: float,
                       pitch: float) -> Tuple[Optional[float], Optional[float]]:
        """Update this track's yaw/pitch history and return its (yaw, pitch) baseline.

        Mirrors the _ear_history pattern: a fixed frame-count window (_EAR_HISTORY_LEN)
        per track, with no baseline until _EAR_HISTORY_MIN frames have accumulated
        (returns (None, None), and the caller falls back to the fixed global
        thresholds — the pose analogue of _EAR_FALLBACK).

        The statistic is the MEDIAN, not the 75th percentile used for EAR: yaw and
        pitch are signed angles scattered around a resting pose, so the middle of the
        distribution is the resting pose. EAR is a one-sided magnitude whose high end
        is the open-eye reference, which is why it uses an upper percentile.

        Every measured frame enters the history unconditionally, exactly as EAR does —
        frames are never excluded on the basis of how they were classified.
        """
        yaw_hist = self._yaw_history.setdefault(track_id, [])
        yaw_hist.append(yaw)
        if len(yaw_hist) > _EAR_HISTORY_LEN:
            del yaw_hist[:-_EAR_HISTORY_LEN]
        pitch_hist = self._pitch_history.setdefault(track_id, [])
        pitch_hist.append(pitch)
        if len(pitch_hist) > _EAR_HISTORY_LEN:
            del pitch_hist[:-_EAR_HISTORY_LEN]
        if len(yaw_hist) < _EAR_HISTORY_MIN:
            return None, None
        return float(np.median(yaw_hist)), float(np.median(pitch_hist))

    def _pose_off_axis(self, yaw: float, pitch: float,
                       baseline_yaw: Optional[float],
                       baseline_pitch: Optional[float]) -> bool:
        """True when head pose is off *this student's own* resting baseline.

        Fixes the side-column bias: a student seated off-center must angle their head
        toward the board to be attentive, so their resting yaw is a large constant that
        a single global threshold reads as distraction. Deviation from their own median
        is what actually signals looking away. Falls back to the fixed global thresholds
        while the track is still warming up and has no baseline yet.

        Yaw is two-sided (turning either way is looking away); pitch stays one-sided,
        preserving the head-down semantics of the original pitch > _POSE_PITCH_HEAD_DOWN
        test — looking down relative to one's own resting pitch counts, looking up
        does not.
        """
        if baseline_yaw is None or baseline_pitch is None:
            return pitch > _POSE_PITCH_HEAD_DOWN or abs(yaw) > _YAW_THRESHOLD
        return (pitch - baseline_pitch > _PITCH_DEVIATION
                or abs(yaw - baseline_yaw) > _YAW_DEVIATION)

    def _distracted_dwell(self, track_id: int, distracted_conf: float,
                          attentive_conf: float, timestamp_ms: float) -> tuple[BehaviorType, float]:
        """Advance the distraction dwell timer and resolve the (behavior, confidence).

        Starts the timer on first entry, then returns DISTRACTED once the gaze has
        been off-axis for _DISTRACTED_SECONDS, otherwise ATTENTIVE while it accrues.
        Uses the video's own timestamp_ms rather than wall-clock time, since
        wall-clock time measures real processing speed, not video content time.
        """
        if self._distracted_since.get(track_id) is None:
            self._distracted_since[track_id] = timestamp_ms
        elapsed = (timestamp_ms - self._distracted_since[track_id]) / 1000.0
        if elapsed >= _DISTRACTED_SECONDS:
            return BehaviorType.DISTRACTED, distracted_conf
        return BehaviorType.ATTENTIVE, attentive_conf

    def _carry_or_unknown(self, track_id: int) -> "BehaviorFrame":
        """Smooth momentary MediaPipe detection failures (no landmarks / no crop).

        Reuse the last confident behavior for up to _CARRY_FORWARD_MAX consecutive
        frames before falling back to UNKNOWN, so a brief detection dropout does not
        spike the UNKNOWN count.
        """
        last  = self._last_behavior.get(track_id)
        count = self._carry_count.get(track_id, 0)
        if last is not None and count < _CARRY_FORWARD_MAX:
            self._carry_count[track_id] = count + 1
            return BehaviorFrame(track_id=track_id, behavior=last, confidence=0.40)
        return BehaviorFrame(track_id=track_id, behavior=BehaviorType.UNKNOWN, confidence=0.0)

    def warmup(self) -> None:
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
        self._lm    = mp_vision.FaceLandmarker.create_from_options(opts)
        if _EYE_STATE_METHOD == "cnn":
            self._eye_clf.warmup()
        self._ready = True
        logger.info("BehaviorAnalyzer: FaceLandmarker ready (crop-then-analyze)")

    def analyze(self, frame: np.ndarray, bbox: np.ndarray, landmarks=None, timestamp_ms: float = 0.0):
        bf = self._analyze_one(frame, np.asarray(bbox), timestamp_ms=timestamp_ms)
        return bf.behavior, bf.confidence

    def analyze_frame(self, frame: np.ndarray, results, timestamp_ms: float) -> List[BehaviorFrame]:
        out: List[BehaviorFrame] = []
        for r in results:
            tid  = getattr(r, "track_id", -1)
            bbox = getattr(r, "bbox_xyxy", None)
            if bbox is None or not self._ready:
                out.append(BehaviorFrame(track_id=tid, behavior=BehaviorType.UNKNOWN, confidence=0.0))
                continue
            out.append(self._analyze_one(frame, bbox, track_id=tid, timestamp_ms=timestamp_ms))
        return out

    def _analyze_one(self, frame: np.ndarray, bbox: np.ndarray, track_id: int = -1, timestamp_ms: float = 0.0) -> BehaviorFrame:
        fallback = BehaviorFrame(track_id=track_id, behavior=BehaviorType.UNKNOWN, confidence=0.0)
        try:
            crop_bgr = _crop_face(frame, bbox)
            if crop_bgr is None:
                return self._carry_or_unknown(track_id)   # carry-forward on crop failure
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
            det    = self._lm.detect(mp_img)
            if not det.face_landmarks:
                return self._carry_or_unknown(track_id)   # carry-forward on no landmarks
            lms = det.face_landmarks[0]
            ear_left  = _ear(lms, _LEFT_EYE)
            ear_right = _ear(lms, _RIGHT_EYE)
            ear = (ear_left + ear_right) / 2.0
            # Landmark-reliability gate: MediaPipe FaceLandmarker (.task) does NOT expose
            # per-landmark visibility/presence (both are None), so we use eye-landmark
            # agreement as a proxy. A large left/right EAR gap means the eye landmarks are
            # unreliable this frame (e.g. a glasses frame sitting over the eye) — do not
            # trust EAR for this frame.
            eye_reliable = abs(ear_left - ear_right) <= _EYE_ASYM_MAX
            history = self._ear_history.setdefault(track_id, [])
            history.append(ear)
            if len(history) > _EAR_HISTORY_LEN:
                del history[:-_EAR_HISTORY_LEN]
            if len(history) < _EAR_HISTORY_MIN:
                baseline_ear = _EAR_FALLBACK
            else:
                baseline_ear = float(np.percentile(history, 75))
            # EAR-variance detection: a high per-track std-dev indicates unstable landmark
            # detection (glasses/occlusion). For such tracks require EAR to drop further
            # before calling SLEEPING, to cut false positives.
            sleep_ratio = _SLEEP_RATIO
            sleeping_secs = _SLEEPING_SECONDS
            unstable = len(history) >= _EAR_HISTORY_MIN and float(np.std(history)) > _EAR_STD_UNSTABLE
            if unstable:
                sleep_ratio   = _SLEEP_RATIO_UNSTABLE
                sleeping_secs = _SLEEPING_SECONDS_UNSTABLE
            # Decide eye state via exactly one method — the two paths never overlap and
            # never fall back to each other. In "cnn" mode EAR is not consulted for this
            # decision at all (the adaptive baseline above is retained only for the
            # head-down heuristic, "ear" mode, and logging).
            if _EYE_STATE_METHOD == "cnn":
                eyes_closed = self._eyes_closed_cnn(frame, bbox, lms)
            else:
                eyes_closed = ear < baseline_ear * sleep_ratio
            pitch = yaw = roll = 0.0
            baseline_yaw = baseline_pitch = None
            if det.facial_transformation_matrixes:
                pitch, yaw, roll = _rotation_to_euler(det.facial_transformation_matrixes[0])
                # Only a real pose measurement updates the baseline. Without a transform
                # matrix, yaw/pitch stay at the placeholder 0.0 — an absent measurement,
                # not a measured zero — and feeding that in would drag every baseline
                # toward 0 and re-create the side-column bias this is fixing.
                baseline_yaw, baseline_pitch = self._pose_baseline(track_id, yaw, pitch)
            if _EYE_STATE_METHOD == "ear" and not eye_reliable:
                # EAR mode only: eye landmarks untrustworthy — cannot judge eye closure,
                # so SLEEPING is not reachable here; classify on head pose alone. CNN reads
                # pixel intensities and does not depend on EAR geometry, so this bypass is
                # skipped in "cnn" mode. Head-down (pitch) and looking-away (yaw) both count
                # as DISTRACTED now (HEAD_DOWN merged in). Prefer UNKNOWN when pose is
                # unremarkable, over a likely misclassification.
                self._sleeping_since[track_id] = None
                if self._pose_off_axis(yaw, pitch, baseline_yaw, baseline_pitch):
                    behavior, confidence = self._distracted_dwell(track_id, 0.70, 0.70, timestamp_ms)
                else:
                    self._distracted_since[track_id] = None
                    behavior, confidence = BehaviorType.UNKNOWN, 0.0
            # SLEEPING is checked FIRST and is fully independent of head pose: a student
            # with their head down is still classified SLEEPING when eyes are confirmed
            # closed. Sustained via the dwell timer to suppress blink false positives.
            elif eyes_closed:
                self._distracted_since[track_id] = None
                if self._sleeping_since.get(track_id) is None:
                    self._sleeping_since[track_id] = timestamp_ms
                elapsed = (timestamp_ms - self._sleeping_since[track_id]) / 1000.0
                if elapsed >= sleeping_secs:
                    behavior = BehaviorType.SLEEPING
                    confidence = 0.97 if _EYE_STATE_METHOD == "cnn" else 0.95
                else:
                    behavior, confidence = BehaviorType.ATTENTIVE, 0.90
            # DISTRACTED: head down (pitch) OR looking away (yaw). HEAD_DOWN is merged here
            # and no longer emitted. The dominant-axis tie-break no longer changes the label
            # (both axes → DISTRACTED); the dwell timer gates it either way.
            elif self._pose_off_axis(yaw, pitch, baseline_yaw, baseline_pitch):
                self._sleeping_since[track_id] = None
                behavior, confidence = self._distracted_dwell(track_id, 0.80, 0.90, timestamp_ms)
            else:
                self._sleeping_since[track_id] = None
                self._distracted_since[track_id] = None
                behavior, confidence = BehaviorType.ATTENTIVE, 0.90
            # Record confident classifications so carry-forward can reuse them on dropouts.
            if behavior is not BehaviorType.UNKNOWN:
                self._record(track_id, behavior)
            return BehaviorFrame(
                track_id     = track_id,
                behavior     = behavior,
                confidence   = confidence,
                ear          = ear,
                yaw          = yaw,
                pitch        = pitch,
                roll         = roll,
                pose_metrics = {"ear": ear, "yaw": yaw, "pitch": pitch, "roll": roll},
                head_pitch   = pitch,
                head_yaw     = yaw,
            )
        except Exception as exc:
            logger.warning("BehaviorAnalyzer track %d failed: %s", track_id, exc)
            return fallback

    def _eyes_closed_cnn(self, frame: np.ndarray, bbox: np.ndarray, lms) -> bool:
        """CNN confirmation that both eyes are closed, cropping the eye regions
        directly from the original full-resolution frame.

        MediaPipe landmarks are normalized to the 256×256 face crop; map them back
        to original-frame pixels so each eye is cropped once at native resolution
        (no upscale-then-downscale blur) before the single 32×32 resize.

        Falls back to True (i.e. trust the EAR decision) if the classifier raises,
        so a model failure never suppresses an otherwise-valid SLEEPING call.
        """
        try:
            box = _padded_face_box(frame, bbox)
            if box is None:
                return True
            x1c, y1c, x2c, y2c = box
            rw, rh = x2c - x1c, y2c - y1c
            # normalized 256×256-crop coords → original-frame pixels
            def to_orig(i):
                lm = lms[i]
                return (x1c + lm.x * rw, y1c + lm.y * rh)
            left  = [to_orig(i) for i in _LEFT_EYE]
            right = [to_orig(i) for i in _RIGHT_EYE]
            return self._eye_clf.both_eyes_closed(frame, left, right)
        except Exception as exc:
            logger.debug("Eye-state CNN failed: %s", exc)
            return True

    def reset(self, track_id: int) -> None:
        self._distracted_since.pop(track_id, None)
        self._sleeping_since.pop(track_id, None)
        self._ear_history.pop(track_id, None)
        self._yaw_history.pop(track_id, None)
        self._pitch_history.pop(track_id, None)
        self._last_behavior.pop(track_id, None)
        self._carry_count.pop(track_id, None)


_analyzer: Optional[BehaviorAnalyzer] = None


def get_analyzer() -> BehaviorAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = BehaviorAnalyzer()
    return _analyzer


def get_behavior_analyzer() -> BehaviorAnalyzer:
    return get_analyzer()
