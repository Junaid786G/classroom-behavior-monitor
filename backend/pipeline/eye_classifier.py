from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from backend.config import get_settings
from backend.utils.gpu import get_onnx_providers

logger = logging.getLogger(__name__)
settings = get_settings()

_MODEL_PATH = str(settings.eye_state_model_path)
_INPUT_NAME = "input.1"
_OUTPUT_NAME = "19"
_INPUT_SIZE = 32          # model expects 32×32
_EYE_PADDING = 0.20       # 20% padding around the eye bounding box


class EyeStateClassifier:
    """
    Wrapper around the open-closed-eye-0001 ONNX classifier.

    The model takes a 32×32 BGR eye crop, preprocessed as (x - 127) / 255 per the
    OMZ model.yml (mean=127, scale=255), and returns two probabilities. NOTE:
    this ONNX's softmax order is [closed, open] — the REVERSE of the OMZ README's
    documented [open, closed] — verified empirically on known-open eyes, so
    _infer swaps the indices and returns (open_prob, closed_prob).

    The session is loaded lazily; call warmup() once during app startup to
    amortise the ONNX initialisation cost before the first frame is processed.
    """

    def __init__(self, model_path: str = _MODEL_PATH) -> None:
        self._model_path = model_path
        self._session = None   # lazy: onnxruntime.InferenceSession

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def warmup(self) -> None:
        """Force model load + one dummy inference – call once in app lifespan."""
        self._ensure_loaded()
        dummy = np.zeros((_INPUT_SIZE, _INPUT_SIZE, 3), dtype=np.uint8)
        self._infer(dummy)
        logger.info("EyeStateClassifier warmed up")

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError("onnxruntime is not installed") from e

        path = Path(self._model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Eye-state ONNX model not found: {path}")

        providers = get_onnx_providers(settings.insightface_ctx_id)
        logger.info("Loading eye-state model %r …", str(path))
        self._session = ort.InferenceSession(str(path), providers=providers)
        logger.info("EyeStateClassifier ready (providers=%s)", self._session.get_providers())

    # ── Public API ────────────────────────────────────────────────────────────

    def both_eyes_closed(
        self,
        frame_bgr: np.ndarray,
        left_landmarks: Sequence,
        right_landmarks: Sequence,
    ) -> bool:
        """
        Classify both eyes and return True only if BOTH are closed.

        Parameters
        ----------
        frame_bgr      : the full-resolution BGR video frame
        left_landmarks : 6 (x, y) left-eye points in frame_bgr pixel coordinates
        right_landmarks: 6 (x, y) right-eye points in frame_bgr pixel coordinates

        A single eye is considered closed when closed_prob > open_prob.
        If either eye crop is degenerate (out of bounds / empty) the eye is
        treated as open, so the method returns False.
        """
        self._ensure_loaded()

        left_closed = self._eye_closed(frame_bgr, left_landmarks)
        if not left_closed:
            return False  # short-circuit: both must be closed
        right_closed = self._eye_closed(frame_bgr, right_landmarks)
        return left_closed and right_closed

    # ── Internal ──────────────────────────────────────────────────────────────

    def _eye_closed(
        self,
        frame_bgr: np.ndarray,
        landmarks: Sequence,
    ) -> bool:
        crop = self._crop_eye(frame_bgr, landmarks)
        if crop is None:
            return False
        open_prob, closed_prob = self._infer(crop)
        return closed_prob > open_prob

    def _crop_eye(
        self,
        frame_bgr: np.ndarray,
        landmarks: Sequence,
    ) -> Optional[np.ndarray]:
        """
        Crop the padded eye region directly from the frame and resize once to 32×32.

        Returns None if the resulting box is empty.
        """
        xs, ys = [], []
        for lm in landmarks:
            x, y = _landmark_to_px(lm)
            xs.append(x)
            ys.append(y)

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # 20% padding around the eye bounding box
        pad_x = (x_max - x_min) * _EYE_PADDING
        pad_y = (y_max - y_min) * _EYE_PADDING

        h, w = frame_bgr.shape[:2]
        x1 = int(round(max(0, x_min - pad_x)))
        y1 = int(round(max(0, y_min - pad_y)))
        x2 = int(round(min(w, x_max + pad_x)))
        y2 = int(round(min(h, y_max + pad_y)))

        if x2 <= x1 or y2 <= y1:
            return None

        eye = frame_bgr[y1:y2, x1:x2]
        if eye.size == 0:
            return None
        return cv2.resize(eye, (_INPUT_SIZE, _INPUT_SIZE))

    def _infer(self, eye_bgr_32: np.ndarray) -> Tuple[float, float]:
        """
        Run the ONNX model on a 32×32 BGR eye crop.
        Returns (open_prob, closed_prob).
        """
        # HWC BGR uint8 → NCHW float32 with the OMZ model.yml preprocessing:
        # mean=127, scale=255  ->  (x - 127) / 255. Feeding raw 0–255 overflows the
        # final softmax (NaN); plain /255 (no mean subtraction) inverts the output.
        blob = ((eye_bgr_32.astype(np.float32) - 127.0) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]
        out = self._session.run([_OUTPUT_NAME], {_INPUT_NAME: blob})[0]
        # This ONNX emits softmax order [closed, open] — the REVERSE of the OMZ
        # README ([open, closed]) — verified empirically on known-open eyes.
        probs = np.asarray(out, dtype=np.float32).reshape(-1)  # [closed, open]
        return float(probs[1]), float(probs[0])                # -> (open_prob, closed_prob)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _landmark_to_px(lm) -> Tuple[float, float]:
    """
    Extract absolute (x, y) pixel coords from one eye point.

    Accepts an object with .x/.y attributes or a plain (x, y) sequence. Callers
    pass coordinates already in frame pixels; no scaling is applied here.
    """
    if hasattr(lm, "x") and hasattr(lm, "y"):
        return float(lm.x), float(lm.y)
    return float(lm[0]), float(lm[1])


# ── Module-level singleton ────────────────────────────────────────────────────

_classifier: Optional[EyeStateClassifier] = None


def get_eye_classifier() -> EyeStateClassifier:
    global _classifier
    if _classifier is None:
        _classifier = EyeStateClassifier()
    return _classifier
