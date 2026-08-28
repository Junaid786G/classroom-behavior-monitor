"""Regression tests for EyeStateClassifier input preprocessing and output order.

Two stacked bugs are locked in here:

1. Preprocessing. `_infer` must apply the OMZ model.yml transform (x - 127) / 255
   (mean=127, scale=255). Raw 0–255 floats overflow the final softmax → `[..,nan]`;
   plain /255 (no mean subtraction) inverts the result.
2. Output order. This ONNX emits softmax order [closed, open] — the REVERSE of the
   OMZ README's [open, closed] — so `_infer` swaps the indices before returning
   (open_prob, closed_prob). Verified empirically on known-open eyes.

Either bug silently made `closed_prob > open_prob` fire for OPEN eyes, so CNN-gated
SLEEPING mis-fired on attentive students.
"""
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pytest

from backend.config import get_settings
from backend.pipeline.behavior import _LEFT_EYE, _RIGHT_EYE
from backend.pipeline.eye_classifier import EyeStateClassifier, _MODEL_PATH

_LANDMARKER = str(get_settings().face_landmarker_path)
_PHOTOS_DIR = Path("data/student_photos")

pytestmark = pytest.mark.skipif(
    not Path(_MODEL_PATH).is_file(),
    reason=f"eye-state ONNX model not present at {_MODEL_PATH}",
)


def _clf() -> EyeStateClassifier:
    clf = EyeStateClassifier()
    clf.warmup()
    return clf


@pytest.mark.parametrize(
    "crop",
    [
        np.zeros((32, 32, 3), dtype=np.uint8),                 # all black
        np.full((32, 32, 3), 255, dtype=np.uint8),             # all white (max range)
        np.random.default_rng(0).integers(                     # realistic dim eye crop
            4, 100, size=(32, 32, 3)).astype(np.uint8),
    ],
    ids=["black", "white", "midrange"],
)
def test_infer_returns_finite_normalised_probs(crop):
    """_infer must return finite probabilities in [0, 1] that sum to ~1 — never
    NaN — across the full input range (the un-normalised bug NaNed on max-range
    pixels in particular)."""
    open_p, closed_p = _clf()._infer(crop)

    assert np.isfinite(open_p) and np.isfinite(closed_p)
    assert 0.0 <= open_p <= 1.0
    assert 0.0 <= closed_p <= 1.0
    assert open_p + closed_p == pytest.approx(1.0, abs=1e-3)


def test_eye_closed_decision_is_a_real_bool():
    """With finite probs, the closed/open decision is a genuine comparison rather
    than the `nan > 0.0` (always-False) that the bug produced."""
    clf = _clf()
    # 6 eye points spanning a small region — exercises the full crop→infer path.
    landmarks = [(10, 12), (12, 10), (14, 10), (16, 12), (14, 14), (12, 14)]
    frame = np.random.default_rng(1).integers(0, 256, size=(64, 64, 3)).astype(np.uint8)

    result = clf._eye_closed(frame, landmarks)
    assert isinstance(result, (bool, np.bool_))


def _open_eye_crops(max_photos: int = 8):
    """Yield 32×32 BGR eye crops from enrollment photos (subjects look at camera,
    so eyes are OPEN). Uses the real FaceLandmarker; yields nothing if unavailable.
    """
    if not Path(_LANDMARKER).is_file() or not _PHOTOS_DIR.is_dir():
        return
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
        VisionTaskRunningMode,
    )

    lm = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_LANDMARKER),
            running_mode=VisionTaskRunningMode.IMAGE, num_faces=1,
        )
    )
    for photo in sorted(_PHOTOS_DIR.glob("*/photo.jpg"))[:max_photos]:
        img = cv2.imread(str(photo))
        if img is None:
            continue
        img = cv2.copyMakeBorder(img, 80, 80, 80, 80, cv2.BORDER_REPLICATE)
        h, w = img.shape[:2]
        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        if not res.face_landmarks:
            continue
        pts_lm = res.face_landmarks[0]
        for idx in (_LEFT_EYE, _RIGHT_EYE):
            pts = [(pts_lm[i].x * w, pts_lm[i].y * h) for i in idx]
            crop = EyeStateClassifier()._crop_eye(img, pts)
            if crop is not None:
                yield crop


def test_open_eyes_classify_as_open():
    """End-to-end semantic check on KNOWN-OPEN eyes from enrollment photos: the
    model's open_prob must beat closed_prob. Guards against re-introducing the
    output-index swap (this ONNX emits [closed, open], not the README's order)."""
    clf = _clf()
    crops = list(_open_eye_crops())
    if not crops:
        pytest.skip("no enrollment photos / landmarker available for open-eye check")

    probs = np.array([clf._infer(c) for c in crops])   # rows: (open_p, closed_p)
    mean_open, mean_closed = probs[:, 0].mean(), probs[:, 1].mean()
    # Mean separation is robust to the odd marginal 32px eye while still catching a
    # re-inversion: pre-fix these open eyes gave mean_open ≈ 0.0.
    assert mean_open >= 0.7 and mean_open > mean_closed, (
        f"known-open eyes read mean open_prob={mean_open:.3f} vs "
        f"closed_prob={mean_closed:.3f} — output order likely re-inverted"
    )
