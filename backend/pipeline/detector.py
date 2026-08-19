from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from backend.config import get_settings
from backend.pipeline.tiling import RawDetection, merge_detections, tile_frame
from backend.utils.gpu import insightface_ctx_id

logger = logging.getLogger(__name__)
settings = get_settings()


class FaceDetector:
    """
    InsightFace RetinaFace wrapper that runs on a single frame or on tiled
    sub-regions for better small-face recall.

    Model loaded lazily on first call so the class can be instantiated cheaply
    at import time; the heavy model load happens once during the first process.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: tuple = (640, 640),
        det_thresh: float = 0.5,
        use_tiling: bool = False,
        tile_size: int = 640,
        tile_overlap: float = 0.2,
    ) -> None:
        self._model_name = model_name
        self._det_size = det_size
        self._det_thresh = det_thresh
        self._use_tiling = use_tiling
        self._tile_size = tile_size
        self._tile_overlap = tile_overlap
        self._app = None   # lazy-loaded insightface.app.FaceAnalysis

    # ── Init ──────────────────────────────────────────────────────────────────

    def warmup(self) -> None:
        """Force model load – call once in app lifespan."""
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._app is not None:
            return
        try:
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise RuntimeError("insightface is not installed") from e

        ctx = insightface_ctx_id(settings.insightface_ctx_id)
        logger.info("Loading InsightFace model %r (ctx=%d) …", self._model_name, ctx)

        self._app = FaceAnalysis(
            name=self._model_name,
            allowed_modules=["detection"],   # skip recognition in detector
            providers=_providers(ctx),
        )
        self._app.prepare(ctx_id=ctx, det_size=self._det_size, det_thresh=self._det_thresh)
        logger.info("FaceDetector ready")

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[RawDetection]:
        """
        Detect faces in *frame* (BGR uint8).
        Returns a list of RawDetection with absolute pixel coordinates.
        """
        self._ensure_loaded()

        if self._use_tiling:
            return self._detect_tiled(frame)
        return self._detect_single(frame)

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[RawDetection]]:
        return [self.detect(f) for f in frames]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _detect_single(self, frame: np.ndarray) -> List[RawDetection]:
        faces = self._app.get(frame)
        return [
            RawDetection(
                bbox=f.bbox.astype(np.float32),
                landmarks=f.kps.astype(np.float32),
                score=float(f.det_score),
            )
            for f in faces
        ]

    def _detect_tiled(self, frame: np.ndarray) -> List[RawDetection]:
        tiles = tile_frame(frame, self._tile_size, self._tile_overlap)
        tile_results = []
        for tile_img, offset in tiles:
            dets = self._detect_single(tile_img)
            tile_results.append((dets, offset))
        return merge_detections(tile_results)


# ── Module-level singleton ────────────────────────────────────────────────────

_detector: Optional[FaceDetector] = None


def get_detector() -> FaceDetector:
    global _detector
    if _detector is None:
        _detector = FaceDetector(
            model_name=settings.insightface_model_name,
            det_thresh=0.5,
            use_tiling=False,
        )
    return _detector


# ── Helper ────────────────────────────────────────────────────────────────────

def _providers(ctx_id: int):
    if ctx_id >= 0:
        return [
            ("CUDAExecutionProvider", {"device_id": ctx_id}),
            "CPUExecutionProvider",
        ]
    return ["CPUExecutionProvider"]
