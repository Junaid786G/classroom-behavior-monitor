from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from backend.config import get_settings
from backend.pipeline.tiling import RawDetection
from backend.utils.gpu import insightface_ctx_id
from backend.utils.image import crop_and_align_face

logger = logging.getLogger(__name__)
settings = get_settings()

_EMBEDDING_DIM = 512


class ArcFaceEmbedder:
    """
    ArcFace recognition model wrapper (InsightFace buffalo_l recognition head).

    The model is loaded lazily; call warmup() once during app startup to
    amortise the ONNX initialisation cost before the first request arrives.

    Output embeddings are L2-normalised 512-d float32 vectors.
    """

    def __init__(self, model_name: str = "buffalo_l") -> None:
        self._model_name = model_name
        self._rec = None   # lazy: insightface recognition model

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def warmup(self) -> None:
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._rec is not None:
            return
        try:
            import insightface.model_zoo as model_zoo
        except ImportError as e:
            raise RuntimeError("insightface is not installed") from e

        ctx = insightface_ctx_id(settings.insightface_ctx_id)
        logger.info("Loading ArcFace recognition model %r (ctx=%d) …", self._model_name, ctx)

        # Load the recognition ONNX directly — avoids FaceAnalysis which
        # requires 'detection' to be present and would load the detector twice.
        from pathlib import Path
        onnx_path = Path.home() / ".insightface" / "models" / self._model_name / "w600k_r50.onnx"
        self._rec = model_zoo.get_model(str(onnx_path), providers=_providers(ctx))
        self._rec.prepare(ctx_id=ctx)
        logger.info("ArcFaceEmbedder ready (dim=%d)", _EMBEDDING_DIM)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_embedding(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        kps: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Extract a 512-d L2-normalised embedding from one face.

        Parameters
        ----------
        frame : BGR uint8 full frame
        bbox  : [x1, y1, x2, y2] absolute pixel coords
        kps   : (5, 2) 5-point landmark coords (optional, improves alignment)
        """
        self._ensure_loaded()
        aligned = crop_and_align_face(frame, bbox, kps, output_size=112)
        return self._embed_aligned(aligned)

    def embed_from_detection(
        self, frame: np.ndarray, det: RawDetection
    ) -> np.ndarray:
        """Convenience wrapper accepting a RawDetection."""
        return self.get_embedding(frame, det.bbox, det.landmarks)

    def embed_batch(
        self,
        frame: np.ndarray,
        detections: List[RawDetection],
    ) -> List[np.ndarray]:
        return [self.embed_from_detection(frame, d) for d in detections]

    def embed_image(self, img_112: np.ndarray) -> np.ndarray:
        """Embed a pre-cropped/aligned 112×112 face crop."""
        self._ensure_loaded()
        return self._embed_aligned(img_112)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _embed_aligned(self, aligned: np.ndarray) -> np.ndarray:
        """Run ArcFace on a pre-aligned 112×112 BGR uint8 crop."""
        emb = np.array(self._rec.get_feat(aligned), dtype=np.float32).flatten()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm
        return emb


# ── Module-level singleton ────────────────────────────────────────────────────

_embedder: Optional[ArcFaceEmbedder] = None


def get_embedder() -> ArcFaceEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = ArcFaceEmbedder(model_name=settings.insightface_model_name)
    return _embedder


def _providers(ctx_id: int):
    if ctx_id >= 0:
        return [
            ("CUDAExecutionProvider", {"device_id": ctx_id}),
            "CPUExecutionProvider",
        ]
    return ["CPUExecutionProvider"]
