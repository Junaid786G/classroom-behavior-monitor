from __future__ import annotations

import asyncio
import json
import logging
from threading import Lock
from typing import List, Optional, Tuple

import faiss
import numpy as np

from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_DIM = 512   # ArcFace embedding dimension


class GalleryManager:
    """
    Thread-safe FAISS IndexFlatIP gallery.

    Embeddings are L2-normalised before insertion so inner-product == cosine
    similarity (range 0–1; 1 = identical).

    Metadata sidecar (meta.json) stores a list of student_ids where
    meta[row_index] == student_id.
    """

    def __init__(self, dim: int = _DIM) -> None:
        self._dim = dim
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(dim)
        self._meta: List[int] = []     # row_index → student_id
        self._lock = Lock()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        idx_path = settings.embeddings_index_path
        meta_path = settings.embeddings_meta_path
        with self._lock:
            faiss.write_index(self._index, str(idx_path))
            meta_path.write_text(json.dumps(self._meta), encoding="utf-8")
        logger.info(f"Gallery saved ({self._index.ntotal} vectors) → {idx_path}")

    def load(self) -> bool:
        idx_path = settings.embeddings_index_path
        meta_path = settings.embeddings_meta_path
        if not idx_path.exists() or not meta_path.exists():
            logger.info("No persisted gallery found – starting empty")
            return False
        with self._lock:
            self._index = faiss.read_index(str(idx_path))
            self._meta = json.loads(meta_path.read_text(encoding="utf-8"))
        logger.info(f"Gallery loaded ({self._index.ntotal} vectors)")
        return True

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add(self, embedding: np.ndarray, student_id: int) -> int:
        """Insert one 512-d vector; return its gallery row index."""
        emb = self._norm(embedding)
        with self._lock:
            idx = self._index.ntotal
            self._index.add(emb)
            self._meta.append(student_id)
        return int(idx)

    def replace(self, student_id: int, embedding: np.ndarray) -> int:
        """
        Remove any existing vectors for student_id and add the new one.
        Returns the new gallery row index.
        """
        self.remove_student(student_id)
        return self.add(embedding, student_id)

    def remove_student(self, student_id: int) -> int:
        """
        Remove all vectors for *student_id* and rebuild the FAISS index.
        Expensive – do not call in hot-path.
        Returns number of vectors removed.
        """
        with self._lock:
            keep = [i for i, sid in enumerate(self._meta) if sid != student_id]
            removed = self._index.ntotal - len(keep)
            if removed == 0:
                return 0

            if keep:
                vecs = np.vstack([
                    self._index.reconstruct(i).reshape(1, -1) for i in keep
                ])
                new_meta = [self._meta[i] for i in keep]
            else:
                vecs = np.empty((0, self._dim), dtype=np.float32)
                new_meta = []

            new_index: faiss.IndexFlatIP = faiss.IndexFlatIP(self._dim)
            if len(vecs):
                new_index.add(vecs)
            self._index = new_index
            self._meta = new_meta

        logger.info(f"Removed {removed} vector(s) for student_id={student_id}")
        return removed

    def rebuild(self, pairs: List[Tuple[int, np.ndarray]]) -> None:
        """
        Atomically replace the entire index with (student_id, embedding) pairs.
        Thread-safe.
        """
        new_index: faiss.IndexFlatIP = faiss.IndexFlatIP(self._dim)
        new_meta: List[int] = []

        if pairs:
            vecs = np.vstack([self._norm(emb) for _, emb in pairs])
            new_index.add(vecs)
            new_meta = [sid for sid, _ in pairs]

        with self._lock:
            self._index = new_index
            self._meta = new_meta

        logger.info(
            f"Gallery rebuilt: {new_index.ntotal} vectors, "
            f"{len(set(new_meta))} unique students"
        )

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        embedding: np.ndarray,
        k: int = 1,
        threshold: Optional[float] = None,
    ) -> List[Tuple[int, float]]:
        """
        Return up to *k* (student_id, score) pairs with score ≥ threshold.
        Empty list if gallery is empty or no match passes threshold.
        """
        if threshold is None:
            threshold = settings.face_recognition_threshold

        with self._lock:
            ntotal = self._index.ntotal
            if ntotal == 0:
                return []
            emb = self._norm(embedding)
            scores, indices = self._index.search(emb, min(k, ntotal))

        results: List[Tuple[int, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            s = float(score)
            if s >= threshold:
                results.append((self._meta[idx], s))
        return results

    def best_match(
        self,
        embedding: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Tuple[Optional[int], float]:
        """Convenience wrapper returning (student_id | None, score)."""
        hits = self.search(embedding, k=1, threshold=threshold)
        return hits[0] if hits else (None, 0.0)

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return self._index.ntotal

    @property
    def student_ids(self) -> List[int]:
        with self._lock:
            return list(set(self._meta))

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _norm(emb: np.ndarray) -> np.ndarray:
        v = emb.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(v)
        return v


# ── Module-level singleton ────────────────────────────────────────────────────

_gallery: Optional[GalleryManager] = None
_init_lock = asyncio.Lock()


async def get_gallery() -> GalleryManager:
    global _gallery
    async with _init_lock:
        if _gallery is None:
            _gallery = GalleryManager()
            _gallery.load()
    return _gallery


async def rebuild_gallery_from_db(db) -> GalleryManager:
    """Pull all stored face embeddings from PostgreSQL and rebuild FAISS."""
    from backend import crud   # local import to break circular dep

    gallery = await get_gallery()
    students = await crud.get_all_students_with_embeddings(db)

    pairs = [
        (s.id, np.array(s.face_embedding, dtype=np.float32))
        for s in students
        if s.face_embedding
    ]
    gallery.rebuild(pairs)
    gallery.save()

    logger.info(f"Gallery rebuilt from DB: {len(pairs)} embeddings, "
                f"{len(set(p[0] for p in pairs))} students")
    return gallery
