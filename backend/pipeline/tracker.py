from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class Track:
    """A confirmed track returned by ByteTrack."""
    track_id: int
    bbox_xyxy: np.ndarray   # [x1, y1, x2, y2] absolute pixels
    confidence: float
    class_id: int = 0


class ByteTracker:
    """
    supervision.ByteTrack wrapper.

    Accepts raw face detections (xyxy boxes + scores) and returns confirmed
    Track objects with stable integer track IDs across frames.
    """

    def __init__(
        self,
        track_thresh: Optional[float] = None,
        track_buffer: Optional[int] = None,
        match_thresh: Optional[float] = None,
        frame_rate: Optional[int] = None,
    ) -> None:
        self._track_thresh = track_thresh or settings.bytetrack_track_thresh
        self._track_buffer = track_buffer or settings.bytetrack_track_buffer
        self._match_thresh = match_thresh or settings.bytetrack_match_thresh
        self._frame_rate = frame_rate or settings.bytetrack_frame_rate
        self._tracker = None   # lazy init

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._tracker is not None:
            return
        try:
            import supervision as sv
        except ImportError as e:
            raise RuntimeError("supervision is not installed") from e

        self._sv = sv
        self._tracker = sv.ByteTrack(
            track_activation_threshold=self._track_thresh,
            lost_track_buffer=self._track_buffer,
            minimum_matching_threshold=self._match_thresh,
            frame_rate=self._frame_rate,
            minimum_consecutive_frames=1,
        )
        logger.info(
            "ByteTracker ready (thresh=%.2f, buffer=%d, match=%.2f, fps=%d)",
            self._track_thresh, self._track_buffer, self._match_thresh, self._frame_rate,
        )

    def reset(self) -> None:
        """Re-initialise the tracker (use between sessions)."""
        self._tracker = None
        self._ensure_loaded()

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        boxes: np.ndarray,      # (N, 4) xyxy float32
        scores: np.ndarray,     # (N,)   float32
        class_ids: Optional[np.ndarray] = None,  # (N,) int; defaults to 0
    ) -> List[Track]:
        """
        Feed detections for one frame; return confirmed tracks.

        Returns an empty list when no detections or no confirmed tracks.
        """
        self._ensure_loaded()

        if len(boxes) == 0:
            # Feed an empty Detections to keep the track buffer updated
            empty = self._sv.Detections.empty()
            self._tracker.update_with_detections(empty)
            return []

        if class_ids is None:
            class_ids = np.zeros(len(boxes), dtype=int)

        dets = self._sv.Detections(
            xyxy=boxes.astype(np.float32),
            confidence=scores.astype(np.float32),
            class_id=class_ids.astype(int),
        )
        tracked = self._tracker.update_with_detections(dets)

        if tracked.tracker_id is None or len(tracked) == 0:
            return []

        return [
            Track(
                track_id=int(tracked.tracker_id[i]),
                bbox_xyxy=tracked.xyxy[i].copy(),
                confidence=float(tracked.confidence[i]),
                class_id=int(tracked.class_id[i]) if tracked.class_id is not None else 0,
            )
            for i in range(len(tracked))
        ]


# ── Module-level singleton ────────────────────────────────────────────────────

_tracker: Optional[ByteTracker] = None


def get_tracker() -> ByteTracker:
    global _tracker
    if _tracker is None:
        _tracker = ByteTracker()
        _tracker._ensure_loaded()
    return _tracker
