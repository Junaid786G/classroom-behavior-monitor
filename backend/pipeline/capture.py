from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# (frame_number, timestamp_ms, frame_bgr)
FrameTuple = Tuple[int, int, np.ndarray]


class VideoCapture:
    """
    Thin wrapper around cv2.VideoCapture supporting files and RTSP streams.

    Usage
    -----
    with VideoCapture("lecture.mp4", skip_frames=2) as cap:
        for frame_no, ts_ms, frame in cap.frames():
            ...
    """

    def __init__(
        self,
        source: str | Path | int,
        skip_frames: int = 0,
        max_frames: Optional[int] = None,
        resize_max_side: Optional[int] = None,
    ) -> None:
        self.source = str(source)
        self.skip_frames = max(0, skip_frames)
        self.max_frames = max_frames
        self.resize_max_side = resize_max_side

        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 25.0
        self._total_frames: int = 0
        self._width: int = 0
        self._height: int = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> "VideoCapture":
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video source: {self.source!r}")

        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            "Opened %r: %dx%d @ %.1f fps, %d total frames",
            self.source, self._width, self._height, self._fps, self._total_frames,
        )
        return self

    def close(self) -> None:
        if self._cap and self._cap.isOpened():
            self._cap.release()
        self._cap = None

    def __enter__(self) -> "VideoCapture":
        return self.open()

    def __exit__(self, *_) -> None:
        self.close()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def duration_seconds(self) -> float:
        return self._total_frames / self._fps if self._fps > 0 else 0.0

    @property
    def resolution(self) -> str:
        return f"{self._width}x{self._height}"

    # ── Iteration ─────────────────────────────────────────────────────────────

    def frames(self) -> Iterator[FrameTuple]:
        """
        Yield (frame_number, timestamp_ms, bgr_frame).

        *frame_number* counts every decoded frame; *skip_frames* controls how
        many are silently dropped between yields.
        """
        if self._cap is None:
            raise RuntimeError("Call open() first or use as a context manager")

        frame_number = 0
        yielded = 0

        while True:
            keep = not self.skip_frames or frame_number % (self.skip_frames + 1) == 0

            if keep:
                # Decode only the frames we actually yield.
                ret, frame = self._cap.read()
            else:
                # Advance past skipped frames without decoding (cheap: no pixel copy).
                ret = self._cap.grab()
            if not ret:
                break

            # Drop interleaved frames without counting them as yielded
            if not keep:
                frame_number += 1
                continue

            if self.resize_max_side:
                frame = _resize_keep_aspect(frame, self.resize_max_side)

            timestamp_ms = int(frame_number / self._fps * 1000)
            yield frame_number, timestamp_ms, frame

            frame_number += 1
            yielded += 1
            if self.max_frames and yielded >= self.max_frames:
                break

    def frames_with_reconnect(
        self,
        reconnect_delay: float = 2.0,
        max_reconnects: int = 10,
    ) -> Iterator[FrameTuple]:
        """Like frames() but silently reconnects on RTSP drop."""
        reconnects = 0
        while reconnects <= max_reconnects:
            try:
                yield from self.frames()
                break       # clean EOF
            except Exception as exc:
                reconnects += 1
                logger.warning(
                    "Capture error (%s) – reconnect %d/%d in %.1fs",
                    exc, reconnects, max_reconnects, reconnect_delay,
                )
                self.close()
                time.sleep(reconnect_delay)
                try:
                    self.open()
                except IOError:
                    continue
        else:
            logger.error("Max reconnect attempts reached for %r", self.source)


# ── Standalone helpers ────────────────────────────────────────────────────────

def get_video_metadata(path: str | Path) -> dict:
    """Return video metadata without iterating frames."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {}
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "fps": fps,
        "total_frames": total,
        "width": w,
        "height": h,
        "duration_seconds": total / fps if fps else 0.0,
        "resolution": f"{w}x{h}",
    }


def _resize_keep_aspect(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), cv2.INTER_LINEAR)
