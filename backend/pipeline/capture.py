from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Callable, Iterator, Optional, Tuple

import cv2
import numpy as np

from backend.config import get_settings

logger = logging.getLogger(__name__)

# (frame_number, timestamp_ms, frame_bgr)
FrameTuple = Tuple[int, int, np.ndarray]

_STREAM_SCHEMES = ("rtsp://", "rtsps://", "http://", "https://", "udp://", "tcp://")

# FFMPEG options applied when opening a network stream. RTSP over UDP (the
# default) silently drops packets on a congested LAN and produces torn frames;
# TCP trades a little latency for intact ones. The timeout is in MICROseconds
# and is what stops cv2.VideoCapture() from blocking forever on a dead host.
#
# The option is `timeout`, NOT the `stimeout` every older RTSP snippet uses:
# FFmpeg renamed it, and the build OpenCV bundles (avformat 59.x behind
# opencv-python 4.9-4.10) no longer recognises the old spelling. An unknown key
# here fails silently — measured against a dead host, `stimeout;5000000` opened
# in 29.9s, exactly matching no options at all, while `timeout;5000000` gave up
# in 5.7s as intended.
_FFMPEG_BASE_OPTS = "timeout;5000000"


def _stream_opts(transport: str) -> str:
    """FFMPEG capture options for one open attempt.

    `auto` omits rtsp_transport entirely and lets FFmpeg negotiate, which is the
    right thing for a server that advertises only one of the two.
    """
    if transport in ("tcp", "udp"):
        return f"rtsp_transport;{transport}|{_FFMPEG_BASE_OPTS}"
    return _FFMPEG_BASE_OPTS


def is_stream_source(source: str | Path | int) -> bool:
    """True for a network stream or a camera device index, False for a file."""
    if isinstance(source, int):
        return True
    text = str(source).strip()
    return text.isdigit() or text.lower().startswith(_STREAM_SCHEMES)


class StreamDropped(RuntimeError):
    """A live source stopped delivering frames.

    Raised instead of ending iteration so callers can tell a genuine end-of-file
    apart from a camera that went away — for a live source they are the same
    cv2 signal (read() returning False) but need opposite responses.
    """


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
        self.is_stream = is_stream_source(source)

        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 25.0
        self._total_frames: int = 0
        self._width: int = 0
        self._height: int = 0
        self._epoch: Optional[float] = None   # wall-clock origin for live sources
        self._frame_offset: int = 0           # keeps frame numbers monotonic across reconnects

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _open_stream(self, transport: str) -> cv2.VideoCapture:
        """One open attempt with a given RTSP transport.

        OpenCV reads this env var when the FFMPEG backend constructs the
        capture, so it has to be set *before* VideoCapture() — there is no
        per-instance API for it. Restored afterwards so a concurrent open of a
        plain file is unaffected.
        """
        prev = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _stream_opts(transport)
        try:
            if self.source.isdigit():
                return cv2.VideoCapture(int(self.source))
            return cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        finally:
            if prev is None:
                os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
            else:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = prev

    def open(self) -> "VideoCapture":
        if self.is_stream:
            # Forcing TCP is right for a congested LAN but fatal against a
            # server that only speaks RTP-over-UDP — VLC's `#rtp{sdp=rtsp://…}`
            # output and plenty of IP cameras refuse the interleave and the
            # open fails in ~0.1s. Falling back costs almost nothing and is the
            # difference between a working camera and a session that dies with
            # zero frames.
            attempts = [get_settings().live_rtsp_transport]
            if attempts[0] == "tcp":
                attempts.append("udp")
            for transport in attempts:
                self._cap = self._open_stream(transport)
                if self._cap.isOpened():
                    if transport != attempts[0]:
                        logger.info("%r: %s refused, opened over %s",
                                    self.source, attempts[0], transport)
                    break
                self._cap.release()
            else:
                raise IOError(
                    f"Cannot open video source: {self.source!r} "
                    f"(tried rtsp_transport {', '.join(attempts)})"
                )
        else:
            self._cap = cv2.VideoCapture(self.source)
            if not self._cap.isOpened():
                raise IOError(f"Cannot open video source: {self.source!r}")

        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.is_stream:
            # Keep only the newest decoded frame. Without this the FFMPEG
            # backend queues frames while inference runs, and a feed processed
            # slower than it arrives drifts steadily further behind real time
            # until "live" is minutes stale.
            try:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:      # not all backends honour it
                pass
            # A stream reports no frame count, and often a nonsense fps.
            if not (1.0 <= self._fps <= 120.0):
                self._fps = 25.0
            # Set once per source, never on reconnect: the timestamps feed
            # behaviour dwell timers, and restarting the clock mid-session would
            # rewind them and reset every in-progress DISTRACTED/SLEEPING timer.
            if self._epoch is None:
                self._epoch = time.monotonic()

        logger.info(
            "Opened %r (%s): %dx%d @ %.1f fps, %d total frames",
            self.source, "stream" if self.is_stream else "file",
            self._width, self._height, self._fps, self._total_frames,
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

        frame_number = self._frame_offset
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
                if self.is_stream:
                    # For a file this is end-of-file; for a live source it means
                    # the feed died. Ending iteration here is what made
                    # frames_with_reconnect() a no-op: it only retries on an
                    # exception, and a returning generator looks like clean EOF.
                    raise StreamDropped(
                        f"Stream stopped delivering frames: {self.source!r}"
                    )
                break

            # Drop interleaved frames without counting them as yielded
            if not keep:
                frame_number += 1
                self._frame_offset = frame_number
                continue

            if self.resize_max_side:
                frame = _resize_keep_aspect(frame, self.resize_max_side)

            if self.is_stream:
                # Wall-clock, not frame_number/fps. Behaviour dwell timers
                # (_DISTRACTED_SECONDS, _SLEEPING_SECONDS) measure elapsed time
                # from these stamps; on a live feed that stalls, reconnects, or
                # runs off-nominal fps, a derived clock drifts from reality and
                # silently rescales every one of those thresholds.
                timestamp_ms = int((time.monotonic() - (self._epoch or time.monotonic())) * 1000)
            else:
                timestamp_ms = int(frame_number / self._fps * 1000)
            yield frame_number, timestamp_ms, frame

            frame_number += 1
            self._frame_offset = frame_number
            yielded += 1
            if self.max_frames and yielded >= self.max_frames:
                break

    def frames_with_reconnect(
        self,
        reconnect_delay: float = 2.0,
        max_reconnects: int = 10,
        max_backoff: float = 30.0,
        on_state: Optional["Callable[..., None]"] = None,
    ) -> Iterator[FrameTuple]:
        """Like frames() but reconnects when a live source drops.

        Backs off exponentially: a camera rebooting or a network blip usually
        recovers in seconds, but hammering a genuinely dead host every 2s for ten
        tries just burns the retry budget in 20 seconds and gives up.

        on_state, if given, is called as on_state(event, attempt=int, delay=float)
        with event in {"dropped", "reconnected", "exhausted"}. Without it the
        retries are invisible to the caller, which would leave an operator
        watching a "connected" dashboard while the camera is actually gone.
        """
        def _notify(event: str, **kw) -> None:
            if on_state is None:
                return
            try:
                on_state(event, **kw)
            except Exception:                 # a broken callback must not stop capture
                logger.debug("on_state callback failed", exc_info=True)

        reconnects = 0
        while reconnects <= max_reconnects:
            try:
                yield from self.frames()
                return                      # genuine end-of-file
            except StreamDropped as exc:
                reconnects += 1
                if reconnects > max_reconnects:
                    break
                delay = min(reconnect_delay * (2 ** (reconnects - 1)), max_backoff)
                logger.warning(
                    "Capture dropped (%s) – reconnect %d/%d in %.1fs",
                    exc, reconnects, max_reconnects, delay,
                )
                _notify("dropped", attempt=reconnects, delay=delay, error=str(exc))
                self.close()
                time.sleep(delay)
                try:
                    self.open()
                except IOError as open_exc:
                    logger.warning("Reconnect %d failed to open: %s", reconnects, open_exc)
                    continue
                _notify("reconnected", attempt=reconnects, delay=0.0)
        _notify("exhausted", attempt=reconnects, delay=0.0)
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
