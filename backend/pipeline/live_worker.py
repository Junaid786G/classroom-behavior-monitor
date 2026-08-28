"""
live_worker — server-side RTSP ingestion running the full recognition pipeline.

The recorded-video path (`stream.py::_process_video_bg`) is an `async def` that
runs cv2 decode and ONNX inference inline on the event loop. That merely stalls
the API for the length of a finite video; for a *continuous* camera feed it would
block the loop for as long as the camera stays connected, freezing health checks,
websockets and every HTTP request. So the live loop runs on its own thread and
marshals its database writes back to the event loop with
`asyncio.run_coroutine_threadsafe`.

Concurrency policy: exactly one live session at a time, enforced by the registry
at the bottom of this module. `FaceRecognizer`, `ByteTracker` and
`BehaviorAnalyzer` are process-wide singletons holding mutable per-track state
(track ids, EAR/yaw history, dwell timers); two concurrent consumers would
interleave into the same dicts and corrupt both. Lifting that restriction means
giving each worker its own tracker and analyzer — see DEPLOYMENT/README notes.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

import cv2

from backend.config import get_settings
from backend.pipeline.annotator import annotate_frame
from backend.pipeline.behavior import get_behavior_analyzer
from backend.pipeline.capture import StreamDropped, VideoCapture, is_stream_source

logger = logging.getLogger(__name__)
settings = get_settings()


class LiveStartError(RuntimeError):
    """A live session could not be started (bad URL, or one already running)."""


@dataclass
class LiveSnapshot:
    """Point-in-time view of a worker, safe to serialise to the UI."""
    session_id: str
    rtsp_url: str
    state: str                      # starting | running | reconnecting | stopping | stopped | error
    frames_processed: int = 0
    frames_persisted: int = 0
    reconnects: int = 0
    connected: bool = False
    last_error: Optional[str] = None
    started_at: Optional[str] = None
    last_frame_at: Optional[str] = None
    processed_fps: float = 0.0
    # Increments once per published frame. Consumers compare it to decide
    # whether anything is actually new; frames_processed is not a substitute,
    # because a frame whose annotate/encode failed advances that counter
    # without producing a new image.
    frame_seq: int = 0
    detections: List[dict] = field(default_factory=list)
    frame_b64: Optional[str] = None


class LiveSessionWorker:
    """Owns one RTSP capture loop on a dedicated thread."""

    def __init__(
        self,
        session_id: UUID,
        rtsp_url: str,
        loop: asyncio.AbstractEventLoop,
        student_map: Dict[int, str],
        roster_ids: frozenset,
        late_after: Optional[datetime] = None,
    ) -> None:
        self.session_id = session_id
        self.rtsp_url = rtsp_url
        self._loop = loop
        self._student_map = student_map
        self._roster_ids = roster_ids
        self._late_after = late_after

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        # ── guarded by _lock ──
        self._state = "starting"
        self._frames = 0
        self._persisted = 0
        self._reconnects = 0
        self._connected = False
        self._last_error: Optional[str] = None
        self._started_at: Optional[datetime] = None
        self._last_frame_at: Optional[datetime] = None
        self._latest_b64: Optional[str] = None
        self._latest_dets: List[dict] = []
        self._frame_seq = 0
        self._recent_times: List[float] = []

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._started_at = datetime.now(timezone.utc)
        self._thread = threading.Thread(
            target=self._run, name=f"live-{self.session_id}", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 15.0) -> None:
        """Ask the capture loop to finish and wait for it.

        Cooperative: the loop checks the flag between frames, so it always stops
        on a frame boundary with its batch flushed rather than mid-write.

        MUST NOT be called from the event-loop thread. This blocks in join(),
        while the capture thread's final flush and end_session need that same
        loop to run — calling it inline deadlocks until the join times out and
        the session is left un-closed. Call it via asyncio.to_thread(), as
        stream.stop_live does.
        """
        if not self.is_running:
            # Already finished. Do NOT stamp "stopping" over the terminal state:
            # that would erase the state and last_error that /live/status exists
            # to report, turning "camera refused the connection" into a blank
            # "stopping" the operator cannot act on.
            self._stop.set()
            return
        self._set(state="stopping")
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("live session=%s thread did not stop in %.0fs",
                               self.session_id, timeout)

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ── status ───────────────────────────────────────────────────────────────

    def _set(self, **kw: Any) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self, f"_{k}", v)

    def snapshot(self, include_frame: bool = True) -> LiveSnapshot:
        with self._lock:
            return LiveSnapshot(
                session_id=str(self.session_id),
                rtsp_url=self.rtsp_url,
                state=self._state,
                frames_processed=self._frames,
                frames_persisted=self._persisted,
                reconnects=self._reconnects,
                connected=self._connected,
                last_error=self._last_error,
                started_at=self._started_at.isoformat() if self._started_at else None,
                last_frame_at=self._last_frame_at.isoformat() if self._last_frame_at else None,
                processed_fps=self._fps_locked(),
                frame_seq=self._frame_seq,
                detections=list(self._latest_dets),
                frame_b64=self._latest_b64 if include_frame else None,
            )

    def _fps_locked(self) -> float:
        """Processed-frame rate over the recent window. Caller holds _lock."""
        if len(self._recent_times) < 2:
            return 0.0
        span = self._recent_times[-1] - self._recent_times[0]
        return (len(self._recent_times) - 1) / span if span > 0 else 0.0

    # ── event-loop bridge ────────────────────────────────────────────────────

    def _await(self, coro, timeout: float = 30.0):
        """Run a coroutine on the API's event loop from this worker thread."""
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)

    # ── thread body ──────────────────────────────────────────────────────────

    def _run(self) -> None:
        from backend.routers import stream as stream_router   # circular at import time

        recognizer = self._await(stream_router.get_recognizer())
        analyzer = get_behavior_analyzer()

        # Single-session policy makes this safe: no other consumer is mutating
        # this state concurrently, so a clean slate here means track ids and
        # dwell timers start from zero rather than inheriting a previous session.
        recognizer.reset()
        analyzer.reset_all()

        # main.py's lifespan normally warms this. Guard anyway: without it
        # _ready stays False and analyze_frame returns UNKNOWN for every track
        # for the entire session, silently producing behaviour rows that carry
        # no information.
        if not getattr(analyzer, "_ready", False):
            logger.info("live session=%s warming BehaviorAnalyzer", self.session_id)
            try:
                analyzer.warmup()
            except Exception as exc:
                logger.error("BehaviorAnalyzer warmup failed: %s", exc)

        cap = VideoCapture(self.rtsp_url, skip_frames=settings.live_frame_skip)
        batch: Dict[str, List[dict]] = {"det": [], "beh": [], "att": []}

        try:
            cap.open()
        except IOError as exc:
            self._set(state="error", last_error=str(exc), connected=False)
            logger.error("live session=%s cannot open %r: %s",
                         self.session_id, self.rtsp_url, exc)
            self._finalise(failed=True)
            return

        self._set(state="running", connected=True)
        logger.info("live session=%s connected to %s (%s @ %.1ffps)",
                    self.session_id, self.rtsp_url, cap.resolution, cap.fps)

        try:
            for frame_number, timestamp_ms, frame in cap.frames_with_reconnect(
                reconnect_delay=settings.live_reconnect_delay,
                max_reconnects=settings.live_max_reconnects,
                on_state=self._on_capture_state,
            ):
                if self._stop.is_set():
                    break

                # Reconnect bookkeeping: frames_with_reconnect logs and retries
                # internally, so surface the transition by watching the counter.
                self._note_frame()

                results = recognizer.process_frame(
                    frame, frame_number, timestamp_ms, self._student_map
                )
                stream_router._gate_to_roster(results, self._roster_ids)
                behaviors = analyzer.analyze_frame(frame, results, timestamp_ms)
                bmap = {b.track_id: b for b in behaviors}

                self._accumulate(batch, frame, frame_number, timestamp_ms, results, bmap)
                self._publish(frame, results, behaviors, bmap)

                if len(batch["det"]) >= settings.live_flush_every:
                    self._flush(batch)

            self._flush(batch)

        except StreamDropped as exc:
            # Escaped the reconnect loop => retry budget exhausted.
            self._set(state="error", connected=False,
                      last_error=f"stream lost, reconnects exhausted: {exc}")
            logger.error("live session=%s gave up: %s", self.session_id, exc)
            self._flush(batch)
        except Exception as exc:
            self._set(state="error", connected=False, last_error=str(exc))
            logger.exception("live session=%s failed: %s", self.session_id, exc)
            self._flush(batch)
        finally:
            cap.close()
            failed = self.snapshot(include_frame=False).state == "error"
            self._finalise(failed=failed)

    # ── per-frame helpers ────────────────────────────────────────────────────

    def _on_capture_state(self, event: str, attempt: int = 0,
                          delay: float = 0.0, error: str = "") -> None:
        """Surface capture retries into the status the UI polls.

        Without this the retry loop is entirely internal to VideoCapture, and a
        dashboard keeps reporting `running / connected` while the camera has been
        gone for however long the backoff lasts.
        """
        with self._lock:
            if event == "dropped":
                self._state = "reconnecting"
                self._connected = False
                self._reconnects = attempt
                self._last_error = f"stream dropped, retry {attempt} in {delay:.0f}s: {error}"
            elif event == "reconnected":
                self._state = "running"
                self._connected = True
                self._last_error = None
            elif event == "exhausted":
                self._state = "error"
                self._connected = False
                self._last_error = f"stream lost; {attempt} reconnect attempts exhausted"

    def _note_frame(self) -> None:
        now = time.monotonic()
        with self._lock:
            self._frames += 1
            self._last_frame_at = datetime.now(timezone.utc)
            self._connected = True
            if self._state == "reconnecting":
                self._state = "running"
            self._recent_times.append(now)
            if len(self._recent_times) > 30:
                del self._recent_times[:-30]

    def _accumulate(self, batch, frame, frame_number, timestamp_ms, results, bmap) -> None:
        from backend.routers.stream import _norm_bbox_xywh

        h, w = frame.shape[:2]
        for r in results:
            x, y, bw, bh = _norm_bbox_xywh(r.bbox_xyxy, (h, w))
            batch["det"].append(dict(
                session_id=self.session_id,
                student_id=r.student_id,
                track_id=r.track_id,
                frame_number=frame_number,
                timestamp_ms=timestamp_ms,
                bbox_x=x, bbox_y=y, bbox_w=bw, bbox_h=bh,
                detection_confidence=r.detection_score,
                recognition_confidence=r.recognition_score if r.student_id else None,
            ))
            bf = bmap.get(r.track_id)
            if bf:
                batch["beh"].append(dict(
                    session_id=self.session_id,
                    student_id=r.student_id,
                    track_id=r.track_id,
                    behavior_type=bf.behavior,
                    start_frame=frame_number,
                    end_frame=frame_number,
                    start_ms=timestamp_ms,
                    end_ms=timestamp_ms,
                    confidence=bf.confidence,
                    pose_metrics=bf.pose_metrics,
                ))
            if r.student_id and r.recognition_score >= settings.attendance_confidence_threshold:
                batch["att"].append(dict(
                    student_id=r.student_id,
                    confidence=r.recognition_score,
                    timestamp=datetime.now(timezone.utc),
                ))

    def _publish(self, frame, results, behaviors, bmap) -> None:
        """Store the newest annotated frame + detections for the UI to poll."""
        from backend.routers.stream import _norm_bbox

        try:
            annotated = annotate_frame(frame, results, behaviors)
            ok, buf = cv2.imencode(
                ".jpg", annotated,
                [cv2.IMWRITE_JPEG_QUALITY, settings.live_jpeg_quality],
            )
            b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok else None
        except Exception as exc:                # never let rendering kill capture
            logger.debug("annotate/encode failed: %s", exc)
            b64 = None

        dets = [
            dict(
                track_id=r.track_id,
                student_id=r.student_id,
                student_name=r.student_name,
                bbox=_norm_bbox(r.bbox_xyxy, frame.shape),
                det_confidence=r.detection_score,
                rec_confidence=r.recognition_score if r.student_id else None,
                behavior=bmap[r.track_id].behavior.value if r.track_id in bmap else None,
            )
            for r in results
        ]
        with self._lock:
            if b64 is not None:
                self._latest_b64 = b64
            self._latest_dets = dets
            # Single-slot conflation: this overwrites the previous frame rather
            # than queueing it. A consumer slower than the camera therefore
            # drops frames and always sees the newest one, and can never apply
            # backpressure to capture — the only thing shared with a reader is
            # this lock, held for these three assignments.
            self._frame_seq += 1

    # ── persistence ──────────────────────────────────────────────────────────

    def _flush(self, batch: Dict[str, List[dict]]) -> None:
        if not any(batch.values()):
            return
        det, beh, att = batch["det"], batch["beh"], batch["att"]
        try:
            frames_in_batch = len({d["frame_number"] for d in det})
            self._await(self._persist(list(det), list(beh), list(att)))
            with self._lock:
                self._persisted += frames_in_batch
        except Exception as exc:
            # A failed flush must not kill the capture loop: the camera keeps
            # running and the next flush may well succeed.
            logger.warning("live session=%s flush failed: %s", self.session_id, exc)
            self._set(last_error=f"persist failed: {exc}")
        finally:
            det.clear(); beh.clear(); att.clear()

    async def _persist(self, det: List[dict], beh: List[dict], att: List[dict]) -> None:
        from backend import crud
        from backend.database import AsyncSessionLocal

        async with AsyncSessionLocal() as db:
            if det:
                await crud.bulk_insert_face_detections(db, det)
            for b in beh:
                await crud.create_behavior_event(db, b)
            for a in att:
                await crud.mark_attendance(
                    db, self.session_id, a["student_id"], a["confidence"],
                    a["timestamp"], late_after=self._late_after,
                )
            if det:
                # by frames, not rows: len(det) counts one row per detected face,
                # so a 15-student room would inflate the counter 15x per frame.
                await crud.increment_frame_count(
                    db, self.session_id, by=len({d["frame_number"] for d in det})
                )

    def _finalise(self, failed: bool) -> None:
        from backend import crud
        from backend.database import AsyncSessionLocal
        from backend.models import SessionStatus

        async def _end() -> None:
            async with AsyncSessionLocal() as db:
                await crud.end_session(
                    db, self.session_id,
                    status=SessionStatus.FAILED if failed else SessionStatus.COMPLETED,
                )

        try:
            self._await(_end(), timeout=20.0)
        except Exception as exc:
            logger.warning("live session=%s could not close out: %s", self.session_id, exc)

        with self._lock:
            self._connected = False
            if self._state != "error":
                self._state = "stopped"
        # Deliberately NOT unregistered. A worker that unregisters itself takes
        # its own post-mortem with it: /live/status then 404s, and the UI can
        # only say "lost contact" for what was really "cannot open video
        # source". Leaving the finished worker in the registry keeps state and
        # last_error readable. This does not hold the live slot — register()
        # reaps workers that are no longer running, and active_worker() filters
        # on is_running — so the next session starts normally.
        logger.info("live session=%s finished: %d frames, %d reconnects",
                    self.session_id, self._frames, self._reconnects)


# ── Registry — one live session at a time ────────────────────────────────────

_workers: Dict[UUID, LiveSessionWorker] = {}
_registry_lock = threading.Lock()


def get_worker(session_id: UUID) -> Optional[LiveSessionWorker]:
    with _registry_lock:
        return _workers.get(session_id)


def active_worker() -> Optional[LiveSessionWorker]:
    with _registry_lock:
        for w in _workers.values():
            if w.is_running:
                return w
    return None


def register(worker: LiveSessionWorker) -> None:
    """Add a worker, refusing a second concurrent live session."""
    with _registry_lock:
        for existing in list(_workers.values()):
            if existing.is_running:
                raise LiveStartError(
                    f"A live session is already running (session {existing.session_id}). "
                    "Stop it before starting another — the recognition pipeline keeps "
                    "per-track state in process-wide singletons and cannot serve two "
                    "live feeds at once."
                )
            _workers.pop(existing.session_id, None)      # reap finished workers
        _workers[worker.session_id] = worker


def unregister(session_id: UUID) -> None:
    with _registry_lock:
        _workers.pop(session_id, None)


def validate_stream_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        raise LiveStartError("Stream URL is required")
    if not is_stream_source(url):
        raise LiveStartError(
            f"Not a live stream URL: {url!r}. Expected rtsp://, rtsps://, http(s)://, "
            "udp://, tcp:// or a camera index."
        )
    return url
