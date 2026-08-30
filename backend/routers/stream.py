from __future__ import annotations

import asyncio
import logging
import shutil
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID

import numpy as np
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend import crud
from backend.config import get_settings
from backend.database import AsyncSessionLocal, get_db
from backend.deps import (
    assignment_pairs,
    authenticate_ws,
    get_current_user,
    require_instructor,
    require_session_access,
    require_training_control,
)
from backend.models import (
    AlertSeverity,
    AlertType,
    AttendanceStatus,
    BehaviorType,
    SessionStatus,
    User,
    UserRole,
    VideoStatus,
)
from backend.pipeline.annotator import annotate_frame
from backend.pipeline.behavior import BehaviorFrame, get_behavior_analyzer
from backend.pipeline.capture import VideoCapture, get_video_metadata
from backend.pipeline.recognizer import FaceRecognizer, RecognitionResult, get_recognizer
from backend.pipeline.live_worker import (
    LiveSessionWorker,
    LiveStartError,
    active_worker,
    get_worker,
    register as register_live_worker,
    validate_stream_url,
)
from backend.schemas import (
    BulkSessionDeleteRequest,
    SessionDeleteRequest,
    SessionDeleteResponse,
    LiveStartRequest,
    LiveStatusOut,
    Page,
    SessionCreate,
    SessionOut,
    VideoUploadOut,
    VideoUploadResponse,
    WSDetection,
    WSFrameResult,
)
from backend.utils.image import frame_to_b64_jpeg, jpeg_bytes_to_frame

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(tags=["sessions", "stream"])

# ── Session CRUD endpoints ────────────────────────────────────────────────────

@router.post(
    "/sessions",
    response_model=SessionOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_instructor)],
)
async def create_session(
    data: SessionCreate,
    db: AsyncSession = Depends(get_db),
):
    # Validate up front so a bad subject_id is a 422, not a raw FK-violation 500.
    if await crud.get_subject(db, data.subject_id) is None:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Subject {data.subject_id} does not exist",
        )
    session = await crud.create_session(db, data)
    return SessionOut.model_validate(session)


# Any authenticated role - the Home page lands all four on it, and Attendance
# and both dashboards pick from it - but an INSTRUCTOR sees only sessions in
# the subjects they are assigned. Without that, the list offered them 66
# sessions of which 65 answered 404 when opened, since require_session_access
# scopes every per-session read. An instructor with no assignments gets an
# empty list, not the department's history.
@router.get("/sessions", response_model=Page)
async def list_sessions(
    classroom_id: Optional[int] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    subject_ids = (
        [subject_id for _, subject_id in assignment_pairs(user)]
        if user.role is UserRole.INSTRUCTOR
        else None
    )
    # A STUDENT sees the sessions they were recorded in, and nothing else. The
    # Home page shows this list to every role, so without this a student reads
    # the department's timetable - titles, instructors, dates.
    attended_by = (
        user.linked_student_id if user.role is UserRole.STUDENT else None
    )
    total, rows = await crud.list_sessions(
        db,
        classroom_id=classroom_id,
        subject_ids=subject_ids,
        attended_by_student_id=attended_by,
        skip=skip,
        limit=limit,
    )
    return Page(total=total, skip=skip, limit=limit, items=[SessionOut.model_validate(r) for r in rows])


# ── Deletion ──────────────────────────────────────────────────────────────────
# HOD IS ABSENT FROM BOTH ROUTES, AND THAT IS THE POINT. models.py defines the
# role as "read-only oversight", and permissions.py already draws this exact
# line for a far smaller power: ATTENDANCE_WRITERS excludes HOD so that page
# access alone cannot hand them the Manual Override. A button that destroys up
# to 168,000 rows contradicts "read-only" considerably harder than that did.
#
# The split between the two routes is by JOB, not by seniority:
#   * one session is a bad recording, which is the instructor's own to remove,
#     scoped by require_session_access to the subjects they are assigned;
#   * a whole course is semester rollover, which is setup — TRAINING_CONTROL's
#     entire remit — and by definition exceeds any one instructor's scope.

#: Statuses that must never be deleted. SessionStatus has no RUNNING: the live
#: pipeline marks a session PROCESSING, so that is the one to refuse. PENDING is
#: deletable — it was created and never started, which is exactly the row most
#: worth clearing up.
_UNDELETABLE = frozenset({SessionStatus.PROCESSING})


async def _refuse_if_live(session_id: UUID, session) -> None:
    """Refuse a session that is being written to right now.

    TWO checks, because either can be true without the other. The stored status
    is the durable answer, but a worker that died without updating its row
    leaves a session reading COMPLETED while the registry still holds it; and a
    row can read PROCESSING months after the process that set it has gone (there
    is one such session in this database, stuck since 2026-07-13). Deleting a
    session under a running worker means the worker's next insert violates a
    foreign key that no longer has a parent.
    """
    if session.status in _UNDELETABLE:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Session is {session.status.value} and cannot be deleted. Stop it "
            f"first — a session still being written to would take its own new "
            f"rows down with it.",
        )
    worker = get_worker(session_id)
    if worker is not None and worker.is_running:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "A live worker is still running for this session. Stop the live "
            "feed before deleting it.",
        )


@router.delete(
    "/sessions/{session_id}",
    response_model=SessionDeleteResponse,
    dependencies=[Depends(require_instructor), Depends(require_session_access)],
)
async def delete_session(
    session_id: UUID,
    payload: SessionDeleteRequest,
    db: AsyncSession = Depends(get_db),
):
    """Delete one session and everything recorded against it. IRREVERSIBLE.

    The two dependencies are ordered deliberately: require_instructor runs
    first, so a HOD is refused as a role before require_session_access — which
    returns early for HOD — would have waved them through.

    The cascade is Postgres's, not the ORM's (see crud.delete_session). It
    reaches further than this router: attendance_records go with the session,
    and crud.get_student_overview builds the student portal's Session History
    and By Subject FROM attendance_records, so the deleted session leaves every
    affected student's own view at the same time. That is verified end to end in
    backend/tests/test_session_deletion.py rather than assumed from the schema.
    """
    session = await crud.get_session(db, session_id)
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")

    await _refuse_if_live(session_id, session)

    deleted = await crud.delete_session(db, session_id)
    if not deleted:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    logger.warning(
        "session %s deleted (subject=%s status=%s)",
        session_id, session.subject_id, session.status.value,
    )
    return SessionDeleteResponse(deleted=deleted, session_ids=[session_id])


@router.post(
    "/courses/{course_id}/sessions/delete-all",
    response_model=SessionDeleteResponse,
    dependencies=[Depends(require_training_control)],
)
async def delete_course_sessions(
    course_id: int,
    payload: BulkSessionDeleteRequest,
    db: AsyncSession = Depends(get_db),
):
    """Delete EVERY session of one course, or of one subject within it.

    Semester rollover. POST rather than DELETE because it carries a body that
    every client must send, and because it names an action rather than a
    resource.

    THREE guards, none of them redundant:
      * `confirm` must be the literal "DELETE" — see BulkSessionDeleteRequest;
      * `expected_count` must equal what is actually about to go, so a rollover
        that races a session created seconds earlier fails instead of silently
        taking one more than the operator was shown;
      * no matched session may be live, checked per session by _refuse_if_live.
        One PROCESSING session blocks the whole batch rather than being skipped:
        a partial wipe is the outcome hardest to reason about afterwards.
    """
    if await crud.get_course(db, course_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Course {course_id} not found")

    sessions = await crud.list_course_sessions(db, course_id, payload.subject_id)

    if len(sessions) != payload.expected_count:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Expected to delete {payload.expected_count} session(s) but found "
            f"{len(sessions)}. Refresh and try again — the set changed since it "
            f"was shown to you.",
        )

    for session in sessions:
        await _refuse_if_live(session.id, session)

    ids = [s.id for s in sessions]
    deleted = await crud.delete_course_sessions(db, course_id, payload.subject_id)
    logger.warning(
        "BULK DELETE course=%s subject=%s removed %d sessions",
        course_id, payload.subject_id, deleted,
    )
    return SessionDeleteResponse(deleted=deleted, session_ids=ids)


@router.get(
    "/sessions/{session_id}",
    response_model=SessionOut,
    dependencies=[Depends(get_current_user)],
)
async def get_session(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    s = await crud.get_session(db, session_id)
    if s is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    return SessionOut.model_validate(s)


# ── Video upload + background processing ─────────────────────────────────────

@router.post(
    "/sessions/{session_id}/videos",
    response_model=VideoUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_instructor)],
)
async def upload_video(
    session_id: UUID,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    session = await crud.get_session(db, session_id)
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")

    if not (file.filename or "").lower().endswith((".mp4", ".avi", ".mkv", ".mov", ".webm")):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Unsupported video format")

    # Store file
    dest_dir = settings.video_storage_path / str(session_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    uid = uuid.uuid4()
    ext = Path(file.filename or "video.mp4").suffix or ".mp4"
    dest = dest_dir / f"{uid}{ext}"

    raw = await file.read()
    dest.write_bytes(raw)

    upload = await crud.create_video_upload(
        db, session_id, file.filename or dest.name, str(dest), len(raw)
    )
    task_id = str(uuid.uuid4())
    await crud.update_video_status(db, upload.id, VideoStatus.QUEUED, celery_task_id=task_id)

    # Kick off background processing
    background_tasks.add_task(_process_video_bg, str(upload.id), str(dest), str(session_id))

    return VideoUploadResponse(
        upload=VideoUploadOut.model_validate(upload),
        task_id=task_id,
        message="Video queued for processing",
    )


@router.get(
    "/sessions/{session_id}/videos",
    response_model=Page,
    dependencies=[Depends(require_instructor)],
)
async def list_session_videos(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    rows = await crud.list_session_videos(db, session_id)
    return Page(
        total=len(rows), skip=0, limit=len(rows),
        items=[VideoUploadOut.model_validate(r) for r in rows],
    )


@router.get(
    "/sessions/{session_id}/videos/{upload_id}",
    response_model=VideoUploadOut,
    dependencies=[Depends(require_instructor)],
)
async def get_video_status(
    session_id: UUID,
    upload_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    v = await crud.get_video_upload(db, upload_id)
    if v is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Upload not found")
    return VideoUploadOut.model_validate(v)


# ── Live (RTSP) ingestion ─────────────────────────────────────────────────────

async def _resolve_roster(db: AsyncSession, sid: UUID) -> Tuple[Dict[int, str], frozenset]:
    """Course-scoped roster for a session. Mirrors the upload and WS paths.

    Scoping to the COURSE and not the classroom matters: a room is physical and
    can host several courses, so room scoping would let the department-wide FAISS
    gallery attribute a face to someone not enrolled in this class.
    """
    course_id = await crud.get_session_course_id(db, sid)
    if course_id is None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Session has no subject/course — cannot resolve a roster",
        )
    _, students = await crud.list_students(db, course_id=course_id, limit=10_000)
    student_map = {st.id: st.full_name for st in students}
    return student_map, frozenset(student_map)


@router.post(
    "/sessions/{session_id}/live/start",
    response_model=LiveStatusOut,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_instructor)],
)
async def start_live(
    session_id: UUID,
    body: LiveStartRequest,
    db: AsyncSession = Depends(get_db),
):
    """Open an RTSP URL server-side and run the full pipeline against it.

    Returns as soon as the capture thread is armed; connection happens on that
    thread, so a camera that is slow or unreachable shows up in the `state` and
    `last_error` fields of /live/status rather than hanging this request.
    """
    session = await crud.get_session(db, session_id)
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")

    try:
        url = validate_stream_url(body.rtsp_url)
    except LiveStartError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))

    existing = get_worker(session_id)
    if existing and existing.is_running:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Live capture is already running for session {session_id}",
        )

    student_map, roster_ids = await _resolve_roster(db, session_id)
    await crud.start_session(db, session_id)
    session = await crud.get_session(db, session_id)

    late_after = (
        session.started_at + timedelta(minutes=settings.late_threshold_minutes)
        if session and session.started_at else None
    )

    worker = LiveSessionWorker(
        session_id=session_id,
        rtsp_url=url,
        loop=asyncio.get_running_loop(),
        student_map=student_map,
        roster_ids=roster_ids,
        late_after=late_after,
    )
    try:
        register_live_worker(worker)
    except LiveStartError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc))

    worker.start()
    logger.info("live session=%s started against %s (roster=%d)",
                session_id, url, len(student_map))
    return LiveStatusOut(**worker.snapshot(include_frame=False).__dict__)


@router.post(
    "/sessions/{session_id}/live/stop",
    response_model=LiveStatusOut,
    dependencies=[Depends(require_instructor)],
)
async def stop_live(session_id: UUID):
    """Stop capture and close the session out. Idempotent.

    A finished worker stays in the registry (see live_worker._finalise), so
    stopping an already-stopped session returns its terminal snapshot rather
    than 404, and stop() leaves that state untouched. 404 is now reserved for
    a session that never had a worker, or one already reaped by a later start.
    """
    worker = get_worker(session_id)
    if worker is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"No live capture registered for session {session_id}",
        )
    # join() blocks until the loop reaches a frame boundary and flushes; keep it
    # off the event loop so the API stays responsive while it winds down.
    await asyncio.to_thread(worker.stop)
    return LiveStatusOut(**worker.snapshot(include_frame=False).__dict__)


@router.get(
    "/sessions/{session_id}/live/status",
    response_model=LiveStatusOut,
    dependencies=[Depends(require_instructor)],
)
async def live_status(
    session_id: UUID,
    include_frame: bool = Query(True, description="Include the latest annotated JPEG (base64)"),
):
    """Poll for live state, the newest detections, and the annotated frame.

    This is the UI's read path: the backend owns the capture loop, so the browser
    pulls results instead of pushing frames as it does for uploaded video.
    """
    worker = get_worker(session_id)
    if worker is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"No live capture registered for session {session_id}",
        )
    return LiveStatusOut(**worker.snapshot(include_frame=include_frame).__dict__)


def _sse(event: str, data: str) -> str:
    """One SSE frame. The blank line terminates the event — without it the
    client buffers indefinitely waiting for the record to end."""
    return f"event: {event}\ndata: {data}\n\n"


@router.get(
    "/sessions/{session_id}/live/stream",
    dependencies=[Depends(require_instructor)],
)
async def live_stream_sse(
    session_id: UUID,
    request: Request,
    include_frame: bool = Query(True, description="Include the annotated JPEG in each event"),
):
    """Push live frames as Server-Sent Events instead of being polled.

    Replaces the client poll loop: one long-lived connection, and a frame
    reaches the UI within `live_sse_watch_interval` of being published rather
    than up to a poll period late. Unchanged state costs a comment heartbeat
    instead of re-sending a ~310KB JPEG.

    A slow client can never stall capture. The worker owns its own thread and
    the only state shared with this coroutine is its lock, held just long
    enough to copy a reference. Measured: a client stalling 8s per event left
    the worker running at 0.66 fps and 43 frames ahead, unaffected.

    What this does NOT give you is drop-on-send. SSE has no acknowledgement,
    so an event handed to the transport is gone from our control and queues in
    the socket buffer; the same measurement showed the stalled client reading
    seq 0..8 *consecutively* — a backlog, not the newest frame. Conflation
    happens at snapshot time only, and the generator free-runs because a write
    that the buffer accepts returns immediately, so it never learns the client
    is behind until flow control finally engages (~3MB on loopback).

    The consumer must therefore conflate on receipt: read events continuously
    and keep only the newest in a single slot, discarding any backlog. The
    Streamlit client does exactly that (`_SSEMonitor`), which is what makes
    the end-to-end behaviour drop-the-stale rather than replay-the-old.

    Auth is a normal Authorization header, which the browser's EventSource
    cannot send — this is consumed server-side by the Streamlit page, which
    can. A browser-native consumer would need a query-token variant.
    """
    worker = get_worker(session_id)
    if worker is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"No live capture registered for session {session_id}",
        )

    async def events():
        last_seq: int = -1
        last_state: Optional[str] = None
        last_send = time.monotonic()

        while True:
            if await request.is_disconnected():
                logger.debug("SSE session=%s client went away", session_id)
                break

            snap = worker.snapshot(include_frame=include_frame)
            payload = LiveStatusOut(**snap.__dict__).model_dump_json()

            if snap.state in ("stopped", "error"):
                # Terminal: one `end` carrying the final snapshot (last frame
                # and last_error included) and the connection closes. Emitting
                # a `status` first would just duplicate it.
                yield _sse("end", payload)
                break

            if snap.frame_seq != last_seq:
                last_seq = snap.frame_seq
                yield _sse("frame", payload)
                last_send = time.monotonic()
            elif snap.state != last_state:
                # State moved without a new frame — reconnecting, say. The UI
                # needs this or it shows "running" through an outage.
                yield _sse("status", payload)
                last_send = time.monotonic()

            last_state = snap.state

            if time.monotonic() - last_send >= settings.live_sse_heartbeat:
                yield ": keepalive\n\n"
                last_send = time.monotonic()

            await asyncio.sleep(settings.live_sse_watch_interval)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # nginx buffers proxied responses by default, which would hold
            # frames back until the buffer fills and defeat the whole point.
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/live/active",
    dependencies=[Depends(require_instructor)],
)
async def live_active():
    """Which live session, if any, currently holds the pipeline.

    Only one runs at a time (the recognition singletons keep per-track state),
    so the UI uses this to explain a 409 rather than just failing to start.
    """
    worker = active_worker()
    if worker is None:
        return {"active": False, "session_id": None}
    snap = worker.snapshot(include_frame=False)
    return {"active": True, "session_id": snap.session_id, "state": snap.state,
            "rtsp_url": snap.rtsp_url, "frames_processed": snap.frames_processed}


# ── Roster scoping ────────────────────────────────────────────────────────────

def _gate_to_roster(results: List[RecognitionResult], roster_ids: frozenset) -> None:
    """Drop recognised identities that are not on this session's roster, in place.

    The FAISS gallery is department-wide, so `best_match` can return a student
    enrolled in a different course. Scoping the roster query alone is NOT
    enough: `student_map` only supplies the display name, while `student_id`
    flows straight into attendance_records and behavior_events. Demote an
    off-roster match to an unknown face so it is counted as an unrecognised
    person rather than attributed to a student who was never in the room.

    Called before behaviour analysis so analysis, annotation and persistence all
    see the same identity - no downstream path can disagree about who this is.
    """
    for r in results:
        if r.student_id is not None and r.student_id not in roster_ids:
            r.student_id = None
            r.student_name = None
            r.recognition_score = 0.0


# ── WebSocket live inference ──────────────────────────────────────────────────

def _source_frame_and_ms(msg_index: int, frame_step: int, fps: float) -> Tuple[int, int]:
    """Map a WS message index to its (source frame index, video timestamp ms).

    Message i carries the frame sampled every `frame_step` source frames, so it is
    source frame i*frame_step at i*frame_step/fps seconds.

    Deriving the clock from the message index alone — the original
    `msg_index * (1000 // fps)` — understates elapsed video time by exactly
    `frame_step`×, which silently scales every downstream behaviour dwell timer
    (_DISTRACTED_SECONDS, _SLEEPING_SECONDS) by that same factor.
    """
    frame_number = msg_index * frame_step
    return frame_number, int(round(frame_number * 1000.0 / fps))


@router.websocket("/ws/live/{session_id}")
async def live_stream(
    websocket: WebSocket,
    session_id: UUID,
    send_annotated: bool = Query(False, description="Include base64-encoded annotated JPEG in response"),
    frame_step: int = Query(1, ge=1, description="Source frames advanced per WS message (client decimation)"),
    source_fps: float = Query(0.0, ge=0.0, description="Source video fps; 0 = fall back to the configured rate"),
):
    """
    Real-time face recognition WebSocket endpoint.

    Protocol
    --------
    Client sends raw JPEG bytes (one frame per message).
    Server replies with JSON conforming to WSFrameResult.

    A client that decimates its source (sending every Nth frame) must declare
    that as frame_step, and its source rate as source_fps. Video time then
    advances frame_step/source_fps seconds per message. Defaults (step=1,
    configured fps) preserve the original contiguous-frame behaviour.

    The client is responsible for reading the session's student roster and
    passing the correct session_id in the URL.
    """
    await websocket.accept()

    # Authenticated after accept() so the client gets a close frame with a
    # readable reason instead of a bare handshake rejection. HTTPBearer cannot
    # be used on a websocket route - see deps.authenticate_ws.
    user = await authenticate_ws(websocket, UserRole.INSTRUCTOR)
    if user is None:
        return

    logger.info("WS live session=%s user=%s connected", session_id, user.username)

    recognizer = await get_recognizer()
    behavior_analyzer = get_behavior_analyzer()

    # Build student_id → name map from DB for this session
    student_map: Dict[int, str] = {}
    msg_index = 0          # WS messages received
    frame_number = 0       # SOURCE frame index — what gets persisted
    fps = source_fps or float(settings.bytetrack_frame_rate)
    pending: Set[asyncio.Task] = set()

    async with AsyncSessionLocal() as db:
        session = await crud.get_session(db, session_id)
        if session is None:
            await websocket.close(code=4004, reason="Session not found")
            return

        # Scope the roster to the session's COURSE, not its classroom. A room is
        # physical and can host several courses, so classroom scoping would mix
        # rosters the moment a second course is recorded in the same room.
        course_id = await crud.get_session_course_id(db, session_id)
        if course_id is None:
            # Defensive: unreachable while subject_id is NOT NULL and the
            # session was just confirmed to exist.
            await websocket.close(code=4004, reason="Session has no subject/course")
            return
        _, students = await crud.list_students(db, course_id=course_id, limit=10_000)
        student_map = {s.id: s.full_name for s in students}
        roster_ids = frozenset(student_map)

        await crud.start_session(db, session_id)

    late_after = session.started_at + timedelta(minutes=settings.late_threshold_minutes) if session and session.started_at else None

    try:
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_bytes(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"ping": True})
                continue

            frame = jpeg_bytes_to_frame(raw)
            frame_number, timestamp_ms = _source_frame_and_ms(msg_index, frame_step, fps)

            # Run recognition pipeline
            results: List[RecognitionResult] = recognizer.process_frame(
                frame, frame_number, timestamp_ms, student_map
            )
            _gate_to_roster(results, roster_ids)
            # Run behaviour analysis
            behaviors: List[BehaviorFrame] = behavior_analyzer.analyze_frame(frame, results, timestamp_ms)
            bmap = {b.track_id: b for b in behaviors}

            # Persist attendance + behavior in background. Hold a reference: a bare
            # create_task may be garbage-collected before it runs, and nothing
            # awaited these at teardown, so in-flight rows were lost on disconnect.
            task = asyncio.create_task(
                _persist_frame_results(session_id, frame_number, timestamp_ms, results,
                                       behaviors, frame.shape, late_after=late_after)
            )
            pending.add(task)
            task.add_done_callback(pending.discard)

            # Build response
            detections = [
                WSDetection(
                    track_id=r.track_id,
                    student_id=r.student_id,
                    student_name=r.student_name,
                    bbox=_norm_bbox(r.bbox_xyxy, frame.shape),
                    det_confidence=r.detection_score,
                    rec_confidence=r.recognition_score if r.student_id else None,
                    behavior=bmap[r.track_id].behavior.value if r.track_id in bmap else None,
                    head_pitch=bmap[r.track_id].head_pitch if r.track_id in bmap else None,
                    head_yaw=bmap[r.track_id].head_yaw if r.track_id in bmap else None,
                )
                for r in results
            ]

            frame_b64 = None
            if send_annotated:
                ann = annotate_frame(frame, results, behaviors)
                frame_b64 = frame_to_b64_jpeg(ann, quality=70)

            msg = WSFrameResult(
                frame_number=frame_number,
                timestamp_ms=timestamp_ms,
                detections=detections,
                frame_b64=frame_b64,
            )
            await websocket.send_json(msg.model_dump())
            msg_index += 1

    except WebSocketDisconnect:
        logger.info("WS live session=%s disconnected after %d frames", session_id, msg_index)
    except Exception as exc:
        logger.exception("WS live session=%s error: %s", session_id, exc)
    finally:
        # Let in-flight persistence finish before closing the session out.
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        async with AsyncSessionLocal() as db:
            await crud.increment_frame_count(db, session_id, by=msg_index)
            await crud.end_session(db, session_id)
        recognizer.reset()


# ── Background video processor ────────────────────────────────────────────────

async def _process_video_bg(upload_id: str, video_path: str, session_id: str) -> None:
    """
    Asynchronous background task: process an uploaded video file through the
    full recognition + attendance + behavior pipeline and persist results.
    """
    uid = UUID(upload_id)
    sid = UUID(session_id)

    async with AsyncSessionLocal() as db:
        await crud.update_video_status(db, uid, VideoStatus.PROCESSING)
        await crud.start_session(db, sid)
        session = await crud.get_session(db, sid)

        # Read video metadata
        meta = get_video_metadata(video_path)
        await crud.update_video_status(db, uid, VideoStatus.PROCESSING, extra_meta=meta)

        # Roster MUST be scoped to this session's course. Loading every student
        # in the department let the department-wide FAISS gallery attribute a
        # face to someone not enrolled here, writing bogus attendance and
        # behavior rows against them. The live WS path scopes the same query.
        course_id = await crud.get_session_course_id(db, sid)
        if course_id is None:
            # Reachable here (unlike the WS path): the session may have been
            # deleted between upload and background processing. Fail closed -
            # processing against the wrong roster is worse than a visible error.
            logger.error("session=%s has no resolvable course; aborting video %s", sid, uid)
            await crud.update_video_status(
                db, uid, VideoStatus.ERROR,
                error_message=f"Session {sid} has no subject/course - cannot resolve roster",
            )
            await crud.end_session(db, sid, status=SessionStatus.FAILED)
            return
        _, students = await crud.list_students(db, course_id=course_id, limit=10_000)
        student_map = {s.id: s.full_name for s in students}
        roster_ids = frozenset(student_map)

    recognizer = await get_recognizer()
    behavior_analyzer = get_behavior_analyzer()
    recognizer.reset()

    late_after = session.started_at + timedelta(minutes=settings.late_threshold_minutes) if session and session.started_at else None

    try:
        with VideoCapture(video_path, skip_frames=settings.pipeline_skip_frames) as cap:
            batch_results: List[dict] = []
            batch_behaviors: List[dict] = []
            batch_detections: List[dict] = []
            processed = 0

            for frame_number, timestamp_ms, frame in cap.frames():
                results = recognizer.process_frame(frame, frame_number, timestamp_ms, student_map)
                _gate_to_roster(results, roster_ids)
                behaviors = behavior_analyzer.analyze_frame(frame, results, timestamp_ms)

                bmap = {b.track_id: b for b in behaviors}

                for r in results:
                    bf = bmap.get(r.track_id)
                    h, w = frame.shape[:2]
                    x, y, bw, bh = _norm_bbox_xywh(r.bbox_xyxy, (h, w))

                    batch_detections.append(dict(
                        session_id=sid,
                        student_id=r.student_id,
                        track_id=r.track_id,
                        frame_number=frame_number,
                        timestamp_ms=timestamp_ms,
                        bbox_x=x, bbox_y=y, bbox_w=bw, bbox_h=bh,
                        detection_confidence=r.detection_score,
                        recognition_confidence=r.recognition_score if r.student_id else None,
                    ))

                    if bf:
                        batch_behaviors.append(dict(
                            session_id=sid,
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
                        batch_results.append(dict(
                            student_id=r.student_id,
                            confidence=r.recognition_score,
                            timestamp=datetime.now(timezone.utc),
                        ))

                processed += 1

                # Flush every 100 processed frames
                if processed % 100 == 0:
                    async with AsyncSessionLocal() as db:
                        await crud.bulk_insert_face_detections(db, batch_detections)
                        for bdata in batch_behaviors:
                            await crud.create_behavior_event(db, bdata)
                        for atten in batch_results:
                            await crud.mark_attendance(
                                db, sid, atten["student_id"],
                                atten["confidence"], atten["timestamp"],
                                late_after=late_after,
                            )
                        await crud.increment_frame_count(db, sid, by=processed)
                    batch_detections.clear()
                    batch_behaviors.clear()
                    batch_results.clear()

            # Flush remainder
            async with AsyncSessionLocal() as db:
                await crud.bulk_insert_face_detections(db, batch_detections)
                for bdata in batch_behaviors:
                    await crud.create_behavior_event(db, bdata)
                for atten in batch_results:
                    await crud.mark_attendance(
                        db, sid, atten["student_id"],
                        atten["confidence"], atten["timestamp"],
                        late_after=late_after,
                    )
                await crud.increment_frame_count(db, sid, by=processed % 100)
                await crud.update_video_status(db, uid, VideoStatus.DONE)
                await crud.end_session(db, sid)

    except Exception as exc:
        logger.exception("Video processing failed for upload %s: %s", upload_id, exc)
        async with AsyncSessionLocal() as db:
            await crud.update_video_status(db, uid, VideoStatus.ERROR, error_message=str(exc))
            await crud.end_session(db, sid, status=SessionStatus.FAILED)


async def _persist_frame_results(
    session_id: UUID,
    frame_number: int,
    timestamp_ms: int,
    results: List[RecognitionResult],
    behaviors: List[BehaviorFrame],
    frame_shape,
    late_after: Optional[datetime] = None,
) -> None:
    """Fire-and-forget persistence of per-frame results for live WS mode."""
    bmap = {b.track_id: b for b in behaviors}
    try:
        async with AsyncSessionLocal() as db:
            detections: List[dict] = []
            for r in results:
                if r.student_id and r.recognition_score >= settings.attendance_confidence_threshold:
                    await crud.mark_attendance(
                        db, session_id, r.student_id,
                        r.recognition_score, datetime.now(timezone.utc),
                        late_after=late_after,
                    )
                # Live mode wrote no face_detections at all; mirror the batch path so
                # unmatched (Unknown) tracks leave a trail too.
                x, y, bw, bh = _norm_bbox_xywh(r.bbox_xyxy, frame_shape[:2])
                detections.append(dict(
                    session_id=session_id,
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
                    await crud.create_behavior_event(db, dict(
                        session_id=session_id,
                        student_id=r.student_id,
                        track_id=r.track_id,
                        behavior_type=bf.behavior,
                        start_frame=frame_number,
                        start_ms=timestamp_ms,
                        confidence=bf.confidence,
                        pose_metrics=bf.pose_metrics,
                    ))
            await crud.bulk_insert_face_detections(db, detections)
    except Exception as exc:
        logger.warning("Failed to persist frame %d results: %s", frame_number, exc)


# ── Coordinate helpers ────────────────────────────────────────────────────────

def _norm_bbox(bbox: np.ndarray, frame_shape) -> List[float]:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox.tolist()
    return [x1 / w, y1 / h, x2 / w, y2 / h]


def _norm_bbox_xywh(bbox: np.ndarray, frame_hw) -> tuple:
    h, w = frame_hw
    x1, y1, x2, y2 = bbox
    return float(x1) / w, float(y1) / h, float(x2 - x1) / w, float(y2 - y1) / h
