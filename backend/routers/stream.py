from __future__ import annotations

import asyncio
import logging
import shutil
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
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from backend import crud
from backend.config import get_settings
from backend.database import AsyncSessionLocal, get_db
from backend.deps import (
    assignment_pairs,
    authenticate_ws,
    get_current_user,
    require_instructor,
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
from backend.schemas import (
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
    total, rows = await crud.list_sessions(
        db, classroom_id=classroom_id, subject_ids=subject_ids, skip=skip, limit=limit
    )
    return Page(total=total, skip=skip, limit=limit, items=[SessionOut.model_validate(r) for r in rows])


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
