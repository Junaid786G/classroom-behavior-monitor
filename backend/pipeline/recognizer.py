from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np

from backend.config import get_settings
from backend.gallery.manager import GalleryManager, get_gallery
from backend.pipeline.detector import FaceDetector, get_detector
from backend.pipeline.embedder import ArcFaceEmbedder, get_embedder
from backend.pipeline.tiling import RawDetection
from backend.pipeline.tracker import ByteTracker, Track, get_tracker

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class RecognitionResult:
    """Fully-annotated output for one detected face in one frame."""
    track_id: int
    bbox_xyxy: np.ndarray          # [x1, y1, x2, y2] absolute pixels
    detection_score: float
    # RetinaFace's own 5 keypoints (eyes, nose, mouth corners) for this face, in
    # absolute pixels. Carried through because they are the ONLY pose signal
    # available on frames where MediaPipe FaceLandmarker returns nothing, which
    # is precisely what happens when a head is turned far enough — see
    # _kps_yaw_index in backend/pipeline/behavior.py. Detections synthesised by
    # _match_tracks_to_dets carry an all-zero placeholder, which _kps_yaw_index
    # rejects as degenerate.
    landmarks: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None
    student_id: Optional[int] = None
    student_name: Optional[str] = None
    recognition_score: float = 0.0
    frame_number: int = 0
    timestamp_ms: int = 0


@dataclass
class _TrackState:
    """Running per-track recognition state across frames."""
    student_id: Optional[int] = None
    best_score: float = 0.0
    vote_counts: Dict[int, int] = field(default_factory=dict)
    confirmed_frames: int = 0


class FaceRecognizer:
    """
    Orchestrates Detection → Tracking → Embedding → Gallery look-up.

    Per-track voting smooths out single-frame misidentifications: a student ID
    must win a majority of frames before it is committed.

    Usage
    -----
    recognizer = FaceRecognizer(gallery, detector, embedder, tracker)
    results = recognizer.process_frame(frame, frame_number, timestamp_ms, student_map)
    """

    def __init__(
        self,
        gallery: GalleryManager,
        detector: FaceDetector,
        embedder: ArcFaceEmbedder,
        tracker: ByteTracker,
        vote_window: int = 5,
    ) -> None:
        self.gallery = gallery
        self.detector = detector
        self.embedder = embedder
        self.tracker = tracker
        self.vote_window = vote_window
        self._track_states: Dict[int, _TrackState] = {}
        self._track_vote_history: Dict[int, List[Optional[int]]] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp_ms: int,
        student_map: Optional[Dict[int, str]] = None,  # id → name
    ) -> List[RecognitionResult]:
        """
        Run full detection → track → embed → recognise pipeline for one frame.

        Parameters
        ----------
        frame         : BGR uint8 image
        frame_number  : sequential frame index in the video
        timestamp_ms  : video-relative timestamp
        student_map   : optional {student_id: full_name} for name annotation
        """
        # 1. Detect raw faces
        detections: List[RawDetection] = self.detector.detect(frame)

        if not detections:
            # Feed empty dets to keep tracker timers running
            self.tracker.update(np.empty((0, 4)), np.empty((0,)))
            return []

        # 2. Track
        boxes = np.array([d.bbox for d in detections], dtype=np.float32)
        scores = np.array([d.score for d in detections], dtype=np.float32)
        tracks: List[Track] = self.tracker.update(boxes, scores)

        if not tracks:
            return []

        # 3. Match tracks back to raw detections by IoU
        matched: List[Tuple[Track, RawDetection]] = _match_tracks_to_dets(
            tracks, detections
        )

        # 4. Embed + gallery look-up + vote
        results: List[RecognitionResult] = []
        for track, det in matched:
            emb = self.embedder.embed_from_detection(frame, det)
            sid, score = self.gallery.best_match(emb)
            voted_sid = self._vote(track.track_id, sid)

            name: Optional[str] = None
            if voted_sid and student_map:
                name = student_map.get(voted_sid)

            results.append(
                RecognitionResult(
                    track_id=track.track_id,
                    bbox_xyxy=track.bbox_xyxy,
                    detection_score=det.score,
                    landmarks=det.landmarks,
                    embedding=emb,
                    student_id=voted_sid,
                    student_name=name,
                    recognition_score=score if voted_sid else 0.0,
                    frame_number=frame_number,
                    timestamp_ms=timestamp_ms,
                )
            )

        return results

    def reset(self) -> None:
        """Clear per-track state – call between sessions."""
        self._track_states.clear()
        self._track_vote_history.clear()
        self.tracker.reset()

    # ── Voting ────────────────────────────────────────────────────────────────

    def _vote(self, track_id: int, candidate_sid: Optional[int]) -> Optional[int]:
        """
        Sliding-window majority vote over the last *vote_window* frames for
        each track.  Returns the winning student_id or None.
        """
        history = self._track_vote_history.setdefault(track_id, [])
        history.append(candidate_sid)
        if len(history) > self.vote_window:
            history.pop(0)

        # Count non-None votes
        counts: Dict[int, int] = {}
        for sid in history:
            if sid is not None:
                counts[sid] = counts.get(sid, 0) + 1

        if not counts:
            return None

        best_sid, best_cnt = max(counts.items(), key=lambda x: x[1])
        # Require at least half the window to agree
        if best_cnt >= max(1, self.vote_window // 2):
            return best_sid
        return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xa1 = max(box_a[0], box_b[0])
    ya1 = max(box_a[1], box_b[1])
    xa2 = min(box_a[2], box_b[2])
    ya2 = min(box_a[3], box_b[3])
    inter = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    if inter == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def _match_tracks_to_dets(
    tracks: List[Track],
    dets: List[RawDetection],
    min_iou: float = 0.3,
) -> List[Tuple[Track, RawDetection]]:
    """Greedy IoU-based matching of confirmed tracks to raw detections."""
    matched: List[Tuple[Track, RawDetection]] = []
    used_dets = set()

    for track in tracks:
        best_iou, best_det_idx = 0.0, -1
        for i, det in enumerate(dets):
            if i in used_dets:
                continue
            iou = _iou(track.bbox_xyxy, det.bbox)
            if iou > best_iou:
                best_iou, best_det_idx = iou, i

        if best_det_idx >= 0 and best_iou >= min_iou:
            matched.append((track, dets[best_det_idx]))
            used_dets.add(best_det_idx)
        else:
            # Track without a matching raw detection – use track bbox directly
            fake_det = RawDetection(
                bbox=track.bbox_xyxy.copy(),
                landmarks=np.zeros((5, 2), dtype=np.float32),
                score=track.confidence,
            )
            matched.append((track, fake_det))

    return matched


# ── Module-level singleton ────────────────────────────────────────────────────

_recognizer: Optional[FaceRecognizer] = None


async def get_recognizer() -> FaceRecognizer:
    global _recognizer
    if _recognizer is None:
        gallery = await get_gallery()
        _recognizer = FaceRecognizer(
            gallery=gallery,
            detector=get_detector(),
            embedder=get_embedder(),
            tracker=get_tracker(),
            vote_window=settings.attendance_min_frames,
        )
    return _recognizer
