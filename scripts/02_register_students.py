#!/usr/bin/env python3
"""
02_register_students.py – Bulk-enroll 15 students from enrollment videos.

Expected video layout
─────────────────────
data/student_photos/
    ali/enrollment.mp4
    aneeq/enrollment.mp4
    …
    uzair/enrollment.mp4

If a video file is missing the student is still registered in the DB
(with no embedding); they can be enrolled later via the Admin panel.

Run from the project root:
    python scripts/02_register_students.py [--video-dir PATH] [--classroom-id N]

Options
───────
  --video-dir    path to per-student photo/video folders  (default: data/student_photos)
  --classroom-id classroom to assign students to  (default: 1)
  --dry-run      print what would happen without writing to DB
  --force        re-enroll already-registered students
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from backend.config import get_settings
from backend.database import AsyncSessionLocal
from backend.gallery.manager import GalleryManager
from backend.models import Student
from backend.pipeline.detector import FaceDetector
from backend.pipeline.embedder import ArcFaceEmbedder
from backend.utils.gpu import insightface_ctx_id

settings = get_settings()

# ── Terminal colours ──────────────────────────────────────────────────────────
_G = "\033[92m"; _R = "\033[91m"; _Y = "\033[93m"
_C = "\033[96m"; _B = "\033[1m";  _RS = "\033[0m"

def ok(msg):   print(f"  {_G}✓{_RS} {msg}")
def err(msg):  print(f"  {_R}✗{_RS} {msg}")
def warn(msg): print(f"  {_Y}!{_RS} {msg}")
def info(msg): print(f"  {_C}▸{_RS} {msg}")


# ── 15-student roster ─────────────────────────────────────────────────────────
STUDENTS: List[dict] = [
    {"code": "CS-001", "name": "ali",     "email": "ali@student.edu"},
    {"code": "CS-002", "name": "Aneeq",   "email": "aneeq@student.edu"},
    {"code": "CS-003", "name": "Awais",   "email": "awais@student.edu"},
    {"code": "CS-004", "name": "daniyal", "email": "daniyal@student.edu"},
    {"code": "CS-005", "name": "hanzala", "email": "hanzala@student.edu"},
    {"code": "CS-006", "name": "hiba",    "email": "hiba@student.edu"},
    {"code": "CS-007", "name": "Junaid",  "email": "junaid@student.edu"},
    {"code": "CS-008", "name": "mushaf",  "email": "mushaf@student.edu"},
    {"code": "CS-009", "name": "saad",    "email": "saad@student.edu"},
    {"code": "CS-010", "name": "saleha",  "email": "saleha@student.edu"},
    {"code": "CS-011", "name": "Samaan",  "email": "samaan@student.edu"},
    {"code": "CS-012", "name": "samaira", "email": "samaira@student.edu"},
    {"code": "CS-013", "name": "sara",    "email": "sara@student.edu"},
    {"code": "CS-014", "name": "tayyab",  "email": "tayyab@student.edu"},
    {"code": "CS-015", "name": "uzair",   "email": "uzair@student.edu"},
]


# ── Video face extraction ─────────────────────────────────────────────────────

def extract_best_frames(
    video_path: Path,
    detector: FaceDetector,
    max_candidates: int = 10,
    sample_every: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Sample frames from *video_path*, detect faces, and return the top
    *max_candidates* (frame, kps, score) tuples ranked by frontalness × sharpness.

    Frontalness is estimated from the 5-point landmark spread: if the
    eye-midpoint roughly aligns with the nose, the face is frontal.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    candidates = []
    frame_idx  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every != 0:
            frame_idx += 1
            continue

        faces = detector.detect(frame)
        if not faces:
            frame_idx += 1
            continue

        # Largest face in this frame
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        det_score = float(largest.score)

        # Sharpness via Laplacian variance of the face crop
        x1, y1, x2, y2 = largest.bbox.astype(int)
        h_f, w_f = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_f, x2), min(h_f, y2)
        crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(crop, cv2.CV_64F).var()) if crop.size > 0 else 0.0

        # Frontalness: nose should be between the two eye corners (horizontal)
        kps = largest.landmarks   # shape (5, 2): re, le, nose, rm, lm
        if kps is not None and len(kps) >= 3:
            eye_mid_x  = (kps[0, 0] + kps[1, 0]) / 2
            nose_x     = kps[2, 0]
            bbox_w     = max(1, x2 - x1)
            frontalness = 1.0 - abs(eye_mid_x - nose_x) / bbox_w
        else:
            frontalness = 0.5

        combined = det_score * 0.4 + (sharpness / 500.0) * 0.3 + frontalness * 0.3
        candidates.append((frame, largest.landmarks, combined))
        frame_idx += 1

    cap.release()
    candidates.sort(key=lambda x: -x[2])
    return candidates[:max_candidates]


def is_frontal(landmarks: np.ndarray, bbox: np.ndarray) -> bool:
    """
    Hard-reject non-frontal poses before embedding.
    Signals from the 5 RetinaFace landmarks:
      yaw_offset  – horizontal nose offset from eye midpoint   (< 0.15)
      pitch_ratio – vertical nose position within bbox         (0.30–0.65)
      eye_tilt    – vertical disparity between the two eyes    (< 0.10)
    Looking DOWN collapses pitch_ratio toward 0 and is the primary rejection path.
    """
    if landmarks is None or len(landmarks) < 3:
        return False
    face_w = float(bbox[2] - bbox[0])
    face_h = float(bbox[3] - bbox[1])
    if face_w <= 0 or face_h <= 0:
        return False
    eye_mid_x = (landmarks[0, 0] + landmarks[1, 0]) / 2.0
    eye_mid_y = (landmarks[0, 1] + landmarks[1, 1]) / 2.0
    nose_x, nose_y = landmarks[2, 0], landmarks[2, 1]
    yaw_offset  = abs(nose_x - eye_mid_x) / face_w
    pitch_ratio = (nose_y - eye_mid_y)    / face_h
    eye_tilt    = abs(landmarks[0, 1] - landmarks[1, 1]) / face_w
    return (
        yaw_offset  < 0.15
        and 0.15 <= pitch_ratio <= 0.70
        and eye_tilt < 0.10
    )


def extract_segmented_embeddings(
    video_path: Path,
    detector: FaceDetector,
    embedder: ArcFaceEmbedder,
    n_segments: int = 8,
    min_det_score: float = 0.65,
    min_sharpness: float = 40.0,
) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
    """
    Divide the video into n_segments equal time slices and extract one embedding
    per slice (the best-scoring detected face within that slice).

    Returns (embeddings, best_embedding):
      embeddings    – list of 1..n_segments normalised 512-d vectors, one per slice
      best_embedding – highest-scored of those, used for backwards-compatible DB storage

    Sampling across time slices (rather than globally picking top-5) ensures the
    set captures natural pose and lighting variation from across the recording.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return [], None

    seg_size   = max(1, total // n_segments)
    dummy_bbox = np.array([0, 0, 112, 112], dtype=np.float32)

    embeddings: List[np.ndarray]     = []
    best_score: float                = -1.0
    best_emb:   Optional[np.ndarray] = None

    for seg in range(n_segments):
        seg_start = seg * seg_size
        seg_end   = min(total - 1, seg_start + seg_size - 1)
        step      = max(1, (seg_end - seg_start) // 5)

        seg_best_score: float                = -1.0
        seg_best_emb:   Optional[np.ndarray] = None

        for fi in range(seg_start, seg_end + 1, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                continue

            faces = detector.detect(frame)
            if not faces:
                continue

            largest   = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            det_score = float(largest.score)
            if det_score < min_det_score:
                continue

            x1, y1, x2, y2 = largest.bbox.astype(int)
            h_f, w_f        = frame.shape[:2]
            x1, y1          = max(0, x1), max(0, y1)
            x2, y2          = min(w_f, x2), min(h_f, y2)
            crop            = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            if crop.size == 0:
                continue
            sharpness = float(cv2.Laplacian(crop, cv2.CV_64F).var())
            if sharpness < min_sharpness:
                continue

            if not is_frontal(largest.landmarks, largest.bbox):
                continue

            kps = largest.landmarks
            try:
                emb = embedder.get_embedding(frame, dummy_bbox, kps)
                if emb is None or np.linalg.norm(emb) == 0:
                    continue
                emb = (emb / np.linalg.norm(emb)).astype(np.float32)
            except Exception:
                continue

            if kps is not None and len(kps) >= 3:
                eye_mid_x   = (kps[0, 0] + kps[1, 0]) / 2
                nose_x      = kps[2, 0]
                bbox_w      = max(1, x2 - x1)
                frontalness = 1.0 - abs(eye_mid_x - nose_x) / bbox_w
            else:
                frontalness = 0.5

            combined = det_score * 0.5 + frontalness * 0.3 + min(1.0, sharpness / 500.0) * 0.2

            if combined > seg_best_score:
                seg_best_score = combined
                seg_best_emb   = emb
                if combined > best_score:
                    best_score = combined
                    best_emb   = emb

        if seg_best_emb is not None:
            embeddings.append(seg_best_emb)

    cap.release()
    return embeddings, best_emb


def save_best_photo(
    frames: List[Tuple[np.ndarray, np.ndarray, float]],
    dest_path: Path,
) -> bool:
    """Save the highest-scoring frame as JPEG to *dest_path*."""
    if not frames:
        return False
    best_frame = frames[0][0]
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(dest_path), best_frame)


# ── DB operations ─────────────────────────────────────────────────────────────

async def get_or_create_student(
    code: str,
    name: str,
    email: str,
    classroom_id: int,
    force: bool = False,
) -> Tuple[Student, bool]:
    """Return (student, created). If force=True, reset the embedding fields."""
    from sqlalchemy import select, update

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Student).where(Student.student_code == code))
        existing = result.scalar_one_or_none()

        if existing:
            if force:
                await db.execute(
                    update(Student)
                    .where(Student.id == existing.id)
                    .values(face_embedding=None, gallery_index=None,
                            photo_path=None, embedding_path=None)
                )
                await db.commit()
                await db.refresh(existing)
            return existing, False

        new = Student(
            student_code=code,
            full_name=name,
            email=email,
            classroom_id=classroom_id,
            is_active=True,
        )
        db.add(new)
        await db.flush()
        await db.commit()
        await db.refresh(new)
        return new, True


async def save_embedding_to_db(
    student_id: int,
    embedding: np.ndarray,
    gallery_index: int,
    photo_path: Optional[str],
    emb_path: Optional[str],
) -> None:
    from datetime import datetime, timezone
    from sqlalchemy import update

    async with AsyncSessionLocal() as db:
        await db.execute(
            update(Student)
            .where(Student.id == student_id)
            .values(
                face_embedding=embedding.tolist(),
                gallery_index=gallery_index,
                photo_path=photo_path,
                embedding_path=emb_path,
                embedding_updated_at=datetime.now(timezone.utc),
            )
        )
        await db.commit()


# ── Main enrollment loop ──────────────────────────────────────────────────────

async def enroll_all(
    video_dir: Path,
    classroom_id: int,
    dry_run: bool,
    force: bool,
    only: Optional[set] = None,
) -> None:
    print(f"\n{_B}{_C}════════════════════════════════════════════════════{_RS}")
    print(f"{_B}{_C}   CLASSROOM MONITOR – Student Enrollment Script    {_RS}")
    print(f"{_B}{_C}════════════════════════════════════════════════════{_RS}")
    info(f"Video directory : {video_dir}")
    info(f"Classroom ID    : {classroom_id}")
    info(f"Dry run         : {dry_run}")
    info(f"Force re-enroll : {force}\n")

    # Load ML models once
    print(f"{_B}Loading ML models …{_RS}")
    detector = FaceDetector(model_name=settings.insightface_model_name)
    embedder = ArcFaceEmbedder(model_name=settings.insightface_model_name)
    gallery  = GalleryManager()
    gallery.load()

    try:
        detector.warmup()
        ok("FaceDetector (RetinaFace) ready")
    except Exception as exc:
        warn(f"FaceDetector load failed: {exc}")
        warn("Will skip face extraction and register students without embeddings")
        detector = None
        embedder = None

    if embedder and detector:
        try:
            embedder.warmup()
            ok("ArcFaceEmbedder ready")
        except Exception as exc:
            warn(f"ArcFaceEmbedder load failed: {exc}")
            embedder = None

    print()

    # Results tracking
    enrolled_ok: List[str] = []
    enrolled_no_video: List[str] = []
    enrolled_no_face: List[str] = []
    failed: List[str] = []

    for stu in STUDENTS:
        code  = stu["code"]
        name  = stu["name"]
        email = stu["email"]

        if only and name.lower() not in only:
            continue

        print(f"{_B}[{code}] {name}{_RS}")

        # ── 1. Register in DB ───────────────────────────────────────────────
        if not dry_run:
            try:
                student, created = await get_or_create_student(
                    code, name, email, classroom_id, force=force
                )
                if created:
                    info(f"Registered in DB (id={student.id})")
                else:
                    if not force:
                        info(f"Already in DB (id={student.id}) – skipping embedding (use --force to redo)")
                        enrolled_ok.append(code)
                        print()
                        continue
                    info(f"Already in DB (id={student.id}) – re-enrolling")

            except Exception as exc:
                err(f"DB error: {exc}")
                failed.append(code)
                print()
                continue
        else:
            student = type("S", (), {"id": 0})()
            info("DRY RUN – skipping DB write")

        # ── 2. Find enrollment video ────────────────────────────────────────
        candidate = video_dir / name.lower() / "enrollment.mp4"
        video_path = candidate if candidate.exists() else None

        if video_path is None:
            warn(f"No enrollment video found at {candidate} – registered without embedding")
            enrolled_no_video.append(code)
            print()
            continue

        info(f"Video: {video_path.name}")

        # ── 3. Extract faces ────────────────────────────────────────────────
        if detector is None:
            warn("Detector unavailable – skipping embedding")
            enrolled_no_video.append(code)
            print()
            continue

        candidates = extract_best_frames(video_path, detector, max_candidates=10, sample_every=5)

        if not candidates:
            warn("No faces detected in video – registered without embedding")
            enrolled_no_face.append(code)
            print()
            continue

        info(f"Extracted {len(candidates)} candidate frames (best score {candidates[0][2]:.3f})")

        # ── 4. Save best photo ──────────────────────────────────────────────
        photo_dest = settings.student_photos_path / str(student.id) / "photo.jpg"
        if not dry_run:
            saved = save_best_photo(candidates, photo_dest)
            if saved:
                info(f"Photo saved → {photo_dest}")
            else:
                warn("Photo save failed")
                photo_dest = None

        # ── 5. Extract per-segment embeddings ───────────────────────────────
        if embedder is None:
            warn("Embedder unavailable – skipping embedding")
            enrolled_no_face.append(code)
            print()
            continue

        embeddings, best_emb = extract_segmented_embeddings(video_path, detector, embedder)
        if not embeddings:
            warn("Embedding extraction failed – registered without embedding")
            enrolled_no_face.append(code)
            print()
            continue

        info(f"Extracted {len(embeddings)} segment embedding(s) (best norm={np.linalg.norm(best_emb):.4f})")

        # ── 6. Add to FAISS gallery ─────────────────────────────────────────
        if not dry_run:
            gallery.remove_student(student.id)
            first_idx: Optional[int] = None
            for emb in embeddings:
                idx = gallery.add(emb, student.id)
                if first_idx is None:
                    first_idx = idx
            info(f"Gallery: {len(embeddings)} vectors added, first_idx={first_idx}")

            # Save per-segment .npy sidecars
            emb_dir = settings.embeddings_cache_path / str(student.id)
            emb_dir.mkdir(parents=True, exist_ok=True)
            for i, emb in enumerate(embeddings):
                np.save(str(emb_dir / f"embedding_{i}.npy"), emb)
            emb_path = emb_dir / "embedding_0.npy"

            # Persist primary embedding to DB (backwards-compatible: one row per student)
            await save_embedding_to_db(
                student.id, best_emb, first_idx,
                str(photo_dest) if photo_dest else None,
                str(emb_path),
            )
            ok(f"Enrolled ✓  ({len(embeddings)} FAISS vectors)")
        else:
            info("DRY RUN – would write embeddings to DB and gallery")
            ok("Dry-run OK")

        enrolled_ok.append(code)
        print()

    # ── 7. Save gallery index ───────────────────────────────────────────────
    if not dry_run and detector is not None:
        gallery.save()
        ok(f"FAISS gallery saved ({gallery.size} vectors)")

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{_B}{'═' * 52}{_RS}")
    print(f"{_B}Enrollment Summary{_RS}")
    print(f"{'═' * 52}")
    print(f"  Total students  : {len(STUDENTS)}")
    print(f"  {_G}Fully enrolled   : {len(enrolled_ok)}{_RS}")
    print(f"  {_Y}No video found  : {len(enrolled_no_video)} → {enrolled_no_video}{_RS}")
    print(f"  {_Y}No face detected : {len(enrolled_no_face)} → {enrolled_no_face}{_RS}")
    print(f"  {_R}Failed           : {len(failed)} → {failed}{_RS}")
    print(f"{'═' * 52}")

    if enrolled_no_video or enrolled_no_face:
        print(f"\n{_Y}Students without embeddings can be enrolled later via:{_RS}")
        print(f"  • The Admin → Enroll from Video page in the Streamlit UI")
        print(f"  • POST /api/v1/students/{{id}}/photo with their photo")

    print(f"\n{_G}{_B}✓ Enrollment complete.{_RS}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bulk-enroll 15 students from enrollment videos"
    )
    p.add_argument(
        "--video-dir",
        default=str(ROOT / "data" / "student_photos"),
        help="Root directory containing per-student folders with enrollment.mp4 (default: data/student_photos/)",
    )
    p.add_argument(
        "--classroom-id",
        type=int,
        default=1,
        help="Classroom DB id to assign all students (default: 1)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without writing to DB or disk",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-enroll students who already have embeddings",
    )
    p.add_argument(
        "--only",
        default=None,
        help="Comma-separated student names to re-enroll, e.g. aneeq,awais,junaid "
             "(case-insensitive; combine with --force to overwrite existing embeddings)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    video_dir = Path(args.video_dir)

    if not video_dir.exists():
        print(f"{_Y}Creating student photos directory: {video_dir}{_RS}")
        video_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"\n{_Y}Place enrollment videos here before running this script:{_RS}\n"
            f"  {video_dir}/ali/enrollment.mp4\n"
            f"  {video_dir}/aneeq/enrollment.mp4\n"
            f"  … (lowercase folder per student, see STUDENTS list in the script)\n\n"
            f"Re-run when videos are in place.  "
            f"Students will still be registered in the DB without embeddings.\n"
        )

    only_set = (
        {n.strip().lower() for n in args.only.split(",")}
        if args.only else None
    )
    asyncio.run(
        enroll_all(
            video_dir=video_dir,
            classroom_id=args.classroom_id,
            dry_run=args.dry_run,
            force=args.force,
            only=only_set,
        )
    )
