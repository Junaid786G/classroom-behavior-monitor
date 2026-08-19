from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from backend import crud
from backend.config import get_settings
from backend.database import get_db
from backend.gallery.manager import get_gallery, rebuild_gallery_from_db
from backend.pipeline.detector import get_detector
from backend.pipeline.embedder import get_embedder
from backend.schemas import (
    EmbeddingResponse,
    GalleryBuildResponse,
    Page,
    StudentCreate,
    StudentOut,
    StudentUpdate,
)
from backend.utils.image import jpeg_bytes_to_frame

router = APIRouter(prefix="/students", tags=["students"])
settings = get_settings()

_ALLOWED_PHOTO_TYPES = {"image/jpeg", "image/png", "image/webp"}


# ── CRUD ──────────────────────────────────────────────────────────────────────

@router.get("", response_model=Page)
async def list_students(
    classroom_id: Optional[int] = Query(None),
    active_only: bool = Query(True),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    total, rows = await crud.list_students(
        db, classroom_id=classroom_id, active_only=active_only, skip=skip, limit=limit
    )
    return Page(total=total, skip=skip, limit=limit, items=[StudentOut.model_validate(r) for r in rows])


@router.post("", response_model=StudentOut, status_code=status.HTTP_201_CREATED)
async def create_student(
    data: StudentCreate,
    db: AsyncSession = Depends(get_db),
):
    existing = await crud.get_student_by_code(db, data.student_code)
    if existing:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Student code {data.student_code!r} already exists")
    student = await crud.create_student(db, data)
    return StudentOut.model_validate(student)


@router.get("/{student_id}", response_model=StudentOut)
async def get_student(
    student_id: int,
    db: AsyncSession = Depends(get_db),
):
    s = await crud.get_student(db, student_id)
    if s is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Student not found")
    return StudentOut.model_validate(s)


@router.patch("/{student_id}", response_model=StudentOut)
async def update_student(
    student_id: int,
    data: StudentUpdate,
    db: AsyncSession = Depends(get_db),
):
    s = await crud.update_student(db, student_id, data)
    if s is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Student not found")
    return StudentOut.model_validate(s)


@router.delete("/{student_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_student(
    student_id: int,
    db: AsyncSession = Depends(get_db),
):
    ok = await crud.delete_student(db, student_id)
    if not ok:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Student not found")

    # Remove from FAISS gallery
    gallery = await get_gallery()
    gallery.remove_student(student_id)
    gallery.save()


# ── Photo upload ──────────────────────────────────────────────────────────────

@router.post("/{student_id}/photo", response_model=StudentOut)
async def upload_photo(
    student_id: int,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    s = await crud.get_student(db, student_id)
    if s is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Student not found")

    if file.content_type not in _ALLOWED_PHOTO_TYPES:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Unsupported image type: {file.content_type}",
        )

    # Save photo
    dest_dir = settings.student_photos_path / str(student_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename or "photo.jpg").suffix or ".jpg"
    dest = dest_dir / f"photo{ext}"

    raw = await file.read()
    dest.write_bytes(raw)

    await crud.update_student_photo(db, student_id, str(dest))

    # Auto-generate embedding
    frame = jpeg_bytes_to_frame(raw)
    detector = get_detector()
    embedder = get_embedder()

    faces = detector.detect(frame)
    if faces:
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = embedder.embed_from_detection(frame, largest)
        gallery = await get_gallery()
        g_idx = gallery.replace(student_id, emb)
        gallery.save()

        emb_path = dest_dir / "embedding.npy"
        np.save(str(emb_path), emb)
        await crud.update_student_embedding(db, student_id, emb.tolist(), g_idx, str(emb_path))

    updated = await crud.get_student(db, student_id)
    return StudentOut.model_validate(updated)


# ── Manual embedding generation ───────────────────────────────────────────────

@router.post("/{student_id}/embed", response_model=EmbeddingResponse)
async def generate_embedding(
    student_id: int,
    db: AsyncSession = Depends(get_db),
):
    s = await crud.get_student(db, student_id)
    if s is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Student not found")
    if not s.photo_path or not Path(s.photo_path).exists():
        return EmbeddingResponse(
            student_id=student_id,
            success=False,
            message="No photo on disk – upload a photo first",
        )

    raw = Path(s.photo_path).read_bytes()
    frame = jpeg_bytes_to_frame(raw)

    detector = get_detector()
    embedder = get_embedder()

    faces = detector.detect(frame)
    if not faces:
        return EmbeddingResponse(
            student_id=student_id,
            success=False,
            message="No face detected in the stored photo",
        )

    largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    emb = embedder.embed_from_detection(frame, largest)

    gallery = await get_gallery()
    g_idx = gallery.replace(student_id, emb)
    gallery.save()

    emb_dir = Path(s.photo_path).parent
    emb_path = emb_dir / "embedding.npy"
    np.save(str(emb_path), emb)
    await crud.update_student_embedding(db, student_id, emb.tolist(), g_idx, str(emb_path))

    return EmbeddingResponse(
        student_id=student_id,
        success=True,
        gallery_index=g_idx,
        message="Embedding generated and indexed",
    )


# ── Gallery management ────────────────────────────────────────────────────────

@router.post("/gallery/rebuild", response_model=GalleryBuildResponse)
async def rebuild_gallery(db: AsyncSession = Depends(get_db)):
    """Rebuild the FAISS gallery from all embeddings stored in PostgreSQL."""
    total, students = await crud.list_students(db, active_only=True, limit=10_000)
    failed: List[str] = []
    indexed = 0

    gallery = await get_gallery()
    gallery.rebuild([])   # clear

    for s in students:
        if not s.face_embedding:
            continue
        try:
            emb = np.array(s.face_embedding, dtype=np.float32)
            gallery.add(emb, s.id)
            indexed += 1
        except Exception as exc:
            failed.append(f"{s.student_code}: {exc}")

    gallery.save()

    return GalleryBuildResponse(
        success=True,
        total_students=total,
        indexed=indexed,
        failed=failed,
        message=f"Gallery rebuilt: {indexed}/{total} students indexed",
    )
