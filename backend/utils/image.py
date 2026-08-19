from __future__ import annotations

import base64
from typing import Optional, Tuple

import cv2
import numpy as np

# ── Codec helpers ─────────────────────────────────────────────────────────────

def frame_to_jpeg_bytes(frame: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


def frame_to_b64_jpeg(frame: np.ndarray, quality: int = 85) -> str:
    return base64.b64encode(frame_to_jpeg_bytes(frame, quality)).decode()


def jpeg_bytes_to_frame(data: bytes) -> np.ndarray:
    buf = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image bytes")
    return frame


def b64_to_frame(b64: str) -> np.ndarray:
    return jpeg_bytes_to_frame(base64.b64decode(b64))


# ── Geometry ──────────────────────────────────────────────────────────────────

def resize_keeping_aspect(
    img: np.ndarray,
    max_side: int,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img
    nh, nw = int(h * scale), int(w * scale)
    return cv2.resize(img, (nw, nh), interpolation=interpolation)


def pad_to_square(img: np.ndarray, fill: int = 0) -> np.ndarray:
    h, w = img.shape[:2]
    if h == w:
        return img
    side = max(h, w)
    out = np.full((side, side, img.shape[2]), fill, dtype=img.dtype)
    out[:h, :w] = img
    return out


def crop_with_margin(
    frame: np.ndarray,
    bbox: np.ndarray,   # [x1, y1, x2, y2] absolute pixels
    margin: float = 0.1,
) -> np.ndarray:
    """Crop a bounding box with a relative margin, clamped to frame edges."""
    x1, y1, x2, y2 = bbox.astype(int)
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * margin), int(bh * margin)
    x1c = max(0, x1 - mx)
    y1c = max(0, y1 - my)
    x2c = min(w, x2 + mx)
    y2c = min(h, y2 + my)
    return frame[y1c:y2c, x1c:x2c]


def crop_and_align_face(
    frame: np.ndarray,
    bbox: np.ndarray,           # [x1, y1, x2, y2]
    kps: Optional[np.ndarray] = None,   # (5, 2) landmarks
    output_size: int = 112,
) -> np.ndarray:
    """
    Align a face to the canonical 112×112 ArcFace input.
    Uses insightface.utils.face_align.norm_crop when landmarks are available;
    falls back to a simple resize crop otherwise.
    """
    if kps is not None:
        try:
            from insightface.utils import face_align
            return face_align.norm_crop(frame, kps, output_size)
        except Exception:
            pass
    # Fallback: plain crop + resize
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, (output_size, output_size))


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalize_imagenet(img: np.ndarray) -> np.ndarray:
    """Normalise a (H,W,3) BGR uint8 frame using ImageNet mean/std → float32."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return (rgb - mean) / std


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ── Coordinate conversions ────────────────────────────────────────────────────

def bbox_xyxy_to_xywh_norm(
    bbox: np.ndarray, frame_hw: Tuple[int, int]
) -> Tuple[float, float, float, float]:
    """Absolute [x1,y1,x2,y2] → normalised (cx, cy, w, h) 0-1."""
    h, w = frame_hw
    x1, y1, x2, y2 = bbox
    return float(x1) / w, float(y1) / h, float(x2 - x1) / w, float(y2 - y1) / h


def bbox_xywh_norm_to_xyxy_abs(
    xn: float, yn: float, wn: float, hn: float,
    frame_hw: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    h, w = frame_hw
    x1 = int(xn * w)
    y1 = int(yn * h)
    x2 = int((xn + wn) * w)
    y2 = int((yn + hn) * h)
    return x1, y1, x2, y2
