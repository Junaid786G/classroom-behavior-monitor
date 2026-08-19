from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class TileOffset:
    x: int   # left pixel of tile within original frame
    y: int   # top pixel of tile within original frame
    w: int   # effective width  (before padding)
    h: int   # effective height (before padding)


@dataclass
class RawDetection:
    """Device-agnostic single-face detection result."""
    bbox: np.ndarray       # [x1, y1, x2, y2] absolute pixels, float32
    landmarks: np.ndarray  # (5, 2) absolute pixels, float32
    score: float


def tile_frame(
    frame: np.ndarray,
    tile_size: int = 640,
    overlap: float = 0.2,
) -> List[Tuple[np.ndarray, TileOffset]]:
    """
    Split *frame* into square, overlapping tiles of size *tile_size*.

    If the entire frame fits in one tile it is returned as-is (zero-padded to
    a square), avoiding the tiling overhead for typical 720p / 1080p inputs.

    Returns
    -------
    List of (tile_img, TileOffset) where tile_img is (tile_size, tile_size, 3).
    """
    h, w = frame.shape[:2]

    if h <= tile_size and w <= tile_size:
        return [(_pad_square(frame, tile_size), TileOffset(0, 0, w, h))]

    stride = max(1, int(tile_size * (1.0 - overlap)))
    tiles: List[Tuple[np.ndarray, TileOffset]] = []

    y = 0
    while y < h:
        x = 0
        while x < w:
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            crop = frame[y:y2, x:x2]
            tiles.append((_pad_square(crop, tile_size), TileOffset(x, y, x2 - x, y2 - y)))
            if x2 >= w:
                break
            x += stride
        if y2 >= h:
            break
        y += stride

    return tiles


def merge_detections(
    tile_results: List[Tuple[List[RawDetection], TileOffset]],
    iou_threshold: float = 0.45,
) -> List[RawDetection]:
    """
    Translate tile-local detections to frame coordinates and apply NMS.
    """
    all_dets: List[RawDetection] = []
    for dets, tile in tile_results:
        for det in dets:
            bbox = det.bbox.copy()
            bbox[[0, 2]] += tile.x
            bbox[[1, 3]] += tile.y
            kps = det.landmarks.copy()
            kps[:, 0] += tile.x
            kps[:, 1] += tile.y
            all_dets.append(RawDetection(bbox, kps, det.score))

    return _nms(all_dets, iou_threshold) if all_dets else []


# ── Internal helpers ──────────────────────────────────────────────────────────

def _nms(dets: List[RawDetection], iou_threshold: float) -> List[RawDetection]:
    if not dets:
        return []

    boxes  = np.array([d.bbox for d in dets], dtype=np.float32)
    scores = np.array([d.score for d in dets], dtype=np.float32)

    # cv2.dnn.NMSBoxes expects (x, y, w, h)
    xywh = [(float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1]))
            for b in boxes]
    idxs = cv2.dnn.NMSBoxes(xywh, scores.tolist(), 0.01, iou_threshold)

    if len(idxs) == 0:
        return []
    flat = idxs.flatten().tolist() if hasattr(idxs, "flatten") else list(idxs)
    return [dets[i] for i in flat]


def _pad_square(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == size and w == size:
        return img
    out = np.zeros((size, size, img.shape[2] if img.ndim == 3 else 1), dtype=img.dtype)
    out[:h, :w] = img
    return out
