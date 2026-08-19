from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.models import BehaviorType
from backend.pipeline.behavior import BehaviorFrame
from backend.pipeline.recognizer import RecognitionResult

# ── Colour palette ────────────────────────────────────────────────────────────

_BEHAVIOR_COLORS: Dict[BehaviorType, Tuple[int, int, int]] = {
    BehaviorType.ATTENTIVE:   (136, 255,   0),   # green  #00ff88
    BehaviorType.DISTRACTED:  (  0, 215, 255),   # yellow #ffd700
    BehaviorType.SLEEPING:    ( 68,  51, 255),   # red    #ff3344
    BehaviorType.HEAD_DOWN:   ( 43, 140, 255),   # orange #ff8c2b
    BehaviorType.USING_PHONE: (128,   0, 128),   # purple
    BehaviorType.TALKING:     (255, 200,   0),   # yellow
    BehaviorType.RAISED_HAND: (0,   255, 255),   # cyan
    BehaviorType.UNKNOWN:     (180, 180, 180),   # grey
}
_DEFAULT_COLOR  = (180, 180, 180)
_TEXT_COLOR     = (255, 255, 255)
_UNKNOWN_COLOR  = (100, 100, 100)

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.45
_THICKNESS  = 1


def annotate_frame(
    frame: np.ndarray,
    recognitions: List[RecognitionResult],
    behaviors: List[BehaviorFrame],
    attention_score: Optional[float] = None,
    show_track_id: bool = True,
    show_confidence: bool = True,
) -> np.ndarray:
    """
    Draw bounding boxes, labels, and behaviour indicators on *frame* (in-place).
    Returns the annotated frame for convenience.
    """
    out = frame.copy()

    # Build track_id → BehaviorFrame map for O(1) lookup
    bmap: Dict[int, BehaviorFrame] = {b.track_id: b for b in behaviors}

    for rec in recognitions:
        bf = bmap.get(rec.track_id)
        behavior = bf.behavior if bf else BehaviorType.UNKNOWN
        color = _BEHAVIOR_COLORS.get(behavior, _DEFAULT_COLOR)

        x1, y1, x2, y2 = rec.bbox_xyxy.astype(int)
        _draw_box(out, x1, y1, x2, y2, color)

        # Build label text
        parts: List[str] = []
        if rec.student_name:
            parts.append(rec.student_name)
        elif rec.student_id:
            parts.append(f"ID:{rec.student_id}")
        else:
            parts.append("Unknown")

        if show_track_id:
            parts.append(f"T{rec.track_id}")

        if show_confidence and rec.recognition_score > 0:
            parts.append(f"{rec.recognition_score:.0%}")

        if bf:
            parts.append(behavior.value)

        label = "  ".join(parts)
        _draw_label(out, label, x1, y1, color)

        # Head-pose arc for attentiveness visualisation
        if bf and bf.head_yaw is not None:
            _draw_pose_indicator(out, rec.bbox_xyxy, bf.head_yaw, bf.head_pitch or 0.0, color)

    # Overlay attention score
    if attention_score is not None:
        _draw_attention_hud(out, attention_score)

    return out


def draw_attendance_overlay(
    frame: np.ndarray,
    present_count: int,
    total_count: int,
    session_elapsed_s: int = 0,
) -> np.ndarray:
    """Overlay attendance stats in the top-right corner."""
    out = frame.copy()
    h, w = out.shape[:2]
    rate = present_count / total_count if total_count else 0.0
    lines = [
        f"Present : {present_count}/{total_count}",
        f"Rate    : {rate:.0%}",
        f"Elapsed : {session_elapsed_s // 60:02d}:{session_elapsed_s % 60:02d}",
    ]
    x = w - 200
    y = 20
    for line in lines:
        cv2.putText(out, line, (x, y), _FONT, _FONT_SCALE, (0, 0, 0), _THICKNESS + 1, cv2.LINE_AA)
        cv2.putText(out, line, (x, y), _FONT, _FONT_SCALE, _TEXT_COLOR, _THICKNESS, cv2.LINE_AA)
        y += 18
    return out


# ── Private helpers ───────────────────────────────────────────────────────────

def _draw_box(img, x1, y1, x2, y2, color, thickness: int = 2) -> None:
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


def _draw_label(img, text: str, x: int, y: int, color) -> None:
    (tw, th), baseline = cv2.getTextSize(text, _FONT, _FONT_SCALE, _THICKNESS)
    pad = 3
    lx1, ly1 = x, max(0, y - th - pad * 2 - baseline)
    lx2, ly2 = x + tw + pad * 2, y
    cv2.rectangle(img, (lx1, ly1), (lx2, ly2), color, -1)
    cv2.putText(
        img, text,
        (x + pad, max(0, y - pad - baseline)),
        _FONT, _FONT_SCALE, _TEXT_COLOR, _THICKNESS, cv2.LINE_AA,
    )


def _draw_pose_indicator(
    img: np.ndarray,
    bbox: np.ndarray,
    yaw: float,
    pitch: float,
    color,
    length: int = 30,
) -> None:
    """Draw a small arrow from the face centre showing approximate gaze direction."""
    x1, y1, x2, y2 = bbox.astype(int)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    yaw_rad   = np.radians(yaw)
    pitch_rad = np.radians(pitch)

    dx = int(length * np.sin(yaw_rad))
    dy = int(length * np.sin(pitch_rad))
    cv2.arrowedLine(img, (cx, cy), (cx + dx, cy + dy), color, 2, cv2.LINE_AA, tipLength=0.35)


def _draw_attention_hud(img: np.ndarray, score: float) -> None:
    h, w = img.shape[:2]
    text  = f"Attention: {score:.0%}"
    bar_w = int(w * 0.2)
    bar_h = 12
    bx, by = 10, h - 30

    # Background
    cv2.rectangle(img, (bx, by), (bx + bar_w, by + bar_h), (50, 50, 50), -1)
    # Fill
    fill_color = (0, 200, 50) if score > 0.6 else (0, 165, 255) if score > 0.35 else (0, 0, 220)
    cv2.rectangle(img, (bx, by), (bx + int(bar_w * score), by + bar_h), fill_color, -1)
    cv2.rectangle(img, (bx, by), (bx + bar_w, by + bar_h), (200, 200, 200), 1)

    cv2.putText(img, text, (bx, by - 5), _FONT, _FONT_SCALE, _TEXT_COLOR, _THICKNESS, cv2.LINE_AA)
