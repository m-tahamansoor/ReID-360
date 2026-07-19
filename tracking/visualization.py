"""
tracking/visualization.py

Drawing routines for annotating frames with bounding boxes, track IDs,
confidence scores, frame numbers, and an FPS counter. Kept free of any
detection/tracking logic so it can be reused or swapped independently.
"""

import cv2
import numpy as np

from detection.config import (
    BOX_COLOR_BGR,
    ID_COLOR_BGR,
    TEXT_COLOR_BGR,
    FONT_SCALE,
    FONT_THICKNESS,
    BOX_THICKNESS,
)


def draw_tracks(frame: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    """
    Draw a green bounding box, blue track ID, and confidence score for
    each tracked person on the frame (in place).

    Args:
        frame: BGR frame to annotate.
        tracks: (M, 7) array [x1, y1, x2, y2, track_id, conf, class_id].

    Returns:
        The same frame object, annotated.
    """
    for row in tracks:
        x1, y1, x2, y2, track_id, conf, _cls = row
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR_BGR, BOX_THICKNESS)

        id_label = f"ID {int(track_id)}"
        conf_label = f"{conf:.2f}"

        cv2.putText(
            frame, id_label, (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, ID_COLOR_BGR, FONT_THICKNESS,
        )
        cv2.putText(
            frame, conf_label, (x1, min(y2 + 18, frame.shape[0] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR_BGR, FONT_THICKNESS,
        )
    return frame


def draw_overlay(frame: np.ndarray, frame_idx: int, fps: float, person_count: int) -> np.ndarray:
    """
    Draw a top-left status overlay with frame number, FPS, and active
    person count.

    Args:
        frame: BGR frame to annotate.
        frame_idx: Current frame number.
        fps: Current smoothed FPS value.
        person_count: Number of currently tracked persons.

    Returns:
        The same frame object, annotated.
    """
    lines = [
        f"Frame: {frame_idx}",
        f"FPS: {fps:.1f}",
        f"Persons: {person_count}",
    ]
    y = 25
    for line in lines:
        cv2.putText(
            frame, line, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR_BGR, FONT_THICKNESS,
        )
        y += 22
    return frame