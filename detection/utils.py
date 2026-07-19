"""
detection/utils.py

Shared utilities: dataset video discovery, video reader/writer helpers,
FPS measurement, and simple file logging. Kept independent of any
detector/tracker specifics so they can be reused across modules.
"""

import time
import logging
from pathlib import Path
from typing import Optional, Iterator, Tuple

import cv2
import numpy as np

from detection.config import SUPPORTED_VIDEO_EXTS


def find_first_video(dataset_dir: Path) -> Optional[Path]:
    """
    Recursively search `dataset_dir` for the first supported video file.

    Args:
        dataset_dir: Root directory to search.

    Returns:
        Path to the first matching video file (sorted for determinism),
        or None if no supported video is found.
    """
    if not dataset_dir.exists():
        return None

    candidates = [
        p for p in dataset_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTS
    ]
    candidates.sort()
    return candidates[0] if candidates else None


def resolve_video_path(cli_path: Optional[str], dataset_dir: Path) -> Path:
    """
    Resolve the video to process: use the CLI-provided path if given,
    otherwise fall back to the first video found recursively in the dataset.

    Args:
        cli_path: Optional path string supplied via command line.
        dataset_dir: Dataset root to search if `cli_path` is None.

    Returns:
        A validated Path to an existing video file.

    Raises:
        FileNotFoundError: If no valid video can be resolved.
    """
    if cli_path:
        path = Path(cli_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Provided video path does not exist: {path}")
        return path

    found = find_first_video(dataset_dir)
    if found is None:
        raise FileNotFoundError(
            f"No supported video files {SUPPORTED_VIDEO_EXTS} found under: {dataset_dir}"
        )
    return found


class VideoReader:
    """Thin wrapper around cv2.VideoCapture exposing metadata and a frame iterator."""

    def __init__(self, video_path: Path):
        """
        Args:
            video_path: Path to the input video file.
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        self.fps: float = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width: int = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames: int = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield (frame_index, frame) tuples until the video ends."""
        idx = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1

    def release(self) -> None:
        """Release the underlying VideoCapture handle."""
        self.cap.release()


class VideoWriter:
    """Thin wrapper around cv2.VideoWriter for saving annotated frames."""

    def __init__(self, output_path: Path, fps: float, frame_size: Tuple[int, int]):
        """
        Args:
            output_path: Destination path for the annotated video (.mp4).
            fps: Output frame rate.
            frame_size: (width, height) of the output frames.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
        if not self.writer.isOpened():
            raise IOError(f"Could not open video writer for: {output_path}")

    def write(self, frame: np.ndarray) -> None:
        """Write a single BGR frame to disk."""
        self.writer.write(frame)

    def release(self) -> None:
        """Flush and close the video writer."""
        self.writer.release()


class FPSMeter:
    """Simple exponential-moving-average FPS counter for live overlay display."""

    def __init__(self, smoothing: float = 0.9):
        self.smoothing = smoothing
        self._last_time: Optional[float] = None
        self.fps: float = 0.0

    def tick(self) -> float:
        """Call once per processed frame; returns the current smoothed FPS."""
        now = time.time()
        if self._last_time is not None:
            instant_fps = 1.0 / max(now - self._last_time, 1e-6)
            self.fps = (
                instant_fps if self.fps == 0.0
                else self.smoothing * self.fps + (1 - self.smoothing) * instant_fps
            )
        self._last_time = now
        return self.fps


def setup_logger(log_path: Path) -> logging.Logger:
    """
    Configure a logger that writes to both console and `log_path`.

    Args:
        log_path: Destination file for persisted logs.

    Returns:
        Configured logging.Logger instance.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("reid360.detection_tracking")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    return logger