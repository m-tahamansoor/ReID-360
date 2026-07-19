"""
scripts/run_detection_tracking.py

Phase 1 entry point: loads a surveillance video, detects persons with
YOLOv8, tracks them with ByteTrack, draws annotations, and saves an
annotated video + CSV of tracking results + a log file.

Usage:
    python scripts/run_detection_tracking.py
    python scripts/run_detection_tracking.py --video path/to/video.mp4
    python scripts/run_detection_tracking.py --conf 0.4 --device cuda:0
"""

import argparse
import sys
import time
from pathlib import Path

# Allow running this script directly (python scripts/run_detection_tracking.py)
# by adding the project root to sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detection.config import (
    DATASET_DIR,
    MODELS_DIR,
    YOLO_MODEL_PATH,
    OUTPUT_VIDEO_PATH,
    OUTPUT_CSV_PATH,
    OUTPUT_LOG_PATH,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    DEVICE,
    TRACK_THRESH,
    TRACK_BUFFER,
    MATCH_THRESH,
)
from detection.utils import (
    resolve_video_path,
    VideoReader,
    VideoWriter,
    FPSMeter,
    setup_logger,
)
from detection.detector import PersonDetector
from tracking.tracker import PersonTracker
from tracking.tracking_utils import TrackingResultWriter
from tracking.visualization import draw_tracks, draw_overlay


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the pipeline."""
    parser = argparse.ArgumentParser(
        description="ReID-360 Phase 1: Person Detection & Within-Camera Tracking"
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to a specific video file. If omitted, the first supported "
             "video found recursively under the dataset directory is used.",
    )
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD,
                         help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD,
                         help="Detection NMS IoU threshold.")
    parser.add_argument("--device", type=str, default=DEVICE,
                         help='Inference device, e.g. "cpu", "0", "cuda:0".')
    parser.add_argument("--track-buffer", type=int, default=TRACK_BUFFER,
                         help="Frames to keep a lost track alive (occlusion tolerance).")
    parser.add_argument("--no-video-out", action="store_true",
                         help="Skip saving the annotated video (CSV still saved).")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    """
    Execute the full detection + tracking pipeline end to end.

    Args:
        args: Parsed CLI arguments controlling thresholds and I/O.
    """
    logger = setup_logger(OUTPUT_LOG_PATH)

    video_path = resolve_video_path(args.video, DATASET_DIR)
    logger.info(f"Using video: {video_path}")

    reader = VideoReader(video_path)
    logger.info(
        f"Video metadata -> fps: {reader.fps:.2f}, "
        f"size: {reader.width}x{reader.height}, frames: {reader.total_frames}"
    )

    detector = PersonDetector(
        model_path=YOLO_MODEL_PATH if YOLO_MODEL_PATH.exists() else None,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
    )
    tracker = PersonTracker(
        track_thresh=args.conf,
        track_buffer=args.track_buffer,
        match_thresh=MATCH_THRESH,
    )

    writer = None
    if not args.no_video_out:
        writer = VideoWriter(
            OUTPUT_VIDEO_PATH, fps=reader.fps, frame_size=(reader.width, reader.height)
        )

    result_writer = TrackingResultWriter()
    fps_meter = FPSMeter()

    start_time = time.time()
    for frame_idx, frame in reader.frames():
        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)

        result_writer.add_frame(frame_idx, tracks)

        annotated = draw_tracks(frame, tracks)
        current_fps = fps_meter.tick()
        annotated = draw_overlay(annotated, frame_idx, current_fps, person_count=len(tracks))

        if writer is not None:
            writer.write(annotated)

        if frame_idx % 50 == 0:
            logger.info(
                f"Frame {frame_idx}/{reader.total_frames} | "
                f"active persons: {len(tracks)} | fps: {current_fps:.1f}"
            )

    reader.release()
    if writer is not None:
        writer.release()

    df = result_writer.save(OUTPUT_CSV_PATH)
    elapsed = time.time() - start_time

    logger.info(f"Processing complete in {elapsed:.1f}s")
    logger.info(f"Total detection rows saved: {len(df)}")
    logger.info(f"Unique track IDs assigned: {result_writer.unique_track_count()}")
    logger.info(f"Annotated video saved to: {OUTPUT_VIDEO_PATH}" if writer else "Video output skipped")
    logger.info(f"Tracking CSV saved to: {OUTPUT_CSV_PATH}")
    logger.info(f"Log file saved to: {OUTPUT_LOG_PATH}")


if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    parsed_args = parse_args()
    run(parsed_args)