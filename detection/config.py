"""
detection/config.py

Centralized configuration for the Detection & Tracking pipeline (Phase 1).
All paths are resolved relative to the project root using pathlib so the
pipeline can be run from any working directory.
"""

from pathlib import Path

# ------------------------------------------------------------------
# Project root (ReID-360/)
# ------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------
# Dataset / model / output locations
# ------------------------------------------------------------------
DATASET_DIR: Path = PROJECT_ROOT / "datasets" / "Video"
MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

OUTPUT_VIDEO_PATH: Path = OUTPUTS_DIR / "annotated_video.mp4"
OUTPUT_CSV_PATH: Path = OUTPUTS_DIR / "tracking_results.csv"
OUTPUT_LOG_PATH: Path = OUTPUTS_DIR / "logs.txt"

# ------------------------------------------------------------------
# Supported video extensions for recursive dataset search
# ------------------------------------------------------------------
SUPPORTED_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

# ------------------------------------------------------------------
# YOLOv8 detection settings
# ------------------------------------------------------------------
YOLO_MODEL_NAME: str = "yolov8n.pt"          # auto-downloaded by ultralytics if not local
YOLO_MODEL_PATH: Path = MODELS_DIR / YOLO_MODEL_NAME

# COCO class index for "person"
PERSON_CLASS_ID: int = 0

CONFIDENCE_THRESHOLD: float = 0.35
IOU_THRESHOLD: float = 0.45
DEVICE: str = "cpu"                          # set to "0" / "cuda:0" if a GPU is available
INFERENCE_IMG_SIZE: int = 640

# ------------------------------------------------------------------
# ByteTrack tracker settings (BoxMOT ByteTrack implementation)
# ------------------------------------------------------------------
TRACK_THRESH: float = 0.5        # confidence threshold to start a new track
TRACK_BUFFER: int = 30           # frames to keep a lost track alive (occlusion tolerance)
MATCH_THRESH: float = 0.8        # IoU matching threshold
FRAME_RATE: int = 30             # fallback FPS assumption if video metadata is unavailable

# ------------------------------------------------------------------
# Visualization settings
# ------------------------------------------------------------------
BOX_COLOR_BGR = (0, 255, 0)        # green bounding box
ID_COLOR_BGR = (255, 0, 0)         # blue track ID text
TEXT_COLOR_BGR = (255, 255, 255)   # white supplementary text
FONT_SCALE: float = 0.55
FONT_THICKNESS: int = 2
BOX_THICKNESS: int = 2