"""
detection/detector.py

YOLOv8-based person detector. Wraps Ultralytics YOLO and exposes a single
`detect` method returning only "person" class detections in a plain
NumPy array, decoupled from any tracking logic.
"""

from pathlib import Path
from typing import Optional

import numpy as np
from ultralytics import YOLO

from detection.config import (
    YOLO_MODEL_NAME,
    PERSON_CLASS_ID,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    DEVICE,
    INFERENCE_IMG_SIZE,
)


class PersonDetector:
    """Detects only the 'person' class using a YOLOv8 model."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        conf_threshold: float = CONFIDENCE_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
        device: str = DEVICE,
        img_size: int = INFERENCE_IMG_SIZE,
    ):
        """
        Args:
            model_path: Local weights path; falls back to auto-downloaded
                `yolov8n.pt` if None or not found.
            conf_threshold: Minimum confidence to keep a detection.
            iou_threshold: NMS IoU threshold.
            device: Inference device, e.g. "cpu", "0", "cuda:0".
            img_size: Inference image size passed to YOLO.
        """
        weights = str(model_path) if model_path and Path(model_path).exists() else YOLO_MODEL_NAME
        self.model = YOLO(weights)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.img_size = img_size

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Run person-only detection on a single BGR frame.

        Args:
            frame: Input image as a numpy array (H, W, 3), BGR.

        Returns:
            np.ndarray of shape (N, 6): [x1, y1, x2, y2, confidence, class_id].
            class_id is always PERSON_CLASS_ID. Empty array if none found.
        """
        results = self.model.predict(
            source=frame,
            classes=[PERSON_CLASS_ID],
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            imgsz=self.img_size,
            verbose=False,
        )

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy().reshape(-1, 1)
        cls = boxes.cls.cpu().numpy().reshape(-1, 1)

        detections = np.hstack([xyxy, conf, cls]).astype(np.float32)
        return detections