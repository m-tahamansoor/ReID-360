"""
tracking/tracker.py

ByteTrack-based multi-object tracker. Prefers the BoxMOT implementation
(pip install boxmot) for a decoupled detector -> tracker pipeline; falls
back to a clear error message if BoxMOT is unavailable so the caller can
install it rather than silently degrading.

The tracker consumes raw detections (from detection.detector.PersonDetector)
and returns detections augmented with persistent track IDs.
"""

from typing import Optional

import numpy as np

from detection.config import (
    TRACK_THRESH,
    TRACK_BUFFER,
    MATCH_THRESH,
    FRAME_RATE,
)


class PersonTracker:
    """Wraps BoxMOT's ByteTrack to assign persistent IDs to person detections."""

    def __init__(
        self,
        track_thresh: float = TRACK_THRESH,
        track_buffer: int = TRACK_BUFFER,
        match_thresh: float = MATCH_THRESH,
        frame_rate: int = FRAME_RATE,
    ):
        """
        Args:
            track_thresh: Confidence threshold to initiate a new track.
            track_buffer: Number of frames a lost track is kept alive for
                (controls tolerance to occlusions).
            match_thresh: IoU threshold used during association.
            frame_rate: Expected video frame rate, used for buffer timing.
        """
        ByteTrack = self._import_bytetrack()

        self._tracker = ByteTrack(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            frame_rate=frame_rate,
        )

    @staticmethod
    def _import_bytetrack():
        """
        Locate the ByteTrack class regardless of BoxMOT's package layout.

        BoxMOT has moved ByteTrack's import path across major versions:
          - <=11.x:  boxmot.ByteTrack               (flat top-level export)
          - 19.x+:   boxmot.trackers.bbox.bytetrack.ByteTrack
          - some releases also expose it under boxmot.trackers.bytetrack.*

        Trying each in order keeps this module working whether the
        environment has an old or a current BoxMOT release installed.

        Returns:
            The ByteTrack class.

        Raises:
            ImportError: If BoxMOT is not installed or none of the known
                import paths resolve to a ByteTrack class.
        """
        candidates = (
            "boxmot:ByteTrack",
            "boxmot.trackers.bbox.bytetrack:ByteTrack",
            "boxmot.trackers.bytetrack.bytetrack:ByteTrack",
            "boxmot.trackers.bytetrack:ByteTrack",
        )
        last_error: Optional[Exception] = None
        for candidate in candidates:
            module_name, attr_name = candidate.split(":")
            try:
                import importlib
                module = importlib.import_module(module_name)
                return getattr(module, attr_name)
            except (ImportError, AttributeError) as exc:
                last_error = exc
                continue

        raise ImportError(
            "Could not locate BoxMOT's ByteTrack class under any known import "
            "path. Install/upgrade BoxMOT with:\n"
            "    pip install -U boxmot\n"
            f"Last error: {last_error}"
        )

    def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """
        Update tracker state with the current frame's detections.

        Args:
            detections: (N, 6) array [x1, y1, x2, y2, conf, class_id] from
                PersonDetector.detect().
            frame: The current BGR frame (required by BoxMOT for internal
                appearance/ReID handling on some trackers).

        Returns:
            np.ndarray of shape (M, 7): [x1, y1, x2, y2, track_id, conf, class_id].
            M <= N; rows without a confirmed track are dropped by ByteTrack.
        """
        if detections.shape[0] == 0:
            # BoxMOT trackers accept an empty (0, 6) array to advance internal state.
            detections = np.empty((0, 6), dtype=np.float32)

        tracks = self._tracker.update(detections, frame)

        if tracks is None or len(tracks) == 0:
            return np.empty((0, 7), dtype=np.float32)

        # BoxMOT returns columns: x1, y1, x2, y2, track_id, conf, cls, [det_ind]
        # Keep only the 7 columns this pipeline needs.
        return np.asarray(tracks, dtype=np.float32)[:, :7]