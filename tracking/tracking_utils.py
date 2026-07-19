"""
tracking/tracking_utils.py

Helpers for converting tracker output into structured records and
persisting them as a CSV file, plus small bookkeeping utilities for
evaluating ID continuity.
"""

from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

CSV_COLUMNS = ["frame", "track_id", "class", "confidence", "x1", "y1", "x2", "y2"]


def tracks_to_records(frame_idx: int, tracks: np.ndarray, class_name: str = "person") -> List[Dict]:
    """
    Convert a single frame's tracker output into a list of row dicts.

    Args:
        frame_idx: Current frame number (0-indexed).
        tracks: (M, 7) array [x1, y1, x2, y2, track_id, conf, class_id].
        class_name: Human-readable class label to store in the CSV.

    Returns:
        List of dicts matching CSV_COLUMNS, one per tracked person.
    """
    records = []
    for row in tracks:
        x1, y1, x2, y2, track_id, conf, _cls = row
        records.append({
            "frame": frame_idx,
            "track_id": int(track_id),
            "class": class_name,
            "confidence": round(float(conf), 4),
            "x1": round(float(x1), 2),
            "y1": round(float(y1), 2),
            "x2": round(float(x2), 2),
            "y2": round(float(y2), 2),
        })
    return records


class TrackingResultWriter:
    """Accumulates per-frame tracking records and writes them to a single CSV."""

    def __init__(self):
        self._rows: List[Dict] = []

    def add_frame(self, frame_idx: int, tracks: np.ndarray, class_name: str = "person") -> None:
        """Append a frame's worth of tracking records to the buffer."""
        self._rows.extend(tracks_to_records(frame_idx, tracks, class_name))

    def save(self, csv_path: Path) -> pd.DataFrame:
        """
        Write all accumulated records to `csv_path`.

        Args:
            csv_path: Destination CSV file path.

        Returns:
            The DataFrame that was written, for further inspection/logging.
        """
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self._rows, columns=CSV_COLUMNS)
        df.to_csv(csv_path, index=False)
        return df

    def unique_track_count(self) -> int:
        """Return the number of distinct track IDs seen across all frames."""
        if not self._rows:
            return 0
        return len({row["track_id"] for row in self._rows})