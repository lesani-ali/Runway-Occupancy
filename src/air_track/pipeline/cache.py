from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from air_track.tracking.geometry import Detection
from air_track.logging import get_logger

logger = get_logger(__name__)


def save_detections(
    path: Path,
    times_sec: np.ndarray,
    sampled_dets: List[List[Detection]],
    fps: float,
    source: str,
) -> None:
    """Persist raw per-frame detections so tracking can be re-run without re-inference.

    Detection is ``((x1, y1, x2, y2), conf)``; boxes are stored pre-ROI-filter exactly
    as the state machine consumes them, so ROI and tracking params can still be retuned.
    """
    frames = [
        {
            "time_sec": float(t),
            "detections": [[list(box), float(conf)] for box, conf in dets],
        }
        for t, dets in zip(times_sec.tolist(), sampled_dets)
    ]
    payload = {"source": str(source), "fps": float(fps), "frames": frames}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    logger.info(f"Saved detection cache ({len(frames)} frames): {path}")


def load_detections(
    path: str | Path,
) -> Tuple[np.ndarray, List[List[Detection]], float]:
    """Load cached detections written by :func:`save_detections`."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    times: List[float] = []
    sampled_dets: List[List[Detection]] = []
    for frame in data["frames"]:
        times.append(float(frame["time_sec"]))
        sampled_dets.append(
            [
                (tuple(float(v) for v in box), float(conf))
                for box, conf in frame["detections"]
            ]
        )

    return (
        np.array(times, dtype=np.float64),
        sampled_dets,
        float(data["fps"]),
    )
