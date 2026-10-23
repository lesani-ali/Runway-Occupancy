from __future__ import annotations

from typing import List, Tuple

import numpy as np
import supervision as sv

from air_track.tracking.geometry import Box, Detection
from air_track.tracking.tracker import BaseTracker, register_tracker


@register_tracker("bytetrack")
class ByteTracker(BaseTracker):
    """ByteTrack from ``supervision``; params are passed to ``sv.ByteTrack``."""

    def __init__(self, frame_rate: float, **params) -> None:
        self._tracker = sv.ByteTrack(frame_rate=round(frame_rate), **params)

    def update(self, dets: List[Detection]) -> List[Tuple[int, Box]]:
        sv_dets = sv.Detections(
            xyxy=np.array([box for box, _ in dets], dtype=float).reshape(-1, 4),
            confidence=np.array([conf for _, conf in dets], dtype=float),
            class_id=np.zeros(len(dets), dtype=int),
        )
        out = self._tracker.update_with_detections(sv_dets)
        return [
            (int(tid), tuple(float(v) for v in box))
            for tid, box in zip(out.tracker_id, out.xyxy)
        ]
