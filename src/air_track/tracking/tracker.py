from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Tuple, Type

import numpy as np
from rich.progress import track

from air_track.tracking.geometry import Box, Detection, ref_point
from air_track.logging import get_console

Track = List[Tuple[int, Box]]  # (frame index, box) per observation

_TRACKERS: Dict[str, Type[BaseTracker]] = {}


class BaseTracker(ABC):
    """Multi-object tracker built as ``cls(frame_rate, **params)``.

    ``update`` is called once per sampled frame, in order.
    """

    @abstractmethod
    def update(self, dets: List[Detection]) -> List[Tuple[int, Box]]:
        """Associate one frame's detections; return (track_id, box) pairs."""


def register_tracker(name: str):
    """Class decorator: register a :class:`BaseTracker` under ``name``."""

    def deco(cls: Type[BaseTracker]) -> Type[BaseTracker]:
        _TRACKERS[name] = cls
        return cls

    return deco


def get_tracker(name: str) -> Type[BaseTracker]:
    """Return the tracker class for ``name`` (module ``tracking/<name>.py``)."""
    if name not in _TRACKERS:
        importlib.import_module(f"air_track.tracking.{name}")
    return _TRACKERS[name]


def run_tracker(
    tracker: BaseTracker, detections: List[List[Detection]]
) -> Dict[int, Track]:
    """Run the tracker over all frames and group observations by track id."""
    tracks: Dict[int, Track] = defaultdict(list)
    for t, dets in enumerate(
        track(detections, description="Tracking", console=get_console())
    ):
        for track_id, box in tracker.update(dets):
            tracks[track_id].append((t, box))
    return dict(tracks)

def stitch_tracks(
    tracks: Dict[int, Track],
    times_sec: np.ndarray,
    max_gap_sec: float,
    max_dist_ratio: float,
) -> Dict[int, Track]:
    """Join a track that ends with one that starts shortly after, close by.

    Re-links fragments the tracker split when IoU matching failed (small
    boxes, a bad box corrupting the motion model). A pair is joined when the
    gap is at most ``max_gap_sec`` and the jump between the last and first
    box centers is at most ``max_dist_ratio`` mean box widths. Each track is
    joined at most once on each side, closest pairs (in time) first.
    """
    pairs = []
    for a, ta in tracks.items():
        end_f, end_box = ta[-1]
        for b, tb in tracks.items():
            start_f, start_box = tb[0]
            gap = times_sec[start_f] - times_sec[end_f]
            if a == b or not 0 < gap <= max_gap_sec:
                continue
            width = (end_box[2] - end_box[0] + start_box[2] - start_box[0]) / 2
            dist = np.linalg.norm(
                np.subtract(ref_point(start_box, "center"),
                            ref_point(end_box, "center"))
            )
            if dist <= max_dist_ratio * width:
                pairs.append((gap, dist, a, b))

    successor: Dict[int, int] = {}
    has_pred = set()
    for _, _, a, b in sorted(pairs):
        if a not in successor and b not in has_pred:
            successor[a] = b
            has_pred.add(b)

    stitched: Dict[int, Track] = {}
    for tid in tracks:
        if tid in has_pred:
            continue
        chain, cur = list(tracks[tid]), tid
        while cur in successor:
            cur = successor[cur]
            chain.extend(tracks[cur])
        stitched[tid] = chain
    return stitched
