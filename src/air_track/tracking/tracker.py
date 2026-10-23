from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Tuple, Type

from rich.progress import track

from air_track.tracking.geometry import Box, Detection
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

