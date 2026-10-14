"""Pluggable detectors.

Each detector family lives in its own subpackage (``detectors/<family>/``) with a
``detector.py`` (inference) and, if fine-tunable, a ``trainer.py`` (training).
They self-register via ``@register_detector`` / ``@register_trainer`` and are
resolved by name through :func:`get_detector` / :func:`get_trainer`, so adding a
new detector needs no edits outside its own folder.
"""

from air_track.detectors.base import BaseDetector, BaseTrainer
from air_track.detectors.registry import (
    get_detector,
    get_trainer,
    register_detector,
    register_trainer,
)

__all__ = [
    "BaseDetector",
    "BaseTrainer",
    "register_detector",
    "register_trainer",
    "get_detector",
    "get_trainer",
]