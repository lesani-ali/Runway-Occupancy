from __future__ import annotations

import importlib
from typing import Dict, Type

from air_track.detectors.base import BaseDetector, BaseTrainer

_DETECTORS: Dict[str, Type[BaseDetector]] = {}
_TRAINERS: Dict[str, Type[BaseTrainer]] = {}


def register_detector(name: str):
    """Class decorator: register a :class:`BaseDetector` under ``name``."""

    def deco(cls: Type[BaseDetector]) -> Type[BaseDetector]:
        cls.name = name
        _DETECTORS[name] = cls
        return cls

    return deco


def register_trainer(name: str):
    """Class decorator: register a :class:`BaseTrainer` under ``name``."""

    def deco(cls: Type[BaseTrainer]) -> Type[BaseTrainer]:
        cls.name = name
        _TRAINERS[name] = cls
        return cls

    return deco


def _ensure_family_imported(family: str, module: str) -> None:
    """Import ``air_track.detectors.<family>.<module>`` so its decorator runs.

    A missing *family* package becomes a friendly error; a missing heavy
    dependency inside the module (e.g. torch, ultralytics) is re-raised as-is.
    """
    pkg = f"air_track.detectors.{family}"
    try:
        importlib.import_module(f"{pkg}.{module}")
    except ModuleNotFoundError as exc:
        if exc.name and (exc.name == pkg or exc.name.startswith(pkg + ".")):
            raise ValueError(
                f"Unknown detector family '{family}'. "
                f"Expected a package at air_track/detectors/{family}/."
            ) from exc
        raise  # a real missing dependency inside the family module


def get_detector(family: str) -> Type[BaseDetector]:
    """Return the detector class for ``family`` (lazy-imports it on first use)."""
    if family not in _DETECTORS:
        _ensure_family_imported(family, "detector")
    if family not in _DETECTORS:
        raise ValueError(
            f"No detector registered for family '{family}'. "
            f"Available: {sorted(_DETECTORS)}"
        )
    return _DETECTORS[family]


def get_trainer(family: str) -> Type[BaseTrainer]:
    """Return the trainer class for ``family`` (lazy-imports it on first use)."""
    if family not in _TRAINERS:
        _ensure_family_imported(family, "trainer")
    if family not in _TRAINERS:
        raise ValueError(
            f"No trainer registered for family '{family}'. "
            f"Available: {sorted(_TRAINERS)}"
        )
    return _TRAINERS[family]
