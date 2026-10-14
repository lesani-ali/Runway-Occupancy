from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import numpy as np

from air_track.tracking.geometry import Detection


class BaseDetector(ABC):
    """Inference interface every detector must implement.

    One instance wraps a ready-to-infer model. Everything model-specific
    (weight loading, tensor prep, SAHI slicing, label conventions) lives behind
    ``predict``; the rest of the pipeline only ever sees ``list[Detection]``.

    Register a concrete subclass with ``@register_detector("<family>")`` so the
    config's ``model.family`` string can resolve to it.
    """

    name: str  # set by @register_detector

    @classmethod
    @abstractmethod
    def load(cls, cfg) -> "BaseDetector":
        """Build a ready-to-infer detector from an inference ``ModelConfig``.

        ``cfg.checkpoint`` may be a known pretrained name or a fine-tuned
        checkpoint path — the detector decides how to interpret it.
        """

    @abstractmethod
    def predict(
        self, frame_bgr: np.ndarray, class_ids: List[int]
    ) -> List[Detection]:
        """Detect on one BGR frame.

        Returns ``[((x1, y1, x2, y2), conf), ...]`` keeping only ``class_ids``.
        """

    @abstractmethod
    def validate(
        self, data_yaml: str, class_ids: List[int], split: str = "val"
    ) -> dict:
        """Evaluate on the ``split`` (``"train"`` or ``"val"``) of a YOLO ``data.yaml``.

        Returns ``{"map_50", "map_50_95", "preds", "targets"}``; ``preds`` and
        ``targets`` are the per-image torchmetrics-style dicts used for the mAP.
        """


class BaseTrainer(ABC):
    """Training interface every fine-tunable detector must implement.

    The dataset is always YOLO detection format; a trainer is responsible for
    adapting it to whatever its model expects (label offsets, box format, …).

    Register a concrete subclass with ``@register_trainer("<family>")``.
    """

    name: str  # set by @register_trainer

    @classmethod
    @abstractmethod
    def train(cls, cfg) -> Path:
        """Fine-tune on a YOLO-format dataset; return the best checkpoint path."""
