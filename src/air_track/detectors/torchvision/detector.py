from __future__ import annotations

from typing import List

import cv2
import numpy as np
import torch

from air_track.detectors.base import BaseDetector
from air_track.detectors.registry import register_detector
from air_track.detectors.sahi import sliced_predict
from air_track.detectors.torchvision.model import (
    MODEL_NAME_TO_CONSTRUCTOR,
    build_torchvision_model,
    is_pretrained_name,
    label_offset_for,
)
from air_track.tracking.geometry import Detection


@register_detector("torchvision")
class TorchvisionDetector(BaseDetector):
    """Faster R-CNN / RetinaNet detector (COCO-pretrained or fine-tuned)."""

    def __init__(
        self,
        model,
        device: str,
        conf: float,
        use_sahi: bool,
        slice_width: int,
        slice_height: int,
        overlap_ratio: float,
        sahi_model=None,
    ):
        self.model = model
        self.device = device
        self.conf = conf
        self.use_sahi = use_sahi
        self.slice_width = slice_width
        self.slice_height = slice_height
        self.overlap_ratio = overlap_ratio
        self.sahi_model = sahi_model

    @classmethod
    def load(cls, cfg) -> "TorchvisionDetector":
        # A registry name loads COCO-pretrained weights; anything else is a
        # fine-tuned checkpoint produced by air-track-train.
        if is_pretrained_name(cfg.checkpoint):
            model = MODEL_NAME_TO_CONSTRUCTOR[cfg.checkpoint](weights="DEFAULT")
            category_mapping = None
        else:
            model, category_mapping = _load_checkpoint(cfg.checkpoint)

        model.eval()

        sahi_model = None
        if cfg.use_sahi:
            sahi_model = _build_sahi_model(
                model, cfg.conf, cfg.device, category_mapping
            )
        else:
            model.to(cfg.device)

        return cls(
            model=model,
            device=cfg.device,
            conf=cfg.conf,
            use_sahi=cfg.use_sahi,
            slice_width=cfg.slice_width,
            slice_height=cfg.slice_height,
            overlap_ratio=cfg.overlap_ratio,
            sahi_model=sahi_model,
        )

    def predict(
        self, frame_bgr: np.ndarray, class_ids: List[int]
    ) -> List[Detection]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self.use_sahi:
            return sliced_predict(
                self.sahi_model,
                frame_rgb,
                class_ids,
                self.slice_width,
                self.slice_height,
                self.overlap_ratio,
            )

        frame_tensor = (
            torch.from_numpy(frame_rgb)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.inference_mode():
            pred = self.model(frame_tensor)[0]

        boxes = pred["boxes"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()

        mask = (scores >= self.conf) & np.isin(labels, class_ids)
        return [
            (tuple(map(float, box)), float(score))
            for box, score in zip(boxes[mask], scores[mask])
        ]


def _load_checkpoint(checkpoint: str):
    """Load a fine-tuned torchvision model; return (model, category_mapping)."""
    ckpt = torch.load(checkpoint, map_location="cpu")
    try:
        weights_name = ckpt["weights_name"]
        num_fg = ckpt["num_fg_classes"]
        state = ckpt["model"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Checkpoint '{checkpoint}' is missing training metadata "
            "(expected 'weights_name', 'num_fg_classes', 'model'). "
            "Use a checkpoint produced by air-track-train."
        ) from exc

    class_names = ckpt.get("class_names") or [
        f"class_{i + 1}" for i in range(num_fg)
    ]

    model = build_torchvision_model(
        weights_name,
        num_fg,
        pretrained=False,
        anchor_aspect_ratios=ckpt.get("anchor_aspect_ratios"),
    )
    model.load_state_dict(state)

    offset = label_offset_for(weights_name)
    category_mapping = {
        str(i + offset): name for i, name in enumerate(class_names)
    }
    return model, category_mapping


def _build_sahi_model(model, conf: float, device: str, category_mapping):
    from sahi.models.torchvision import TorchVisionDetectionModel

    kwargs = dict(model=model, confidence_threshold=conf, device=device)
    if category_mapping is not None:
        try:
            return TorchVisionDetectionModel(
                category_mapping=category_mapping, **kwargs
            )
        except TypeError:  # older SAHI without category_mapping kwarg
            pass
    return TorchVisionDetectionModel(**kwargs)
