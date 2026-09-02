from __future__ import annotations

from typing import List

import cv2
import numpy as np

from air_track.detectors.base import BaseDetector
from air_track.detectors.registry import register_detector
from air_track.detectors.sahi import sliced_predict
from air_track.tracking.geometry import Detection


@register_detector("ultralytics")
class UltralyticsDetector(BaseDetector):
    """YOLO detector. A .pt bundles architecture + weights + names, so a
    pretrained name and a fine-tuned path load the same way."""

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
    def load(cls, cfg) -> "UltralyticsDetector":
        from ultralytics import YOLO

        model = YOLO(cfg.checkpoint)

        sahi_model = None
        if cfg.use_sahi:
            from sahi.models.ultralytics import UltralyticsDetectionModel

            sahi_model = UltralyticsDetectionModel(
                model=model,
                confidence_threshold=cfg.conf,
                device=cfg.device,
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
        if self.use_sahi:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            return sliced_predict(
                self.sahi_model,
                frame_rgb,
                class_ids,
                self.slice_width,
                self.slice_height,
                self.overlap_ratio,
            )

        # Direct (non-SAHI) YOLO. ultralytics expects BGR numpy input.
        results = self.model.predict(
            frame_bgr,
            conf=self.conf,
            classes=list(class_ids),
            device=self.device,
            verbose=False,
        )
        dets: List[Detection] = []
        if results and results[0].boxes is not None:
            r = results[0].boxes
            for xyxy, score in zip(
                r.xyxy.cpu().numpy(), r.conf.cpu().numpy()
            ):
                dets.append((tuple(float(v) for v in xyxy), float(score)))
        return dets
