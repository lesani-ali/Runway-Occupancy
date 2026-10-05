from __future__ import annotations

from pathlib import Path

from air_track.detectors.base import BaseTrainer
from air_track.detectors.registry import register_trainer
from air_track.logging import get_logger

logger = get_logger(__name__)


@register_trainer("ultralytics")
class UltralyticsTrainer(BaseTrainer):
    """Fine-tune a YOLO detector. Ultralytics consumes the YOLO-format
    ``data.yaml`` directly, so no dataset adaptation is needed. The optimizer
    and learning rate are left to Ultralytics' ``optimizer="auto"``."""

    @classmethod
    def train(cls, cfg) -> Path:
        from ultralytics import YOLO

        model = YOLO(cfg.weights)
        model.train(
            data=cfg.data_yaml,
            epochs=cfg.epochs,
            imgsz=cfg.img_size,
            batch=cfg.batch_size,
            workers=cfg.workers,
            fliplr=cfg.hflip_prob,
            device=cfg.device,
            project=(Path(cfg.output_dir).resolve()),
            name=cfg.run_name,
            seed=cfg.seed,
        )
        best = Path(cfg.output_dir) / cfg.run_name / "weights" / "best.pt"
        logger.info(f"YOLO training complete. Best weights at {best}")
        return best
