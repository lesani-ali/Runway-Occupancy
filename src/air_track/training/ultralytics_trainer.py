from __future__ import annotations

from ultralytics import YOLO
from air_track.logging import get_logger

logger = get_logger(__name__)


def train(cfg) -> None:
    model = YOLO(cfg.weights)
    model.train(
        data=cfg.data_yaml,
        epochs=cfg.epochs,
        imgsz=cfg.img_size,
        batch=cfg.batch_size,
        device=cfg.device,
        project=cfg.output_dir,
        name=cfg.run_name,
        seed=cfg.seed,
    )
    logger.info(
        "YOLO training complete. Best weights at "
        f"{cfg.output_dir}/{cfg.run_name}/weights/best.pt"
    )
