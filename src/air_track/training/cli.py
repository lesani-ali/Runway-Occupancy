from __future__ import annotations

import click

from air_track.training.config import TrainConfig
from air_track.logging import setup_logger, get_logger

logger = get_logger(__name__)


@click.command()
@click.option(
    "-c", "--config", "config_path",
    type=click.Path(exists=True), required=True,
    help="Path to training config YAML.",
)
def train(config_path) -> None:
    """Fine-tune a detector (torchvision or YOLO) on a YOLO-format dataset."""
    cfg = TrainConfig.from_yaml(config_path)
    setup_logger(name=cfg.package_name, level="INFO")
    logger.info(f"Loaded training config: {config_path} (family={cfg.family})")

    if cfg.family == "ultralytics":
        from air_track.training import ultralytics_trainer

        ultralytics_trainer.train(cfg)
    else:
        from air_track.training import torchvision_trainer

        torchvision_trainer.train(cfg)


if __name__ == "__main__":
    train()
