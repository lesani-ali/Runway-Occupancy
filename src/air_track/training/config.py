from __future__ import annotations

import yaml
from pathlib import Path
from typing import Literal
from pydantic import BaseModel


class TrainConfig(BaseModel):
    package_name: str = "air_track"

    # "torchvision" -> Faster R-CNN / RetinaNet; "ultralytics" -> YOLO
    family: Literal["torchvision", "ultralytics"] = "torchvision"
    weights: str = "fasterrcnn_resnet50_fpn"
    device: str = "cuda:0"

    # Schedule
    epochs: int = 50
    batch_size: int = 2
    lr: float = 0.005
    momentum: float = 0.9          # torchvision SGD
    weight_decay: float = 5e-4     # torchvision SGD
    lr_step_size: int = 20         # torchvision StepLR
    lr_gamma: float = 0.1          # torchvision StepLR
    workers: int = 4
    seed: int = 0

    # Ultralytics-only
    img_size: int = 640

    # Output
    output_dir: str = "runs/train"
    run_name: str = "finetune"

    data_yaml: str

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.model_validate(data)
