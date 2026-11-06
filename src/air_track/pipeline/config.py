from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class VideoConfig(BaseModel):
    frame_stride: int = 1
    max_frames: Optional[int] = None


class ModelConfig(BaseModel):
    # "torchvision" -> Faster R-CNN / RetinaNet; "ultralytics" -> YOLO (.pt)
    family: Literal["torchvision", "ultralytics"] = "torchvision"
    checkpoint: str = "fasterrcnn_resnet50_fpn"
    device: str = "cpu"
    conf: float = Field(default=0.3, ge=0.0, le=1.0)

    use_sahi: bool = False
    slice_width: int = Field(default=650, gt=0)
    slice_height: int = Field(default=650, gt=0)
    overlap_ratio: float = Field(default=0.2, ge=0.0, le=1.0)

    airplane_class_ids: List[int] = Field(default_factory=lambda: [5])


class RoiConfig(BaseModel):
    polygon: List[List[float]]  # runway surface, [x, y] vertices
    mask_frame: bool = False
    drift_every_sec: float = 0.0  # re-fit ROI to camera drift; 0 = static camera


class TrackingConfig(BaseModel):
    tracker: str = "bytetrack"  # module name in air_track/tracking/
    params: Dict[str, Any] = Field(default_factory=dict)
    stitch_max_gap_sec: float = 3.0
    stitch_max_dist_ratio: float = 3.0


class RotConfig(BaseModel):
    ref_point: Literal["bottom", "center"] = "bottom"
    air_height_ratio: float = 8.0
    min_segment_frames: int = 3
    max_off_gap_sec: float = 2.0
    min_track_frames: int = 5
    min_displacement_px: float = 100.0


class OutputsConfig(BaseModel):
    output_dir: str = "./data"
    save_csv: bool = False
    save_plot: bool = False
    save_img: bool = False
    hide_conf: bool = False
    generate_video: bool = False


class EvaluationConfig(BaseModel):
    iou_threshold: float = 0.5


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_file: Optional[str] = None
    verbose: bool = False


class Config(BaseModel):
    package_name: str = "air_track"
    video: VideoConfig = Field(default_factory=VideoConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    roi: RoiConfig
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    rot: RotConfig = Field(default_factory=RotConfig)
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> Config:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.model_validate(data)
