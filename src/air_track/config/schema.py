from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class VideoConfig(BaseModel):
    frame_stride: int = 1
    max_frames: Optional[int] = None


class ModelConfig(BaseModel):
    weights: str = "fasterrcnn_resnet50_fpn"
    device: str = "cpu"
    conf: float = 0.3

    use_sahi: bool = False
    slice_width: int = 650
    slice_height: int = 650
    overlap_ratio: float = 0.2

    airplane_class_ids: List[int] = Field(default_factory=lambda: [5])


class RoiConfig(BaseModel):
    polygon: Optional[List[List[int]]] = None
    mask_frame: bool = False


class TrackingConfig(BaseModel):
    # IDLE → ACTIVE trigger
    start_conf: float = 0.5
    start_streak_k: int = 3
    start_streak_n: int = 5

    # ACTIVE keep-alive
    keep_conf: float = 0.4
    gate_px: float = 500.0
    max_pos_uncertainty: float = 5000.0

    # LOST recovery
    recover_conf: float = 0.5
    lost_timeout: int = 30

    # Kalman filter
    process_var: float = 70.0
    meas_var: float = 100.0


class OutputsConfig(BaseModel):
    output_dir: str = "./data"
    save_csv: bool = False
    save_plot: bool = False
    save_img: bool = False
    hide_conf: bool = False
    generate_video: bool = False


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_file: Optional[str] = None
    verbose: bool = False


class Config(BaseModel):
    video: VideoConfig = Field(default_factory=VideoConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    roi: RoiConfig = Field(default_factory=RoiConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
