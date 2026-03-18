import yaml
from pathlib import Path

from air_track.config.schema import Config


def load_config(config_path: str) -> Config:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config_dict = yaml.safe_load(f) or {}

    return Config(**config_dict)
