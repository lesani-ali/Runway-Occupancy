import yaml
from pathlib import Path
from types import SimpleNamespace


def load_config(config_path: str) -> SimpleNamespace:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return SimpleNamespace(**config_dict)
