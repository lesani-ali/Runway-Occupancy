import yaml
from pathlib import Path
from types import SimpleNamespace


def _to_namespace(obj):
    """Recursively convert dicts to SimpleNamespace."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    return obj


def load_config(config_path: str) -> SimpleNamespace:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return _to_namespace(yaml.safe_load(f))
