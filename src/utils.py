from pathlib import Path
import os

from rich.console import Console


# Shared console
console = Console()


def hms(sec: float) -> str:
    """Convert seconds → HH:MM:SS string."""
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def increment_path(path, exist_ok: bool = False) -> Path:
    """Increment path if it exists, e.g. runs/exp → runs/exp-2."""
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        for n in range(2, 9999):
            p = f"{path}-{n}{suffix}"
            if not os.path.exists(p):
                break
        path = Path(p)
    return path