from pathlib import Path
import os

def increment_path(path, exist_ok=False):
    """Increment path if it exists, e.g. runs/exp -> runs/exp2
    
    Args:
        path (Path): Path to the directory.
        exist_ok (bool): If True, do not increment the path.
    
    Returns:
        Path: Incremented path.
    """
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")

        # Method 1
        for n in range(2, 9999):
            p = f"{path}-{n}{suffix}"  # increment path
            if not os.path.exists(p):
                break
        path = Path(p)

    return path