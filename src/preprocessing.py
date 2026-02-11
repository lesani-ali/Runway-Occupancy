import cv2
import numpy as np
from typing import List, Tuple, Union


ROIType = Union[Tuple[int, int, int, int], List[Tuple[int, int]]]


def apply_roi_mask(frame_bgr: np.ndarray, roi: ROIType) -> np.ndarray:
    """
    Mask everything outside ROI to black. ROI can be:
      - (x1, y1, x2, y2) rectangle tuple
      - list of (x, y) polygon points
    """
    h, w = frame_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if isinstance(roi, tuple) and len(roi) == 4:
        # Rectangle ROI
        x1, y1, x2, y2 = roi
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    else:
        # Polygon ROI
        pts = np.array(roi, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    out = frame_bgr.copy()
    out[mask == 0] = 0
    return out
