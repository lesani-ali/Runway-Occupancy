"""Pixel-level ROI masking applied to frames before detection."""

import cv2
import numpy as np
from typing import List, Tuple, Union


ROIType = Union[Tuple[int, int, int, int], List[Tuple[int, int]]]


def apply_roi_mask(frame_bgr: np.ndarray, roi: ROIType) -> np.ndarray:
    """
    Zero out all pixels outside *roi*.

    *roi* can be:
      - (x1, y1, x2, y2)  rectangular tuple
      - [(x, y), …]       polygon vertices
    """
    h, w = frame_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if (
        isinstance(roi, (tuple, list))
        and len(roi) == 4
        and isinstance(roi[0], (int, float))
    ):
        x1, y1, x2, y2 = roi
        cv2.rectangle(
            mask,
            (max(0, int(x1)), max(0, int(y1))),
            (min(w, int(x2)), min(h, int(y2))),
            255, -1,
        )
    else:
        pts = np.array(roi, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    out = frame_bgr.copy()
    out[mask == 0] = 0
    return out
