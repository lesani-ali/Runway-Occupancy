from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
import numpy as np

Box = Tuple[float, float, float, float]
Detection = Tuple[Box, float]
Point = Tuple[float, float]
Polygon = Sequence[Point]


def box_center(box: Box) -> Point:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_in_poly(pt: Point, poly: Polygon) -> bool:
    """Ray casting point-in-polygon. poly is a list of (x, y) vertices."""
    n = len(poly)
    if n < 3:
        return False
    x, y = pt
    inside = False
    x0, y0 = poly[0]
    for i in range(1, n + 1):
        x1, y1 = poly[i % n]
        # Check if point is between y0 and y1
        if (y0 > y) != (y1 > y):
            # Compute x-coordinate of intersection with the ray
            x_intersect = (x1 - x0) * (y - y0) / (y1 - y0 + 1e-12) + x0
            if x < x_intersect:
                inside = not inside
        x0, y0 = x1, y1
    return inside


def filter_by_roi(
    dets: List[Detection],
    roi_poly: Optional[Polygon],
) -> List[Detection]:
    """Filter detections by ROI polygon using box center."""
    if not roi_poly or len(roi_poly) < 3:
        return dets
    return [
        (box, conf) for box, conf in dets if point_in_poly(box_center(box), roi_poly)
    ]


def passes_gating(box: Box, pred_xy: Point, gate_dist_px: float) -> bool:
    """Return True if the box-center is within gate_dist_px of pred_xy."""
    cx, cy = box_center(box)
    dist = float(np.hypot(cx - pred_xy[0], cy - pred_xy[1]))
    return dist <= gate_dist_px


def select_highest_conf(cands: List[Detection]) -> Optional[Detection]:
    """Return the detection with the highest confidence, or None."""
    return max(cands, key=lambda d: d[1]) if cands else None


def select_best_near_prediction(
    cands: List[Detection],
    pred_xy: Point,
) -> Optional[Detection]:
    """Return the detection whose center is closest to pred_xy, or None."""
    if not cands:
        return None
    px, py = pred_xy
    return min(
        cands,
        key=lambda d: np.hypot(box_center(d[0])[0] - px, box_center(d[0])[1] - py),
        default=None,
    )
