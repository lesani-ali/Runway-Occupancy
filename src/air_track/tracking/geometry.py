from __future__ import annotations

from enum import Enum
from typing import Optional, Sequence, Tuple

Box = Tuple[float, float, float, float]
Detection = Tuple[Box, float]
Point = Tuple[float, float]
Polygon = Sequence[Point]


class Zone(str, Enum):
    GROUND = "ground"  # on the runway surface (inside the ROI)
    AIR = "air"        # in the airspace right above the runway
    OFF = "off"        # anywhere else (taxiway, apron, far from runway)


def ref_point(box: Box, mode: str = "bottom") -> Point:
    """Reference point of a box: bottom-center (ground contact) or center."""
    x1, y1, x2, y2 = box
    y = y2 if mode == "bottom" else (y1 + y2) / 2.0
    return ((x1 + x2) / 2.0, y)


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


def top_edge_y(x: float, poly: Polygon) -> Optional[float]:
    """Smallest y where the vertical line at x meets the polygon (None if outside)."""
    ys = []
    n = len(poly)
    for i in range(n):
        (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % n]
        if x0 != x1 and min(x0, x1) <= x <= max(x0, x1):
            ys.append(y0 + (y1 - y0) * (x - x0) / (x1 - x0))
    return min(ys) if ys else None


def zone_of(pt: Point, poly: Polygon, air_height_px: float) -> Zone:
    """GROUND inside the ROI, AIR within air_height_px above it, OFF otherwise."""
    if point_in_poly(pt, poly):
        return Zone.GROUND
    top = top_edge_y(pt[0], poly)
    if top is not None and top - air_height_px <= pt[1] < top:
        return Zone.AIR
    return Zone.OFF
