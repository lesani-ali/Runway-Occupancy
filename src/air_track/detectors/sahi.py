from __future__ import annotations

from typing import List

import numpy as np
from sahi.predict import get_sliced_prediction

from air_track.tracking.geometry import Detection


def sliced_predict(
    sahi_model,
    frame_rgb: np.ndarray,
    class_ids: List[int],
    slice_width: int,
    slice_height: int,
    overlap_ratio: float,
) -> List[Detection]:
    """Run SAHI sliced inference and return typed detections.

    The slicing loop is family-agnostic; only the ``sahi_model`` adapter differs
    between detectors, so each detector builds its own and shares this helper.
    """
    result = get_sliced_prediction(
        frame_rgb,
        sahi_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        verbose=0,
    )
    dets: List[Detection] = []
    for pred in result.object_prediction_list:
        if pred.category.id in class_ids:
            box = pred.bbox.to_voc_bbox()
            dets.append((tuple(float(v) for v in box), float(pred.score.value)))
    return dets
