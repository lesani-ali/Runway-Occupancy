from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    track,
)

from air_track.detectors import get_detector
from air_track.logging import get_console, get_logger
from air_track.tracking.geometry import Detection, Polygon
from air_track.utils.preprocessing import apply_roi_mask

logger = get_logger(__name__)

DRIFT_PATCH_W, DRIFT_PATCH_H = 512, 256  # image patch around each ROI corner
DRIFT_MIN_CONF = 0.2  # weaker phase-correlation peaks (e.g. a passing aircraft) are ignored


def process_video(
    video_path: str,
    config,
    save_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[List[Detection]], float]:
    """Run a detector over a video, returning (times_sec, per-frame dets, fps)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    vid_cfg = config.video
    mdl_cfg = config.model
    roi_cfg = config.roi
    out_cfg = config.outputs

    stride = max(1, int(vid_cfg.frame_stride))

    logger.info(
        f"Loading {mdl_cfg.family} detector {mdl_cfg.checkpoint} on "
        f"{mdl_cfg.device} (SAHI: {'yes' if mdl_cfg.use_sahi else 'no'})"
    )
    detector = get_detector(mdl_cfg.family).load(mdl_cfg)
    class_ids = mdl_cfg.airplane_class_ids

    roi = roi_cfg.polygon

    times: List[float] = []
    sampled_dets: List[List[Detection]] = []

    frame_index = 0
    processed = 0
    pbar_total = total_frames // stride if total_frames > 0 else None

    with Progress(
        TextColumn("[green]{task.description}[/green]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=get_console(),
    ) as progress:
        task = progress.add_task("Processing frames", total=pbar_total)

        while cap.isOpened():
            # Skipped frames are only grabbed (no retrieve/convert to BGR)
            if frame_index % stride != 0:
                if not cap.grab():
                    break
                frame_index += 1
                continue

            ok, frame = cap.read()
            if not ok:
                break

            frame_for_det = frame
            if roi is not None and roi_cfg.mask_frame:
                frame_for_det = apply_roi_mask(frame, roi)

            dets = detector.predict(frame_for_det, class_ids)

            times.append(frame_index / fps)
            sampled_dets.append(dets)

            if out_cfg.save_img and save_dir is not None:
                _save_annotated_frame(
                    frame, dets, processed, save_dir, out_cfg.hide_conf
                )

            processed += 1
            progress.update(task, advance=1)

            if (
                vid_cfg.max_frames is not None
                and processed >= vid_cfg.max_frames
            ):
                break

            frame_index += 1

    cap.release()

    return (
        np.array(times, dtype=np.float64),
        sampled_dets,
        fps,
    )


def _save_annotated_frame(
    frame: np.ndarray,
    dets: List[Detection],
    frame_num: int,
    save_dir: Path,
    hide_conf: bool,
) -> None:
    """Draw detection boxes on a frame copy and save to disk."""
    vis = frame.copy()
    for box, score in dets:
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if not hide_conf:
            cv2.putText(
                vis,
                f"{score:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
    cv2.imwrite(str(save_dir / f"img_{frame_num + 1}.jpg"), vis)


def roi_per_frame(
    video_path: str,
    polygon: Polygon,
    times_sec: np.ndarray,
    every_sec: float,
) -> List[Polygon]:
    """ROI for each sampled frame, following slow camera drift.

    Every `every_sec` seconds each polygon corner moves down/up by how much the
    image around it shifted since the first frame (phase correlation). The shift
    differs along the runway, so corners move separately. 0 = static camera.
    """
    if not every_sec:
        return [polygon] * len(times_sec)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    def gray(sec: float) -> Optional[np.ndarray]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(sec * fps))
        ok, frame = cap.read()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) if ok else None

    ref = gray(0.0)
    img_h, img_w = ref.shape
    window = cv2.createHanningWindow((DRIFT_PATCH_W, DRIFT_PATCH_H), cv2.CV_32F)
    # Top-left corner of the patch around each polygon vertex, kept inside the image
    patches = [
        (int(np.clip(x - DRIFT_PATCH_W / 2, 0, img_w - DRIFT_PATCH_W)),
         int(np.clip(y - DRIFT_PATCH_H / 2, 0, img_h - DRIFT_PATCH_H)))
        for x, y in polygon
    ]

    # Only the sample times the frames actually need
    keys = np.round(np.asarray(times_sec) / every_sec).astype(int)
    shifts = np.zeros(len(polygon))
    rois = {}
    for key in track(
        np.unique(keys), "Measuring camera drift", console=get_console()
    ):
        img = gray(key * every_sec)
        for i, (x, y) in enumerate(patches):
            if img is None:
                break
            # Copies: phaseCorrelate writes the windowed patch back into a slice
            (_, dy), conf = cv2.phaseCorrelate(
                ref[y:y + DRIFT_PATCH_H, x:x + DRIFT_PATCH_W].copy(),
                img[y:y + DRIFT_PATCH_H, x:x + DRIFT_PATCH_W].copy(),
                window,
            )
            if conf >= DRIFT_MIN_CONF:  # otherwise keep the last good shift
                shifts[i] = dy
        rois[key] = [(x, y + dy) for (x, y), dy in zip(polygon, shifts)]
    cap.release()

    logger.info(f"ROI corners moved {shifts.min():+.1f} to {shifts.max():+.1f} px by the last frame")
    return [rois[key] for key in keys]
