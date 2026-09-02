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
)

from air_track.detectors import get_detector
from air_track.logging import get_console, get_logger
from air_track.tracking.geometry import Detection
from air_track.utils.preprocessing import apply_roi_mask

logger = get_logger(__name__)


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
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % stride == 0:
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
