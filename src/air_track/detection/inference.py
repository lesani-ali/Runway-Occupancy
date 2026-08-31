from __future__ import annotations

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Tuple

from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)

from sahi.predict import get_sliced_prediction

from air_track.utils.preprocessing import apply_roi_mask
from air_track.detection.models import load_detection_model
from air_track.tracking.geometry import Detection
from air_track.logging import get_logger, get_console

logger = get_logger(__name__)


def detect_aircraft_in_frame(
    frame_bgr: np.ndarray,
    detection_model: dict,
    airplane_class_ids: List[int],
    slice_width: int,
    slice_height: int,
    overlap_ratio: float,
) -> Tuple[List[Detection], dict]:
    """Run detection on one frame, return typed detections and raw results."""
    model = detection_model["model"]
    use_sahi = detection_model.get("use_sahi", False)
    conf_threshold = detection_model["conf"]

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    if use_sahi:
        result = get_sliced_prediction(
            frame_rgb,
            model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            verbose=0,
        )
        boxes, labels, scores = [], [], []
        for pred in result.object_prediction_list:
            if pred.category.id in airplane_class_ids:
                boxes.append(pred.bbox.to_voc_bbox())
                labels.append(pred.category.id)
                scores.append(pred.score.value)
    elif detection_model.get("family") == "ultralytics":
        # Direct (non-SAHI) YOLO. ultralytics expects BGR numpy input.
        boxes, labels, scores = _predict_ultralytics_direct(
            model,
            frame_bgr,
            conf_threshold,
            airplane_class_ids,
            detection_model["device"],
        )
    else:
        device = detection_model["device"]
        frame_tensor = (
            torch.from_numpy(frame_rgb)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(device)
        )
        with torch.inference_mode():
            predictions = model(frame_tensor)[0]

        all_boxes = predictions["boxes"].cpu().numpy()
        all_labels = predictions["labels"].cpu().numpy()
        all_scores = predictions["scores"].cpu().numpy()

        mask = (all_scores >= conf_threshold) & np.isin(
            all_labels, airplane_class_ids
        )
        boxes = all_boxes[mask].tolist()
        labels = all_labels[mask].tolist()
        scores = all_scores[mask].tolist()

    dets = [
        (tuple(map(float, box)), float(score))
        for box, score in zip(boxes, scores)
    ]
    raw_results = {
        "boxes": np.array(boxes, dtype=np.float32),
        "labels": np.array(labels, dtype=np.int32),
        "scores": np.array(scores, dtype=np.float32),
    }
    return dets, raw_results


def _predict_ultralytics_direct(
    model,
    frame_bgr: np.ndarray,
    conf: float,
    airplane_class_ids: List[int],
    device: str,
) -> Tuple[list, list, list]:
    """Run a YOLO model directly (no SAHI) and return (boxes, labels, scores)."""
    results = model.predict(
        frame_bgr,
        conf=conf,
        classes=list(airplane_class_ids),
        device=device,
        verbose=False,
    )
    boxes, labels, scores = [], [], []
    if results and results[0].boxes is not None:
        r = results[0].boxes
        for xyxy, score, cls in zip(
            r.xyxy.cpu().numpy(),
            r.conf.cpu().numpy(),
            r.cls.cpu().numpy(),
        ):
            boxes.append([float(v) for v in xyxy])
            scores.append(float(score))
            labels.append(int(cls))
    return boxes, labels, scores


def process_video(
    video_path: str,
    config,
    save_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[List[Detection]], float]:

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
    use_sahi = mdl_cfg.use_sahi

    logger.info(
        f"Loading model {mdl_cfg.checkpoint} on {mdl_cfg.device} (SAHI: {'yes' if use_sahi else 'no'})",
    )
    detection_model = load_detection_model(
        checkpoint=mdl_cfg.checkpoint,
        device=mdl_cfg.device,
        conf=mdl_cfg.conf,
        use_sahi=use_sahi,
        family=mdl_cfg.family,
    )
    logger.info(f"Device: {detection_model['device']}")

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

                dets, raw_results = detect_aircraft_in_frame(
                    frame_bgr=frame_for_det,
                    detection_model=detection_model,
                    airplane_class_ids=mdl_cfg.airplane_class_ids,
                    slice_width=mdl_cfg.slice_width,
                    slice_height=mdl_cfg.slice_height,
                    overlap_ratio=mdl_cfg.overlap_ratio,
                )

                times.append(frame_index / fps)
                sampled_dets.append(dets)

                if out_cfg.save_img and save_dir is not None:
                    _save_annotated_frame(
                        frame,
                        raw_results,
                        processed,
                        save_dir,
                        out_cfg.hide_conf,
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
    raw_results: dict,
    frame_num: int,
    save_dir: Path,
    hide_conf: bool,
) -> None:
    """Draw bounding boxes on a frame copy and save to disk."""
    vis = frame.copy()
    for box, score in zip(raw_results["boxes"], raw_results["scores"]):
        x1, y1, x2, y2 = box.astype(int)
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
