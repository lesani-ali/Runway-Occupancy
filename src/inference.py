import cv2
import numpy as np
import torch
from typing import List
from tqdm import tqdm
from rich.console import Console

from sahi.predict import get_sliced_prediction

from preprocessing import apply_roi_mask
from models import load_detection_model

console = Console()


def detect_aircraft_in_frame(
    frame_bgr: np.ndarray,
    detection_model,
    airplane_class_ids: List[int],
    slice_width: int,
    slice_height: int,
    overlap_ratio: float,
) -> tuple[bool, dict]:
    """
    Run Faster R-CNN detection on one frame (with optional SAHI slicing).
    Returns (present, results) where present is True if any airplane is detected.
    """
    use_sahi = detection_model.get('use_sahi', False)
    conf_threshold = detection_model['conf']
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    if use_sahi:
        # Use SAHI sliced prediction
        sahi_model = detection_model['sahi_model']
        result = get_sliced_prediction(
            frame_rgb,
            sahi_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            verbose=0,
        )
        
        # Extract predictions from SAHI result
        boxes = []
        labels = []
        scores = []
        
        for pred in result.object_prediction_list:
            if pred.category.id in airplane_class_ids:
                bbox = pred.bbox.to_voc_bbox()  # [x1, y1, x2, y2]
                boxes.append(bbox)
                labels.append(pred.category.id)
                scores.append(pred.score.value)
        
        filtered_boxes = np.array(boxes) if boxes else np.empty((0, 4))
        filtered_labels = np.array(labels) if labels else np.empty((0,))
        filtered_scores = np.array(scores) if scores else np.empty((0,))
        
    else:
        # Standard inference (full frame)
        model = detection_model['model']
        device = detection_model['device']
        
        # Convert to tensor and normalize
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            predictions = model(frame_tensor)[0]
        
        # Filter by confidence and airplane class (COCO class 5 is airplane)
        boxes = predictions['boxes'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        
        # Filter detections
        mask = (scores >= conf_threshold) & np.isin(labels, airplane_class_ids)
        filtered_boxes = boxes[mask]
        filtered_labels = labels[mask]
        filtered_scores = scores[mask]
    
    # Check if any airplanes were found
    is_present = len(filtered_boxes) > 0
    
    # Store results
    results = {
        'boxes': filtered_boxes,
        'labels': filtered_labels,
        'scores': filtered_scores,
        'frame_shape': frame_bgr.shape
    }

    return is_present, results


def process_video(
    video_path: str,
    config,
    save_img: bool = False,
    save_dir=None,
    hide_conf: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Process video and detect aircraft presence frame by frame.
    Returns (times_sec, present_array, fps).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, int(config.frame_stride))

    # Load detection model
    use_sahi = getattr(config, 'use_sahi', False)
    sahi_status = "enabled" if use_sahi else "disabled"
    console.print(
        f"[cyan][INFO][/cyan]\t\tLoading model: [yellow]{config.weights}[/yellow] on [magenta]{config.device}[/magenta] (SAHI: {sahi_status})"
    )
    
    detection_model = load_detection_model(
        weights=config.weights, device=config.device, conf=config.conf, use_sahi=use_sahi
    )

    console.print(
        f"[cyan][INFO][/cyan]\t\tRunning on: [yellow]{detection_model['device']}[/yellow]"
    )

    times = []
    present = []

    frame_index = 0
    processed = 0

    pbar_total = total_frames // stride if total_frames > 0 else None
    pbar = tqdm(total=pbar_total, desc="Processing frames", unit="frame", ncols=100)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame_index % stride == 0:
            frame_for_det = frame
            if config.roi is not None:
                frame_for_det = apply_roi_mask(frame_for_det, config.roi)

            is_present, results = detect_aircraft_in_frame(
                frame_bgr=frame_for_det,
                detection_model=detection_model,
                airplane_class_ids=config.airplane_class_ids,
                slice_width=config.slice_width,
                slice_height=config.slice_height,
                overlap_ratio=config.overlap_ratio,
            )

            times.append(frame_index / fps)
            present.append(1 if is_present else 0)

            if save_img and save_dir is not None:
                # Draw bounding boxes on frame
                vis_frame = frame.copy()
                for box, score in zip(results['boxes'], results['scores']):
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    if not hide_conf:
                        label = f"{score:.2f}"
                        cv2.putText(vis_frame, label, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Save image
                img_path = f"{save_dir}/img_{processed + 1}.jpg"
                cv2.imwrite(img_path, vis_frame)

            processed += 1
            pbar.update(1)

            if config.max_frames is not None and processed >= config.max_frames:
                break

        frame_index += 1

    pbar.close()
    cap.release()

    times_sec = np.array(times, dtype=np.float64)
    present_array = np.array(present, dtype=np.int32)

    return times_sec, present_array, fps
