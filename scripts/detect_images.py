#!/usr/bin/env python3
"""Run an aircraft detector on images and save the results.

Takes one image or a folder of images. For each image it writes a copy with
the detected boxes drawn on it, and all boxes go into one detections.json.
Outputs land in a fresh folder under --out (runs/detect/<source>, -2, -3, ...).

    python scripts/detect_images.py -s data/model_train/data-final/images/val
    python scripts/detect_images.py -s frame.jpg --sahi --conf 0.4
    python scripts/detect_images.py -s frame.jpg -w yolo.pt -f ultralytics
"""
from __future__ import annotations

import json
from pathlib import Path

import click
import cv2
import torch

from air_track.detectors import get_detector
from air_track.detectors.torchvision.model import is_pretrained_name, label_offset_for
from air_track.pipeline.config import ModelConfig
from air_track.utils import increment_path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
BOX_COLOR = (0, 255, 0)  # BGR


def default_class_id(family: str, weights: str) -> int:
    """Aircraft class id as each model type outputs it."""
    if family == "torchvision":
        # COCO "airplane" = 5; fine-tuned: Faster R-CNN 1 (0 = background), RetinaNet 0
        if is_pretrained_name(weights):
            return 5
        return label_offset_for(torch.load(weights, map_location="cpu")["weights_name"])
    # YOLO checkpoints store their class names: COCO {4: "airplane"}, ours {0: "aircraft"}
    from ultralytics import YOLO

    names = YOLO(weights).names
    for class_id, name in names.items():
        if name in ("aircraft", "airplane"):
            return class_id
    raise click.ClickException(f"No aircraft class in {weights}: {names}")


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def draw_boxes(image, dets, hide_conf: bool):
    vis = image.copy()
    for (x1, y1, x2, y2), score in dets:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(vis, p1, p2, BOX_COLOR, 2)
        if not hide_conf:
            org = (p1[0], max(p1[1] - 6, 12))
            cv2.putText(
                vis, f"{score:.2f}", org,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 2,
            )
    return vis


@click.command()
@click.option(
    "-s", "--source", type=click.Path(exists=True, path_type=Path),
    required=True, help="An image file or a folder of images.",
)
@click.option(
    "-w", "--weights", default="runs/Fast-RCNN/best.pt", show_default=True,
    help="Checkpoint path, or a pretrained name (e.g. fasterrcnn_resnet50_fpn).",
)
@click.option(
    "-f", "--family", type=click.Choice(["torchvision", "ultralytics"]),
    default="torchvision", show_default=True,
)
@click.option("--conf", default=0.5, show_default=True, help="Min score to draw/save.")
@click.option("--device", default="cuda:0", show_default=True)
@click.option(
    "--class-id", type=int, default=None,
    help="Aircraft class id. Default: picked from family and weights.",
)
@click.option(
    "--sahi", is_flag=True,
    help="Sliced inference, for large full frames (e.g. fisheye).",
)
@click.option("--slice-size", default=640, show_default=True)
@click.option(
    "-o", "--out", type=click.Path(path_type=Path),
    default="runs/detect", show_default=True,
)
@click.option("--hide-conf", is_flag=True, help="Don't print scores on boxes.")
def main(
    source, weights, family, conf, device, class_id, sahi, slice_size, out, hide_conf,
):
    """Detect aircraft in images."""
    images = collect_images(source)
    if not images:
        raise click.ClickException(f"No images found in {source}")

    if class_id is None:
        class_id = default_class_id(family, weights)

    model_cfg = ModelConfig(
        family=family,
        checkpoint=weights,
        device=device,
        conf=conf,
        use_sahi=sahi,
        slice_width=slice_size,
        slice_height=slice_size,
        airplane_class_ids=[class_id],
    )
    click.echo(f"Loading {family} model: {weights} (class id {class_id})")
    detector = get_detector(family).load(model_cfg)

    save_dir = increment_path(out / source.stem)
    save_dir.mkdir(parents=True)

    results = {}
    with click.progressbar(images, label="Detecting") as bar:
        for path in bar:
            image = cv2.imread(str(path))
            if image is None:
                click.echo(f"\nSkipping unreadable image: {path}", err=True)
                continue

            dets = detector.predict(image, [class_id])
            results[path.name] = [
                {"box": [round(v, 1) for v in box], "score": round(score, 4)}
                for box, score in dets
            ]
            cv2.imwrite(str(save_dir / path.name), draw_boxes(image, dets, hide_conf))

    (save_dir / "detections.json").write_text(json.dumps(results, indent=2))

    n_boxes = sum(len(d) for d in results.values())
    n_hit = sum(1 for d in results.values() if d)
    click.echo(
        f"{n_boxes} aircraft in {n_hit}/{len(results)} images. "
        f"Saved to {save_dir}"
    )


if __name__ == "__main__":
    main()
