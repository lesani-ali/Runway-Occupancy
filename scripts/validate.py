#!/usr/bin/env python3
"""Validate an aircraft detector on a YOLO data.yaml split.

Runs the detector's validate() on --split (val by default). Writes mAP@.5,
mAP@.5:.95 and the best-F1 confidence to <out>/<split>.txt, and a precision /
recall / F1 vs confidence plot to <out>/<split>_pr_f1_vs_conf.png.

    python scripts/validate.py -d data/model_train/data-final/data.yaml
    python scripts/validate.py -d data.yaml --split train
    python scripts/validate.py -d data.yaml -w runs/yolo26s/weights/best.pt -f ultralytics
"""
from __future__ import annotations

from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.ops import box_iou

from air_track.detectors import get_detector
from air_track.detectors.torchvision.model import is_pretrained_name, label_offset_for
from air_track.pipeline.config import ModelConfig


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


def match_detections(preds, targets, iou_thr: float = 0.5):
    """Mark each prediction TP/FP at ``iou_thr``; return (scores, is_tp, n_gt)."""
    scores, is_tp, n_gt = [], [], 0
    for pred, target in zip(preds, targets):
        n_gt += len(target["boxes"])
        order = pred["scores"].argsort(descending=True)
        boxes, conf = pred["boxes"][order], pred["scores"][order]
        ious = box_iou(boxes, target["boxes"])
        matched = torch.zeros(len(target["boxes"]), dtype=torch.bool)
        # Greedy, highest score first: each GT box is matched at most once
        for i in range(len(boxes)):
            hit = False
            if ious.shape[1]:
                iou = ious[i].masked_fill(matched, -1)
                j = int(iou.argmax())
                if iou[j] >= iou_thr:
                    matched[j] = hit = True
            scores.append(float(conf[i]))
            is_tp.append(hit)
    return np.array(scores), np.array(is_tp), n_gt


def pr_vs_conf(preds, targets):
    """Precision, recall and F1 at each confidence threshold (IoU 0.5)."""
    scores, is_tp, n_gt = match_detections(preds, targets)
    confs = np.linspace(0, 1, 101)
    tp = np.array([is_tp[scores >= c].sum() for c in confs])
    n_det = np.array([(scores >= c).sum() for c in confs])
    precision = np.divide(tp, n_det, out=np.ones(len(confs)), where=n_det > 0)
    recall = tp / max(n_gt, 1)
    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros(len(confs)), where=(precision + recall) > 0,
    )
    return confs, precision, recall, f1


def plot_pr_vs_conf(confs, precision, recall, f1, best, title, path: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(confs, precision, label="Precision")
    ax.plot(confs, recall, label="Recall")
    ax.plot(confs, f1, label="F1")
    ax.axvline(confs[best], color="gray", linestyle="--",
               label=f"Best F1 {f1[best]:.3f} @ conf {confs[best]:.2f}")
    ax.set(xlabel="Confidence", ylabel="Score", xlim=(0, 1), ylim=(0, 1.02), title=title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@click.command()
@click.option(
    "-d", "--data-yaml", type=click.Path(exists=True, dir_okay=False),
    required=True, help="YOLO data.yaml to validate on.",
)
@click.option(
    "--split", type=click.Choice(["train", "val"]),
    default="val", show_default=True, help="data.yaml split to validate on.",
)
@click.option(
    "-w", "--weights", default="runs/Fast-RCNN/best.pt", show_default=True,
    help="Checkpoint path, or a pretrained name (e.g. fasterrcnn_resnet50_fpn).",
)
@click.option(
    "-f", "--family", type=click.Choice(["torchvision", "ultralytics"]),
    default="torchvision", show_default=True,
)
@click.option("--device", default="cuda:0", show_default=True)
@click.option(
    "--class-id", type=int, default=None,
    help="Aircraft class id. Default: picked from family and weights.",
)
@click.option(
    "-o", "--out", type=click.Path(path_type=Path),
    default="runs/val", show_default=True,
)
def main(data_yaml, split, weights, family, device, class_id, out):
    """Compute mAP of a detector on a data.yaml split."""
    if class_id is None:
        class_id = default_class_id(family, weights)

    model_cfg = ModelConfig(
        family=family, checkpoint=weights, device=device,
        airplane_class_ids=[class_id],
    )
    click.echo(f"Loading {family} model: {weights} (class id {class_id})")
    detector = get_detector(family).load(model_cfg)

    click.echo(f"Validating on the {split} split of {data_yaml}")
    metrics = detector.validate(data_yaml, [class_id], split)

    confs, precision, recall, f1 = pr_vs_conf(metrics["preds"], metrics["targets"])
    best = int(f1.argmax())

    report = (
        f"weights:    {weights}\n"
        f"family:     {family}\n"
        f"data:       {data_yaml}\n"
        f"split:      {split}\n"
        f"mAP@.5:     {metrics['map_50']:.4f}\n"
        f"mAP@.5:.95: {metrics['map_50_95']:.4f}\n",
        f"best conf:  {confs[best]:.2f} (IoU .5: P {precision[best]:.4f}, "
        f"R {recall[best]:.4f}, F1 {f1[best]:.4f})\n"
    )

    out.mkdir(parents=True, exist_ok=True)
    (out / f"{split}.txt").write_text(report)
    plot_path = out / f"{split}_pr_f1_vs_conf.png"
    plot_pr_vs_conf(
        confs, precision, recall, f1, best, f"{weights} ({split})", plot_path
    )

    click.echo(report + f"Saved to {out / f'{split}.txt'} and {plot_path}")


if __name__ == "__main__":
    main()
