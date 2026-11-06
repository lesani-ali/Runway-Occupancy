from __future__ import annotations

from functools import partial
from typing import Optional, Sequence

import torch
import torchvision

MODEL_NAME_TO_CONSTRUCTOR = {
    "fasterrcnn_resnet50_fpn": torchvision.models.detection.fasterrcnn_resnet50_fpn,
    "fasterrcnn_resnet50_fpn_v2": torchvision.models.detection.fasterrcnn_resnet50_fpn_v2,
    "retinanet_resnet50_fpn": torchvision.models.detection.retinanet_resnet50_fpn,
}

# Both Faster R-CNN (P2-P6) and RetinaNet (P3-P7) predict on 5 FPN levels.
NUM_FPN_LEVELS = 5


def is_pretrained_name(checkpoint: str) -> bool:
    """True if ``checkpoint`` is a known torchvision model name (vs a file path)."""
    return checkpoint in MODEL_NAME_TO_CONSTRUCTOR


def label_offset_for(weights_name: str) -> int:
    """Map YOLO's 0-based class ids to the model's label convention.

    Faster R-CNN labels are 1..K (0 = background); RetinaNet labels are 0..K-1
    (focal loss, no background class).
    """
    return 0 if weights_name.startswith("retinanet") else 1


def build_torchvision_model(
    weights: str,
    num_fg_classes: int,
    pretrained: bool = True,
    anchor_sizes: Optional[Sequence[int]] = None,
    anchor_aspect_ratios: Optional[Sequence[float]] = None,
):
    """Build a torchvision detector with its head resized to ``num_fg_classes``.

    Shared by training (``pretrained=True``, fine-tune from COCO) and checkpoint
    loading (``pretrained=False``, weights come from the state dict).
    ``anchor_sizes`` (one base size per FPN level, smallest first) and
    ``anchor_aspect_ratios`` (h/w) override the default anchors; None keeps them.
    """
    if weights not in MODEL_NAME_TO_CONSTRUCTOR:
        raise ValueError(
            f"Unknown torchvision model '{weights}'. "
            f"Choose from: {list(MODEL_NAME_TO_CONSTRUCTOR)}"
        )
    # Without pretrained weights, also skip the ImageNet backbone download:
    # a checkpoint's state dict overwrites every weight anyway.
    model = MODEL_NAME_TO_CONSTRUCTOR[weights](
        weights="DEFAULT" if pretrained else None,
        weights_backbone=None,
    )

    if weights.startswith("fasterrcnn"):
        # Faster R-CNN counts background, so head size = classes + 1
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_fg_classes + 1
        )
        if anchor_sizes or anchor_aspect_ratios:
            _set_rpn_anchors(model, anchor_sizes, anchor_aspect_ratios)
    elif weights.startswith("retinanet"):
        if anchor_sizes or anchor_aspect_ratios:
            _set_retinanet_anchors(model, anchor_sizes, anchor_aspect_ratios)

        # RetinaNet uses focal loss; num_classes = foreground classes only
        from torchvision.models.detection.retinanet import (
            RetinaNetClassificationHead,
        )

        num_anchors = model.anchor_generator.num_anchors_per_location()[0]
        in_channels = model.backbone.out_channels
        try:
            model.head.classification_head = RetinaNetClassificationHead(
                in_channels,
                num_anchors,
                num_fg_classes,
                norm_layer=partial(torch.nn.GroupNorm, 32),
            )
        except TypeError:  # older torchvision without norm_layer kwarg
            model.head.classification_head = RetinaNetClassificationHead(
                in_channels, num_anchors, num_fg_classes
            )
    else:
        raise ValueError(f"Fine-tuning not supported for '{weights}'.")

    return model


def _make_anchor_generator(sizes, aspect_ratios):
    from torchvision.models.detection.anchor_utils import AnchorGenerator

    if len(sizes) != NUM_FPN_LEVELS:
        raise ValueError(
            f"anchor_sizes needs one size per FPN level ({NUM_FPN_LEVELS}), "
            f"got {len(sizes)}."
        )
    return AnchorGenerator(
        tuple(sizes), (tuple(aspect_ratios),) * len(sizes)
    )


def _set_rpn_anchors(model, anchor_sizes, aspect_ratios) -> None:
    """Replace the Faster R-CNN RPN anchors (one size per level).

    The RPN head's pretrained conv trunk is kept. Its 1x1 objectness and box
    outputs are re-initialised only if the anchors-per-location count changes.
    """
    generator = model.rpn.anchor_generator
    sizes = (
        tuple((s,) for s in anchor_sizes) if anchor_sizes else generator.sizes
    )
    model.rpn.anchor_generator = _make_anchor_generator(
        sizes, aspect_ratios or generator.aspect_ratios[0]
    )

    head = model.rpn.head
    num_anchors = model.rpn.anchor_generator.num_anchors_per_location()[0]
    if head.cls_logits.out_channels == num_anchors:
        return

    in_channels = head.cls_logits.in_channels
    head.cls_logits = torch.nn.Conv2d(in_channels, num_anchors, 1)
    head.bbox_pred = torch.nn.Conv2d(in_channels, num_anchors * 4, 1)
    for layer in (head.cls_logits, head.bbox_pred):
        torch.nn.init.normal_(layer.weight, std=0.01)
        torch.nn.init.constant_(layer.bias, 0)


def _set_retinanet_anchors(model, anchor_sizes, aspect_ratios) -> None:
    """Replace the RetinaNet anchors (3 octave scales per level, as torchvision).

    The box regression head is pretrained for 9 anchors per location; it is
    re-initialised only if the new anchors change that count.
    """
    generator = model.anchor_generator
    sizes = (
        tuple(
            (x, int(x * 2 ** (1 / 3)), int(x * 2 ** (2 / 3)))
            for x in anchor_sizes
        )
        if anchor_sizes
        else generator.sizes
    )
    model.anchor_generator = _make_anchor_generator(
        sizes, aspect_ratios or generator.aspect_ratios[0]
    )

    num_anchors = model.anchor_generator.num_anchors_per_location()[0]
    if num_anchors == generator.num_anchors_per_location()[0]:
        return

    from torchvision.models.detection.retinanet import RetinaNetRegressionHead

    model.head.regression_head = RetinaNetRegressionHead(
        model.backbone.out_channels, num_anchors
    )
