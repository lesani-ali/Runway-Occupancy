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
    anchor_aspect_ratios: Optional[Sequence[float]] = None,
):
    """Build a torchvision detector with its head resized to ``num_fg_classes``.

    Shared by training (``pretrained=True``, fine-tune from COCO) and checkpoint
    loading (``pretrained=False``, weights come from the state dict).
    ``anchor_aspect_ratios`` (h/w) overrides the Faster R-CNN RPN anchors.
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
        if anchor_aspect_ratios:
            _set_rpn_aspect_ratios(model, anchor_aspect_ratios)
    elif weights.startswith("retinanet"):
        if anchor_aspect_ratios:
            raise ValueError(
                "anchor_aspect_ratios is only supported for Faster R-CNN."
            )
        # RetinaNet uses focal loss; num_classes = foreground classes only
        from torchvision.models.detection.retinanet import (
            RetinaNetClassificationHead,
        )

        num_anchors = model.head.classification_head.num_anchors
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


def _set_rpn_aspect_ratios(model, aspect_ratios: Sequence[float]) -> None:
    """Replace the RPN anchors' aspect ratios (h/w), keeping the anchor sizes.

    The RPN head's pretrained conv trunk is kept. Its 1x1 objectness and box
    outputs are re-initialised only if the anchors-per-location count changes.
    """
    from torchvision.models.detection.anchor_utils import AnchorGenerator

    sizes = model.rpn.anchor_generator.sizes
    model.rpn.anchor_generator = AnchorGenerator(
        sizes, (tuple(aspect_ratios),) * len(sizes)
    )

    head = model.rpn.head
    num_anchors = len(aspect_ratios) * len(sizes[0])
    if head.cls_logits.out_channels == num_anchors:
        return

    in_channels = head.cls_logits.in_channels
    head.cls_logits = torch.nn.Conv2d(in_channels, num_anchors, 1)
    head.bbox_pred = torch.nn.Conv2d(in_channels, num_anchors * 4, 1)
    for layer in (head.cls_logits, head.bbox_pred):
        torch.nn.init.normal_(layer.weight, std=0.01)
        torch.nn.init.constant_(layer.bias, 0)
