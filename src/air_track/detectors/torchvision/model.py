from __future__ import annotations

from functools import partial

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
    weights: str, num_fg_classes: int, pretrained: bool = True
):
    """Build a torchvision detector with its head resized to ``num_fg_classes``.

    Shared by training (``pretrained=True``, fine-tune from COCO) and checkpoint
    loading (``pretrained=False``, weights come from the state dict).
    """
    if weights not in MODEL_NAME_TO_CONSTRUCTOR:
        raise ValueError(
            f"Unknown torchvision model '{weights}'. "
            f"Choose from: {list(MODEL_NAME_TO_CONSTRUCTOR)}"
        )
    model = MODEL_NAME_TO_CONSTRUCTOR[weights](
        weights="DEFAULT" if pretrained else None
    )

    if weights.startswith("fasterrcnn"):
        # Faster R-CNN counts background, so head size = classes + 1
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_fg_classes + 1
        )
    elif weights.startswith("retinanet"):
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