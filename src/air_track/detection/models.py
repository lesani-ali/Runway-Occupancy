from __future__ import annotations

from functools import partial

import torch
import torchvision
from sahi.models.torchvision import TorchVisionDetectionModel
from sahi.models.ultralytics import UltralyticsDetectionModel

MODEL_NAME_TO_CONSTRUCTOR = {
    "fasterrcnn_resnet50_fpn": torchvision.models.detection.fasterrcnn_resnet50_fpn,
    "fasterrcnn_resnet50_fpn_v2": torchvision.models.detection.fasterrcnn_resnet50_fpn_v2,
    "retinanet_resnet50_fpn": torchvision.models.detection.retinanet_resnet50_fpn,
}


def load_detection_model(
    checkpoint: str,
    device: str,
    conf: float,
    use_sahi: bool = False,
    family: str = "torchvision",
) -> dict:
    """Load a detection model (torchvision or YOLO) with optional SAHI slicing.

    ``checkpoint`` selects the model:
      * a known pretrained name (e.g. ``fasterrcnn_resnet50_fpn`` for torchvision,
        ``yolo11n.pt`` for YOLO) loads COCO-pretrained weights;
      * anything else is treated as a fine-tuned checkpoint from ``air-track-train``.
    """
    if family == "ultralytics":
        # A YOLO .pt bundles architecture + weights + names, so a pretrained name
        # and a fine-tuned path are loaded the same way.
        return _load_ultralytics_model(checkpoint, device, conf, use_sahi)

    # torchvision: a registry name -> pretrained; otherwise a fine-tuned file.
    if checkpoint in MODEL_NAME_TO_CONSTRUCTOR:
        return _load_torchvision_model(checkpoint, device, conf, use_sahi)
    return _load_torchvision_checkpoint(checkpoint, device, conf, use_sahi)


def _load_ultralytics_model(
    weights: str, device: str, conf: float, use_sahi: bool
) -> dict:
    """Load a YOLO model (``weights`` is a .pt path or name, e.g. ``yolo11n.pt``)."""
    from ultralytics import YOLO

    detection_model = YOLO(weights)

    if use_sahi:
        sahi_model = UltralyticsDetectionModel(
            model=detection_model,
            confidence_threshold=conf,
            device=device,
        )
        return {
            "model": sahi_model,
            "device": device,
            "conf": conf,
            "use_sahi": use_sahi,
            "family": "ultralytics",
        }

    detection_model.to(device)
    return {
        "model": detection_model,
        "device": device,
        "conf": conf,
        "use_sahi": use_sahi,
        "family": "ultralytics",
    }

    
def _load_torchvision_model(
    weights: str, device: str, conf: float, use_sahi: bool
) -> dict:
    detection_model = MODEL_NAME_TO_CONSTRUCTOR[weights](weights="DEFAULT")

    if use_sahi:
        sahi_model = TorchVisionDetectionModel(
            model=detection_model,
            confidence_threshold=conf,
            device=device,
        )
        return {
            "model": sahi_model,
            "device": device,
            "conf": conf,
            "use_sahi": use_sahi,
            "family": "torchvision",
        }

    detection_model.eval()
    detection_model.to(device)
    return {
        "model": detection_model,
        "device": device,
        "conf": conf,
        "use_sahi": use_sahi,
        "family": "torchvision",
    }


def _load_torchvision_checkpoint(
    checkpoint: str, device: str, conf: float, use_sahi: bool
) -> dict:
    """Load a fine-tuned torchvision model."""
    ckpt = torch.load(checkpoint, map_location="cpu")
    try:
        weights_name = ckpt["weights_name"]
        num_fg = ckpt["num_fg_classes"]
        state = ckpt["model"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Checkpoint '{checkpoint}' is missing training metadata "
            "(expected 'weights_name', 'num_fg_classes', 'model'). "
            "Use a checkpoint produced by air-track-train."
        ) from exc

    class_names = ckpt.get("class_names") or [
        f"class_{i + 1}" for i in range(num_fg)
    ]

    model = build_torchvision_model(weights_name, num_fg, pretrained=False)
    model.load_state_dict(state)
    model.eval()

    # Faster R-CNN labels are 1..K (0 = background); RetinaNet labels are
    # 0..K-1 (focal loss, no background class).
    label_offset = 0 if weights_name.startswith("retinanet") else 1
    category_mapping = {
        str(i + label_offset): name for i, name in enumerate(class_names)
    }

    if use_sahi:
        try:
            sahi_model = TorchVisionDetectionModel(
                model=model,
                confidence_threshold=conf,
                device=device,
                category_mapping=category_mapping,
            )
        except TypeError:  # older SAHI without category_mapping kwarg
            sahi_model = TorchVisionDetectionModel(
                model=model, confidence_threshold=conf, device=device
            )
        return {
            "model": sahi_model,
            "device": device,
            "conf": conf,
            "use_sahi": use_sahi,
            "family": "torchvision",
        }

    model.to(device)
    return {
        "model": model,
        "device": device,
        "conf": conf,
        "use_sahi": use_sahi,
        "family": "torchvision",
    }
    

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