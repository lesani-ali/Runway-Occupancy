import torchvision
from sahi.models.torchvision import TorchVisionDetectionModel


MODEL_NAME_TO_CONSTRUCTOR = {
    "fasterrcnn_resnet50_fpn": torchvision.models.detection.fasterrcnn_resnet50_fpn,
    "fasterrcnn_resnet50_fpn_v2": torchvision.models.detection.fasterrcnn_resnet50_fpn_v2,
    "retinanet_resnet50_fpn": torchvision.models.detection.retinanet_resnet50_fpn,
}


def load_detection_model(weights: str, device: str, conf: float, use_sahi: bool = False) -> dict:
    """Load a torchvision detection model with an optional SAHI slicing wrapper."""
    detection_model = MODEL_NAME_TO_CONSTRUCTOR[weights](weights="DEFAULT")

    if use_sahi:
        sahi_model = TorchVisionDetectionModel(
            model=detection_model,
            confidence_threshold=conf,
            device=device,
        )
        return {"sahi_model": sahi_model, "device": device, "conf": conf, "use_sahi": use_sahi}

    detection_model.eval()
    detection_model.to(device)
    return {"model": detection_model, "device": device, "conf": conf, "use_sahi": use_sahi}