import torchvision
from sahi.models.torchvision import TorchVisionDetectionModel


MODEL_NAME_TO_CONSTRUCTOR = {
    "fasterrcnn_resnet50_fpn": torchvision.models.detection.fasterrcnn_resnet50_fpn,
    "fasterrcnn_resnet50_fpn_v2": torchvision.models.detection.fasterrcnn_resnet50_fpn_v2,
    "retinanet_resnet50_fpn": torchvision.models.detection.retinanet_resnet50_fpn,
}


def load_detection_model(weights: str, device: str, conf: float, use_sahi: bool = False):
    """Load a Faster R-CNN model with optional SAHI wrapper.""" 

    detection_model = MODEL_NAME_TO_CONSTRUCTOR[weights](weights="DEFAULT")

    # If SAHI is enabled, create SAHI wrapper
    if use_sahi:
        sahi_model = TorchVisionDetectionModel(
            model= detection_model,
            confidence_threshold=conf,
            device=device,
        )
        result = {'sahi_model': sahi_model, 'device': device, 'conf': conf, 'use_sahi': use_sahi}

        return result

    detection_model.eval()
    detection_model.to(device)
    
    result = {'model': detection_model, 'device': device, 'conf': conf, 'use_sahi': use_sahi}

    return result