from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import torch
import yaml
from torch.utils.data import Dataset

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _parse_names(names) -> List[str]:
    """YOLO ``names`` may be a list or an id->name mapping; return an ordered list."""
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=int)]
    return list(names)


def _resolve(root: Path, entry: str) -> Path:
    p = Path(entry)
    return p if p.is_absolute() else root / p


def _images_to_labels_dir(images_dir: Path) -> Path:
    """YOLO convention: labels mirror images with the 'images' segment -> 'labels'."""
    parts = list(images_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    # No 'images' segment -> assume a sibling 'labels' dir.
    return images_dir.parent / "labels"


def build_yolo_datasets(
    data_yaml: str, label_offset: int, hflip_prob: float = 0.0
) -> Tuple["YoloDetectionDataset", Optional["YoloDetectionDataset"]]:
    """Build train (and optional val) datasets from a YOLO ``data.yaml``.

    Augmentation (``hflip_prob``) is applied to the train split only.
    """
    yaml_path = Path(data_yaml)
    with yaml_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    root = Path(cfg["path"]) if cfg.get("path") else yaml_path.parent
    class_names = _parse_names(cfg["names"])

    train_images = _resolve(root, cfg["train"])
    train_ds = YoloDetectionDataset(
        train_images, class_names, label_offset, hflip_prob
    )

    val_ds = None
    if cfg.get("val"):
        val_images = _resolve(root, cfg["val"])
        if val_images.resolve() != train_images.resolve():
            val_ds = YoloDetectionDataset(val_images, class_names, label_offset)

    return train_ds, val_ds


class YoloDetectionDataset(Dataset):
    """YOLO-format dataset for torchvision detection fine-tuning.

    Reads images from ``images_dir`` and normalized ``cls cx cy w h`` labels from
    the mirrored ``labels`` tree. Returns ``(image, target)`` where image is a CHW
    float tensor in [0, 1] and target has ``boxes`` (pixel xyxy) and ``labels``.
    ``label_offset`` maps YOLO's 0-based class ids to the model's convention
    (1 for Faster R-CNN, whose 0 is background; 0 for RetinaNet).
    ``hflip_prob`` is the chance of horizontally flipping an image and its boxes.
    """

    def __init__(
        self,
        images_dir,
        class_names: List[str],
        label_offset: int = 1,
        hflip_prob: float = 0.0,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = _images_to_labels_dir(self.images_dir)
        self.class_names = list(class_names)
        self.label_offset = label_offset
        self.hflip_prob = hflip_prob
        self.image_paths = sorted(
            p
            for p in self.images_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.images_dir}")

    @property
    def num_fg_classes(self) -> int:
        return len(self.class_names)

    def _read_labels(self, stem: str, width: int, height: int):
        label_path = self.labels_dir / f"{stem}.txt"
        boxes, labels = [], []
        if not label_path.exists():
            return boxes, labels  # background image (no objects)

        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                print(f"Invalid label format in {label_path}: {line}")
                continue
            class_id, cx, cy, nw, nh = parts
            cx, cy, nw, nh = float(cx), float(cy), float(nw), float(nh)
            if nw <= 0 or nh <= 0:
                continue
            x1 = (cx - nw / 2) * width
            y1 = (cy - nh / 2) * height
            x2 = (cx + nw / 2) * width
            y2 = (cy + nh / 2) * height
            boxes.append([x1, y1, x2, y2])
            labels.append(int(float(class_id)) + self.label_offset)
        return boxes, labels

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]

        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        image = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)

        boxes, labels = self._read_labels(path.stem, width, height)

        if boxes:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)

        if torch.rand(1).item() < self.hflip_prob:
            image = image.flip(-1)
            boxes_t[:, [0, 2]] = width - boxes_t[:, [2, 0]]

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
        }
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))
