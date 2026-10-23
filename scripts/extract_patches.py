#!/usr/bin/env python3
"""Extract training patches from fisheye frames.

Positive patches are cropped around annotated aircraft (the aircraft sits at a
random offset inside the patch, not always centred).  Negative patches are
cropped from negative frames and from aircraft-free regions of positive frames.
Output is a YOLO-format dataset (images/labels per split) that drops straight
into the existing training pipeline: positives carry a remapped box, negatives
get an empty label file.

    python scripts/extract_patches.py
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2

CONFIG = {
    "annotated_dir": Path("data/model_train/annotated"),
    "frames_dir": Path("data/model_train/frames"),
    "output_dir": Path("data/model_train/patches"),

    # Patch size in pixels, (width, height).
    "patch_size": (640, 640),

    # How many patches to sample per split.
    "num_samples": {
        "train": {"positive": 400, "negative": 300},
        "val": {"positive": 60, "negative": 80},
    },

    # Region to sample patches from: None = whole image, else (x1, y1, x2, y2) in pixels.
    "roi": {
        "positive": None,
        "negative": (870, 0, 7800, 1100),
    },
    
    # Attempts to find a non-overlapping negative patch in a positive frame.
    "max_tries": 20,
    # Keep a neighbouring box only if this fraction of its area is in the patch.
    "min_visibility": 0.8,
    "seed": 0,
}

SPLITS = ("train", "val")
CLASS_ID = 0


# Geometry helpers
def read_boxes(label_path: Path, w: int, h: int) -> list[tuple[int, int, int, int]]:
    """Read YOLO labels into pixel ``(x1, y1, x2, y2)`` boxes."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = (float(p) for p in parts)
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return boxes


def resolve_roi(roi, w: int, h: int) -> tuple[int, int, int, int]:
    """Clamp a configured ROI to the image, defaulting to the whole image."""
    if roi is None:
        return 0, 0, w, h
    x1, y1, x2, y2 = roi
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def sample_origin(roi, pw: int, ph: int, rng: random.Random):
    """Random top-left corner for a patch that fits inside ``roi``."""
    rx1, ry1, rx2, ry2 = roi
    if rx2 - rx1 < pw or ry2 - ry1 < ph:
        return None
    px = rng.randint(rx1, rx2 - pw)
    py = rng.randint(ry1, ry2 - ph)
    return px, py


def overlaps(px: int, py: int, pw: int, ph: int, boxes) -> bool:
    """True if the patch intersects any box."""
    for bx1, by1, bx2, by2 in boxes:
        if px < bx2 and bx1 < px + pw and py < by2 and by1 < py + ph:
            return True
    return False


def remap_boxes(px: int, py: int, pw: int, ph: int, boxes, min_vis: float) -> list[str]:
    """Shift boxes into patch space and format as YOLO label lines.

    A box is kept only if at least ``min_vis`` of its area falls inside the
    patch; neighbouring aircraft that are mostly outside are dropped rather than
    labelled as a clipped sliver.
    """
    lines = []
    for bx1, by1, bx2, by2 in boxes:
        cx1, cy1 = max(bx1 - px, 0), max(by1 - py, 0)
        cx2, cy2 = min(bx2 - px, pw), min(by2 - py, ph)
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        full_area = (bx2 - bx1) * (by2 - by1)
        visible_area = (cx2 - cx1) * (cy2 - cy1)
        if full_area <= 0 or visible_area / full_area < min_vis:
            continue
        cx = (cx1 + cx2) / 2 / pw
        cy = (cy1 + cy2) / 2 / ph
        bw = (cx2 - cx1) / pw
        bh = (cy2 - cy1) / ph
        lines.append(f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


# Patch sampling
def positive_patch(img, boxes, roi, pw: int, ph: int, rng: random.Random, min_vis: float):
    """Crop a patch that contains one aircraft at a random in-patch offset."""
    bx1, by1, bx2, by2 = rng.choice(boxes)
    rx1, ry1, rx2, ry2 = roi

    # Origin range so the patch still covers the box and stays inside the ROI.
    lo_x, hi_x = max(rx1, bx2 - pw), min(bx1, rx2 - pw)
    lo_y, hi_y = max(ry1, by2 - ph), min(by1, ry2 - ph)
    px = rng.randint(lo_x, hi_x) if lo_x <= hi_x else (bx1 + bx2 - pw) // 2
    py = rng.randint(lo_y, hi_y) if lo_y <= hi_y else (by1 + by2 - ph) // 2
    px = max(rx1, min(px, rx2 - pw))
    py = max(ry1, min(py, ry2 - ph))

    patch = img[py:py + ph, px:px + pw]
    return patch, remap_boxes(px, py, pw, ph, boxes, min_vis)


def negative_patch(img, boxes, roi, pw: int, ph: int, rng: random.Random, tries: int):
    """Crop a patch from ``roi`` that overlaps no box (None if none found)."""
    for _ in range(tries):
        origin = sample_origin(roi, pw, ph, rng)
        if origin is None:
            return None
        px, py = origin
        if not overlaps(px, py, pw, ph, boxes):
            return img[py:py + ph, px:px + pw]
    return None


# Dataset assembly
def load_positive(image_path: Path, labels_dir: Path):
    """Return ``(image, boxes)`` for a positive frame."""
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    boxes = read_boxes(labels_dir / f"{image_path.stem}.txt", w, h)
    return img, boxes


def save(patch, lines: list[str], name: str, img_dir: Path, lbl_dir: Path, write_label: bool=True) -> None:
    cv2.imwrite(str(img_dir / f"{name}.jpg"), patch)
    if write_label:
        (lbl_dir / f"{name}.txt").write_text("\n".join(lines))


def build_split(split: str, cfg: dict, rng: random.Random) -> None:
    pw, ph = cfg["patch_size"]
    img_dir = cfg["output_dir"] / "images" / split
    lbl_dir = cfg["output_dir"] / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    pos_images = sorted((cfg["annotated_dir"] / "images" / split).glob("*.jpg"))
    neg_frames = sorted((cfg["frames_dir"] / split / "negative").glob("*.jpg"))
    labels_dir = cfg["annotated_dir"] / "labels" / split
    pos_roi = cfg["roi"]["positive"]
    neg_roi = cfg["roi"]["negative"]

    # Positives: pick a random annotated frame, crop around an aircraft.
    made = 0
    target = cfg["num_samples"][split]["positive"]
    while made < target and pos_images:
        src = rng.choice(pos_images)
        img, boxes = load_positive(src, labels_dir)
        if not boxes:
            continue
        roi = resolve_roi(pos_roi, img.shape[1], img.shape[0])
        patch, lines = positive_patch(img, boxes, roi, pw, ph, rng, cfg["min_visibility"])
        if lines:
            save(patch, lines, f"{src.stem}_pos_{made:05d}", img_dir, lbl_dir)
            made += 1

    # Negatives: from negative frames and aircraft-free positive regions.
    made = 0
    target = cfg["num_samples"][split]["negative"]
    while made < target and (neg_frames or pos_images):
        from_positive = pos_images and (not neg_frames or rng.random() < 0.5)
        if from_positive:
            src = rng.choice(pos_images)
            img, boxes = load_positive(src, labels_dir)
        else:
            src = rng.choice(neg_frames)
            img, boxes = cv2.imread(str(src)), []
        roi = resolve_roi(neg_roi, img.shape[1], img.shape[0])
        patch = negative_patch(img, boxes, roi, pw, ph, rng, cfg["max_tries"])
        if patch is not None:
            save(patch, [], f"{src.stem}_neg_{made:05d}", img_dir, lbl_dir, write_label=False)
            made += 1

    print(f"[{split}] wrote patches to {img_dir}")


def write_data_yaml(cfg: dict) -> None:
    (cfg["output_dir"] / "data.yaml").write_text(
        "path: .\n\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        f"  {CLASS_ID}: aircraft\n"
    )


def main() -> None:
    cfg = CONFIG
    rng = random.Random(cfg["seed"])
    for split in SPLITS:
        build_split(split, cfg, rng)
    write_data_yaml(cfg)
    print(f"Done. Dataset at {cfg['output_dir']}")


if __name__ == "__main__":
    main()
