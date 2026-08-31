from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

from air_track.detection.models import build_torchvision_model
from air_track.training.torch_dataset import build_yolo_datasets, collate_fn
from air_track.logging import get_logger

logger = get_logger(__name__)


def build_dataloaders(cfg):
    # YOLO class ids are 0-based; Faster R-CNN reserves 0 for background, so its
    # targets are shifted by 1, while RetinaNet (focal loss) stays 0-based.
    label_offset = 0 if cfg.weights.startswith("retinanet") else 1
    train_ds, val_ds = build_yolo_datasets(cfg.data_yaml, label_offset)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        collate_fn=collate_fn,
    )

    val_loader = None

    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.workers,
            collate_fn=collate_fn,
        )

    return train_ds, train_loader, val_loader


def build_optimizer(model, cfg):
    params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    return torch.optim.SGD(
        params,
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )


def build_scheduler(optimizer, cfg):
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.lr_step_size,
        gamma=cfg.lr_gamma,
    )


def _resolve_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


def _move_to_device(images, targets, device):
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    return images, targets


def train_one_epoch(model, loader, optimizer, device):
    model.train()

    running_loss = 0.0

    for images, targets in loader:
        images, targets = _move_to_device(
            images,
            targets,
            device,
        )

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / max(len(loader), 1)


def save_checkpoint(model, epoch, meta, path):
    checkpoint = {
        "model": model.state_dict(),
        "epoch": epoch,
        **meta,
    }

    torch.save(checkpoint, path)


@torch.no_grad()
def validate_one_epoch(model, loader, device) -> float:
    # torchvision detectors only return losses in train() mode; no_grad avoids
    # gradient work, and the FPN backbone uses frozen BN so stats don't drift.
    model.train()
    total, n = 0.0, 0
    for images, targets in loader:
        images, targets = _move_to_device(images, targets, device)
        loss_dict = model(images, targets)
        total += float(sum(loss_dict.values()))
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate_detection(model, loader, device) -> dict:
    # eval() mode makes torchvision detectors return predictions (not losses),
    # which MeanAveragePrecision accumulates into COCO mAP@[.5:.95] and mAP@.5.
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy")
    for images, targets in loader:
        images, targets = _move_to_device(images, targets, device)
        preds = model(images)
        metric.update(
            [
                {
                    "boxes": p["boxes"].cpu(),
                    "scores": p["scores"].cpu(),
                    "labels": p["labels"].cpu(),
                }
                for p in preds
            ],
            [
                {"boxes": t["boxes"].cpu(), "labels": t["labels"].cpu()}
                for t in targets
            ],
        )
    result = metric.compute()
    return {"map": float(result["map"]), "map_50": float(result["map_50"])}


def build_metadata(cfg, train_ds):
    return {
        "weights_name": cfg.weights,
        "num_fg_classes": train_ds.num_fg_classes,
        "class_names": train_ds.class_names,
        "label_offset": train_ds.label_offset,
    }

        
def train(cfg):
    torch.manual_seed(cfg.seed)

    device = _resolve_device(cfg.device)

    train_ds, train_loader, val_loader = (
        build_dataloaders(cfg)
    )

    num_fg = train_ds.num_fg_classes

    logger.info(
        f"Train images: {len(train_ds)} | "
        f"{num_fg} class(es): {train_ds.class_names}"
    )

    model = build_torchvision_model(cfg.weights,num_fg).to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    out_dir = Path(cfg.output_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = build_metadata(cfg, train_ds)

    best_score = float("-inf")
    for epoch in range(1, cfg.epochs + 1):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        scheduler.step()

        msg = f"Epoch {epoch}/{cfg.epochs} | train_loss={train_loss:.4f}"

        val_loss = None
        val_map = None

        if val_loader is not None:
            val_loss = validate_one_epoch(model, val_loader, device)
            metrics = evaluate_detection(model, val_loader, device)
            val_map = metrics["map"]
            msg += (
                f" | val_loss={val_loss:.4f}"
                f" | mAP={val_map:.4f} | mAP@.5={metrics['map_50']:.4f}"
            )

        logger.info(msg)

        save_checkpoint(model, epoch, meta, out_dir / "last.pt")

        # Rank on mAP when a val set exists; else fall back to (negated) train
        # loss, so "higher is better" holds for both.
        score = val_map if val_map is not None else -train_loss

        if score > best_score:
            best_score = score
            best_meta = {**meta, "val_map": val_map, "val_loss": val_loss}
            save_checkpoint(model, epoch, best_meta, out_dir / "best.pt")

            tag = (
                f"mAP={val_map:.4f}"
                if val_map is not None
                else f"train_loss={train_loss:.4f}"
            )
            logger.info(f" ↳ new best ({tag})")

    logger.info(f"Done. Checkpoints saved to {out_dir} (best.pt, last.pt)")




