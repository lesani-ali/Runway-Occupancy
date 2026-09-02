from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

from air_track.detectors.base import BaseTrainer
from air_track.detectors.registry import register_trainer
from air_track.detectors.torchvision.dataset import (
    build_yolo_datasets,
    collate_fn,
)
from air_track.detectors.torchvision.model import (
    build_torchvision_model,
    label_offset_for,
)
from air_track.logging import get_logger

logger = get_logger(__name__)


@register_trainer("torchvision")
class TorchvisionTrainer(BaseTrainer):
    """Fine-tune a torchvision detector on a YOLO-format dataset.

    Training is a stateful process, so the run's model, optimizer, scheduler and
    bookkeeping live on the instance. The class-level :meth:`train` keeps the
    stateless ``BaseTrainer`` contract by wrapping one throwaway instance.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = self._resolve_device(cfg.device)

        self.train_ds, self.train_loader, self.val_loader = (
            self.build_dataloaders(cfg)
        )
        self.model = build_torchvision_model(
            cfg.weights, self.train_ds.num_fg_classes
        ).to(self.device)
        self.optimizer = self.build_optimizer(self.model, cfg)
        self.scheduler = self.build_scheduler(self.optimizer, cfg)

        self.out_dir = Path(cfg.output_dir) / cfg.run_name
        self.meta = self.build_metadata(cfg, self.train_ds)

    @classmethod
    def train(cls, cfg) -> Path:
        torch.manual_seed(cfg.seed)
        return cls(cfg)._train()

    def _train(self) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Train images: {len(self.train_ds)} | "
            f"{self.train_ds.num_fg_classes} class(es): "
            f"{self.train_ds.class_names}"
        )

        best_path = self.out_dir / "best.pt"
        best_score = float("-inf")
        for epoch in range(1, self.cfg.epochs + 1):

            train_loss = self.train_one_epoch()
            self.scheduler.step()

            msg = (
                f"Epoch {epoch}/{self.cfg.epochs} | train_loss={train_loss:.4f}"
            )

            val_loss = None
            val_map = None

            if self.val_loader is not None:
                val_loss = self.validate_one_epoch()
                metrics = self.evaluate_detection()
                val_map = metrics["map"]
                msg += (
                    f" | val_loss={val_loss:.4f}"
                    f" | mAP={val_map:.4f} | mAP@.5={metrics['map_50']:.4f}"
                )

            logger.info(msg)

            self.save_checkpoint(epoch, self.meta, self.out_dir / "last.pt")

            # Rank on mAP when a val set exists; else fall back to (negated)
            # train loss, so "higher is better" holds for both.
            score = val_map if val_map is not None else -train_loss

            if score > best_score:
                best_score = score
                best_meta = {
                    **self.meta, "val_map": val_map, "val_loss": val_loss
                }
                self.save_checkpoint(epoch, best_meta, best_path)

                tag = (
                    f"mAP={val_map:.4f}"
                    if val_map is not None
                    else f"train_loss={train_loss:.4f}"
                )
                logger.info(f" ↳ new best ({tag})")

        logger.info(
            f"Done. Checkpoints saved to {self.out_dir} (best.pt, last.pt)"
        )
        return best_path

    def train_one_epoch(self) -> float:
        self.model.train()

        running_loss = 0.0

        for images, targets in self.train_loader:
            images, targets = self._move_to_device(images, targets)

            loss_dict = self.model(images, targets)
            loss = sum(loss_dict.values())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / max(len(self.train_loader), 1)

    @torch.no_grad()
    def validate_one_epoch(self) -> float:
        # torchvision detectors only return losses in train() mode; no_grad avoids
        # gradient work, and the FPN backbone uses frozen BN so stats don't drift.
        self.model.train()
        total, n = 0.0, 0
        for images, targets in self.val_loader:
            images, targets = self._move_to_device(images, targets)
            loss_dict = self.model(images, targets)
            total += float(sum(loss_dict.values()))
            n += 1
        return total / max(n, 1)

    @torch.no_grad()
    def evaluate_detection(self) -> dict:
        # eval() mode makes torchvision detectors return predictions (not losses),
        # which MeanAveragePrecision accumulates into COCO mAP@[.5:.95] and mAP@.5.
        self.model.eval()
        metric = MeanAveragePrecision(box_format="xyxy")
        for images, targets in self.val_loader:
            images, targets = self._move_to_device(images, targets)
            preds = self.model(images)
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

    def save_checkpoint(self, epoch, meta, path):
        checkpoint = {
            "model": self.model.state_dict(),
            "epoch": epoch,
            **meta,
        }

        torch.save(checkpoint, path)

    def _move_to_device(self, images, targets):
        images = [img.to(self.device) for img in images]
        targets = [
            {k: v.to(self.device) for k, v in t.items()} for t in targets
        ]
        return images, targets

    @staticmethod
    def build_dataloaders(cfg):
        # YOLO class ids are 0-based; Faster R-CNN reserves 0 for background, so
        # its targets are shifted by 1, while RetinaNet (focal loss) stays 0-based.
        label_offset = label_offset_for(cfg.weights)
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

    @staticmethod
    def build_optimizer(model, cfg):
        params = [p for p in model.parameters() if p.requires_grad]

        return torch.optim.SGD(
            params,
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )

    @staticmethod
    def build_scheduler(optimizer, cfg):
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.lr_step_size,
            gamma=cfg.lr_gamma,
        )

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA not available; falling back to CPU.")
            return torch.device("cpu")
        return torch.device(device)

    @staticmethod
    def build_metadata(cfg, train_ds):
        return {
            "weights_name": cfg.weights,
            "num_fg_classes": train_ds.num_fg_classes,
            "class_names": train_ds.class_names,
            "label_offset": train_ds.label_offset,
        }
