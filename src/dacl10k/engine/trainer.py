"""Multi-GPU Distributed Data Parallel (DDP) training engine using Hugging Face Accelerate."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dacl10k.utils.metrics import MultiLabelIoUMetric


class Trainer:
    """Distributed-ready trainer for multi-label semantic segmentation."""

    def __init__(
        self,
        cfg: dict[str, Any],
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module | None = None,
        accelerator: Accelerator | None = None,
    ) -> None:
        """Initialize the Trainer.

        Args:
            cfg: Configuration dictionary.
            model: PyTorch segmentation model returning raw logits.
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            criterion: Loss function (defaults to nn.BCEWithLogitsLoss()).
            accelerator: Optional Hugging Face Accelerator instance.
        """
        self.cfg = cfg
        if accelerator is None:
            mp = cfg.get("mixed_precision", "no") if torch.cuda.is_available() else "no"
            self.accelerator = Accelerator(mixed_precision=mp)
        else:
            self.accelerator = accelerator

        self.criterion = criterion if criterion is not None else nn.BCEWithLogitsLoss()

        lr = float(cfg.get("lr", 1e-4))
        weight_decay = float(cfg.get("weight_decay", 1e-2))
        epochs = max(int(cfg.get("epochs", 30)), 1)

        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.scheduler,
        ) = self.accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)

        num_classes = int(cfg.get("num_classes", 19))
        threshold = float(cfg.get("threshold", 0.5))
        self.metric = MultiLabelIoUMetric(num_classes=num_classes, threshold=threshold)

    @staticmethod
    def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract (images, masks) tensors from a dict or tuple/list batch."""
        if isinstance(batch, dict):
            return batch["image"], batch["mask"]
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return batch[0], batch[1]
        raise TypeError(f"Unsupported batch format: {type(batch)}")

    def train_one_epoch(self, epoch: int) -> float:
        """Run one training epoch.

        Args:
            epoch: Current epoch index.

        Returns:
            Mean training loss across batches.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            images, targets = self._unpack_batch(batch)
            targets = targets.to(dtype=torch.float32)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, targets)
            self.accelerator.backward(loss)
            self.optimizer.step()

            total_loss += float(loss.detach().item())
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Run validation and compute loss and mIoU.

        Returns:
            Dictionary containing "val_loss" and "miou".
        """
        self.model.eval()
        self.metric.reset()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            images, targets = self._unpack_batch(batch)
            targets = targets.to(dtype=torch.float32)

            logits = self.model(images)
            loss = self.criterion(logits, targets)

            gathered_logits, gathered_targets = self.accelerator.gather_for_metrics(
                (logits, targets)
            )
            self.metric.update(gathered_logits, gathered_targets)

            total_loss += float(loss.detach().item())
            num_batches += 1

        metrics = self.metric.compute()
        val_loss = total_loss / max(num_batches, 1)
        return {
            "val_loss": val_loss,
            "miou": float(metrics["miou"]),
        }

    def save_checkpoint(self, path: str | Path, epoch: int, best_miou: float) -> None:
        """Save model, optimizer, and scheduler state on the main process.

        Args:
            path: Destination file path for the checkpoint.
            epoch: Completed epoch index.
            best_miou: Best validation mIoU achieved so far.
        """
        if not self.accelerator.is_main_process:
            return

        ckpt_path = Path(path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        payload = {
            "model_state_dict": unwrapped_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": epoch,
            "best_miou": best_miou,
        }
        torch.save(payload, ckpt_path)

    def load_checkpoint(self, path: str | Path) -> tuple[int, float]:
        """Load model, optimizer, and scheduler state from a checkpoint.

        Args:
            path: Checkpoint file path to load.

        Returns:
            Tuple of (epoch, best_miou) stored in the checkpoint.
        """
        ckpt_path = Path(path)
        checkpoint = torch.load(ckpt_path, map_location=self.accelerator.device, weights_only=False)

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        epoch = int(checkpoint.get("epoch", 0))
        best_miou = float(checkpoint.get("best_miou", 0.0))
        return epoch, best_miou

    def fit(self, resume_path: str | Path | None = None) -> dict[str, float]:
        """Execute the full training and validation loop across epochs.

        Args:
            resume_path: Optional path to a checkpoint to resume training from.

        Returns:
            Dictionary containing the final training/validation metrics and best_miou.
        """
        epochs = int(self.cfg.get("epochs", 30))
        checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints"))
        use_wandb = bool(self.cfg.get("use_wandb", False))

        start_epoch = 0
        best_miou = -1.0

        if resume_path is not None:
            loaded_epoch, best_miou = self.load_checkpoint(resume_path)
            start_epoch = loaded_epoch + 1

        wandb_run = None
        if use_wandb and self.accelerator.is_main_process:
            import wandb

            wandb_run = wandb.init(
                project=self.cfg.get("wandb_project", "dacl10k-segmentation"),
                config=self.cfg,
                resume="allow",
            )

        last_metrics: dict[str, float] = {"train_loss": 0.0, "val_loss": 0.0, "miou": 0.0}

        for epoch in range(start_epoch, epochs):
            train_loss = self.train_one_epoch(epoch)
            val_metrics = self.validate()
            self.scheduler.step()

            val_loss = val_metrics["val_loss"]
            miou = val_metrics["miou"]

            is_best = miou > best_miou
            if is_best:
                best_miou = miou

            self.save_checkpoint(checkpoint_dir / "last.pt", epoch=epoch, best_miou=best_miou)
            if is_best:
                self.save_checkpoint(checkpoint_dir / "best.pt", epoch=epoch, best_miou=best_miou)

            if wandb_run is not None and self.accelerator.is_main_process:
                import wandb

                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_miou": miou,
                        "best_miou": best_miou,
                    }
                )

            last_metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "miou": miou,
                "best_miou": best_miou,
            }

        if wandb_run is not None and self.accelerator.is_main_process:
            wandb_run.finish()

        return last_metrics
