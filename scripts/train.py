"""CLI entry point for training dacl10k multi-label segmentation models."""

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from dacl10k.data.dataset import DACL10kDataset
from dacl10k.data.transforms import get_transforms
from dacl10k.engine.trainer import Trainer
from dacl10k.models.factory import build_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train dacl10k segmentation model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default_config.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional path to a checkpoint file to resume training from.",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Run a 1-epoch smoke test on synthetic TensorDataset data.",
    )
    return parser.parse_args()


def main() -> None:
    """Load config, build datasets and model, and execute training."""
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)

    num_classes = int(config.get("num_classes", 19))

    if args.dummy:
        config["epochs"] = 1
        config["encoder_weights"] = None
        if config.get("arch", "segformer").lower() == "segformer":
            config["encoder_name"] = "mit_b0"
        else:
            config["encoder_name"] = "resnet18"

        dummy_images = torch.randn(4, 3, 64, 64)
        dummy_masks = torch.randint(0, 2, (4, num_classes, 64, 64)).to(torch.float32)
        train_ds = TensorDataset(dummy_images, dummy_masks)
        val_ds = TensorDataset(dummy_images, dummy_masks)
        train_loader = DataLoader(train_ds, batch_size=2, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)
    else:
        data_dir = Path(config.get("data_dir", "data/dacl10k"))
        img_size_cfg = config.get("image_size", [512, 512])
        image_size = (int(img_size_cfg[0]), int(img_size_cfg[1]))
        batch_size = int(config.get("batch_size", 8))
        num_workers = int(config.get("num_workers", 4))

        train_ds = DACL10kDataset(
            root_dir=data_dir,
            split="train",
            transforms=get_transforms("train", image_size=image_size),
        )
        val_ds = DACL10kDataset(
            root_dir=data_dir,
            split="val",
            transforms=get_transforms("val", image_size=image_size),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    model = build_model(config)
    trainer = Trainer(
        cfg=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    metrics = trainer.fit(resume_path=args.resume)
    print(f"Training complete: {metrics}")


if __name__ == "__main__":
    main()
