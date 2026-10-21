"""Unit tests for the dacl10k dataset, rasterizer, and transforms."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from dacl10k.data import (
    CLASS_TO_IDX,
    NUM_CLASSES,
    DACL10kDataset,
    get_transforms,
    rasterize_annotations,
)


def test_rasterize_exterior_and_interior() -> None:
    """Verify polygon rasterization with exterior contours, interior holes, and multi-hot output."""
    synthetic_annotation = {
        "size": {"height": 100, "width": 100},
        "objects": [
            {
                "classTitle": "Crack",
                "points": {
                    "exterior": [[10, 10], [50, 10], [50, 50], [10, 50]],
                    "interior": [],
                },
            },
            {
                "classTitle": "Spalling",
                "points": {
                    "exterior": [[20, 20], [80, 20], [80, 80], [20, 80]],
                    "interior": [[[40, 40], [60, 40], [60, 60], [40, 60]]],
                },
            },
            {
                "classTitle": "UnknownDefect",
                "points": {
                    "exterior": [[0, 0], [99, 0], [99, 99], [0, 99]],
                    "interior": [],
                },
            },
        ],
    }

    mask = rasterize_annotations(synthetic_annotation, height=100, width=100)

    assert mask.shape == (100, 100, NUM_CLASSES)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 1})

    crack_idx = CLASS_TO_IDX["Crack"]
    spalling_idx = CLASS_TO_IDX["Spalling"]

    # Crack interior point should be 1
    assert mask[30, 30, crack_idx] == 1
    assert mask[5, 5, crack_idx] == 0

    # Spalling exterior region outside the hole should be 1
    assert mask[25, 25, spalling_idx] == 1
    # Spalling interior hole should be 0
    assert mask[50, 50, spalling_idx] == 0

    # Overlapping pixel at (30, 30) has both Crack and Spalling active
    assert mask[30, 30, crack_idx] == 1
    assert mask[30, 30, spalling_idx] == 1

    # All other channels must be strictly 0
    for c in range(NUM_CLASSES):
        if c not in (crack_idx, spalling_idx):
            assert np.all(mask[:, :, c] == 0)


def test_dataset_output_shapes_and_types(tmp_path: Path) -> None:
    """Verify DACL10kDataset output dictionary shapes, types, and overlapping multi-hot masks."""
    split = "train"
    img_dir = tmp_path / "images" / split
    ann_dir = tmp_path / "annotations" / split
    img_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    for i in range(2):
        dummy_img = np.random.randint(0, 256, size=(200, 200, 3), dtype=np.uint8)
        img_path = img_dir / f"sample_{i}.jpg"
        cv2.imwrite(str(img_path), dummy_img)

        dummy_ann = {
            "size": {"height": 200, "width": 200},
            "objects": [
                {
                    "classTitle": "Crack",
                    "points": {
                        "exterior": [[0, 0], [199, 0], [199, 199], [0, 199]],
                        "interior": [],
                    },
                },
                {
                    "classTitle": "Rust",
                    "points": {
                        "exterior": [[0, 0], [199, 0], [199, 199], [0, 199]],
                        "interior": [],
                    },
                },
            ],
        }
        ann_path = ann_dir / f"sample_{i}.json"
        ann_path.write_text(json.dumps(dummy_ann), encoding="utf-8")

    transforms = get_transforms("train", image_size=(128, 128))
    dataset = DACL10kDataset(root_dir=tmp_path, split=split, transforms=transforms)

    assert len(dataset) == 2

    sample = dataset[0]
    assert isinstance(sample["id"], str)
    assert sample["id"] == "sample_0"

    image_tensor = sample["image"]
    mask_tensor = sample["mask"]

    assert isinstance(image_tensor, torch.Tensor)
    assert image_tensor.shape == torch.Size([3, 128, 128])
    assert image_tensor.dtype == torch.float32

    assert isinstance(mask_tensor, torch.Tensor)
    assert mask_tensor.shape == torch.Size([19, 128, 128])
    assert mask_tensor.dtype == torch.float32
    assert mask_tensor.min() >= 0.0
    assert mask_tensor.max() <= 1.0

    # Verify overlapping multi-hot labels at the same spatial locations
    crack_idx = CLASS_TO_IDX["Crack"]
    rust_idx = CLASS_TO_IDX["Rust"]
    assert torch.all(mask_tensor[crack_idx] == 1.0)
    assert torch.all(mask_tensor[rust_idx] == 1.0)
    assert torch.any(mask_tensor.sum(dim=0) >= 2.0)
