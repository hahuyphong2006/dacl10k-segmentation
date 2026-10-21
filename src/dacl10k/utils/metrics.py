"""Evaluation metrics for multi-label semantic segmentation."""

import torch


class MultiLabelIoUMetric:
    """Vectorized multi-label Intersection-over-Union (IoU) metric accumulator."""

    def __init__(self, num_classes: int = 19, threshold: float = 0.5, eps: float = 1e-7) -> None:
        """Initialize the MultiLabelIoUMetric.

        Args:
            num_classes: Number of target classes.
            threshold: Probability threshold for binary mask prediction.
            eps: Smoothing epsilon to prevent division by zero.
        """
        self.num_classes = num_classes
        self.threshold = threshold
        self.eps = eps
        self.intersection = torch.zeros(self.num_classes, dtype=torch.float64)
        self.union = torch.zeros(self.num_classes, dtype=torch.float64)

    def reset(self) -> None:
        """Reset intersection and union accumulators to zero."""
        self.intersection.zero_()
        self.union.zero_()

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate per-class intersection and union from a batch of logits and targets.

        Args:
            logits: Raw model output tensor of shape [B, C, H, W].
            targets: Ground truth binary tensor of shape [B, C, H, W].
        """
        probs = torch.sigmoid(logits)
        preds = (probs >= self.threshold).to(torch.float64)
        targets_f = (targets >= 0.5).to(torch.float64)

        # Sum across (Batch, H, W) -> shape [C]
        dims = (0, 2, 3)
        batch_inter = (preds * targets_f).sum(dim=dims).cpu()
        batch_union = (preds + targets_f - preds * targets_f).sum(dim=dims).cpu()

        self.intersection += batch_inter
        self.union += batch_union

    def compute(self) -> dict[str, float | list[float]]:
        """Compute mean IoU and per-class IoU across all accumulated batches.

        Returns:
            Dictionary with keys "miou" (float) and "per_class_iou" (list of floats).
        """
        per_class_iou = (self.intersection + self.eps) / (self.union + self.eps)
        miou = float(per_class_iou.mean().item())
        return {
            "miou": miou,
            "per_class_iou": [float(x) for x in per_class_iou.tolist()],
        }
