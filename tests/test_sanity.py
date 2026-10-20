import torch

import dacl10k


def test_version() -> None:
    assert dacl10k.__version__ == "0.1.0"


def test_dummy_sigmoid_bounds() -> None:
    logits = torch.randn(2, 19, 64, 64)
    probs = torch.sigmoid(logits)
    assert probs.shape == (2, 19, 64, 64)
    assert torch.all((probs >= 0.0) & (probs <= 1.0))
    assert torch.all((probs > 0.0) & (probs < 1.0))
