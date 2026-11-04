"""Deterministic synthetic cases; never presented as trained-model accuracy."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from support import cross_root

INPUT_SHAPE = (784,)


def build_model(hidden=50):
    """The fixed serving network: Linear -> square -> Linear, depth 3.

    Written out rather than imported so the ranked architecture cannot drift
    with the demos. `x * x` is a traced `call_function` node that
    `nn.vectorize` claims as a polynomial activation; writing `nn.ReLU` here
    would substitute to the same x**2 but emit a warning, so the square is
    literal and the traced graph matches the scored function exactly.
    """
    import torch
    from torch import nn as tnn

    class MnistMLP(tnn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = tnn.Linear(784, hidden)
            self.fc2 = tnn.Linear(hidden, 10)

        def forward(self, x):
            x = self.fc1(x)
            x = x * x
            return self.fc2(x)

    return MnistMLP()


def make_case(case):
    cross_root()
    import numpy as np
    import torch

    torch.set_num_threads(1)
    model = build_model(case.get("hidden", 50)).double().eval()
    rng = np.random.default_rng(case["weight_seed"])
    with torch.no_grad():
        for name, p in model.named_parameters():
            values = rng.normal(0, 0.04 if name.endswith("weight") else 0.01, p.shape)
            p.copy_(torch.from_numpy(values))
    rng = np.random.default_rng(case["input_seed"])
    x = rng.uniform(-1.0, 1.0, (case["batch"], *INPUT_SHAPE))
    kind = case.get("input_kind")
    if kind == "zeros":
        x.fill(0)
    elif kind == "alternating":
        x = np.where(np.indices(x.shape).sum(0) % 2, 1.0, -1.0)
    elif kind == "distinct_lanes":
        # every batch lane different, so a per-lane bug cannot hide
        x = (np.arange(x.size).reshape(x.shape) % 17) / 8.0 - 1.0
    with torch.no_grad():
        logits = model(torch.from_numpy(x)).numpy()
    return model, INPUT_SHAPE, x, logits
