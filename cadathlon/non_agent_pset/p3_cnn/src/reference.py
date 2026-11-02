"""Deterministic synthetic cases; never presented as trained-model accuracy."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from support import cross_root


def make_case(case):
    cross_root()
    import numpy as np
    import torch
    from lenet_train import QuadLeNet
    from alexnet_infer import QuadAlexNetTinyInfer

    torch.set_num_threads(1)
    if case["model"] == "lenet":
        model, shape = QuadLeNet().double().eval(), (1, 28, 28)
    elif case["model"] == "alexnet_tiny":
        model = QuadAlexNetTinyInfer(client_downsample=False).double().eval()
        shape = (3, 16, 16)
    else:
        raise ValueError("Unknown model")
    rng = np.random.default_rng(case["weight_seed"])
    with torch.no_grad():
        for name, p in model.named_parameters():
            values = rng.normal(0, 0.04 if name.endswith("weight") else 0.01, p.shape)
            p.copy_(torch.from_numpy(values))
    rng = np.random.default_rng(case["input_seed"])
    x = rng.uniform(-1.0, 1.0, (case["batch"], *shape))
    if case.get("input_kind") == "zeros":
        x.fill(0)
    elif case.get("input_kind") == "alternating":
        x = np.where(np.indices(x.shape).sum(0) % 2, 1.0, -1.0)
    with torch.no_grad():
        logits = model(torch.from_numpy(x)).numpy()
    return model, shape, x, logits
