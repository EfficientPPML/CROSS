"""Fixed public NN workloads and an independent PyTorch numerical oracle."""
import numpy as np


def parameters(case):
    rng = np.random.default_rng(case["weight_seed"])
    def weights(shape):
        return rng.normal(0, .25 / np.sqrt(np.prod(shape[1:])), shape)
    if case["network"] == "mlp":
        a, h, c = case["inputs"], case["hidden"], case["classes"]
        return {"w1": weights((h, a)), "b1": rng.normal(0, .01, h),
                "w2": weights((c, h)), "b2": rng.normal(0, .01, c)}
    if case["network"] == "cnn":
        channels, side, c = case["channels"], case["side"], case["classes"]
        return {"kernel": weights((channels, 1, 3, 3)), "b1": rng.normal(0, .01, channels),
                "w2": weights((c, channels * (side // 2) ** 2)), "b2": rng.normal(0, .01, c)}
    raise ValueError("unknown network")


def input_values(case, seed=None):
    rng = np.random.default_rng(case["input_seed"] if seed is None else seed)
    shape = ((case["batch"], case["inputs"]) if case["network"] == "mlp"
             else (case["batch"], 1, case["side"], case["side"]))
    return rng.uniform(-1, 1, shape)


def reference(case, x, weights=None):
    import torch
    import torch.nn.functional as f
    p = {k: torch.from_numpy(v) for k, v in (weights or parameters(case)).items()}
    x = torch.as_tensor(x, dtype=torch.float64)
    with torch.no_grad():
        if case["network"] == "mlp":
            y = f.linear(x, p["w1"], p["b1"]).square()
        else:
            y = f.conv2d(x, p["kernel"], p["b1"], padding=1).square()
            y = f.avg_pool2d(y, 2).flatten(1)
        return f.linear(y, p["w2"], p["b2"]).numpy()
