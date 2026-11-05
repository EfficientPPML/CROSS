"""Native WISE Winograd transforms checked against independent convolution."""
import os
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent_pset/p5_wise/src"))
from schedule import NumpyArithmetic
from wise_source import load_wise
from winograd import execute_winograd, rotations


@pytest.mark.skipif(not os.environ.get("WISE_ROOT"), reason="requires pinned WISE")
@pytest.mark.parametrize("number,slots", [(4, 32), (4, 256), (5, 256), (6, 512)])
@pytest.mark.parametrize("convolution", ["winograd", "tiled-toeplitz"])
def test_complete_native_winograd_cnn_preserves_fixed_network(number, slots, convolution):
    import json
    from schedule import build_schedule, execute, pack_inputs
    from workloads import input_values, reference
    folder = Path(__file__).resolve().parents[1] / "agent_pset/p5_wise"
    case = json.loads((folder / f"test/test{number}.in").read_text())
    plan = build_schedule(case, slots, convolution=convolution)
    digest = plan.identity()
    for seed in [case["input_seed"], 9271]:
        x = input_values(case, seed)
        result = execute(plan, pack_inputs(plan, x), NumpyArithmetic())
        actual = np.concatenate(result)[:plan.output_size].reshape(case["batch"], case["classes"])
        np.testing.assert_allclose(actual, reference(case, x), rtol=1e-11, atol=1e-11)
    assert digest == plan.identity()
    assert plan.summary()["layers"][0]["operation"] == ("winograd" if convolution == "winograd" else "linear")


@pytest.mark.skipif(not os.environ.get("WISE_ROOT"), reason="requires pinned WISE")
@pytest.mark.parametrize("slots", [16, 32, 64])
def test_actual_native_winograd_with_distinct_images_channels_and_padding(slots):
    import torch
    wise, _ = load_wise()
    rng = np.random.default_rng(781)
    order = wise.Order(C=2, H=8, W=8, H_real=6, W_real=6, batchsize=(2, 1, 1), tilesize=(2, 2))
    target = wise.Order(C=4, H=8, W=8, H_real=6, W_real=6, batchsize=(2, 1, 1), tilesize=(2, 2))
    x = rng.normal(size=(2, 2, 6, 6))
    kernel = rng.normal(size=(4, 2, 3, 3))
    bias = rng.normal(size=4)
    layer = wise.pack_conv2d_winograd(order, target, kernel, bn_b=bias).pack(slots)
    raw = np.zeros(2 * 2 * 8 * 8)
    for b, c, h, w in np.ndindex(x.shape):
        raw[order.idx(b, c, h, w)] = x[b, c, h, w]
    out = np.concatenate(execute_winograd(layer, raw.reshape(-1, slots), slots, NumpyArithmetic()))
    actual = np.array([out[target.idx(b, c, h, w)] for b, c, h, w in np.ndindex((2, 4, 6, 6))]).reshape(2, 4, 6, 6)
    expected = torch.nn.functional.conv2d(torch.from_numpy(x), torch.from_numpy(kernel), torch.from_numpy(bias), padding=1).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert all(0 < r < slots for r in rotations(layer, slots))
