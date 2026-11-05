"""Real secure CKKS proof of native WISE Winograd; a component, not a full NN."""
import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys
import time
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("cpu", "tpu"), default="cpu")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("choose a fresh output path")
    os.environ["JAX_PLATFORMS"] = args.backend
    import jax
    import torch
    from cross_backend import create_contexts, make_arithmetic, profile
    from schedule import NumpyArithmetic
    from wise_source import load_wise
    from winograd import execute_winograd, rotations
    from support import require_secure_execution
    started = time.perf_counter()
    spec = profile(1)
    slots = spec["degree"] // 2
    side = int(np.sqrt(4 * slots))
    if side * side != 4 * slots:
        raise ValueError("probe requires square tile plane")
    wise, provenance = load_wise()
    order = wise.Order(C=1, H=side, W=side, H_real=side-2, W_real=side-2, tilesize=(2, 2))
    rng = np.random.default_rng(5107)
    x = rng.uniform(-.25, .25, (1, 1, side-2, side-2))
    kernel = rng.uniform(-.15, .15, (1, 1, 3, 3))
    bias = np.array([.01])
    with redirect_stdout(sys.stderr):
        layer = wise.pack_conv2d_winograd(order, order, kernel, bn_b=bias).pack(slots)
    raw = np.zeros(4 * slots)
    positions = np.array([order.idx(*idx) for idx in np.ndindex(x.shape)])
    raw[positions] = x.ravel()
    packed = raw.reshape(-1, slots)
    expected = torch.nn.functional.conv2d(torch.from_numpy(x), torch.from_numpy(kernel),
                                          torch.from_numpy(bias), padding=1).numpy().ravel()
    simulated = np.concatenate(execute_winograd(layer, packed, slots, NumpyArithmetic()))
    np.testing.assert_allclose(simulated[positions], expected, rtol=0, atol=1e-12)
    devices = jax.devices()
    if devices[0].platform != args.backend:
        raise RuntimeError("wrong backend")
    with redirect_stdout(sys.stderr), jax.default_device(devices[0]):
        client, server = create_contexts(spec, rotations(layer, slots))
        arithmetic = make_arithmetic(server)
        encrypted = [client.encrypt_slots(v) for v in packed]
        jax.block_until_ready([v.polynomial for v in encrypted])
        require_secure_execution(arithmetic, encrypted)
        begin = time.perf_counter()
        output = execute_winograd(layer, encrypted, slots, arithmetic)
        jax.block_until_ready([v.polynomial for v in output])
        server_s = time.perf_counter() - begin
        require_secure_execution(arithmetic, output)
        decoded = np.concatenate([client.decrypt_slots(v) for v in output])
    error = float(np.max(np.abs(decoded[positions] - expected)))
    # Every physical output slot is checked against the native ideal schedule,
    # including the two-pixel padding and cross-tile routing boundaries.
    full_error = float(np.max(np.abs(decoded - simulated)))
    passed = np.isfinite(full_error) and full_error <= 1e-5 and error <= 1e-5
    result = {"status": "completed" if passed else "failed", "passed": bool(passed),
              "scope": "one native WISE Winograd convolution, not a complete NN or calibrated timing",
              "backend": args.backend, "device": str(devices[0]), "crypto_profile": spec,
              "wise": provenance, "logical_shape": list(x.shape), "ciphertexts": len(output),
              "server_cold_s": server_s, "elapsed_s": time.perf_counter()-started,
              "max_abs_error": error, "all_physical_slots_max_abs_error": full_error}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
