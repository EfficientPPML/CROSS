"""Check bounded rescale against the original optimized function on CPU.

Put bounded-rescale-patched Jaxite and CROSS's jaxite_word on PYTHONPATH.
Pass the rescale.py from PR147 with only jaxite-pr147-optimized.patch applied.
Full encrypted TPU correctness is checked separately through run_common.py.
"""

import argparse
import ast
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jaxite.jaxite_ckks import rescale, types
from ship_test import primes

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("reference_rescale", type=Path)
args = parser.parse_args()
jax.config.update("jax_enable_x64", True)
if jax.default_backend() != "cpu":
    raise RuntimeError("Run original-vs-bounded residue checks on CPU")
module = ast.parse(args.reference_rescale.read_text())
fn = next(
    node
    for node in ast.walk(module)
    if isinstance(node, ast.FunctionDef) and node.name == "_rescale_fused_jax"
)
if "fori_loop" in ast.unparse(fn):
    raise ValueError("The reference must be the original unrolled rescale")
module = ast.Module(body=[fn], type_ignores=[])
namespace = dict(jax=jax, jnp=jnp, types=types, Ciphertext=types.Ciphertext)
exec(compile(module, str(args.reference_rescale), "exec"), namespace)
original = namespace["_rescale_fused_jax"]
for drops in (3, 5, 24):
    n = 16
    q, _ = primes(n, q_count=drops + 3, p_count=1)
    kernel = rescale.Rescale()
    kernel.precompute_constants(q, num_rescales=drops, r=4, c=4)
    data = np.random.default_rng(193).integers(0, q, (2, n, len(q)), dtype=np.uint32)
    ct = types.Ciphertext(jnp.asarray(data), jnp.asarray(q, jnp.uint32))
    expected = jax.jit(lambda k, c: original(k, c).data)(kernel, ct)
    actual = jax.jit(lambda k, c: k._rescale_fused_jax(c).data)(kernel, ct)
    np.testing.assert_array_equal(actual, expected)
    print(f"{drops} drops: bounded recurrence preserves every residue", flush=True)
