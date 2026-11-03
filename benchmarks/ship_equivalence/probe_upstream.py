"""Check pristine upstream SHIP compatibility; these are not latency results.

Run with upstream Jaxite on PYTHONPATH. Each invocation uses a fresh process,
so upstream mutable kernel defaults cannot leak between cases.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import traceback

import gmpy2
import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import bootstrapping, encode, encrypt, key_gen, ntt_cpu, random, types
import numpy as np

jax.config.update('jax_enable_x64', True)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--degree', type=int, default=16)
parser.add_argument('--q-count', type=int, default=5)
parser.add_argument('--p-count', type=int, default=1)
parser.add_argument('--weight', type=int, default=1)
parser.add_argument('--theta', type=int, default=2)
parser.add_argument('--q0-bits', type=int, default=30)
parser.add_argument('--upstream-fixture', action='store_true',
                    help='Use upstream hw1 test constants and zero noise as a control')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.degree < 8 or args.degree & (args.degree - 1):
    parser.error('degree must be a power of two >= 8')
if not 1 <= args.weight <= args.degree or args.p_count < 1:
    parser.error('weight must be in [1,degree] and p-count must be positive')
if args.q_count < (args.weight-1).bit_length()+3:
    parser.error('Q chain must leave at least two output limbs')
if not 1 <= args.theta <= args.degree//2 or not 4 <= args.q0_bits <= 31:
    parser.error('invalid theta or q0-bits')
source = Path(bootstrapping.__file__).resolve()
root = source.parents[2]
sha = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
if sha != '1c35ddd29179e946ab7d08515053cefd7e76c46e':
    raise RuntimeError(f'Unexpected upstream revision: {sha}')
if subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
    raise RuntimeError('Baseline checkout has tracked modifications')
used = set()
def primes(bits, count):
    value = ((1 << bits)-2)//(2*args.degree)*(2*args.degree)+1
    result = []
    while len(result) < count:
        if value < 3:
            raise ValueError('Not enough distinct NTT primes for this configuration')
        if value not in used and gmpy2.is_prime(value):
            result.append(value)
            used.add(value)
        value -= 2*args.degree
    return result
q = primes(args.q0_bits, 1) + primes(30, args.q_count-1)
p = primes(30, args.p_count)
if args.upstream_fixture:
    if args.degree != 16 or args.weight != 1 or args.q_count != 5 or args.theta != 2:
        parser.error('upstream fixture requires N=16, weight=1, Q count=5, theta=2')
    from jaxite.jaxite_ckks.bootstrapping_test import TEST_PRIMES
    q = TEST_PRIMES[:-1]
    p = TEST_PRIMES[-1:] + [v for v in p if v not in TEST_PRIMES][:args.p_count-1]
scale = float(2**30 if args.upstream_fixture else q[-1])
gamma = 1.0 if args.upstream_fixture else 8.0
report = dict(source_commit=sha, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              degree=args.degree, q=q, p=p, weight=args.weight, theta=args.theta,
              backend=jax.default_backend(), jax_version=jax.__version__,
              scope='coefficient-to-slot compatibility probe, not a full-bootstrap benchmark',
              input_key='sparse binary', output_key='same sparse binary',
              scale=scale, gamma=gamma, error_gate=1e-3 if args.upstream_fixture else 2**-10,
              upstream_fixture=args.upstream_fixture,
              noise='zero (upstream control only)' if args.upstream_fixture else 'Gaussian sigma=3.19')
try:
    n = args.degree
    rng = random.ZeroNoiseRandomSource() if args.upstream_fixture else random.TestRandomSource(42)
    coeffs = np.zeros(n, dtype=np.int64)
    coeffs[[0] if args.upstream_fixture else rng.rng.sample(range(n), args.weight)] = 1
    residues = np.column_stack([coeffs % v for v in q]).astype(np.uint64)
    sk = types.SecretKey(ntt_cpu.ntt_negacyclic_poly(residues, q), np.array(q, dtype=np.uint32))
    sk0 = types.SecretKey(sk.data[:, :1], sk.moduli[:1])
    pk, _ = key_gen.keygen(n, q[:1], random_source=rng, sk=sk0)
    kernel = bootstrapping.SHIP()
    report['stage'] = 'setup'
    kernel.precompute_constants(q, p, n, 1, sk, args.theta, rng)
    values = np.full(n//2, 0.1) if args.upstream_fixture else np.linspace(-0.125, 0.125, n//2)
    coeff_values = np.pad(np.rint(values*q[0]/gamma).astype(np.int64), (0, n//2))
    data = ntt_cpu.ntt_negacyclic_poly((coeff_values[:, None] % q[0]).astype(np.uint64), q[:1])
    pt = types.Plaintext(jnp.array(data, dtype=jnp.uint32), jnp.array(q[:1], dtype=jnp.uint32))
    ct = encrypt.Encrypt(pk).encrypt(pt, random_source=rng)
    report['stage'] = 'half_bootstrap'
    out = jax.jit(lambda x: kernel.half_bootstrap(x, args.theta, scale, gamma))(ct)
    jax.block_until_ready(out.data)
    report['output_q'] = np.asarray(out.moduli).tolist()
    report['stage'] = 'decode'
    actual = np.asarray(encode.Decode(scale, n//2).decode(encrypt.Decrypt(sk).decrypt(out)))
    expected = gamma/(2*np.pi)*np.sin(2*np.pi*values/gamma) if args.upstream_fixture else values
    gate = 1e-3 if args.upstream_fixture else 2**-10
    error = float(np.max(np.abs(actual-expected)))
    report.update(max_absolute_error=error, error_gate=gate,
                  error_reference='sine approximation' if args.upstream_fixture else 'original plaintext',
                  passed=bool(np.isfinite(error) and error<=gate), stage='complete')
except Exception as exc:
    report.update(passed=False, error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('traceback','q','p')}, indent=2))
raise SystemExit(0 if report['passed'] else 1)
