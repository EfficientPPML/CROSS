"""Actual secure CPU packed-transformer components; not full encrypted models."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
if os.environ['JAX_PLATFORMS'] != 'cpu':
    raise RuntimeError('this component validation requires CPU')
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'cadathlon/common')]
import jax
import numpy as np
from jaxite_word import CKKSContext, he_params, key_gen, util, ckks_ctx
from jaxite_word.composite_prime_gen import composite_prime_gen
from encrypted_refresh import ClientRotationKeyStore
from encrypted_packed_transformer import PackedPairLayout, PackedTransformer, plan_packed_transformer
from encrypted_transformer import public_toy_approximations
from support import load_module

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--report', type=Path, default=Path(__file__).with_suffix('.json'))
parser.add_argument('--scale-bits', type=int, choices=(44, 56), default=56)
parser.add_argument('--only', choices=('all', 'attention'), default='all')
args = parser.parse_args()
jax.config.update('jax_enable_x64', True)
started = time.monotonic()
sources = [Path(__file__), *sorted((ROOT/'jaxite_word').glob('*.py'))]
sources += [ROOT/'cadathlon/common'/name for name in ('encrypted_packed_transformer.py', 'encrypted_packed_qwen.py',
    'encrypted_polynomial.py', 'encrypted_transformer.py', 'encrypted_primitives.py', 'encrypted_compilation.py',
    'encrypted_rotation.py', 'encrypted_refresh.py', 'support.py')]
folder = ROOT/'cadathlon/agent_pset/p2_bert'
sources += [folder/'src/reference.py', folder/'test/test1.in']
def hashes():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
report = dict(scope=__doc__, passed=False, cpu_only=True, full_model=False, source_hashes_start=hashes(), checks={})
client = server = ops = key_store = None

try:
    case = json.loads((folder/'test/test1.in').read_text())
    weights, embeddings, public_mask, _ = load_module(folder/'src/reference.py').make_case(case)
    degree, depth, dnum = 32768, 10, 4
    slots = degree//2
    layout = PackedPairLayout(1, 4, 32, slots)
    plan = plan_packed_transformer('bert', case['config'], weights, batch=1, sequence=4, slots=slots)
    rotations = tuple(sorted(set(plan['rotations']) | {3, slots-3}))
    q = composite_prime_gen(2, 2*(depth+1), 61, args.scale_bits, 2*degree, 31)
    p = util.generate_p_towers(q, dnum, degree, aux_bits=30)
    assert he_params.qp_is_secure(q, p, degree, he_params.SecurityTarget(128, 'classical'))
    he_params.assert_p_coverage(q, p, dnum)
    report['ring'] = dict(degree=degree, depth=depth, dnum=dnum, q_count=len(q), p_count=len(p),
                          actual_log2_qp=sum(map(math.log2, q+p)), security_bits=128, rotations=rotations, scale_bits=args.scale_bits)
    keys = key_gen.gen_pke_pair(q, [], degree)
    evaluation = key_gen.gen_evaluation_key(keys['secret_key'], q=q, P=p, dnum=dnum)
    key_store = ClientRotationKeyStore.generate(keys['secret_key'], q, p, degree=degree, dnum=dnum, rotations=rotations)
    scale = float(q[-2])*q[-1]
    params = dict(degree=degree, num_slots=slots, scaling_factor=scale, output_scale=scale, q_towers=q, p_towers=p,
                  composite_degree=2, degree_layout=(128, 256), p=args.scale_bits, CKKS_M_FACTOR=1, max_bits_in_word=61, noise_scale_degree=1)
    client = CKKSContext(dict(params, **keys))
    server = CKKSContext(dict(params, evaluation_key=[
        jax.numpy.asarray(evaluation[name], dtype='uint32').transpose(0, 2, 1) for name in ('a', 'b')]))
    server.program_initialization(rotations, dnum=dnum, r=128, c=256, batch=1, pregenerated_rotation_keys=key_store.keys)
    client._polynomial_ntt_ctx = server._param_cache.ntt_q_max
    assert server.secret_key is None and server._param_cache.secret_key is None and server.parameters.get('secret_key') is None
    def forbidden(*a, **kw):
        raise AssertionError('server encryption/decryption is forbidden')
    server.encrypt = server.decrypt = server.encrypt_slots = server.decrypt_slots = forbidden
    ops = PackedTransformer(server, layout, low_memory=True)
    report['keyless_evaluator'] = True

    def pack(values):
        return layout.pack_client_embeddings(values)[0]
    def encrypt(values):
        return client.encrypt_slots(pack(values))
    def verify(name, actual, expected, level, tolerance=2e-5):
        actual.polynomial.block_until_ready()
        decoded = client.decrypt_slots(actual)
        errors = np.abs(decoded-expected)
        error = float(np.max(errors))
        actual_level = ops.arithmetic._level(actual)
        metadata_ok = actual_level == level and actual._ckks_nsd == 1 and math.isclose(actual._ckks_scale, scale, rel_tol=1e-12)
        report['checks'][name] = dict(passed=bool(error < tolerance and metadata_ok), max_abs_error=error, tolerance=tolerance, output_level=actual_level, expected_level=level, noise_scale_degree=actual._ckks_nsd, error_at_zero_expected_slots=float(np.max(errors[expected==0])) if np.any(expected==0) else None)
        print(f'[{time.monotonic()-started:.2f}s] {name}: error={error:.3g}, level={level}', flush=True)
        ops.arithmetic.checkpoint(actual, name)

    rng = np.random.default_rng(361)
    if args.only == 'all':
        original = encrypt(embeddings)
        name = 'bert.encoder.layer.0.intermediate.dense'
        matrix, bias = weights[name+'.weight'], weights[name+'.bias']
        actual = ops.linear(original, matrix, bias)
        verify('sequential_BSGS_rectangular_bias', actual, pack(embeddings@matrix.T+bias), depth-1)
        del actual
        name = 'bert.pooler.dense'
        matrix, bias = weights[name+'.weight'], weights[name+'.bias']
        mask = layout.tile((np.arange(layout.width)==0)[None, :, None, None])
        actual = ops.linear(original, matrix, bias, input_mask=mask)
        expected = embeddings.copy(); expected[:, 1:] = 0
        verify('CLS_mask_preserves_public_bias', actual, pack(expected@matrix.T+bias), depth-1)
        del actual
        actual = ops.linear(original, np.zeros((3, 16)), np.array([.2, -.1, .3]))
        verify('zero_matrix_public_bias', actual, pack(np.broadcast_to([.2, -.1, .3], (1, 4, 3))), depth-1)
        del actual, original

        rng = np.random.default_rng(361)
        values = rng.normal(size=(1, 4, 24))
        values = (values-values.mean(-1, keepdims=True))/values.std(-1, keepdims=True) + np.arange(4)[None, :, None]*.2
        gamma, beta = rng.uniform(.8, 1.2, 24), rng.uniform(-.1, .1, 24)
        inverse = public_toy_approximations('bert')['inverse_sqrt']
        actual = ops.normalize(encrypt(values), gamma, 1e-12, inverse, bias=beta, center=True)
        centered = values-values.mean(-1, keepdims=True)
        variance = np.mean(centered*centered, axis=-1, keepdims=True)+1e-12
        expected = centered*inverse.evaluate_public(variance)*gamma+beta
        verify('LayerNorm_H24_zero_padding', actual, pack(expected), depth-8, tolerance=1e-4)
        del actual

    if args.only == 'attention':
        # Reproduce the same head inputs as the complete validation. Skipping
        # earlier expensive operations must not silently change the fixture.
        rng.normal(size=(1, 4, 24))
        rng.uniform(.8, 1.2, 24)
        rng.uniform(-.1, .1, 24)
    # Three six-coordinate heads in eight-coordinate segments leave both
    # per-head padding and a completely inactive fourth head segment.
    head_values = rng.uniform(-.5, .5, (1, 4, 3, 6))
    padded = np.zeros((1, 4, 32))
    for head in range(3): padded[..., head*8:head*8+6] = head_values[..., head, :]
    gamma = rng.uniform(.8, 1.2, 6)
    inverse = public_toy_approximations('qwen')['inverse_sqrt']
    actual = ops.normalize_heads(encrypt(padded), gamma, 1e-6, inverse, 3)
    variance = np.mean(head_values*head_values, axis=-1, keepdims=True)+1e-6
    normalized = head_values*inverse.evaluate_public(variance)*gamma
    expected = np.zeros_like(padded)
    for head in range(3): expected[..., head*8:head*8+6] = normalized[..., head, :]
    verify('head_RMSNorm_padded_and_inactive_segments', actual, pack(expected), 0, tolerance=2e-4)
    del actual

    original = encrypt(padded)
    actual = ops.transpose_pairs(original)
    packed = pack(padded).reshape(-1, 4, 4, 32)
    verify('query_key_transpose', actual, packed.transpose(0, 2, 1, 3).reshape(-1), depth-1)
    del actual
    actual = ops.rope(original, 3, 6, 1000000.)
    positions = np.arange(4)[None, :, None, None]
    frequencies = 1000000.**(-2*np.arange(3)/6)
    angles = np.concatenate([positions*frequencies, positions*frequencies], axis=-1)
    rotated = np.concatenate([-head_values[..., 3:], head_values[..., :3]], axis=-1)
    rope = head_values*np.cos(angles)+rotated*np.sin(angles)
    expected = np.zeros_like(padded)
    for head in range(3): expected[..., head*8:head*8+6] = rope[..., head, :]
    verify('RoPE_non_power_of_two_head_width', actual, pack(expected), depth-1)
    del actual, original
    report['passed'] = all(check['passed'] for check in report['checks'].values())
except BaseException as error:
    report['error'] = f'{type(error).__name__}: {error}'
    traceback.print_exc()
finally:
    for context in (client, server):
        if context is not None: ckks_ctx.evict_context_caches(context)
    for name in ('client', 'server', 'ops', 'actual', 'original', 'context'):
        globals().pop(name, None)
    jax.clear_caches(); gc.collect()
    if key_store is not None: key_store.close()
    report['source_hashes_end'] = hashes()
    report['source_unchanged'] = report['source_hashes_start'] == report['source_hashes_end']
    report['passed'] = report['passed'] and report['source_unchanged']
    report['wall_s'] = time.monotonic()-started
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
raise SystemExit(0 if report['passed'] else 1)
