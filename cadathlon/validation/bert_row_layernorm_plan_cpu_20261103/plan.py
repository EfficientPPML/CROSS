"""Public full-width BERT row-LayerNorm schedule; no keys or ciphertexts."""
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import hashlib
import io
import json
import math
import os
import struct
import sys
import tarfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/jianming_gatech/CROSS')
sys.path[:0] = [str(ROOT), str(ROOT / 'cadathlon/common')]
if os.environ.get('JAX_PLATFORMS') != 'cpu':
    raise RuntimeError('explicit CPU backend required for public import planning')
from encrypted_polynomial import EncryptedPolynomialEvaluator, nonlinear_approximation
from jaxite_word import CKKSContext, he_params, key_gen, util
from jaxite_word.composite_prime_gen import composite_prime_gen
import jax

assert {device.platform for device in jax.devices()} == {'cpu'}


def forbidden(*args, **kwargs):
    raise AssertionError('public plan must not generate keys or ciphertext contexts')


CKKSContext.__init__ = forbidden
for module in {key_gen, sys.modules.get('key_gen')} - {None}:
    for name in ('gen_pke_pair', 'gen_evaluation_key', 'gen_rotation_key', 'key_switch_gen'):
        setattr(module, name, forbidden)

FEATURES = 4096
WIDTH = 768
SLOTS = 16384
ROWS = SLOTS // FEATURES
MEAN_SHIFTS = tuple(1 << bit for bit in range(8)) + (256, 512)
BROADCAST_SHIFTS = tuple(-(1 << bit) for bit in range(12))


@dataclass
class Value:
    data: np.ndarray
    level: int


class PublicArithmetic(EncryptedPolynomialEvaluator):
    """Run the existing evaluator's power schedule on public NumPy arrays."""

    def __init__(self):
        self.strategy = 'power'
        self.refresher = None
        self.ctx = SimpleNamespace(num_slots=SLOTS)
        self.counts = Counter()
        self.rotations = set()

    def _level(self, value):
        return value.level

    def weighted_sum(self, values, weights, *, bias=0.):
        values, weights = tuple(values), tuple(weights)
        self.counts['plaintext_products'] += len(values)
        self.counts['composite_rescales'] += 1
        self.counts['weighted_sums'] += 1
        return Value(sum(v.data * w for v, w in zip(values, weights)) + bias,
                     min(v.level for v in values) - 1)

    def multiply(self, left, right):
        self.counts['ciphertext_products'] += 1
        self.counts['composite_rescales'] += 1
        return Value(left.data * right.data, min(left.level, right.level) - 1)

    def add(self, left, right, *, subtract=False):
        self.counts['additions'] += 1
        return Value(left.data - right.data if subtract else left.data + right.data,
                     min(left.level, right.level))

    def rotate(self, value, shift):
        self.counts['rotations'] += 1
        self.rotations.add(shift % SLOTS)
        return Value(np.roll(value.data, -shift), value.level)


def prefix768(arithmetic, value):
    """Rolling sum is used only at anchors; other positions are discarded."""
    block256 = value
    for bit in range(8):
        block256 = arithmetic.add(block256, arithmetic.rotate(block256, 1 << bit))
    block512 = arithmetic.add(block256, arithmetic.rotate(block256, 256))
    return arithmetic.add(block512, arithmetic.rotate(block256, 512))


def broadcast_prefix768(arithmetic, anchors):
    block256 = anchors
    for bit in range(8):
        block256 = arithmetic.add(block256, arithmetic.rotate(block256, -(1 << bit)))
    block512 = arithmetic.add(block256, arithmetic.rotate(block256, -256))
    return arithmetic.add(block512, arithmetic.rotate(block256, -512))


def broadcast4096(arithmetic, anchors):
    result = anchors
    for bit in range(12):
        result = arithmetic.add(result, arithmetic.rotate(result, -(1 << bit)))
    return result


def evaluate(arithmetic, value, *, valid_rows, approximation, epsilon, gamma, beta):
    if type(valid_rows) is not int or not 1 <= valid_rows <= ROWS:
        raise ValueError('valid_rows outside the physical tile')
    if np.shape(value.data) != (SLOTS,):
        raise ValueError('one full physical tile required')
    anchors = np.zeros((ROWS, FEATURES))
    anchors[:valid_rows, 0] = 1 / WIDTH
    mean_anchors = arithmetic.weighted_sum([prefix768(arithmetic, value)], [anchors.ravel()])
    mean = broadcast_prefix768(arithmetic, mean_anchors)
    centered = arithmetic.add(value, mean, subtract=True)
    squares = arithmetic.multiply(centered, centered)
    bias = np.zeros((ROWS, FEATURES))
    bias[:valid_rows, 0] = epsilon
    bias[valid_rows:, 0] = sum(approximation.domain) / 2
    variance_anchors = arithmetic.weighted_sum([prefix768(arithmetic, squares)],
                                               [anchors.ravel()], bias=bias.ravel())
    variance = broadcast4096(arithmetic, variance_anchors)
    inverse = arithmetic.evaluate(variance, approximation)
    normalized = arithmetic.multiply(centered, inverse)
    gamma_slots = np.zeros((ROWS, FEATURES))
    beta_slots = np.zeros((ROWS, FEATURES))
    gamma_slots[:valid_rows, :WIDTH] = gamma
    beta_slots[:valid_rows, :WIDTH] = beta
    result = arithmetic.weighted_sum([normalized], [gamma_slots.ravel()], bias=beta_slots.ravel())
    return {'mean': mean, 'centered': centered, 'variance': variance,
            'inverse': inverse, 'normalized': normalized, 'result': result}


def read_tensor(path, name):
    with path.open('rb') as stream:
        count = struct.unpack('<Q', stream.read(8))[0]
        entry = json.loads(stream.read(count))[name]
        assert entry['dtype'] == 'F32'
        lo, hi = entry['data_offsets']
        stream.seek(8 + count + lo)
        data = stream.read(hi - lo)
    return (np.frombuffer(data, '<f4').reshape(entry['shape']).astype(np.float64),
            hashlib.sha256(data).hexdigest())


def main():
    started = time.perf_counter()
    paths = [Path(__file__), ROOT / 'cadathlon/common/encrypted_polynomial.py',
             ROOT / 'cadathlon/validation/compose_pretrained_bert_arithmetic.json',
             ROOT / 'cadathlon/validation/plan_pretrained_bert_operators.json']
    paths += sorted((ROOT / 'jaxite_word').glob('*.py'))
    raw = {('plan.py' if path == Path(__file__) else str(path.relative_to(ROOT))): path.read_bytes()
           for path in paths}
    before = {name: hashlib.sha256(data).hexdigest() for name, data in raw.items()}
    with tarfile.open(HERE / 'sources.tar.gz', 'w:gz') as archive:
        for name, data in raw.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    manifest = json.loads((ROOT / 'cadathlon/validation/compose_pretrained_bert_arithmetic.json').read_text())
    prior = json.loads((ROOT / 'cadathlon/validation/plan_pretrained_bert_operators.json').read_text())
    asset = Path('/home/jianming_gatech/cross_validation_assets/bert/f150c1d609d1e50dd5e2e5408661cfac8339277c')
    checkpoint = asset / 'model.safetensors'
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert checkpoint_sha == prior['asset_sha256']['model.safetensors']
    config = json.loads((asset / 'config.json').read_text())
    epsilon = float(config['layer_norm_eps'])
    cases = []
    tensor_hashes = {}
    for phase, prefix in [('attention', 'attention.output'), ('mlp', 'output')]:
        rule = manifest['operator_manifest']['norms']['0.' + phase]
        assert rule['method'] == 'power' and rule['degree'] in (12, 16)
        approximation = nonlinear_approximation('inverse_sqrt', rule['domain'], rule['degree'])
        gamma, gamma_sha = read_tensor(checkpoint, f'bert.encoder.layer.0.{prefix}.LayerNorm.weight')
        beta, beta_sha = read_tensor(checkpoint, f'bert.encoder.layer.0.{prefix}.LayerNorm.bias')
        tensor_hashes[phase] = {'gamma': gamma_sha, 'beta': beta_sha}
        degree = rule['degree']
        grid = np.linspace(*approximation.domain, 16385)
        grid_error = float(np.max(np.abs(approximation.evaluate_public(grid) - 1 / np.sqrt(grid))))
        for valid_rows in (1, 2, 3, 4):
            rng = np.random.default_rng(20261103 + degree * 100 + valid_rows)
            tile = rng.uniform(-100., 100., (ROWS, FEATURES))
            logical = rng.normal(size=(valid_rows, WIDTH))
            logical -= logical.mean(axis=1, keepdims=True)
            logical /= np.sqrt((logical * logical).mean(axis=1, keepdims=True))
            lo, hi = approximation.domain
            target_variance = np.linspace(lo + (hi-lo) / 8, hi - (hi-lo) / 8, valid_rows)
            logical *= np.sqrt(target_variance[:, None] - epsilon)
            logical += np.linspace(-9., 11., valid_rows)[:, None]
            tile[:valid_rows, :WIDTH] = logical
            original = tile.copy()
            engine = PublicArithmetic()
            intermediates = evaluate(engine, Value(tile.ravel(), 12), valid_rows=valid_rows,
                                     approximation=approximation, epsilon=epsilon, gamma=gamma, beta=beta)
            expected_mean = logical.mean(axis=1, keepdims=True)
            expected_centered = logical - expected_mean
            expected_variance = (expected_centered ** 2).mean(axis=1, keepdims=True) + epsilon
            expected_inverse = approximation.evaluate_public(expected_variance)
            expected = np.zeros_like(tile)
            expected[:valid_rows, :WIDTH] = expected_centered * expected_inverse * gamma + beta
            actual = intermediates['result'].data.reshape(ROWS, FEATURES)
            actual_variance = intermediates['variance'].data.reshape(ROWS, FEATURES)
            inverse_input_ok = bool(np.all(actual_variance >= lo) and np.all(actual_variance <= hi))
            error = float(np.max(np.abs(actual - expected)))
            assert error < 1e-9 and inverse_input_ok
            assert np.array_equal(tile, original)
            assert not actual[:valid_rows, WIDTH:].any() and not actual[valid_rows:].any()
            assert intermediates['result'].level == 1 and engine.counts['rotations'] == 42
            cases.append(dict(phase=phase, valid_rows=valid_rows, all_slots_checked=SLOTS,
                max_abs_error_against_independent_numpy=error, all_inverse_inputs_in_fixed_domain=inverse_input_ok,
                variance_min=float(actual_variance.min()), variance_max=float(actual_variance.max()),
                input_unchanged=True, exact_zero_ideal_output_padding=True,
                levels={key: value.level for key, value in intermediates.items()},
                counts=dict(engine.counts), rotation_indices=sorted(engine.rotations),
                rule=rule, epsilon=epsilon, grid_inverse_abs_error=grid_error,
                coefficients=list(approximation.coefficients),
                monomial_coefficient_l1=float(np.abs(np.polynomial.chebyshev.cheb2poly(approximation.coefficients)).sum())))
    q = composite_prime_gen(2, 26, 61, 56, 65536, 31)
    p = util.generate_p_towers(q, 8, 32768, aux_bits=30)
    target = he_params.SecurityTarget(128, 'classical')
    he_params.validate_kernel_moduli(q, p, 8)
    he_params.assert_p_coverage(q, p, 8)
    assert he_params.qp_is_secure(q, p, 32768, target)
    alpha = (len(q) + 7) // 8
    partitions = (len(q) + alpha - 1) // alpha
    key_bytes = 2 * partitions * 32768 * (len(q) + len(p)) * 4
    rotations = sorted(set(MEAN_SHIFTS) | {shift % SLOTS for shift in BROADCAST_SHIFTS})
    assert all(case['rotation_indices'] == rotations for case in cases)
    after = {('plan.py' if path == Path(__file__) else str(path.relative_to(ROOT))):
             hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    assert before == after
    report = dict(scope=__doc__, passed=True, encrypted_execution=False,
        source_hashes_before=before, source_hashes_after=after, source_files_unchanged=True,
        model_revision=asset.name, model_checkpoint_sha256=checkpoint_sha,
        config_sha256=hashlib.sha256((asset/'config.json').read_bytes()).hexdigest(),
        layernorm_tensor_sha256=tensor_hashes,
        layout=dict(features=FEATURES, width=WIDTH, slots=SLOTS, rows=ROWS),
        physical_input_padding='Hostile random values in unused features and inactive rows; never included in logical moments.',
        inverse_padding='Valid-row variances broadcast to all4096features; inactive rows receive fixed public domain midpoint.',
        plaintext_fixed_absolute_gate=1e-9, future_he_fixed_absolute_gate=1e-6,
        cases=cases, ring=dict(degree=32768, depth=12, composite_degree=2, dnum=8,
            q=q, p=p, actual_log2_qp=sum(map(math.log2, q+p)), integer_qp_bits=math.prod(q+p).bit_length(),
            security_ceiling=he_params.security_ceiling(32768,target), secure=True,
            p_covers_q=True, actual_partitions=partitions, scale=float(q[-2])*q[-1]),
        raw_storage=dict(rotation_key_count=len(rotations), rotation_pair_bytes=key_bytes,
            rotation_keys_bytes=len(rotations)*key_bytes, relin_key_bytes=key_bytes,
            input_ciphertext_bytes=2*32768*len(q)*4),
        rotation_indices=rotations, cpu_affinity=sorted(os.sched_getaffinity(0)),
        elapsed_s=time.perf_counter()-started,
        limitations=['Public arithmetic only; no encrypted LayerNorm, native input fixture, model-quality or timing result.',
            'Fixed domains come from earlier public planning; sampled activation domains are not universal hidden-input bounds.',
            'No refresh is planned; chain scales/noise, controls and peak memory still need actual validation.',
            'Intermediate centered/normalized padding may be nonzero; only final learned affine masks all output padding.'])
    (HERE/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('passed','layout','ring','raw_storage','elapsed_s')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
