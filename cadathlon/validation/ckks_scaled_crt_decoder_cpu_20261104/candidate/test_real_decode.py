"""Clear CPU arithmetic tests: genuine RNS/CRT/FFT, no ciphertext or key use."""
import ast
import hashlib
import importlib
import importlib.abc
import json
import math
import os
from pathlib import Path
import sys
from fractions import Fraction
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent
for key, value in {'JAX_PLATFORMS': 'cpu', 'JAX_PLATFORM_NAME': 'cpu',
                   'JAX_ENABLE_X64': 'True', 'PYTHONDONTWRITEBYTECODE': '1'}.items():
    assert os.environ.get(key) == value, (key, 'must be set before importing JAX')
assert not any(name == 'jax' or name.startswith('jax.') for name in sys.modules)
IMPORT_GUARD_DENIALS = []
DEVICE_ACCESS_DENIALS = []
class CPUOnlyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('libtpu', 'torch', 'transformers', 'tensorflow', 'flax'):
            IMPORT_GUARD_DENIALS.append(fullname)
            raise ModuleNotFoundError('CPU clear-arithmetic test forbids ' + fullname)
CPU_IMPORT_GUARD = CPUOnlyImports()
sys.meta_path.insert(0, CPU_IMPORT_GUARD)
def device_audit(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes)):
        path = os.fsdecode(args[0])
        if path.startswith(('/dev/accel', '/dev/vfio', '/dev/tpu')):
            DEVICE_ACCESS_DENIALS.append([event, path])
            raise PermissionError('accelerator device access forbidden')
    if event == 'ctypes.dlopen' and args and 'libtpu' in str(args[0]):
        DEVICE_ACCESS_DENIALS.append([event, str(args[0])])
        raise PermissionError('TPU library loading forbidden')
sys.addaudithook(device_audit)
sys.path.insert(0, str(ROOT / 'checkout'))
sys.path.insert(0, str(ROOT / 'checkout/jaxite_word'))
import numpy as np
import jax
import ckks_ctx as ckks

GEOMETRY = json.loads((ROOT / 'public_geometry.json').read_bytes())
Q = GEOMETRY['q_towers']
BIG = math.prod(Q)
SCALE = GEOMETRY['scale']
RECORDS = []


def old_decoder():
    """Only the old decode definition; real current dependencies are unchanged."""
    module = ast.parse((ROOT / 'before/ckks_ctx.py').read_bytes())
    node = next(n for n in module.body if isinstance(n, ast.FunctionDef)
                and n.name == '_fast_decode')
    namespace = dict(vars(ckks))
    exec(compile(ast.Module(body=[node], type_ignores=[]),
                 str(ROOT / 'before/ckks_ctx.py'), 'exec'), namespace)
    return namespace['_fast_decode']


OLD = old_decoder()


def residues(degree, entries):
    """Direct integer residues, independent of production CRT reconstruction."""
    out = np.zeros((degree, len(Q)), dtype=np.uint64)
    for index, value in entries.items():
        assert type(value) is int and abs(value) <= BIG // 2
        out[index, :] = [value % q for q in Q]
    return out


def decode(matrix, *, old=False, slots=None):
    ctx = SimpleNamespace(q_towers=Q, num_slots=matrix.shape[0] // 2)
    return (OLD if old else ckks._fast_decode)(matrix, ctx, SCALE,
                                             slots_to_decode=slots)


def record(name, values, **extra):
    assert np.isfinite(values).all()
    RECORDS.append({'name': name, 'slots': int(values.size),
                    'all_finite': True, 'minimum_hex': float(values.min()).hex(),
                    'maximum_hex': float(values.max()).hex(), **extra})


@pytest.fixture(autouse=True)
def no_crypto(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('context/key/encryption/decryption is outside clear arithmetic scope')
    for name in ('__init__', 'encrypt', 'encrypt_slots', 'encrypt_slots_batch',
                 'decrypt', 'decrypt_slots', 'decrypt_slots_batch'):
        monkeypatch.setattr(ckks.CKKSContext, name, forbidden)
    for name in ('gen_pke_pair', 'gen_evaluation_key', 'gen_rotation_key'):
        monkeypatch.setattr(ckks.kg, name, forbidden)
    monkeypatch.setattr(ckks, '_fast_decrypt_to_rns_coeffs_array', forbidden)


def test_cpu_and_source_origin():
    assert jax.default_backend() == 'cpu'
    assert all(device.platform == 'cpu' for device in jax.devices())
    assert jax.config.jax_enable_x64 is True
    from jax._src.lib import _jax
    assert not _jax.pjrt_plugin_loaded('tpu')
    assert not DEVICE_ACCESS_DENIALS
    assert Path(ckks.__file__).resolve() == ROOT / 'checkout/jaxite_word/ckks_ctx.py'
    assert not any(name == 'libtpu' or name.startswith('libtpu.') for name in sys.modules)
    assert not any(name == 'transformers' or name.startswith('transformers.') for name in sys.modules)
    assert len(Q) == 40 and BIG.bit_length() == 1096
    assert SCALE.hex() == '0x1.ea3b603ea0001p+55'
    assert GEOMETRY['degree'] == 131072 and GEOMETRY['num_slots'] == 65536
    assert ckks._DECODE_STDDEV_LIMIT == 1 / 32


def test_exact_source_delta():
    manifest = json.loads((ROOT / 'source_origin.json').read_bytes())['source_hashes']
    assert len(manifest) == 199
    changed = []
    for name, digest in manifest.items():
        if hashlib.sha256((ROOT / 'checkout' / name).read_bytes()).hexdigest() != digest:
            changed.append(name)
    assert changed == ['jaxite_word/ckks_ctx.py']
    assert hashlib.sha256((ROOT / 'before/ckks_ctx.py').read_bytes()).hexdigest() == manifest[changed[0]]
    original = ast.parse((ROOT / 'before/ckks_ctx.py').read_bytes())
    successor = ast.parse((ROOT / 'checkout/jaxite_word/ckks_ctx.py').read_bytes())
    helpers = [n for n in successor.body if isinstance(n, ast.FunctionDef)
               and n.name == '_scale_centered_crt_coefficient']
    assert len(helpers) == 1
    successor.body.remove(helpers[0])
    class Restore(ast.NodeTransformer):
        count = 0
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == '_scale_centered_crt_coefficient':
                assert not node.keywords and len(node.args) == 3
                assert isinstance(node.args[0], ast.Name) and node.args[0].id in ('r', 'im')
                assert [n.id for n in node.args[1:]] == ['scale', 'scale_inv']
                self.count += 1
                return ast.BinOp(left=ast.Call(func=ast.Name(id='float', ctx=ast.Load()),
                    args=[node.args[0]], keywords=[]), op=ast.Mult(), right=node.args[2])
            return self.generic_visit(node)
    restore = Restore()
    successor = restore.visit(successor)
    assert restore.count == 2
    assert ast.dump(original, include_attributes=False) == ast.dump(successor, include_attributes=False)


@pytest.mark.parametrize('value', [0, 1, -1, 2**53 + 1, -(2**53 + 1), 2**1023, -(2**1023)])
def test_normal_helper_path_is_legacy_bitwise(value):
    got = ckks._scale_centered_crt_coefficient(value, SCALE, 1.0 / SCALE)
    expected = float(value) * (1.0 / SCALE)
    assert got.hex() == expected.hex()


@pytest.mark.parametrize('value', [2**1024, -(2**1024)])
def test_real_crt_reconstructs_overflowing_centered_integer(value):
    matrix = residues(16, {0: value, 3: -value})
    actual = ckks._crt_combine_rns(matrix, Q)
    centered = [v - BIG if v > BIG // 2 else v for v in actual]
    assert centered == [value, 0, 0, -value] + [0] * 12
    with pytest.raises(OverflowError):
        float(value)


@pytest.mark.parametrize('degree', [16, 131072])
@pytest.mark.parametrize('sign', [-1, 1])
def test_real_crt_fft_symmetric_large_constant(degree, sign):
    value = sign * 2**1024
    matrix = residues(degree, {0: value})
    with pytest.raises(OverflowError, match='int too large'):
        decode(matrix, old=True)
    expected = float(Fraction(value) / Fraction(SCALE))
    got = decode(matrix)
    assert got.shape == (degree // 2,)
    assert np.isfinite(got).all()
    assert np.array_equal(got, np.full(degree // 2, expected))
    record('large_symmetric_constant', got, degree=degree, sign=sign,
           independent_fraction_expected_hex=expected.hex(), old_conversion_overflow=True)


@pytest.mark.parametrize('degree', [16, 131072])
def test_real_crt_fft_zero_and_output_trimming(degree):
    matrix = residues(degree, {})
    full = decode(matrix)
    assert np.array_equal(full, np.zeros(degree // 2))
    trim = decode(matrix, slots=3)
    assert np.array_equal(trim, full[:3])
    record('zero_full_and_trim', full, degree=degree)


@pytest.mark.parametrize('value', [2**1023, -(2**1023), 1, -1])
def test_real_crt_fft_ordinary_constant_matches_old(value):
    matrix = residues(32, {0: value})
    got, expected = decode(matrix), decode(matrix, old=True)
    assert np.array_equal(got.view(np.uint64), expected.view(np.uint64))
    assert np.isfinite(got).all()
    record('legacy_constant', got, coefficient_bits=abs(value).bit_length(), sign=1 if value > 0 else -1)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_real_crt_fft_mixed_representable_matches_old(seed):
    rng = np.random.default_rng(seed)
    values = [int(x) for x in rng.integers(-2**20, 2**20, size=32)]
    matrix = residues(32, dict(enumerate(values)))
    got, expected = decode(matrix), decode(matrix, old=True)
    assert np.array_equal(got.view(np.uint64), expected.view(np.uint64))
    assert np.isfinite(got).all()
    record('legacy_mixed', got, public_seed=seed)


@pytest.mark.parametrize('degree', [16, 131072])
@pytest.mark.parametrize('coordinate', ['real1', 'imag0'])
def test_real_crt_asymmetry_still_rejected_before_fft(degree, coordinate):
    index = 1 if coordinate == 'real1' else degree // 2
    matrix = residues(degree, {index: 2**1024})
    with np.errstate(over='ignore', invalid='ignore'):
        with pytest.raises(ckks.ApproximationErrorTooHigh, match='stddev=inf'):
            decode(matrix)
    RECORDS.append({'name': 'asymmetry_rejected', 'degree': degree,
                    'coordinate': coordinate, 'unchanged_limit': 1 / 32})


@pytest.mark.parametrize('sign', [-1, 1])
def test_true_scaled_overflow_is_not_rescued(sign):
    matrix = residues(16, {0: sign * (BIG // 2)})
    with pytest.raises(OverflowError, match='unit-scale coefficient is not representable'):
        decode(matrix)
    with pytest.raises(OverflowError):
        float(Fraction(sign * (BIG // 2)) / Fraction(SCALE))


@pytest.mark.parametrize('stddev, accepted', [(0.0, True), (1/32, True),
    (math.nextafter(1/32, math.inf), False), (math.inf, False), (math.nan, False)])
def test_original_gate_unchanged(stddev, accepted):
    if accepted:
        ckks._check_decode_stddev(stddev)
    else:
        with pytest.raises(ckks.ApproximationErrorTooHigh):
            ckks._check_decode_stddev(stddev)
