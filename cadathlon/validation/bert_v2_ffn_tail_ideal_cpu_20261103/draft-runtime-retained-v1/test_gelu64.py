"""Frozen polynomial and real binary64 metadata equations; no encrypted run."""
from pathlib import Path
import math
import numpy as np
import pytest

from test_routing import backend_free, SLOTS
from test_attention_tile import ActualScaleArithmetic, value, Q, SCALE
from encrypted_bert_v2_gelu import BertV2Gelu, Gelu64Contract, gelu64_plan

MANIFEST = (Path(__file__).parent/'candidate_manifest.json').read_bytes()


class Arithmetic(ActualScaleArithmetic):
    def __init__(self):
        super().__init__()
        self.ctx._param_cache.q_moduli_at_level = lambda level: Q[:2*(level+1)]
    def weighted_sum(self, values, weights, *, bias=0.):
        self.calls.append(('weighted_sum',))
        level = min(v.level for v in values)
        divisor = math.prod(Q[2*level:2*level+2])
        terms = [float(v._ckks_scale)*float((self.scale*divisor)/v._ckks_scale) for v in values]
        assert all(math.isclose(s, terms[0], rel_tol=1e-12) for s in terms)
        return value(sum(v.data*w for v, w in zip(values, weights))+bias,
                     level-1, terms[0]/divisor)


@pytest.mark.parametrize('level', [16, 24, 49])
@pytest.mark.parametrize('input_scale', [math.nextafter(SCALE, -math.inf), SCALE, math.nextafter(SCALE, math.inf), SCALE*.93])
def test_every_physical_slot_and_actual_scale(level, input_scale):
    contract = Gelu64Contract(MANIFEST)
    data = np.random.default_rng(2026110306).uniform(-21, 14, SLOTS)
    data[::13] = 0.
    data[:5] = [-21., 14., 0., -1., 1.]
    source = value(data, level, input_scale)
    ops = Arithmetic(); owner = BertV2Gelu(ops, contract)
    output = owner.evaluate_tile(source)
    expected = np.polynomial.chebyshev.chebval((2/35)*data+(7/35), contract.coefficients)
    np.testing.assert_allclose(output.data, expected, rtol=0, atol=2e-13)
    assert output.level == level-14
    assert output._ckks_scale == owner.last_plan['output_scale']
    assert math.isclose(output._ckks_scale, SCALE, rel_tol=1e-12)
    assert sum(c[0] == 'multiply' for c in ops.calls) == 63
    assert sum(c[0] == 'weighted_sum' for c in ops.calls) == 65
    assert not any(c[0] == 'rotate' for c in ops.calls)
    np.testing.assert_array_equal(source.data, data)
    assert source._public_bound == 'deliberately stale' and not hasattr(output, '_public_bound')
    assert not owner._active


@pytest.mark.parametrize('event', ['domain map', 'T64', 'output'])
@pytest.mark.parametrize('mutation', ['original', 'result', 'contract', 'context', 'owner'])
def test_callback_mutations_fail_before_next_primitive(event, mutation):
    source = value(np.zeros(SLOTS), 24, SCALE)
    ops = Arithmetic(); contract = Gelu64Contract(MANIFEST); calls = []
    def progress(label, result, counts):
        if label != event: return
        calls.append(len(ops.calls))
        if mutation == 'original': source.polynomial = source.polynomial.copy()
        elif mutation == 'result': result._ckks_nsd = True
        elif mutation == 'contract': object.__setattr__(contract, 'degree', 64.)
        elif mutation == 'context': ops.ctx.secret_key = object()
        else: owner.arithmetic = Arithmetic()
    owner = BertV2Gelu(ops, contract, progress=progress)
    with pytest.raises(ValueError): owner.evaluate_tile(source)
    assert calls == [len(ops.calls)] and not owner._active and owner.last_plan is None


@pytest.mark.parametrize('mutation', ['scale', 'payload'])
def test_retained_earlier_chebyshev_branch_is_guarded(mutation):
    ops = Arithmetic(); saved = []; calls = []
    def progress(label, result, counts):
        if label == 'T2': saved.append(result)
        if label == 'T63':
            calls.append(len(ops.calls))
            if mutation == 'scale': saved[0]._ckks_scale *= 2
            else: saved[0].polynomial = saved[0].polynomial.copy()
    owner = BertV2Gelu(ops, Gelu64Contract(MANIFEST), progress=progress)
    with pytest.raises(ValueError): owner.evaluate_tile(value(np.zeros(SLOTS), 24, SCALE))
    assert calls == [len(ops.calls)] and owner.last_plan is None and not owner._active


def test_primary_cancellation_and_reentry_release_state():
    ops = Arithmetic(); source = value(np.zeros(SLOTS), 24, SCALE)
    error = KeyboardInterrupt('same exception')
    def progress(*args):
        with pytest.raises(ValueError, match='reentered'): owner.evaluate_tile(source)
        raise error
    owner = BertV2Gelu(ops, Gelu64Contract(MANIFEST), progress=progress)
    with pytest.raises(KeyboardInterrupt) as caught: owner.evaluate_tile(source)
    assert caught.value is error and len(ops.calls) == 1 and not owner._active


@pytest.mark.parametrize('mutation', ['manifest', 'coefficients', 'degree', 'domain', 'low_level', 'nsd', 'implicit_refresh'])
def test_invalid_contract_or_input_rejected_before_work(mutation):
    ops = Arithmetic(); source = value(np.zeros(SLOTS), 24, SCALE)
    with pytest.raises((ValueError, TypeError)):
        contract = Gelu64Contract(MANIFEST+b' ' if mutation == 'manifest' else MANIFEST)
        if mutation == 'coefficients': object.__setattr__(contract, 'coefficients', tuple([0.]*65))
        if mutation == 'degree': object.__setattr__(contract, 'degree', 64.)
        if mutation == 'domain': object.__setattr__(contract, 'domain', [-21., 14.])
        if mutation == 'low_level': source.level = 15
        if mutation == 'nsd': source._ckks_nsd = True
        if mutation == 'implicit_refresh': ops.refresher = object()
        BertV2Gelu(ops, contract).evaluate_tile(source)
    assert not ops.calls


@pytest.mark.parametrize('level', range(16, 50))
def test_public_depth_inventory_all_usable_common_levels(level):
    plan = gelu64_plan(Q[:2*(level+1)], input_level=level,
                       canonical_scale=SCALE, input_scale=SCALE)
    assert plan['output_level'] == level-14
    assert len([x for x in plan['stages'] if x['kind'] == 'ciphertext_product']) == 63
    assert len([x for x in plan['stages'] if x['kind'] == 'weighted']) == 65
