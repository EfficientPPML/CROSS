#!/usr/bin/env python3
"""Reproduce conditional ideal-arithmetic BERT bounds and counterexamples.

No encryption, key generation, refresh, CKKS error, or fixture execution occurs.
Public weights and polynomial contracts supply the bounds. Counterexample
vectors satisfy normalization-energy contracts; they are not claimed to come
from the model's embedding table or to fail a public fixture.
"""
from fractions import Fraction as F
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
if os.environ['JAX_PLATFORMS'] != 'cpu':
    raise RuntimeError('public bound review requires JAX_PLATFORMS=cpu')
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'cadathlon/common')]
import numpy as np
from encrypted_transformer import public_toy_approximations
from support import load_module


def upper_float(value):
    result = float(value)
    while F(result) < value:
        result = math.nextafter(result, math.inf)
    return result


def lower_float(value):
    result = float(value)
    while F(result) > value:
        result = math.nextafter(result, -math.inf)
    return result


def upper_sqrt(value):
    result = math.sqrt(float(value))
    while F(result)**2 < value:
        result = math.nextafter(result, math.inf)
    return result


def squared_norm(values):
    return sum(value*value for value in values)


def matrix_bound(matrix):
    """Outward operator bound from exact binary-rational Gram/Frobenius sums."""
    gram = [[sum(x*y for x, y in zip(left, right)) for right in matrix] for left in matrix]
    gershgorin = max(sum(abs(x) for x in row) for row in gram)
    frobenius = sum(squared_norm(row) for row in matrix)
    return upper_sqrt(min(gershgorin, frobenius))


def normalized_inverse_envelope(approximation):
    """Reviewed grid-plus-global-derivative formula, with explicit FP slack.

    The grid alone is not an upper bound. For f(x)=sqrt(x)*p(x), use
    |T_n|<=1 and |T_n'|<=n² to bound |f'| over the positive interval.
    The analytic argument is exact; the additional1e-11 floating margin is
    an engineering allowance, not a formal interval-arithmetic certificate.
    """
    lo, hi = approximation.domain
    if lo <= 0:
        raise ValueError('positive variance domain required')
    coefficients = np.asarray(approximation.coefficients)
    magnitude = float(np.abs(coefficients).sum())
    derivative = float(np.sum(np.arange(len(coefficients))**2*np.abs(coefficients)))
    lipschitz = magnitude/(2*math.sqrt(lo)) + math.sqrt(hi)*2*derivative/(hi-lo)
    points = 65537
    values = np.linspace(lo, hi, points)
    normalized = (2*values-lo-hi)/(hi-lo)
    sampled = float(np.max(np.abs(np.sqrt(values)*np.polynomial.chebyshev.chebval(normalized, coefficients))))
    margin = lipschitz*(hi-lo)/(2*(points-1)) + 1e-11*max(1., magnitude*math.sqrt(hi))
    return sampled+margin


def affine_head(weight, bias, gamma, beta, energy):
    """For x=diag(gamma)u+beta, bound W*x+b with ||u||₂<=energy."""
    rows = [[F(float(x)) for x in row] for row in weight]
    gamma, beta = [F(float(x)) for x in gamma], [F(float(x)) for x in beta]
    matrix = [[x*g for x, g in zip(row, gamma)] for row in rows]
    offset = [sum(x*b for x, b in zip(row, beta))+F(float(bias_value))
              for row, bias_value in zip(rows, bias)]
    operator = matrix_bound(matrix)
    coordinate = upper_float(F(energy)*F(max(upper_sqrt(squared_norm(row)) for row in matrix))
                             + max(map(abs, offset)))
    l2 = upper_float(F(energy)*F(operator)+F(upper_sqrt(squared_norm(offset))))
    return dict(operator_upper=operator, coordinate_upper=coordinate, l2_upper=l2,
                numpy_spectral_diagnostic=float(np.linalg.norm(np.asarray(matrix, dtype=float), ord=2)))


def score_layers(case, weights, ratio):
    config = case['config']
    hidden, heads = config['hidden_size'], config['num_attention_heads']
    dimension, layers = hidden//heads, []
    for layer in range(config['num_hidden_layers']):
        norm = ('bert.embeddings.LayerNorm' if layer == 0
                else f'bert.encoder.layer.{layer-1}.output.LayerNorm')
        gamma, beta = weights[norm+'.weight'], weights[norm+'.bias']
        energy = upper_sqrt(F(hidden)*F(1. if layer == 0 else ratio)**2)
        prefix, rows = f'bert.encoder.layer.{layer}.attention.self.', []
        for index in range(heads):
            selection = slice(index*dimension, (index+1)*dimension)
            query, key = [affine_head(weights[prefix+name+'.weight'][selection],
                weights[prefix+name+'.bias'][selection], gamma, beta, energy) for name in ('query', 'key')]
            product = F(query['l2_upper'])*F(key['l2_upper'])
            divisor = math.sqrt(dimension)
            while F(divisor)**2 > dimension:
                divisor = math.nextafter(divisor, -math.inf)
            rows.append(dict(head=index, query=query, key=key,
                head_product_l1_upper=upper_float(product),
                aligned_score_upper=upper_float(product/F(divisor)),
                raw_sliding_product_upper=upper_float(2*product)))
        score = max(row['aligned_score_upper'] for row in rows)
        layers.append(dict(layer=layer, normalization=norm, normalized_energy_upper=energy,
            heads=rows, score_upper=score, valid_key_mean_upper=score,
            centered_score_upper=upper_float(2*F(score))))
    return dict(hidden_size=hidden, head_dim=dimension, layers=layers,
                maximum_score_upper=max(row['score_upper'] for row in layers))


def norm_only_witness(case, weights):
    """Find a witness numerically, then verify its energy and score exactly."""
    config = case['config']
    hidden, heads = config['hidden_size'], config['num_attention_heads']
    dimension = hidden//heads
    gamma, beta = [weights['bert.embeddings.LayerNorm.'+name] for name in ('weight', 'bias')]
    center, best = np.eye(hidden)-np.ones((hidden, hidden))/hidden, None
    for index in range(heads):
        prefix, selection = 'bert.encoder.layer.0.attention.self.', slice(index*dimension, (index+1)*dimension)
        wq, wk = [weights[prefix+name+'.weight'][selection] for name in ('query', 'key')]
        bq, bk = [weights[prefix+name+'.bias'][selection] for name in ('query', 'key')]
        left_matrix, right_matrix = (wq*gamma)@center, (wk*gamma)@center
        left_bias, right_bias = wq@beta+bq, wk@beta+bk
        singular_vectors, _, _ = np.linalg.svd(left_matrix.T@right_matrix)
        for sign in (-1, 1):
            query = sign*singular_vectors[:, 0]*math.sqrt(hidden)*.999
            for _ in range(30):
                key = right_matrix.T@(left_matrix@query+left_bias)
                key *= math.sqrt(hidden)*.999/np.linalg.norm(key)
                query = left_matrix.T@(right_matrix@key+right_bias)
                query *= math.sqrt(hidden)*.999/np.linalg.norm(query)
            score = float((left_matrix@query+left_bias)@(right_matrix@key+right_bias)/math.sqrt(dimension))
            if best is None or score > best[0]:
                best = score, index, center@query, center@key, wq, wk, bq, bk
    _, index, query, key, wq, wk, bq, bk = best

    def exact_center(values):
        values = [F(float(x)) for x in values]
        mean = sum(values)/len(values)
        return [x-mean for x in values]

    query, key = exact_center(query), exact_center(key)
    assert sum(query) == sum(key) == 0
    assert squared_norm(query) < hidden and squared_norm(key) < hidden

    def project(matrix, bias, value):
        affine = [F(float(g))*v+F(float(b)) for g, v, b in zip(gamma, value, beta)]
        return [sum(F(float(w))*x for w, x in zip(row, affine))+F(float(b))
                for row, b in zip(matrix, bias)]

    dot = sum(x*y for x, y in zip(project(wq, bq, query), project(wk, bk, key)))
    assert dot > 0 and dot*dot > 16*dimension  # Exact proof of score > 4.
    score_lower = lower_float(dot/F(upper_sqrt(F(dimension))))
    return dict(head=index, exact_energy_and_zero_mean_checks=True,
        score_lower_bound=score_lower,
        masked_centered_slot_upper_bound_if_valid_keys_repeat_this_key=-score_lower,
        query_rational=[[x.numerator, x.denominator] for x in query],
        key_rational=[[x.numerator, x.denominator] for x in key],
        claim='Norm-contract witness only; not a public fixture, embedding-table input, or encrypted failure.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_suffix('.json'))
    args = parser.parse_args()
    folder = ROOT/'cadathlon/agent_pset/p2_bert'
    sources = [Path(__file__), folder/'src/reference.py', *sorted((folder/'test').glob('test*.in'))]
    sources += [ROOT/'cadathlon/common'/name for name in ('encrypted_transformer.py', 'encrypted_polynomial.py', 'support.py')]

    def snapshot():
        return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}

    before = snapshot()
    oracle = load_module(folder/'src/reference.py')
    approximation = public_toy_approximations('bert')['inverse_sqrt']
    ratio = normalized_inverse_envelope(approximation)
    report = dict(scope=__doc__, encrypted_execution=False, fixture_execution=False,
        norm_ratio_envelope=ratio, source_sha256=before, cases={}, norm_only_softmax_counterexamples={},
        approximation=dict(domain=approximation.domain, coefficients=approximation.coefficients),
        derivation=dict(input='x=diag(gamma)u+beta, ||u||2<=sqrt(H)*R',
            projection='A=W_head diag(gamma), c=W_head beta+b_head; E_head<=E*upper(||A||2)+||c||2',
            operator='upper(||A||2)=sqrt(min(max_i sum_j |(A A^T)_ij|, ||A||F^2))',
            score='sum_i |q_i k_i|<=E_query*E_key; divide by sqrt(actual head_dim)',
            reductions='Raw windows intersect at most two aligned groups; use group absolute-sum bounds. After masked broadcast restore the aligned bound. Valid-count score means have the same bound as scores.'),
        limitations=['Conditional ideal arithmetic only; CKKS and refresh errors are excluded.',
            'Exact rational Gram and affine products plus outward rounding certify operator arithmetic for the supplied energy bound.',
            'The separate normalized-inverse envelope uses an analytic derivative margin and engineering floating slack; no full interval-arithmetic certificate is claimed.',
            'NumPy SVD is used for diagnostics and witness search only, never as an upper bound.',
            'Boolean masks and correctly matched positive valid-key counts are required.',
            'Public-fixture domain observations and full encrypted numerical gates remain separate evidence.'],
        versions=dict(python=sys.version.split()[0], numpy=np.__version__))
    for number in range(1, 7):
        case = json.loads((folder/f'test/test{number}.in').read_text())
        weights = oracle.toy_weights(case['config'], case['weight_seed'])
        report['cases'][f'bert/test{number}.in'] = score_layers(case, weights, ratio)
        if number >= 4:
            report['norm_only_softmax_counterexamples'][f'bert/test{number}.in public weights'] = norm_only_witness(case, weights)

    assert F(math.sqrt(3))**2 < 3 <= F(upper_sqrt(F(3)))**2
    assert matrix_bound([[F(1)]*16 for _ in range(4)]) == 8.
    signed = np.array([100., -100., -100., 100.])
    sliding = signed+np.roll(signed, -1)
    assert np.array_equal(sliding[::2], [0., 0.]) and max(abs(sliding)) == 200.
    report['counterexamples'] = dict(ordinary_sqrt3_underestimates=True,
        ones4x16_maximum_row_l2=4., ones4x16_operator_norm=8.,
        signed_groups=signed.tolist(), signed_aligned_sums=sliding[::2].tolist(),
        signed_sliding_sums=sliding.tolist(),
        padding_note='BERT H24/head4 has actual dimension6 padded8; divide scores by sqrt6.')
    report['source_unchanged_at_recording'] = before == snapshot()
    if not report['source_unchanged_at_recording']:
        raise RuntimeError('source changed during review; rerun against stable files')
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({case: row['maximum_score_upper'] for case, row in report['cases'].items()}, indent=2))


if __name__ == '__main__':
    main()
