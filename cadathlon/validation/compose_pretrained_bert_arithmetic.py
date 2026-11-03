#!/usr/bin/env python3
"""Complete approximate plaintext BERT planning on three fixed public pairs.

The operator manifest is chosen from the prior analytic-function assessment,
before logits are compared. No clipping, private-value stopping rule, labels,
key generation, ciphertext execution or MRPC accuracy evaluation is used.
Domain observations are validation instrumentation, not proposed server code.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'cadathlon/common')]
import numpy as np
from encrypted_polynomial import nonlinear_approximation
from support import load_module


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@dataclass
class Value:
    data: np.ndarray
    depth: int = 0


class Arithmetic:
    """Float64 values plus an explicit, abstract level/operation accounting."""
    def __init__(self):
        self.counts = Counter()
        self.domains = {}
        self.peak = dict(magnitude=0., operation=None)
        self.phase_costs = []

    def node(self, data, depth, label):
        data = np.asarray(data, dtype=np.float64)
        finite = data[np.isfinite(data)]
        magnitude = float(np.max(np.abs(finite))) if finite.size else None
        if magnitude is not None and magnitude > self.peak['magnitude']:
            self.peak = dict(magnitude=magnitude, operation=label)
        return Value(data, depth)

    def observe(self, label, data, domain):
        data = np.asarray(data)
        finite = np.isfinite(data)
        finite_values = data[finite]
        lo, hi = domain
        self.domains[label] = dict(domain=list(domain), count=int(data.size),
            minimum=float(finite_values.min()) if finite_values.size else None,
            maximum=float(finite_values.max()) if finite_values.size else None,
            below=int(np.sum(data < lo)), above=int(np.sum(data > hi)),
            nonfinite=int(np.sum(~finite)))

    def domain_failures(self):
        return {name: row for name, row in self.domains.items()
                if row['below'] or row['above'] or row['nonfinite']}

    def affine(self, value, weight=1., bias=0., label='affine'):
        self.counts['plaintext_tensor_multiplications'] += 1
        self.counts['affine_stages'] += 1
        return self.node(value.data*weight+bias, value.depth+1, label)

    def weighted_sum(self, values, weights, bias=0., label='weighted_sum'):
        self.counts['plaintext_tensor_multiplications'] += len(values)
        self.counts['affine_stages'] += 1
        result = values[0].data*weights[0]
        for value, weight in zip(values[1:], weights[1:]):
            result = result+value.data*weight
        return self.node(result+bias, max(value.depth for value in values)+1, label)

    def add(self, left, right, subtract=False, label='add'):
        self.counts['ciphertext_tensor_additions'] += 1
        return self.node(left.data-right.data if subtract else left.data+right.data,
                         max(left.depth, right.depth), label)

    def multiply(self, left, right, label='multiply'):
        data = left.data*right.data
        self.counts['ciphertext_tensor_multiplications'] += 1
        self.counts['elementwise_ciphertext_scalar_products'] += data.size
        return self.node(data, max(left.depth, right.depth)+1, label)

    def reduce(self, value, axis, factor=1., keepdims=True, bias=0., label='reduce'):
        self.counts['plaintext_tensor_multiplications'] += 1
        self.counts['masked_reduction_stages'] += 1
        return self.node(np.sum(value.data, axis=axis, keepdims=keepdims)*factor+bias,
                         value.depth+1, label)

    def linear(self, value, weight, bias, label):
        self.counts['public_matrix_maps'] += 1
        return self.node(value.data@weight.T+bias, value.depth+1, label)

    def view(self, value, data):
        return Value(data, value.depth)

    def layout_move(self, value, label):
        self.counts['masked_layout_move_stages'] += 1
        return Value(value.data, value.depth+1)

    def polynomial(self, value, contract, strategy, label, input_scale=1.):
        lo, hi = contract.domain
        self.observe(label, value.data*input_scale, contract.domain)
        # Public input scaling is folded into the existing affine mapper.
        normalized = self.affine(value, 2/(hi-lo)*input_scale, -(hi+lo)/(hi-lo), label+'.map')
        terms = {1: normalized}
        coefficients = np.asarray(contract.coefficients)
        if strategy == 'power':
            coefficients = np.polynomial.chebyshev.cheb2poly(coefficients)
            def basis(n):
                if n not in terms:
                    terms[n] = self.multiply(basis(n//2), basis((n+1)//2), label+'.power')
                return terms[n]
        else:
            def basis(n):
                if n not in terms:
                    product = self.multiply(basis(n//2), basis((n+1)//2), label+'.chebyshev_product')
                    terms[n] = (self.affine(product, 2., -1., label+'.chebyshev_even') if n % 2 == 0
                        else self.weighted_sum([product, normalized], [2., -1.], label=label+'.chebyshev_odd'))
                return terms[n]
        indices = [n for n in range(1, len(coefficients)) if strategy == 'chebyshev' or coefficients[n] != 0]
        result = self.weighted_sum([basis(n) for n in indices], [coefficients[n] for n in indices],
                                   coefficients[0], label+'.sum')
        return result

    def inverse(self, value, domain, iterations, label):
        self.observe(label, value.data, domain)
        initial = 2/sum(domain)
        # The first Newton update is a public affine map; no encrypted
        # initializer or client callback is needed.
        result = self.affine(value, -initial*initial, 2*initial, label+'.first')
        for iteration in range(1, iterations):
            product = self.multiply(value, result, label+'.product')
            residual = self.affine(product, -1., 2., label+'.residual')
            result = self.multiply(result, residual, label+'.update')
        return result

    def inverse_sqrt(self, value, rule, label):
        domain = rule['domain']
        if rule['method'] == 'power':
            contract = nonlinear_approximation('inverse_sqrt', domain, rule['degree'])
            return self.polynomial(value, contract, 'power', label)
        self.observe(label, value.data, domain)
        initial = 1/math.sqrt(domain[1])
        result = self.affine(value, -.5*initial**3, 1.5*initial, label+'.first')
        for _ in range(1, rule['iterations']):
            square = self.multiply(result, result, label+'.square')
            product = self.multiply(value, square, label+'.product')
            residual = self.affine(product, -.5, 1.5, label+'.residual')
            result = self.multiply(result, residual, label+'.update')
        return result

    def normalize(self, value, gamma, beta, epsilon, rule, label):
        mean = self.reduce(value, -1, factor=1/value.data.shape[-1], label=label+'.mean')
        centered = self.add(value, mean, subtract=True, label=label+'.center')
        square = self.multiply(centered, centered, label+'.square')
        variance = self.reduce(square, -1, factor=1/value.data.shape[-1], bias=epsilon, label=label+'.variance')
        inverse = self.inverse_sqrt(variance, rule, label+'.inverse_sqrt')
        normalized = self.multiply(centered, inverse, label+'.normalized')
        return self.affine(normalized, gamma, beta, label+'.learned_affine')

    def softmax(self, scores, rule, label):
        keys = scores.data.shape[-1]
        mean = self.reduce(scores, -1, factor=1/keys, label=label+'.mean')
        centered = self.add(scores, mean, subtract=True, label=label+'.center')
        self.observe(label+'.centered_scores', centered.data, rule['centered_domain'])
        base = self.polynomial(centered, nonlinear_approximation('exp', (-.5, .5), 8), 'power',
                               label+'.base_exp', input_scale=2.**-rule['squarings'])
        denominator = self.reduce(base, -1, factor=1/keys, label=label+'.initial_mean')
        inverse = self.inverse(denominator, (.5, 2.), 6, label+'.initial_reciprocal')
        probabilities = self.affine(self.multiply(base, inverse, label+'.initial_normalized'),
                                    1/keys, label=label+'.initial_probability')
        for iteration in range(rule['squarings']):
            prefix = label+f'.round{iteration}'
            square = self.multiply(probabilities, probabilities, prefix+'.square')
            denominator = self.reduce(square, -1, label=prefix+'.denominator')
            inverse = self.inverse(denominator, (.5/keys, 2.), 10, prefix+'.reciprocal')
            probabilities = self.multiply(square, inverse, prefix+'.normalized')
        return probabilities

    def tanh(self, value, rule, label):
        self.observe(label+'.input', value.data, rule['domain'])
        result = self.polynomial(value, nonlinear_approximation('tanh', (-1., 1.), rule['base_degree']),
            'power', label+'.base', input_scale=2.**-rule['doublings'])
        for iteration in range(rule['doublings']):
            square = self.multiply(result, result, label+'.square')
            # Constant addition could be a same-level add_plain; use the
            # conservative existing weighted-affine convention here.
            denominator = self.affine(square, 1., 1., label+'.denominator')
            inverse = self.inverse(denominator, (1., 2.1), rule['reciprocal_iterations'],
                                   label+f'.reciprocal{iteration}')
            result = self.affine(self.multiply(result, inverse, label+'.quotient'), 2., label=label+'.double')
        return result


def choose_manifest(planning):
    """Only prior function-grid accuracy and conditioning decide the methods."""
    nonlinear = planning['nonlinear_operator_candidates']
    norms, gelu = {}, {}
    for layer in range(12):
        for phase in ('attention', 'mlp'):
            row = nonlinear[f'{layer}.{phase}.inverse_sqrt']
            choice = row['sweep'][-1]
            norms[f'{layer}.{phase}'] = dict(domain=row['empirical_candidate_domain'],
                method='power' if choice['current_power_degree_and_coefficient_budget_supported'] else 'newton',
                degree=row['selected_degree'], iterations=row['fixed_inverse_sqrt_newton']['iterations'])
        row = nonlinear[f'{layer}.gelu']
        gelu[str(layer)] = dict(domain=row['empirical_candidate_domain'], degree=row['selected_degree'],
            strategy='balanced Chebyshev', rejected_monomial_coefficient_l1=row['sweep'][-1]['normalized_monomial_coefficient_l1'])
    softmax = {layer: dict(centered_domain=row['empirical_centered_input_domain'],
        squarings=row['variants']['normalized_repeated_squaring']['squaring_rounds'])
        for layer, row in planning['softmax_operator_candidates'].items()}
    alternatives = nonlinear['pooler.tanh']['fixed_rational_doubling_alternative']
    selected = next(row for row in alternatives if row['max_abs_grid_error'] <= 1e-4)
    tanh = dict(domain=nonlinear['pooler.tanh']['empirical_candidate_domain'],
        base_degree=selected['base_degree'], doublings=selected['doubling_rounds'],
        reciprocal_iterations=selected['reciprocal_iterations_per_round'])
    return dict(norms=norms, gelu=gelu, softmax=softmax, tanh=tanh,
        selection='Fixed from prior analytic-function domains/targets and power conditioning, before comparing any full-model logits.')


def forward(config, weights, embeddings, manifest):
    engine = Arithmetic()
    value = Value(np.asarray(embeddings, dtype=float))
    batch, sequence, hidden = value.data.shape
    heads, dimension = config['num_attention_heads'], hidden//config['num_attention_heads']
    epsilon = config['layer_norm_eps']
    for layer in range(12):
        start_counts, start_depth = engine.counts.copy(), value.depth
        prefix = f'bert.encoder.layer.{layer}.'
        projections = []
        for name in ('query', 'key', 'value'):
            projected = engine.linear(value, weights[prefix+'attention.self.'+name+'.weight'],
                weights[prefix+'attention.self.'+name+'.bias'], f'{layer}.{name}')
            projections.append(engine.view(projected, projected.data.reshape(batch, sequence, heads, dimension).transpose(0, 2, 1, 3)))
        query, key, projected = projections
        key = engine.layout_move(key, f'{layer}.key_layout')
        projected = engine.layout_move(projected, f'{layer}.value_layout')
        products = engine.multiply(engine.view(query, query.data[:, :, :, None, :]),
            engine.view(key, key.data[:, :, None, :, :]), f'{layer}.query_key')
        scores = engine.reduce(products, -1, factor=1/math.sqrt(dimension), keepdims=False, label=f'{layer}.scores')
        probabilities = engine.softmax(scores, manifest['softmax'][str(layer)], f'{layer}.softmax')
        products = engine.multiply(engine.view(probabilities, probabilities.data[..., None]),
            engine.view(projected, projected.data[:, :, None, :, :]), f'{layer}.probability_value')
        attended = engine.reduce(products, -2, keepdims=False, label=f'{layer}.attended')
        attended = engine.view(attended, attended.data.transpose(0, 2, 1, 3).reshape(batch, sequence, hidden))
        branch = engine.linear(attended, weights[prefix+'attention.output.dense.weight'],
            weights[prefix+'attention.output.dense.bias'], f'{layer}.attention_output')
        value = engine.add(value, branch, label=f'{layer}.attention_residual')
        name = prefix+'attention.output.LayerNorm'
        value = engine.normalize(value, weights[name+'.weight'], weights[name+'.bias'], epsilon,
            manifest['norms'][f'{layer}.attention'], f'{layer}.attention_norm')
        middle = engine.linear(value, weights[prefix+'intermediate.dense.weight'],
            weights[prefix+'intermediate.dense.bias'], f'{layer}.mlp_input')
        rule = manifest['gelu'][str(layer)]
        middle = engine.polynomial(middle, nonlinear_approximation('gelu', rule['domain'], rule['degree']),
            'chebyshev', f'{layer}.gelu')
        branch = engine.linear(middle, weights[prefix+'output.dense.weight'],
            weights[prefix+'output.dense.bias'], f'{layer}.mlp_output')
        value = engine.add(value, branch, label=f'{layer}.mlp_residual')
        name = prefix+'output.LayerNorm'
        value = engine.normalize(value, weights[name+'.weight'], weights[name+'.bias'], epsilon,
            manifest['norms'][f'{layer}.mlp'], f'{layer}.mlp_norm')
        engine.phase_costs.append(dict(layer=layer, critical_depth_increment=value.depth-start_depth,
            critical_depth_after=value.depth, operations=dict(engine.counts-start_counts)))
    cls = engine.view(value, value.data[:, 0])
    pooled = engine.linear(cls, weights['bert.pooler.dense.weight'], weights['bert.pooler.dense.bias'], 'pooler')
    pooled = engine.tanh(pooled, manifest['tanh'], 'pooler.tanh')
    output = engine.linear(pooled, weights['classifier.weight'], weights['classifier.bias'], 'classifier')
    return output, engine


def gate(actual, expected):
    finite = bool(np.isfinite(actual).all())
    shape = actual.shape == expected.shape
    numeric = bool(shape and finite and np.all(np.abs(actual-expected) <= .02+.01*np.abs(expected)))
    ties = bool(shape and finite and np.array_equal(actual[:, 1] >= actual[:, 0], expected[:, 1] >= expected[:, 0]))
    return dict(finite=finite, shape_matches=shape, logit_tolerance_pass=numeric, binary_tie_rule_pass=ties,
        max_abs_logit_error=float(np.max(np.abs(actual-expected))) if finite and shape else None,
        original_logit_tolerance='abs(approx-clear)<=0.02+0.01*abs(clear)',
        original_tie_rule='class1 iff logit1>=logit0')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--planning', type=Path, default=Path(__file__).with_name('plan_pretrained_bert_operators.json'))
    parser.add_argument('--output', type=Path, default=Path(__file__).with_suffix('.json'))
    args = parser.parse_args()
    started = time.perf_counter()
    sources = [Path(__file__), Path(__file__).with_name('test_pretrained_bert_arithmetic.py'), args.planning, ROOT/'cadathlon/validation/check_pretrained_bert_domains.py',
        ROOT/'cadathlon/agent_pset/p2_bert/src/reference.py', ROOT/'cadathlon/agent_pset/p2_bert/src/runner.py',
        ROOT/'cadathlon/agent_pset/p2_bert/doc/problem.md', ROOT/'cadathlon/common/encrypted_polynomial.py',
        ROOT/'cadathlon/common/support.py']
    def hashes():
        return {str(path.relative_to(ROOT)): digest(path) for path in sources}
    before = hashes()
    planning = json.loads(args.planning.read_text())
    manifest = choose_manifest(planning)
    audit = load_module(ROOT/'cadathlon/validation/check_pretrained_bert_domains.py')
    reference = load_module(ROOT/'cadathlon/agent_pset/p2_bert/src/reference.py')
    assets = {name: digest(args.snapshot/name) for name in audit.EXPECTED_ASSETS}
    assert assets == audit.EXPECTED_ASSETS
    import torch
    from transformers import AutoTokenizer, BertForSequenceClassification
    torch.set_num_threads(4)
    model, loading = BertForSequenceClassification.from_pretrained(args.snapshot, local_files_only=True,
        use_safetensors=True, torch_dtype=torch.float64, attn_implementation='eager', output_loading_info=True)
    assert not any(loading.get(name) for name in ('missing_keys', 'mismatched_keys', 'error_msgs'))
    assert not (set(loading['unexpected_keys'])-{'bert.embeddings.position_ids'})
    model.eval()
    config = model.config.to_dict()
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    tokens = tokenizer([a for a, _ in audit.EXAMPLES], [b for _, b in audit.EXAMPLES],
        max_length=128, truncation=True, padding='max_length', return_tensors='pt')
    with torch.no_grad():
        native = model(**tokens).logits.numpy()
    weights = {name: value.detach().numpy() for name, value in model.state_dict().items()}
    embeddings = reference.embed(config, weights, tokens['input_ids'].numpy(), tokens['token_type_ids'].numpy())
    mask = tokens['attention_mask'].numpy().astype(bool)
    exact = reference.forward(config, weights, embeddings, mask)
    np.testing.assert_allclose(exact, native, atol=1e-8, rtol=1e-8)
    cases = []
    for index in range(len(embeddings)):
        compact = embeddings[index, mask[index]][None]
        with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
            output, engine = forward(config, weights, compact, manifest)
        failures = engine.domain_failures()
        checks = gate(output.data, exact[index:index+1])
        cases.append(dict(example=index, valid_tokens=int(mask[index].sum()), gates=checks,
            all_original_gates_pass=all(checks[name] for name in ('finite', 'shape_matches', 'logit_tolerance_pass', 'binary_tie_rule_pass')),
            every_declared_domain_observed=engine.domains, propagated_domain_failures=failures,
            domain_observation_count=len(engine.domains),
            cost=dict(critical_level_depth=output.depth, operations=dict(engine.counts), by_layer=engine.phase_costs,
                observed_tensor_magnitude_peak=engine.peak,
                ideal_critical_path_refresh_lower_bound_depth49_output25=max(0, math.ceil((output.depth-49)/23)))))
        print(json.dumps(dict(example=index, max_abs_logit_error=checks['max_abs_logit_error'],
            original_gates_pass=cases[-1]['all_original_gates_pass'], domain_failure_count=len(failures),
            critical_depth=output.depth)), flush=True)
    # Tie behavior is part of the original gate, including exact equality.
    assert gate(np.array([[1., 1.]]), np.array([[1., 1.]]))['binary_tie_rule_pass']
    assert not gate(np.array([[1.+1e-6, 1.]]), np.array([[1., 1.]]))['binary_tie_rule_pass']
    report = dict(scope=__doc__, model_id=planning['model_id'], revision=planning['revision'],
        asset_sha256=assets, examples=audit.EXAMPLES, operator_manifest=manifest,
        approximate_complete_forward_evaluated=True, encrypted_execution=False, mrpc_accuracy_evaluated=False,
        input_boundary='Client computes the complete embedding module including embedding LayerNorm, as permitted by the P2 input contract. The approximate server-side graph covers all 12 encoder layers, pooler and classifier.',
        labels_or_logits_fitted=False, exact_numpy_vs_native_max_abs_error=float(np.max(np.abs(exact-native))),
        cases=cases, all_original_gates_pass=all(row['all_original_gates_pass'] for row in cases),
        all_observed_domains_pass=all(not row['propagated_domain_failures'] for row in cases),
        cost_model=dict(ct_product='One level per elementwise secret tensor product; counts are tensor calls and scalar elements, not realized ciphertext counts.',
            public_maps='One level per weighted affine stage, public matrix map, masked reduction or key/value layout movement; additions and public reshape views cost zero.',
            polynomial='Actual balanced monomial or Chebyshev algebra is executed. High-degree GELU uses Chebyshev recurrence, not rejected high-conditioning monomials.',
            reciprocal='First fixed Newton update is fused public affine; later iterations use two CT products plus one affine stage.',
            inverse_sqrt='First fixed Newton update is fused public affine; later iterations use three CT products plus one affine stage.',
            refresh='ceil((critical_depth-49)/23) is only a critical-path lower bound for this chosen abstract schedule, assuming initial49 levels, refresh output25 and input reserve2. It is not a lower bound for other optimized circuits. Branches, scheduling, precision and real packing can require more.'),
        limitations=['Same three public examples used for empirical function domains; no unseen-input or MRPC guarantee.',
            'No clipping, secret-dependent candidate selection, or data-dependent iteration stopping is part of the proposed arithmetic.',
            'Recorded domain checks are host validation only. Their absence of failure does not certify arbitrary inputs or CKKS error.',
            'No encryption, quantization, rescale error, bootstrap error or ciphertext-noise simulation is included.',
            'Cost estimates assume an abstract SIMD tensor layout; no complete matrix/rotation/key inventory or concrete full-model FHE packing is implemented.',
            'Observed tensor magnitudes exclude internal linear-map partial sums and cannot certify refresh scaling bounds.',
            'Chebyshev coefficient conditioning is improved, but operation count, depth and accumulated encrypted error remain substantial unresolved issues.'],
        source_hashes_before=before, source_hashes_after=hashes(),
        command=[sys.executable, *sys.argv],
        versions={name: importlib.metadata.version(name) for name in ('numpy', 'torch', 'transformers', 'safetensors', 'tokenizers')},
        python=sys.version.split()[0], cpu_affinity=sorted(os.sched_getaffinity(0)), elapsed_s=time.perf_counter()-started)
    report['source_files_unchanged'] = before == report['source_hashes_after']
    assert report['source_files_unchanged']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(all_original_gates_pass=report['all_original_gates_pass'],
        all_observed_domains_pass=report['all_observed_domains_pass'], source_files_unchanged=report['source_files_unchanged'],
        elapsed_s=report['elapsed_s']), indent=2))


if __name__ == '__main__':
    main()
