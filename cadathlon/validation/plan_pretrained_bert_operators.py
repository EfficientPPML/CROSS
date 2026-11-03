#!/usr/bin/env python3
"""Offline operator planning for three fixed public pretrained BERT examples.

This is not MRPC evaluation, an encrypted reference, or full approximate-model
validation. Polynomial coefficients approximate analytic functions only; no
logits or labels are fitted. Candidate domains are empirical calibration from
these three examples, not certified bounds for other inputs or CKKS errors.
"""
from __future__ import annotations
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


def summary(values):
    values = np.asarray(values)
    return dict(min=float(values.min()), max=float(values.max()), count=int(values.size))


def joined(arrays):
    return np.concatenate([np.asarray(array).reshape(-1) for array in arrays])


def inverse_newton(values, lo, hi, tolerance=1e-12):
    """Fixed public iteration count; interval checks are audit instrumentation."""
    assert np.min(values) >= lo and np.max(values) <= hi
    log_ratio = math.log1p(-2*lo/(lo+hi))
    iterations = max(0, math.ceil(math.log2(math.log(tolerance)/log_ratio)))
    result = np.full_like(values, 2/(lo+hi))
    for _ in range(iterations):
        result = result*(2-values*result)
    return result, iterations


def reduced_exp(values, squarings):
    contract = nonlinear_approximation('exp', (-.5, .5), 8)
    result = contract.evaluate_public(values/2**squarings)
    for _ in range(squarings):
        result = result*result
    return result


def polynomial_upper_max(scores, degree, public_span):
    """Polynomial binary maximum with a public uniform |x| tail allowance.

    |t|=2/pi+4/pi*sum((-1)^(k+1)*T_(2k)(t)/(4k²-1)). Truncating at
    degree2K has tail L1=2/(pi*(2K+1)) by telescoping. Add that allowance
    before each binary max. Floating1e-12 slack is engineering margin.
    """
    width = 1 << (scores.shape[-1]-1).bit_length()
    values = np.concatenate([scores, np.repeat(scores[..., -1:], width-scores.shape[-1], axis=-1)], axis=-1)
    coefficients = np.zeros(degree+1)
    coefficients[0] = 2/math.pi
    for k in range(1, degree//2+1):
        coefficients[2*k] = 4/math.pi*(-1)**(k+1)/(4*k*k-1)
    allowance = 2/(math.pi*(degree+1))+1e-12
    excess = 0.
    while values.shape[-1] > 1:
        scale = public_span+excess
        left, right = values[..., ::2], values[..., 1::2]
        argument = (left-right)/scale
        assert np.max(np.abs(argument)) <= 1
        values = (left+right+scale*np.polynomial.chebyshev.chebval(argument, coefficients)+scale*allowance)/2
        excess += scale*allowance
    conditioning = float(np.abs(np.polynomial.chebyshev.cheb2poly(coefficients)).sum())
    return values, excess, conditioning


def assess_softmax(scores_by_layer):
    result = {}
    for layer, scores in scores_by_layer.items():
        centered = [value-value.mean(axis=-1, keepdims=True) for value in scores]
        low, high = min(value.min() for value in centered), max(value.max() for value in centered)
        span = high-low
        lo, hi = math.floor(low-.1*span), math.ceil(high+.1*span)
        references = []
        for value in centered:
            exp = np.exp(value-value.max(axis=-1, keepdims=True))
            references.append(exp/exp.sum(axis=-1, keepdims=True))
        variants = {}
        for degree in (8, 16, 32):
            errors, gaps, denominators, reciprocal_counts, squaring_counts = [], [], [], [], []
            for value, reference in zip(centered, references):
                upper, excess, conditioning = polynomial_upper_max(value, degree, hi-lo)
                gap = upper-value.max(axis=-1, keepdims=True)
                assert gap.min() >= -1e-12 and gap.max() <= excess+1e-10
                squarings = max(0, math.ceil(math.log2((hi-lo+excess)/.5)))
                exp = reduced_exp(value-upper, squarings)
                denominator = exp.sum(axis=-1, keepdims=True)
                # In ideal arithmetic max<=upper<=max+excess gives
                # exp(-excess)<=sum(exp(score-upper))<=number_of_keys.
                inverse, count = inverse_newton(denominator, .9*math.exp(-excess), 1.1*value.shape[-1])
                errors.append(float(np.max(np.abs(exp*inverse-reference))))
                gaps.append(float(gap.max()))
                denominators.append(float(denominator.min()))
                reciprocal_counts.append(count)
                squaring_counts.append(squarings)
            variants[f'polynomial_upper_max_degree{degree}'] = dict(
                max_abs_probability_error=max(errors), observed_max_shift_excess=max(gaps),
                theoretical_ideal_shift_excess_bound=excess, observed_min_denominator_sum=min(denominators),
                fixed_reciprocal_iterations=max(reciprocal_counts), exp_squarings=max(squaring_counts),
                abs_polynomial_monomial_coefficient_l1=conditioning,
                current_power_degree_and_coefficient_budget_supported=degree <= 32 and conditioning <= 1e8,
                tournament_stages=max((value.shape[-1]-1).bit_length() for value in centered))
        squarings = max(0, math.ceil(math.log2(max(abs(lo), abs(hi))/.5)))
        errors, iteration_schedules, denominator_ranges = [], [], []
        for value, reference in zip(centered, references):
            keys = value.shape[-1]
            base = nonlinear_approximation('exp', (-.5, .5), 8).evaluate_public(value/2**squarings)
            inverse, count = inverse_newton(base.mean(axis=-1, keepdims=True), .5, 2.)
            probabilities = base*inverse/keys
            schedule, ranges = [count], []
            for _ in range(squarings):
                squared = probabilities*probabilities
                denominator = squared.sum(axis=-1, keepdims=True)
                # Exact probabilities imply1/keys<=sum(p²)<=1. Wider public
                # bounds leave numerical slack, but do not certify HE error.
                inverse, count = inverse_newton(denominator, .5/keys, 2.)
                ranges.append(summary(denominator))
                probabilities = squared*inverse
                schedule.append(count)
            errors.append(float(np.max(np.abs(probabilities-reference))))
            iteration_schedules.append(schedule)
            denominator_ranges.append(ranges)
        variants['normalized_repeated_squaring'] = dict(max_abs_probability_error=max(errors),
            private_max_required=False, squaring_rounds=squarings,
            fixed_reciprocal_iterations_by_example=iteration_schedules,
            observed_squared_probability_denominators=denominator_ranges,
            conditional_denominator_domains_by_example=[[.5/value.shape[-1], 2.] for value in centered])
        result[layer] = dict(empirical_centered_input_domain=[lo, hi], variants=variants)
    return result


def assess_functions(values_by_operator, reference):
    functions = dict(inverse_sqrt=lambda x: 1/np.sqrt(x), gelu=reference.gelu, tanh=np.tanh)
    result = {}
    for operator, values in values_by_operator.items():
        kind = operator.split('.')[-1]
        low, high = float(values.min()), float(values.max())
        domain = ((.8*low, 1.2*high) if kind == 'inverse_sqrt' else
                  (math.floor(low-.1*(high-low)), math.ceil(high+.1*(high-low))))
        grid = np.linspace(*domain, 16385)
        exact = functions[kind](grid)
        sweep = []
        for degree in (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256):
            contract = nonlinear_approximation(kind, domain, degree)
            approximation = contract.evaluate_public(grid)
            absolute = float(np.max(np.abs(approximation-exact)))
            relative = float(np.max(np.abs(approximation/exact-1))) if kind == 'inverse_sqrt' else None
            powers = np.polynomial.chebyshev.cheb2poly(contract.coefficients)
            conditioning = float(np.abs(powers).sum())
            target = relative <= 1e-5 if relative is not None else absolute <= (1e-3 if kind == 'gelu' else 1e-4)
            sweep.append(dict(degree=degree, max_abs_grid_error=absolute, max_relative_grid_error=relative,
                normalized_monomial_coefficient_l1=conditioning, sampled_grid_target_met=bool(target),
                current_power_degree_and_coefficient_budget_supported=degree <= 32 and conditioning <= 1e8,
                nominal_balanced_chebyshev_depth=contract.depth,
                nominal_balanced_power_depth=(degree-1).bit_length()+2))
            if target:
                break
        observed_error = float(np.max(np.abs(contract.evaluate_public(values)-functions[kind](values))))
        row = dict(observed_input_range=[low, high], empirical_candidate_domain=domain,
                   sweep=sweep, selected_degree=sweep[-1]['degree'], max_abs_observed_error=observed_error)
        if kind == 'inverse_sqrt':
            error, count = 1-math.sqrt(domain[0]/domain[1]), 0
            while error > 1e-6:
                error = 1.5*error*error-.5*error*error*error
                count += 1
            inverse = np.full_like(values, 1/math.sqrt(domain[1]))
            for _ in range(count):
                inverse = .5*inverse*(3-values*inverse*inverse)
            row['fixed_inverse_sqrt_newton'] = dict(iterations=count,
                public_initial_value=1/math.sqrt(domain[1]),
                max_relative_observed_error=float(np.max(np.abs(inverse*np.sqrt(values)-1))),
                ideal_relative_error_bound=error,
                ct_multiplications_per_iteration=3)
        result[operator] = row
    pooler = values_by_operator['pooler.tanh']
    domain = result['pooler.tanh']['empirical_candidate_domain']
    grid = np.linspace(*domain, 16385)
    squarings = max(0, math.ceil(math.log2(max(map(abs, domain)))))
    doubling = []
    for degree in (8, 12, 16):
        contract = nonlinear_approximation('tanh', (-1., 1.), degree)
        errors = []
        for values in (grid, pooler):
            approximation = contract.evaluate_public(values/2**squarings)
            for _ in range(squarings):
                denominator = 1+approximation*approximation
                inverse, count = inverse_newton(denominator, 1., 2.1)
                approximation = 2*approximation*inverse
            errors.append(float(np.max(np.abs(approximation-np.tanh(values)))))
        doubling.append(dict(base_degree=degree, doubling_rounds=squarings,
            reciprocal_iterations_per_round=count, max_abs_grid_error=errors[0],
            max_abs_observed_error=errors[1], base_monomial_coefficient_l1=float(np.abs(
                np.polynomial.chebyshev.cheb2poly(contract.coefficients)).sum())))
    result['pooler.tanh']['fixed_rational_doubling_alternative'] = doubling
    return result


def trace_exact(config, weights, embeddings, mask, reference):
    batch, sequence, hidden = embeddings.shape
    heads, dimension = config['num_attention_heads'], hidden//config['num_attention_heads']
    value, ranges, scores_by_layer, nonlinear, valid_hidden = embeddings.copy(), {}, {}, {}, []
    cls = np.zeros(mask.shape, bool)
    cls[:, 0] = True
    for layer in range(config['num_hidden_layers']):
        prefix = f'bert.encoder.layer.{layer}.'
        query, key, projected = [reference.linear(value, weights, prefix+'attention.self.'+name)
            .reshape(batch, sequence, heads, dimension).transpose(0, 2, 1, 3) for name in ('query', 'key', 'value')]
        scores = query@key.swapaxes(-1, -2)/math.sqrt(dimension)
        compact = [scores[index][:, mask[index]][:, :, mask[index]].copy() for index in range(batch)]
        scores_by_layer[str(layer)] = compact
        centered, denominators, masked_arguments = [], [], []
        padding_centered, padding_denominators = [], []
        max_shift_denominators = []
        for index in range(batch):
            for query_mask, centered_out, denominator_out in (
                (mask[index], centered, denominators), (~mask[index], padding_centered, padding_denominators)):
                raw = scores[index][:, query_mask][:, :, mask[index]]
                mean = raw.mean(axis=-1, keepdims=True)
                adjusted = raw-mean
                centered_out.append(adjusted)
                denominator_out.append(np.exp(adjusted).mean(axis=-1))
            raw = compact[index]
            masked_arguments.append(-raw.mean(axis=-1))
            max_shift_denominators.append(np.exp(raw-raw.max(axis=-1, keepdims=True)).sum(axis=-1))
        row = dict(raw_scores_valid_query_key=summary(joined(compact)),
            mean_centered_exp_valid_query_valid_key=summary(joined(centered)),
            mean_centered_exp_padding_query_valid_key=summary(joined(padding_centered)),
            masked_key_exp_argument_valid_queries=summary(joined(masked_arguments)),
            mean_exp_denominator_valid_queries=summary(joined(denominators)),
            mean_exp_denominator_padding_queries=summary(joined(padding_denominators)),
            exact_max_shift_sum_denominator_valid_queries=summary(joined(max_shift_denominators)))
        masked_scores = np.where(mask[:, None, None, :], scores, -np.inf)
        probabilities = np.exp(masked_scores-masked_scores.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        attended = (probabilities@projected).transpose(0, 2, 1, 3).reshape(batch, sequence, hidden)
        for phase in ('attention', 'mlp'):
            if phase == 'attention':
                value = value+reference.linear(attended, weights, prefix+'attention.output.dense')
                norm = prefix+'attention.output.LayerNorm'
            else:
                intermediate = reference.linear(value, weights, prefix+'intermediate.dense')
                nonlinear[f'{layer}.gelu'] = intermediate[mask].copy()
                row['gelu_input'] = {name: summary(intermediate[index]) for name, index in
                                    (('valid_tokens', mask), ('padding_tokens', ~mask), ('cls', cls))}
                value = value+reference.linear(reference.gelu(intermediate), weights, prefix+'output.dense')
                norm = prefix+'output.LayerNorm'
            center = value-value.mean(axis=-1, keepdims=True)
            variance = np.mean(center*center, axis=-1, keepdims=True)+config['layer_norm_eps']
            nonlinear[f'{layer}.{phase}.inverse_sqrt'] = variance[mask].copy()
            row[phase+'_layernorm_variance'] = {name: summary(variance[index]) for name, index in
                (('valid_tokens', mask), ('padding_tokens', ~mask), ('cls', cls))}
            value = center/np.sqrt(variance)*weights[norm+'.weight']+weights[norm+'.bias']
        ranges[str(layer)] = row
        valid_hidden.append([value[index, mask[index]].copy() for index in range(batch)])
    pooler = reference.linear(value[:, 0], weights, 'bert.pooler.dense')
    nonlinear['pooler.tanh'] = pooler.copy()
    logits = reference.linear(np.tanh(pooler), weights, 'classifier')
    return logits, ranges, scores_by_layer, nonlinear, valid_hidden


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_suffix('.json'))
    args = parser.parse_args()
    started = time.perf_counter()
    audit_path = ROOT/'cadathlon/validation/check_pretrained_bert_domains.py'
    reference_path = ROOT/'cadathlon/agent_pset/p2_bert/src/reference.py'
    sources = [Path(__file__), audit_path, reference_path,
        ROOT/'cadathlon/common/encrypted_polynomial.py', ROOT/'cadathlon/common/support.py']
    def snapshot():
        return {str(path.relative_to(ROOT)): digest(path) for path in sources}
    before = snapshot()
    audit, reference = load_module(audit_path), load_module(reference_path)
    assets = {name: digest(args.snapshot/name) for name in audit.EXPECTED_ASSETS}
    if assets != audit.EXPECTED_ASSETS:
        raise ValueError('checkpoint assets differ from the pinned audit snapshot')
    import torch
    from transformers import AutoTokenizer, BertForSequenceClassification
    torch.set_num_threads(4)
    model, loading = BertForSequenceClassification.from_pretrained(args.snapshot, local_files_only=True,
        use_safetensors=True, torch_dtype=torch.float64, attn_implementation='eager', output_loading_info=True)
    if any(loading.get(name) for name in ('missing_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError(f'incomplete checkpoint load: {loading}')
    if set(loading['unexpected_keys'])-{'bert.embeddings.position_ids'}:
        raise ValueError(f'unexpected checkpoint tensors: {loading}')
    model.eval()
    config = model.config.to_dict()
    assert (config['num_hidden_layers'], config['hidden_size'], model.num_labels) == (12, 768, 2)
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    tokens = tokenizer([a for a, _ in audit.EXAMPLES], [b for _, b in audit.EXAMPLES],
        max_length=128, truncation=True, padding='max_length', return_tensors='pt')
    with torch.no_grad():
        expected = model(**tokens).logits.numpy()
    weights = {name: value.detach().numpy() for name, value in model.state_dict().items()}
    mask = tokens['attention_mask'].numpy().astype(bool)
    assert mask[:, 0].all()
    embeddings = reference.embed(config, weights, tokens['input_ids'].numpy(), tokens['token_type_ids'].numpy())
    logits, ranges, scores, nonlinear, valid_hidden = trace_exact(config, weights, embeddings, mask, reference)
    np.testing.assert_allclose(logits, expected, atol=1e-8, rtol=1e-8)
    compact_checks = []
    for index in range(len(mask)):
        value = embeddings[index, mask[index]][None].copy()
        public_mask = np.ones(value.shape[:2], bool)
        per_layer = []
        for layer in range(12):
            value = reference.encoder_block(config, weights, value, public_mask, layer)
            error = float(np.max(np.abs(value[0]-valid_hidden[layer][index])))
            assert error < 1e-10
            per_layer.append(error)
        compact_logits = reference.linear(np.tanh(reference.linear(value[:, 0], weights, 'bert.pooler.dense')), weights, 'classifier')
        error = float(np.max(np.abs(compact_logits[0]-logits[index])))
        assert error < 1e-10
        compact_checks.append(dict(valid_length=int(mask[index].sum()),
            all_valid_hidden_max_error_by_layer=per_layer, cls_logit_max_error=error))
    report = dict(scope=__doc__, model_id='google-bert/bert-base-cased-finetuned-mrpc',
        revision='f150c1d609d1e50dd5e2e5408661cfac8339277c', asset_sha256=assets,
        examples=audit.EXAMPLES, encrypted_execution=False, mrpc_accuracy_evaluated=False,
        approximate_full_forward_evaluated=False, labels_or_logits_fitted=False,
        exact_full_forward_max_abs_error_vs_transformers=float(np.max(np.abs(logits-expected))),
        per_layer_operator_ranges=ranges,
        padding_compaction=dict(checks=compact_checks, original_token_rows=int(mask.size),
            retained_valid_token_rows=int(mask.sum()), original_dense_attention_pairs=int(mask.shape[0]*mask.shape[1]**2),
            retained_valid_attention_pairs=int(sum(length*length for length in mask.sum(axis=1))),
            reasoning='Masked keys never contribute to valid queries; LayerNorm/MLP are rowwise. Removing masked queries/keys while retaining original position-enriched embeddings preserves every valid-token layer output and CLS. The attention mask must already be public.',
            limitation='Logical work counts are not measured encrypted speedups; layouts, keys and compaction circuit costs remain unimplemented.'),
        softmax_operator_candidates=assess_softmax(scores),
        nonlinear_operator_candidates=assess_functions(nonlinear, reference),
        candidate_rules=dict(domains='Each positive variance interval is widened20%; other observed per-operator intervals receive10% span slack and outward integer endpoints.',
            analytic_grid_points=16385, targets=dict(inverse_sqrt_relative=1e-5, gelu_absolute=1e-3, tanh_absolute=1e-4),
            softmax_reciprocal_relative_target=1e-12,
            no_secret_branching='Candidate circuits use fixed public loop counts and arithmetic. Host interval checks and exact max comparisons are validation instrumentation only.'),
        limitations=['Three public examples are calibration observations, not certified domains for unseen inputs.',
            'Operator candidates run on exact-reference intermediates. Approximation errors are not propagated through the full model.',
            'Dense-grid errors are sampled diagnostics, not interval certificates.',
            'Candidate degrees and local error targets are planning choices, not exhaustive approximation impossibility results or model-level accuracy gates.',
            'Polynomial upper-max tail reasoning is ideal arithmetic; floating allowances and CKKS errors need separate proof.',
            'Normalized squaring reciprocal domains depend on probability normalization; their numerical checks do not certify encrypted error propagation.',
            'No FHE level/ring/bootstrap plan, performance measurement, MRPC accuracy, or encrypted full-checkpoint readiness is claimed.'],
        source_hashes_before=before, source_hashes_after=snapshot(),
        versions={name: importlib.metadata.version(name) for name in ('numpy', 'torch', 'transformers', 'safetensors', 'tokenizers')},
        python=sys.version.split()[0], cpu_affinity=sorted(os.sched_getaffinity(0)),
        elapsed_planning_s=time.perf_counter()-started)
    report['source_files_unchanged'] = before == report['source_hashes_after']
    assert report['source_files_unchanged']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(exact_error=report['exact_full_forward_max_abs_error_vs_transformers'],
        valid_lengths=[row['valid_length'] for row in compact_checks],
        compaction_max_hidden_error=max(max(row['all_valid_hidden_max_error_by_layer']) for row in compact_checks),
        source_files_unchanged=report['source_files_unchanged'], elapsed_planning_s=report['elapsed_planning_s']), indent=2))


if __name__ == '__main__':
    main()
