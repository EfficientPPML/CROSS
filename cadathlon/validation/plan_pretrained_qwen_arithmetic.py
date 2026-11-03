#!/usr/bin/env python3
"""Fixed arithmetic planning for three public full-checkpoint Qwen3 prompts.

Analytic-function candidates are selected before approximate logits are
computed. Exact-reference domains are empirical, not certified bounds. This
executes float64 arithmetic only: no ciphertexts, cache, decoding or generation.
"""
from __future__ import annotations
from collections import Counter
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
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT/'cadathlon/common')]
from compose_pretrained_bert_arithmetic import Arithmetic, Value
from encrypted_polynomial import nonlinear_approximation
from support import load_module
import numpy as np


INVERSE_RELATIVE_TARGET = 1e-6
SILU_ABSOLUTE_TARGET = 1e-4
GRID_SIZE = 16385


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def metric(actual, expected):
    finite = bool(np.isfinite(actual).all())
    shape = actual.shape == expected.shape
    if not finite or not shape:
        return dict(finite=finite, shape_matches=shape, max_abs_error=None, nrmse=None)
    difference = actual-expected
    return dict(finite=True, shape_matches=True, max_abs_error=float(np.max(np.abs(difference))),
        nrmse=float(np.sqrt(np.mean(difference*difference))/max(np.sqrt(np.mean(expected*expected)), .01)))


def summary(values):
    values = np.asarray(values)
    return dict(min=float(values.min()), max=float(values.max()), count=int(values.size))


def add_values(target, label, values):
    target.setdefault(label, []).append(np.array(values, copy=True))


def trace_exact(config, weights, embeddings, reference):
    x = embeddings.copy()
    batch, sequence, _ = x.shape
    nq, nkv, dimension = config['num_attention_heads'], config['num_key_value_heads'], config['head_dim']
    nonlinear, centered_scores, ranges, blocks, operators = {}, {}, {}, [], {}

    def norm(value, weight, label):
        variance = np.mean(value*value, axis=-1, keepdims=True)+config['rms_norm_eps']
        nonlinear[label] = variance.copy()
        result = value/np.sqrt(variance)*weight
        operators[label] = result.copy()
        return result

    for layer in range(config['num_hidden_layers']):
        p = f'model.layers.{layer}.'
        a = p+'self_attn.'
        z = norm(x, weights[p+'input_layernorm.weight'], f'{layer}.input.inverse_sqrt')
        projections = []
        for name, heads in (('q', nq), ('k', nkv), ('v', nkv)):
            projected = z@weights[a+name+'_proj.weight'].T
            operators[f'{layer}.{name}_linear'] = projected.copy()
            projections.append(projected.reshape(batch, sequence, heads, dimension).transpose(0, 2, 1, 3))
        q, k, v = projections
        q = reference.rope(norm(q, weights[a+'q_norm.weight'], f'{layer}.q.inverse_sqrt'), config['rope_theta'])
        k = reference.rope(norm(k, weights[a+'k_norm.weight'], f'{layer}.k.inverse_sqrt'), config['rope_theta'])
        operators[f'{layer}.q_rope'], operators[f'{layer}.k_rope'] = q.copy(), k.copy()
        k, v = np.repeat(k, nq//nkv, axis=1), np.repeat(v, nq//nkv, axis=1)
        scores = q@k.swapaxes(-1, -2)/math.sqrt(dimension)
        probabilities = np.zeros_like(scores)
        centered_scores[str(layer)] = []
        raw, denominators = [], []
        for query in range(sequence):
            valid = scores[..., query, :query+1]
            centered = valid-valid.mean(axis=-1, keepdims=True)
            centered_scores[str(layer)].append(centered.copy())
            raw.append(valid.reshape(-1))
            denominators.append(np.exp(centered).mean(axis=-1).reshape(-1))
            exponential = np.exp(valid-valid.max(axis=-1, keepdims=True))
            probabilities[..., query, :query+1] = exponential/exponential.sum(axis=-1, keepdims=True)
        attended = (probabilities@v).transpose(0, 2, 1, 3).reshape(batch, sequence, nq*dimension)
        operators[f'{layer}.gqa'] = attended.copy()
        branch = attended@weights[a+'o_proj.weight'].T
        operators[f'{layer}.output_linear'] = branch.copy()
        x = x+branch
        z = norm(x, weights[p+'post_attention_layernorm.weight'], f'{layer}.post.inverse_sqrt')
        m = p+'mlp.'
        gate = z@weights[m+'gate_proj.weight'].T
        up = z@weights[m+'up_proj.weight'].T
        operators[f'{layer}.gate_linear'], operators[f'{layer}.up_linear'] = gate.copy(), up.copy()
        nonlinear[f'{layer}.silu'] = gate.copy()
        activation = gate*np.exp(-np.logaddexp(0, -gate))
        operators[f'{layer}.silu'] = activation.copy()
        branch = (activation*up)@weights[m+'down_proj.weight'].T
        operators[f'{layer}.gated_mlp'] = branch.copy()
        x = x+branch
        blocks.append(x.copy())
        ranges[str(layer)] = dict(raw_valid_scores=summary(np.concatenate(raw)),
            centered_valid_scores=summary(np.concatenate([a.reshape(-1) for a in centered_scores[str(layer)]])),
            mean_exp_denominators=summary(np.concatenate(denominators)), residual=summary(x),
            silu_input=summary(gate))
    last = norm(x[:, -1], weights['model.norm.weight'], 'final.inverse_sqrt')
    logits = last@weights['model.embed_tokens.weight'].T
    return dict(logits=logits, nonlinear=nonlinear, centered_scores=centered_scores,
                ranges=ranges, blocks=blocks, operators=operators)


def reciprocal_iterations(lo, hi, target=1e-12):
    # Uniform ideal residual after t Newton steps: rho**(2**t).
    return max(1, math.ceil(math.log2(math.log(target)/math.log1p(-2*lo/(lo+hi)))))


def fixed_reciprocal(values, lo, hi, iterations):
    result = np.full_like(values, 2/(lo+hi))
    for _ in range(iterations):
        result *= 2-values*result
    return result


def silu_rational(values, rule):
    base = nonlinear_approximation('tanh', (-1., 1.), rule['base_degree'])
    result = base.evaluate_public(values/2**(rule['doublings']+1))
    for _ in range(rule['doublings']):
        result = 2*result*fixed_reciprocal(1+result*result, 1., 2.1, rule['reciprocal_iterations'])
    return .5*values*(1+result)


def choose_manifest(traces):
    """Only analytic local-function errors and public conditioning select it."""
    nonlinear, scores = {}, {}
    for trace in traces:
        for label, values in trace['nonlinear'].items():
            add_values(nonlinear, label, values)
        for layer, rows in trace['centered_scores'].items():
            scores.setdefault(layer, []).extend(rows)
    manifest, assessments = {}, {}
    for label, arrays in nonlinear.items():
        values = np.concatenate([v.reshape(-1) for v in arrays])
        kind = 'silu' if label.endswith('silu') else 'inverse_sqrt'
        low, high = float(values.min()), float(values.max())
        domain = ((.8*low, 1.2*high) if kind == 'inverse_sqrt' else
                  (math.floor(low-.1*(high-low)), math.ceil(high+.1*(high-low))))
        grid = np.linspace(*domain, GRID_SIZE)
        exact = 1/np.sqrt(grid) if kind == 'inverse_sqrt' else grid*np.exp(-np.logaddexp(0, -grid))
        sweep = []
        for degree in (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256):
            contract = nonlinear_approximation(kind, domain, degree)
            approximation = contract.evaluate_public(grid)
            absolute = float(np.max(np.abs(approximation-exact)))
            relative = float(np.max(np.abs(approximation/exact-1))) if kind == 'inverse_sqrt' else None
            conditioning = float(np.abs(np.polynomial.chebyshev.cheb2poly(contract.coefficients)).sum())
            passed = relative <= INVERSE_RELATIVE_TARGET if relative is not None else absolute <= SILU_ABSOLUTE_TARGET
            sweep.append(dict(degree=degree, max_abs_grid_error=absolute, max_relative_grid_error=relative,
                monomial_coefficient_l1=conditioning, local_target_pass=bool(passed)))
            if passed:
                break
        rule = dict(domain=domain)
        if kind == 'inverse_sqrt':
            if passed and degree <= 32 and conditioning <= 1e8:
                rule.update(method='power', degree=degree)
            else:
                error, count = 1-math.sqrt(domain[0]/domain[1]), 0
                while error > INVERSE_RELATIVE_TARGET:
                    error = 1.5*error**2-.5*error**3
                    count += 1
                rule.update(method='newton', iterations=count, ideal_relative_error_bound=error)
        elif passed:
            rule.update(method='chebyshev', degree=degree)
        else:
            doublings = max(0, math.ceil(math.log2(max(map(abs, domain))/2)))
            variants = []
            for base_degree in (8, 12, 16, 20, 24):
                rational = dict(doublings=doublings, base_degree=base_degree,
                                reciprocal_iterations=reciprocal_iterations(1., 2.1))
                error = float(np.max(np.abs(silu_rational(grid, rational)-exact)))
                variants.append(dict(**rational, max_abs_grid_error=error))
                if error <= SILU_ABSOLUTE_TARGET:
                    break
            if error > SILU_ABSOLUTE_TARGET:
                raise ValueError(f'No analytic SiLU candidate met the fixed local target for {label}')
            rule.update(method='rational_tanh', **rational)
        manifest[label] = rule
        assessments[label] = dict(observed_range=[low, high], empirical_domain=domain, sweep=sweep,
            selected=rule, rational_variants=variants if kind == 'silu' and not passed else [])
    softmax = {}
    for layer, rows in scores.items():
        values = np.concatenate([row.reshape(-1) for row in rows])
        low, high = float(values.min()), float(values.max())
        domain = (math.floor(low-.1*(high-low)), math.ceil(high+.1*(high-low)))
        squarings = max(0, math.ceil(math.log2(max(1., *map(abs, domain))/.5)))
        softmax[layer] = dict(centered_domain=domain, squarings=squarings,
            initial_reciprocal_iterations=reciprocal_iterations(.5, 2.),
            reciprocal_iterations_by_keys={str(keys): reciprocal_iterations(.5/keys, 2.)
                                          for keys in sorted({row.shape[-1] for row in rows})})
    return dict(nonlinear=manifest, softmax=softmax,
        selection='Fixed from empirical exact-reference domains and analytic-function targets before approximate logits are computed.',
        local_targets=dict(inverse_sqrt_relative=INVERSE_RELATIVE_TARGET, silu_absolute=SILU_ABSOLUTE_TARGET,
                           reciprocal_ideal_residual=1e-12, grid_size=GRID_SIZE)), assessments


class QwenArithmetic(Arithmetic):
    def norm(self, value, weight, epsilon, rule, label):
        square = self.multiply(value, value, label+'.square')
        variance = self.reduce(square, -1, factor=1/value.data.shape[-1], bias=epsilon, label=label+'.variance')
        inverse = self.inverse_sqrt(variance, rule, label)
        return self.affine(self.multiply(value, inverse, label+'.normalized'), weight, label=label+'.weight')

    def rope(self, value, theta, label):
        dimension, sequence = value.data.shape[-1], value.data.shape[-2]
        angles = np.arange(sequence)[:, None]*theta**(-np.arange(0, dimension, 2)/dimension)[None, :]
        angles = np.concatenate((angles, angles), axis=-1)[None, None]
        rotated = self.view(value, np.concatenate((-value.data[..., dimension//2:], value.data[..., :dimension//2]), axis=-1))
        return self.weighted_sum([value, rotated], [np.cos(angles), np.sin(angles)], label=label)

    def causal_attention(self, query, key, projected, rule, label):
        key, projected = self.layout_move(key, label+'.key_layout'), self.layout_move(projected, label+'.value_layout')
        products = self.multiply(self.view(query, query.data[:, :, :, None, :]),
            self.view(key, key.data[:, :, None, :, :]), label+'.qk')
        scores = self.reduce(products, -1, factor=1/math.sqrt(query.data.shape[-1]), keepdims=False, label=label+'.scores')
        rows = []
        for row in range(query.data.shape[-2]):
            if row == 0:
                # The sole legal key has public probability one. This is a
                # public shape simplification, not a private-value branch.
                rows.append(self.view(projected, projected.data[:, :, 0]))
                continue
            row_label = label+f'.query{row}'
            valid = self.view(scores, scores.data[..., row, :row+1])
            mean = self.reduce(valid, -1, factor=1/(row+1), label=row_label+'.mean')
            centered = self.add(valid, mean, subtract=True, label=row_label+'.center')
            self.observe(row_label+'.centered', centered.data, rule['centered_domain'])
            base = self.polynomial(centered, nonlinear_approximation('exp', (-.5, .5), 8), 'power',
                row_label+'.exp', input_scale=2.**-rule['squarings'])
            denominator = self.reduce(base, -1, factor=1/(row+1), label=row_label+'.initial_mean')
            inverse = self.inverse(denominator, (.5, 2.), rule['initial_reciprocal_iterations'], row_label+'.initial_inverse')
            probability = self.affine(self.multiply(base, inverse, row_label+'.initial_normalized'),
                1/(row+1), label=row_label+'.probability')
            for iteration in range(rule['squarings']):
                prefix = row_label+f'.round{iteration}'
                square = self.multiply(probability, probability, prefix+'.square')
                denominator = self.reduce(square, -1, label=prefix+'.sum')
                inverse = self.inverse(denominator, (.5/(row+1), 2.), rule['reciprocal_iterations_by_keys'][str(row+1)], prefix+'.inverse')
                probability = self.multiply(square, inverse, prefix+'.normalize')
            products = self.multiply(self.view(probability, probability.data[..., None]),
                self.view(projected, projected.data[:, :, :row+1]), row_label+'.pv')
            rows.append(self.reduce(products, -2, keepdims=False, label=row_label+'.attended'))
        return self.node(np.stack([row.data for row in rows], axis=-2), max(row.depth for row in rows), label+'.assembled')

    def silu(self, value, rule, label):
        if rule['method'] == 'chebyshev':
            return self.polynomial(value, nonlinear_approximation('silu', rule['domain'], rule['degree']), 'chebyshev', label)
        self.observe(label, value.data, rule['domain'])
        result = self.polynomial(value, nonlinear_approximation('tanh', (-1., 1.), rule['base_degree']),
            'power', label+'.tanh_base', input_scale=2.**-(rule['doublings']+1))
        for iteration in range(rule['doublings']):
            square = self.multiply(result, result, label+'.square')
            denominator = self.affine(square, 1., 1., label+'.denominator')
            inverse = self.inverse(denominator, (1., 2.1), rule['reciprocal_iterations'], label+f'.inverse{iteration}')
            result = self.affine(self.multiply(result, inverse, label+'.quotient'), 2., label=label+'.double')
        sigmoid = self.affine(result, .5, .5, label+'.sigmoid')
        return self.multiply(value, sigmoid, label+'.activated')


def approximate_forward(config, weights, embeddings, manifest):
    engine, value = QwenArithmetic(), Value(embeddings.copy())
    batch, sequence, _ = embeddings.shape
    nq, nkv, dimension = config['num_attention_heads'], config['num_key_value_heads'], config['head_dim']
    nonlinear, blocks, operators = manifest['nonlinear'], [], {}
    for layer in range(config['num_hidden_layers']):
        start_counts, start_depth = engine.counts.copy(), value.depth
        p, label = f'model.layers.{layer}.', str(layer)
        a = p+'self_attn.'
        name = label+'.input.inverse_sqrt'
        normalized = engine.norm(value, weights[p+'input_layernorm.weight'], config['rms_norm_eps'], nonlinear[name], name)
        operators[name] = normalized.data.copy()
        projections = []
        for name, heads in (('q', nq), ('k', nkv), ('v', nkv)):
            projected = engine.linear(normalized, weights[a+name+'_proj.weight'], 0., label+'.'+name)
            operators[label+'.'+name+'_linear'] = projected.data.copy()
            projections.append(engine.view(projected, projected.data.reshape(batch, sequence, heads, dimension).transpose(0, 2, 1, 3)))
        query, key, projected = projections
        for kind, operand in (('q', query), ('k', key)):
            name = label+'.'+kind+'.inverse_sqrt'
            result = engine.norm(operand, weights[a+kind+'_norm.weight'], config['rms_norm_eps'], nonlinear[name], name)
            operators[name] = result.data.copy()
            result = engine.rope(result, config['rope_theta'], label+'.'+kind+'_rope')
            operators[label+'.'+kind+'_rope'] = result.data.copy()
            if kind == 'q':
                query = result
            else:
                key = result
        key = engine.view(key, np.repeat(key.data, nq//nkv, axis=1))
        projected = engine.view(projected, np.repeat(projected.data, nq//nkv, axis=1))
        attended = engine.causal_attention(query, key, projected, manifest['softmax'][label], label+'.attention')
        attended = engine.view(attended, attended.data.transpose(0, 2, 1, 3).reshape(batch, sequence, nq*dimension))
        operators[label+'.gqa'] = attended.data.copy()
        branch = engine.linear(attended, weights[a+'o_proj.weight'], 0., label+'.output')
        operators[label+'.output_linear'] = branch.data.copy()
        value = engine.add(value, branch, label=label+'.attention_residual')
        name = label+'.post.inverse_sqrt'
        normalized = engine.norm(value, weights[p+'post_attention_layernorm.weight'], config['rms_norm_eps'], nonlinear[name], name)
        operators[name] = normalized.data.copy()
        m = p+'mlp.'
        gate = engine.linear(normalized, weights[m+'gate_proj.weight'], 0., label+'.gate')
        up = engine.linear(normalized, weights[m+'up_proj.weight'], 0., label+'.up')
        operators[label+'.gate_linear'], operators[label+'.up_linear'] = gate.data.copy(), up.data.copy()
        activated = engine.silu(gate, nonlinear[label+'.silu'], label+'.silu')
        operators[label+'.silu'] = activated.data.copy()
        product = engine.multiply(activated, up, label+'.gated')
        branch = engine.linear(product, weights[m+'down_proj.weight'], 0., label+'.down')
        operators[label+'.gated_mlp'] = branch.data.copy()
        value = engine.add(value, branch, label=label+'.mlp_residual')
        blocks.append(value.data.copy())
        engine.phase_costs.append(dict(layer=layer, depth_increment=value.depth-start_depth,
            depth_after=value.depth, operations=dict(engine.counts-start_counts)))
    last = engine.norm(engine.view(value, value.data[:, -1]), weights['model.norm.weight'], config['rms_norm_eps'],
                       nonlinear['final.inverse_sqrt'], 'final.inverse_sqrt')
    operators['final.inverse_sqrt'] = last.data.copy()
    output = engine.linear(last, weights['model.embed_tokens.weight'], 0., 'vocabulary')
    return output, engine, blocks, operators


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_suffix('.json'))
    args = parser.parse_args()
    started = time.perf_counter()
    audit = load_module(ROOT/'cadathlon/validation/check_pretrained_qwen_domains.py')
    reference = load_module(ROOT/'cadathlon/agent_pset/p4_qwen/src/reference.py')
    sources = [Path(__file__), Path(__file__).with_name('test_pretrained_qwen_arithmetic.py'), Path(__file__).with_name('compose_pretrained_bert_arithmetic.py'),
        ROOT/'cadathlon/validation/check_pretrained_qwen_domains.py', ROOT/'cadathlon/agent_pset/p4_qwen/src/reference.py',
        ROOT/'cadathlon/agent_pset/p4_qwen/src/runner.py', ROOT/'cadathlon/agent_pset/p4_qwen/doc/problem.md',
        ROOT/'cadathlon/agent_pset/p4_qwen/test/model_source.json', ROOT/'cadathlon/common/encrypted_polynomial.py',
        ROOT/'cadathlon/common/support.py']
    def hashes():
        return {str(path.relative_to(ROOT)): digest(path) for path in sources}
    before = hashes()
    assets = {name: digest(args.snapshot/name) for name in audit.EXPECTED_ASSETS}
    if assets != audit.EXPECTED_ASSETS:
        raise ValueError('Pinned Qwen assets differ')
    import torch
    from transformers import AutoTokenizer, Qwen3ForCausalLM
    torch.set_num_threads(4)
    model, loading = Qwen3ForCausalLM.from_pretrained(args.snapshot, local_files_only=True, use_safetensors=True,
        torch_dtype=torch.float64, attn_implementation='eager', output_loading_info=True)
    if any(loading.get(name) for name in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError(f'Checkpoint load mismatch: {loading}')
    model.eval()
    config = model.config.to_dict()
    if (config['num_hidden_layers'], config['hidden_size'], config['vocab_size']) != (28, 1024, 151936):
        raise ValueError('Expected the full pinned Qwen3-0.6B architecture')
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    weights = {name: value.detach().numpy() for name, value in model.state_dict().items()}
    traces, embeddings, natives, cases = [], [], [], []
    for prompt in audit.PROMPTS:
        tokens = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
        with torch.no_grad():
            native = model(**tokens, use_cache=False).logits[:, -1].numpy()
        embedding = weights['model.embed_tokens.weight'][tokens['input_ids'].numpy()]
        trace = trace_exact(config, weights, embedding, reference)
        oracle = reference.forward(config, weights, embedding)
        native_check, oracle_check = metric(trace['logits'], native), metric(trace['logits'], oracle)
        native_top1 = bool(np.array_equal(native.argmax(-1), trace['logits'].argmax(-1)))
        if native_check['max_abs_error'] > 1e-3 or native_check['nrmse'] > 1e-5 or oracle_check['max_abs_error'] > 1e-8 or not native_top1:
            raise ValueError('Exact trace does not match preserved semantic gates')
        traces.append(trace)
        embeddings.append(embedding)
        natives.append(native)
        cases.append(dict(prompt=prompt, tokens=embedding.shape[1], all_final_logits=trace['logits'].shape[-1],
            exact_vs_native=native_check, exact_vs_numpy_oracle=oracle_check,
            exact_native_top1_agreement=native_top1))
    manifest, assessments = choose_manifest(traces)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Freeze and preserve selection before running approximate model outputs.
    manifest_path = args.output.with_name(args.output.stem+'_operators.json')
    manifest_path.write_text(json.dumps(dict(manifest=manifest, assessments=assessments), indent=2, allow_nan=False)+'\n')
    baseline_path = args.output.with_name(args.output.stem+'_exact_logits.npz')
    np.savez_compressed(baseline_path, numpy_logits=np.concatenate([trace['logits'] for trace in traces]), native_logits=np.concatenate(natives))
    for index, (embedding, trace, native) in enumerate(zip(embeddings, traces, natives)):
        with np.errstate(over='ignore', invalid='ignore', under='ignore', divide='ignore'):
            output, engine, blocks, operators = approximate_forward(config, weights, embedding, manifest)
        checks = metric(output.data, trace['logits'])
        top1 = bool(checks['finite'] and np.array_equal(output.data.argmax(-1), trace['logits'].argmax(-1)))
        block_metrics = [dict(layer=layer, **metric(actual, expected)) for layer, (actual, expected) in enumerate(zip(blocks, trace['blocks']))]
        operator_metrics = {label: metric(actual, trace['operators'][label]) for label, actual in operators.items()}
        def within(row, bound):
            return bool(row['finite'] and row['shape_matches'] and row['nrmse'] <= bound)
        cases[index].update(approximate_vs_exact=checks, approximate_vs_native=metric(output.data, native),
            full_logit_nrmse_gate_pass=within(checks, .10), public_prompt_top1_agreement=top1,
            bundled_toy_numeric_threshold_diagnostic_pass=within(checks, .10) and checks['max_abs_error'] <= .20,
            block_metrics=block_metrics, all_block_nrmse_gates_pass=all(within(row, .05) for row in block_metrics),
            operator_metrics=operator_metrics, all_operator_nrmse_gates_pass=all(within(row, .05) for row in operator_metrics.values()),
            exact_ranges=trace['ranges'], observed_domains=engine.domains, domain_failures=engine.domain_failures(),
            cost=dict(critical_level_depth=output.depth, operations=dict(engine.counts), layers=engine.phase_costs,
                observed_tensor_magnitude_peak=engine.peak,
                chosen_schedule_critical_path_refresh_lower_bound_depth49_output25=max(0, math.ceil((output.depth-49)/23))))
        print(json.dumps(dict(example=index, max_abs_error=checks['max_abs_error'], nrmse=checks['nrmse'],
            top1=top1, domain_failure_count=len(engine.domain_failures()), critical_depth=output.depth)), flush=True)
    provenance = json.loads((ROOT/'cadathlon/agent_pset/p4_qwen/test/model_source.json').read_text())
    report = dict(scope=__doc__, model_id=provenance['repo_id'], revision=provenance['revision'], asset_sha256=assets,
        architecture={key: config[key] for key in ('num_hidden_layers', 'hidden_size', 'intermediate_size', 'num_attention_heads', 'num_key_value_heads', 'head_dim', 'vocab_size')},
        approximate_full_forward_evaluated=True, encrypted_execution=False, generation_evaluated=False,
        labels_or_logits_fitted=False, cases=cases, operator_manifest_file=manifest_path.name,
        operator_manifest_sha256=digest(manifest_path), exact_logits_file=baseline_path.name, exact_logits_sha256=digest(baseline_path),
        original_full_logit_nrmse_threshold=.10, original_operator_block_nrmse_threshold=.05,
        all_public_prompt_numeric_gates_pass=all(c['full_logit_nrmse_gate_pass'] and c['all_block_nrmse_gates_pass'] and c['all_operator_nrmse_gates_pass'] for c in cases),
        all_public_prompt_top1_agree=all(c['public_prompt_top1_agreement'] for c in cases),
        all_observed_domains_pass=all(not c['domain_failures'] for c in cases),
        cost_scope='Abstract SIMD tensor accounting, not realized ciphertext counts. Each secret product/public affine/map/reduction/layout stage costs one level. Depth and refresh lower bounds apply only to this chosen schedule, not optimized implementations; publicly indexed causal query-row branches run in parallel in the depth model.',
        limitations=['Three public calibration prompts of3/3/4tokens do not establish Q3-Q5 prefill or90%aggregate argmax gates.',
            'No key generation, ciphertext arithmetic, CKKS/bootstrap error, decoding, KV cache or generation was evaluated.',
            'Public empirical operator domains are not certified bounds for unseen inputs or propagated encrypted error.',
            'The same prompts supply calibration ranges and final observations; this is not an independent generalization test.',
            'Public causal rows are evaluated with their legal key count, avoiding evaluation in unused future-key slots. A real packed implementation must mask or safely initialize every extra slot.',
            'No full-model layout, matrix tiling, rotations, keys, memory plan or precision/refresh schedule exists for this graph.',
            'Observed magnitudes omit internal linear-map partial sums; small final logit errors do not establish refresh scaling bounds.',
            'The depth49/output25/reserve2 refresh count is only an optimistic critical-path lower bound for the chosen abstract schedule.'],
        source_hashes_before=before, source_hashes_after=hashes(), source_files_unchanged=before==hashes(),
        versions={name: importlib.metadata.version(name) for name in ('numpy', 'torch', 'transformers', 'safetensors', 'tokenizers')},
        command=[sys.executable, *sys.argv], cpu_affinity=sorted(os.sched_getaffinity(0)), elapsed_s=time.perf_counter()-started)
    if not report['source_files_unchanged']:
        raise ValueError('Sources changed during planning')
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: report[key] for key in ('all_public_prompt_numeric_gates_pass', 'all_public_prompt_top1_agree', 'all_observed_domains_pass', 'source_files_unchanged', 'elapsed_s')}, indent=2))


if __name__ == '__main__':
    main()
