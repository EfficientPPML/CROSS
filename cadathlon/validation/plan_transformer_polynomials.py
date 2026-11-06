"""Public numerical planning only: no encrypted inference or runtime fallback.

Evaluate analytic Chebyshev approximations in a copy of the toy transformer
circuits. Domains/degrees are fixed across all fixtures in a sweep, never fitted
to their expected outputs. Report every domain violation rather than clipping
values. Observed fixture ranges are not certified bounds for arbitrary inputs.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'common'))
from support import load_module, metrics
from encrypted_polynomial import nonlinear_approximation
from encrypted_transformer import public_toy_approximations

BERT = load_module(ROOT / 'agent_pset/p2_bert/src/reference.py')
QWEN = load_module(ROOT / 'agent_pset/p4_qwen/src/reference.py')
FUNCTIONS = {
    'inverse_sqrt': lambda x: 1 / np.sqrt(x),
    'gelu': BERT.gelu, 'silu': lambda x: x * np.exp(-np.logaddexp(0, -x)),
    'tanh': np.tanh, 'exp': np.exp, 'reciprocal': lambda x: 1 / x,
}


@lru_cache(None)
def approximation(name, domain, degree):
    # Use the runtime's interpolation/roundoff policy, still evaluating only
    # public plaintext values here. Domain violations are recorded by Plan.
    contract = nonlinear_approximation(name, domain, degree)
    return np.polynomial.Chebyshev(contract.coefficients, domain=domain)


@lru_cache(None)
def approximation_statistics(name, domain, degree, input_power_basis=False):
    """Dense-grid diagnostics, explicitly not certified interval bounds."""
    series = approximation(name, domain, degree)
    t = np.linspace(-1., 1., 100001)
    x = (domain[1] + domain[0]) / 2 + t * (domain[1] - domain[0]) / 2
    exact = FUNCTIONS[name](x)
    chebyshev = np.polynomial.chebyshev.chebval(t, series.coef)
    powers = np.polynomial.chebyshev.cheb2poly(series.coef)
    monomial = np.polynomial.polynomial.polyval(t, powers)
    if input_power_basis and (domain[0] != -domain[1] or domain[1] > 2 or degree != 2):
        raise ValueError('input-power statistics require the runtime centered quadratic contract')
    selected_powers = powers / domain[1] ** np.arange(len(powers)) if input_power_basis else powers
    selected_values = np.polynomial.polynomial.polyval(x if input_power_basis else t, selected_powers)
    return {
        'name': name, 'domain': domain, 'degree': degree,
        'sampled_max_abs_error': float(np.max(np.abs(chebyshev - exact))),
        'sampled_max_relative_error': (float(np.max(np.abs(chebyshev / exact - 1)))
                                       if name in ('exp', 'reciprocal', 'inverse_sqrt') else None),
        'sampled_minimum': float(np.min(chebyshev)),
        'normalized_monomial_coefficient_l1': float(np.sum(np.abs(powers))),
        'normalized_monomial_max_coefficient': float(np.max(np.abs(powers))),
        'sampled_monomial_vs_chebyshev_deviation': float(np.max(np.abs(monomial - chebyshev))),
        'balanced_monomial_depth_including_affine_map_and_final_sum': math.ceil(math.log2(degree)) + 2,
        'input_power_basis': bool(input_power_basis),
        'selected_balanced_power_depth': math.ceil(math.log2(degree)) + 2 - int(input_power_basis),
        'selected_monomial_coefficient_l1': float(np.sum(np.abs(selected_powers))),
        'sampled_selected_monomial_vs_chebyshev_deviation': float(np.max(np.abs(selected_values - chebyshev))),
    }


def runtime_approximation_statistics():
    """Keep actual runtime evaluation schedules distinct from degree sweeps."""
    return {
        problem: {
            name: approximation_statistics(name, contract.domain, len(contract.coefficients)-1,
                                           contract.input_power_basis)
            for name, contract in public_toy_approximations(problem).items()
        }
        for problem in ('bert', 'qwen')
    }


class Plan:
    def __init__(self, problem, degrees=None, exact=False):
        self.exact, self.ranges, self.violations = exact, {}, {}
        runtime = public_toy_approximations(problem)
        self.contracts = {name: (contract.domain, (degrees or {}).get(name, len(contract.coefficients) - 1))
                          for name, contract in runtime.items()}

    def observe(self, name, values):
        low, high = float(np.min(values)), float(np.max(values))
        old = self.ranges.setdefault(name, {'min': low, 'max': high, 'count': 0})
        old.update(min=min(low, old['min']), max=max(high, old['max']), count=old['count'] + values.size)

    def apply(self, name, values, label=None):
        values = np.asarray(values)
        self.observe(label or name, values)
        domain, degree = self.contracts[name]
        outside = (~np.isfinite(values)) | (values < domain[0]) | (values > domain[1])
        if np.any(outside):
            self.violations[label or name] = self.violations.get(label or name, 0) + int(outside.sum())
        if self.exact:
            return FUNCTIONS[name](values)
        return approximation(name, domain, degree)(values)

    def softmax(self, scores, mask):
        scores = np.asarray(scores)
        mask = np.broadcast_to(mask, scores.shape)
        batch, _, queries, keys = scores.shape
        if queries != keys:
            raise ValueError('paired layout requires square self-attention scores')
        # Observe one complete PairLayout period, including duplicated padded
        # queries/batches and invalid padded keys. Full CKKS slots repeat this
        # period; reporting only the dense unpadded matrix can miss arguments
        # that still enter a ciphertext polynomial before its output is masked.
        padded_batch = 1 << (batch - 1).bit_length()
        width = 1 << (keys - 1).bit_length()
        batch_indices = np.arange(padded_batch) % batch
        positions = np.minimum(np.arange(width), keys - 1)
        scores = scores[batch_indices][:, :, positions][:, :, :, positions]
        mask = mask[batch_indices][:, :, positions][:, :, :, positions].copy()
        mask[..., keys:] = False
        count = mask.sum(axis=-1, keepdims=True)
        if np.any(count == 0):
            raise ValueError('fully masked attention row')
        # Mean centering uses only arithmetic and the public attention mask.
        # No private max/branch/lookup is required by the planned HE circuit.
        center = np.where(mask, scores, 0.).sum(axis=-1, keepdims=True) / count
        # Runtime first masks the scores, subtracts the row mean in every
        # slot, then evaluates exp before applying the output mask. Invalid
        # slots therefore see -center, not zero.
        exps = self.apply('exp', np.where(mask, scores, 0.) - center) * mask
        mean_exp = exps.sum(axis=-1, keepdims=True) / count
        inverse = self.apply('reciprocal', np.broadcast_to(mean_exp, exps.shape))
        return (exps * inverse / count)[:batch, :, :queries, :keys]

    def norm(self, x, weight, bias, eps, *, center, label):
        if center:
            x = x - x.mean(axis=-1, keepdims=True)
        variance = np.mean(x*x, axis=-1, keepdims=True) + eps
        return x * self.apply('inverse_sqrt', variance, label) * weight + bias


def bert_forward(config, w, x, mask, plan):
    b, s, h = x.shape
    heads, d = config['num_attention_heads'], h // config['num_attention_heads']
    for layer in range(config['num_hidden_layers']):
        p = f'bert.encoder.layer.{layer}.'
        q, k, v = [BERT.linear(x, w, p + 'attention.self.' + name).reshape(b,s,heads,d).transpose(0,2,1,3)
                   for name in ('query','key','value')]
        probs = plan.softmax(q @ k.swapaxes(-1,-2) / np.sqrt(d), mask[:,None,None,:])
        attended = (probs @ v).transpose(0,2,1,3).reshape(b,s,h)
        for sublayer in ('attention', 'mlp'):
            if sublayer == 'attention':
                x = x + BERT.linear(attended, w, p + 'attention.output.dense')
                prefix = p + 'attention.output.LayerNorm'
            else:
                activated = plan.apply('gelu', BERT.linear(x, w, p + 'intermediate.dense'))
                x = x + BERT.linear(activated, w, p + 'output.dense')
                prefix = p + 'output.LayerNorm'
            x = plan.norm(x, w[prefix+'.weight'], w[prefix+'.bias'], config['layer_norm_eps'],
                          center=True, label='layernorm_variance')
    pooled = plan.apply('tanh', BERT.linear(x[:,0,:], w, 'bert.pooler.dense'))
    return BERT.linear(pooled, w, 'classifier')


def qwen_forward(c, w, x, plan):
    b, s, h = x.shape
    nq, nkv, d = c['num_attention_heads'], c['num_key_value_heads'], c['head_dim']
    eps = c['rms_norm_eps']
    for layer in range(c['num_hidden_layers']):
        p = f'model.layers.{layer}.'
        a = p + 'self_attn.'
        z = plan.norm(x, w[p+'input_layernorm.weight'], 0., eps, center=False, label='residual_variance')
        q = (z @ w[a+'q_proj.weight'].T).reshape(b,s,nq,d).transpose(0,2,1,3)
        k = (z @ w[a+'k_proj.weight'].T).reshape(b,s,nkv,d).transpose(0,2,1,3)
        v = (z @ w[a+'v_proj.weight'].T).reshape(b,s,nkv,d).transpose(0,2,1,3)
        q = QWEN.rope(plan.norm(q,w[a+'q_norm.weight'],0.,eps,center=False,label='qk_variance'),c['rope_theta'])
        k = QWEN.rope(plan.norm(k,w[a+'k_norm.weight'],0.,eps,center=False,label='qk_variance'),c['rope_theta'])
        k, v = np.repeat(k,nq//nkv,axis=1), np.repeat(v,nq//nkv,axis=1)
        probs = plan.softmax(q @ k.swapaxes(-1,-2) / np.sqrt(d), np.tril(np.ones((s,s),dtype=bool)))
        attended = (probs @ v).transpose(0,2,1,3).reshape(b,s,nq*d)
        x = x + attended @ w[a+'o_proj.weight'].T
        z = plan.norm(x,w[p+'post_attention_layernorm.weight'],0.,eps,center=False,label='residual_variance')
        m = p+'mlp.'
        activated = plan.apply('silu',z @ w[m+'gate_proj.weight'].T)
        x = x + (activated * (z @ w[m+'up_proj.weight'].T)) @ w[m+'down_proj.weight'].T
    last = plan.norm(x[:,-1,:],w['model.norm.weight'],0.,eps,center=False,label='residual_variance')
    return last @ w['model.embed_tokens.weight'].T


def run(problem, degrees, exact=False):
    reference, path = (BERT,'p2_bert') if problem == 'bert' else (QWEN,'p4_qwen')
    plan = Plan(problem, degrees, exact)
    cases = []
    for source in sorted((ROOT/'agent_pset'/path/'test').glob('test*.in')):
        case = json.loads(source.read_text())
        args = reference.make_case(case)
        w, x, expected = args[0], args[1], args[-1]
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            actual = (bert_forward(case['config'],w,x,args[2],plan) if problem=='bert'
                      else qwen_forward(case['config'],w,x,plan))
        try:
            errors = metrics(actual,expected)
            if problem == 'bert':
                numeric = bool(np.all(np.abs(actual-expected) <= .02+.01*np.abs(expected)))
                predictions = bool(np.array_equal(actual[:,1]>=actual[:,0],expected[:,1]>=expected[:,0]))
                passed = numeric and predictions
            else:
                passed = errors['nrmse'] <= .10 and errors['max_abs_error'] <= .20
            cases.append(dict(case=source.name,passed=passed,**errors))
        except ValueError as error:
            cases.append(dict(case=source.name,passed=False,error=str(error)))
    return dict(problem=problem,exact=exact,contracts=plan.contracts,ranges=plan.ranges,
                domain_violations=plan.violations,cases=cases,
                all_numeric_gates_pass=all(c['passed'] for c in cases),
                within_planned_domains=not plan.violations)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'validation/transformer_polynomial_plan.json')
    args=parser.parse_args()
    source_paths = (Path(__file__).resolve(), ROOT / 'common/encrypted_transformer.py',
                    ROOT / 'common/encrypted_polynomial.py')
    source_before = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in source_paths}
    reports=[run(problem,{},exact=True) for problem in ('bert','qwen')]
    for problem in ('bert','qwen'):
        for inv in ((2,4,8) if problem=='bert' else (12,16,24,32)):
            for activation in (2,4,8,12):
                reports.append(run(problem,{'inverse_sqrt':inv,'gelu':activation,'silu':activation,
                                            'tanh':activation,'exp':12,'reciprocal':24}))
    softmax_sweep = []
    for problem in ('bert', 'qwen'):
        for exp_degree in (2, 4, 6, 8, 12):
            for reciprocal_degree in (2, 4, 6, 8, 12, 16, 24):
                softmax_sweep.append(run(problem, {
                    'inverse_sqrt': 2 if problem == 'bert' else 12,
                    'gelu': 8, 'tanh': 8, 'silu': 2,
                    'exp': exp_degree, 'reciprocal': reciprocal_degree}))
    current_runtime = [run(problem, {}) for problem in ('bert', 'qwen')]
    historical_alternatives = [run('bert',dict(inverse_sqrt=4,gelu=8,tanh=8,exp=8,reciprocal=8)),
                               run('qwen',dict(inverse_sqrt=16,silu=4,exp=8,reciprocal=8))]
    statistics = []
    seen = set()
    for report in reports + softmax_sweep + current_runtime + historical_alternatives:
        for name, (domain, degree) in report['contracts'].items():
            key = name, tuple(domain), degree
            if key not in seen:
                seen.add(key)
                statistics.append(approximation_statistics(*key))
    output={'scope':'Plaintext-only circuit planning. No HE execution, refresh, scoring baseline or certified domain bounds.',
            'method':'Analytic Chebyshev interpolation, public mask and mean-centered softmax; no fixture-output fitting.',
            'source_sha256': source_before,
            'low_depth_cautions': 'Passing toy logit gates alone is insufficient: degree4/6 exp on[-4,4] becomes negative; degree2 reciprocal on[.5,8] has sampled41% relative error. The candidate plans use positive exponential approximations on the sampled domain. All sampled bounds still require independent certification or domain enforcement.',
            'reports':reports,'softmax_sweep':softmax_sweep,
            'current_runtime_contract_plans':current_runtime,
            'historical_alternative_degree_plans':historical_alternatives,
            'contract_plan_note':'Current runtime contracts are read from public_toy_approximations. Historical alternatives preserve prior degree choices for comparison; none of these plaintext runs is encrypted validation.',
            'softmax_observation_scope':'One complete padded PairLayout period, including masked-slot polynomial inputs; unused full-ring slot periods repeat these values.',
            'approximation_statistics':statistics,
            'current_runtime_approximation_statistics':runtime_approximation_statistics(),
            'power_schedule_note':'Degree-sweep statistics retain the normalized affine-map schedule. Current runtime statistics also identify input-power quadratics, which fold the domain scale into coefficients and save one level.'}
    output['source_unchanged_at_recording'] = source_before == {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths}
    if not output['source_unchanged_at_recording']:
        raise RuntimeError('runtime source changed during planning; repeat against frozen sources')
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    for report in reports:
        print(report['problem'],report['exact'],{k:v[1] for k,v in report['contracts'].items()},
              report['all_numeric_gates_pass'],report['within_planned_domains'],report['domain_violations'])
