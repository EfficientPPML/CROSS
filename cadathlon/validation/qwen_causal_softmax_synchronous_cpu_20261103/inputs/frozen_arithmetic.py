"""Strict float64 candidate circuit and explicit abstract level accounting.

Domain/nonfinite checks are validation instrumentation. They never replace
an invalid activation by an original-model value and are not server fallback.
"""
from collections import Counter
from dataclasses import dataclass
import math
import numpy as np
class InvalidCircuit(ValueError):
    def __init__(self,label,record):
        super().__init__('Invalid approximate trajectory at '+label);self.label=label;self.record=record
class FrozenPolynomial:
    def __init__(self,rule):self.rule=rule;self.domain=rule['domain'];self.coefficients=rule['chebyshev_coefficients']

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
        self.operators = {}
        self.blocks = []

    def node(self, data, depth, label):
        data = np.asarray(data, dtype=np.float64)
        if not np.isfinite(data).all():raise InvalidCircuit(label, {'nonfinite':int(np.sum(~np.isfinite(data)))})
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
        row=self.domains[label]
        if row['below'] or row['above'] or row['nonfinite']:raise InvalidCircuit(label,row)

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
        if strategy!=contract.rule['strategy']:raise ValueError('Frozen polynomial strategy mismatch')
        coefficients = np.asarray(contract.rule['evaluation_coefficients'])
        if strategy == 'power':
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

    def inverse(self, value, rule, label):
        self.observe(label, value.data, rule['domain'])
        result = self.affine(value, rule['rounded_first_weight'], rule['rounded_first_bias'], label+'.first')
        for iteration in range(1, rule['iterations']):
            product = self.multiply(value, result, label+'.product')
            residual = self.affine(product, -1., 2., label+'.residual')
            result = self.multiply(result, residual, label+'.update')
        return result

    def inverse_sqrt(self, value, rule, label):
        if rule['method']=='polynomial':return self.polynomial(value,FrozenPolynomial(rule),rule['strategy'],label)
        self.observe(label, value.data, rule['domain'])
        result=self.affine(value,rule['rounded_first_weight'],rule['rounded_first_bias'],label+'.first')
        for _ in range(1,rule['iterations']):
            square=self.multiply(result,result,label+'.square')
            product=self.multiply(value,square,label+'.product')
            residual=self.affine(product,-.5,1.5,label+'.residual')
            result=self.multiply(result,residual,label+'.update')
        return result

class QwenArithmetic(Arithmetic):
    def __init__(self, manifest):
        super().__init__();self.manifest=manifest

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
            base = self.polynomial(centered, FrozenPolynomial(self.manifest['exp']), 'power',
                row_label+'.exp', input_scale=2.**-rule['squarings'])
            denominator = self.reduce(base, -1, factor=1/(row+1), label=row_label+'.initial_mean')
            inverse = self.inverse(denominator, self.manifest['initial_reciprocal'], row_label+'.initial_inverse')
            probability = self.affine(self.multiply(base, inverse, row_label+'.initial_normalized'),
                1/(row+1), label=row_label+'.probability')
            for iteration in range(rule['squarings']):
                prefix = row_label+f'.round{iteration}'
                square = self.multiply(probability, probability, prefix+'.square')
                denominator = self.reduce(square, -1, label=prefix+'.sum')
                inverse = self.inverse(denominator, self.manifest['reciprocal_by_keys'][str(row+1)], prefix+'.inverse')
                probability = self.multiply(square, inverse, prefix+'.normalize')
            products = self.multiply(self.view(probability, probability.data[..., None]),
                self.view(projected, projected.data[:, :, :row+1]), row_label+'.pv')
            rows.append(self.reduce(products, -2, keepdims=False, label=row_label+'.attended'))
        return self.node(np.stack([row.data for row in rows], axis=-2), max(row.depth for row in rows), label+'.assembled')

    def silu(self, value, rule, label):
        if rule['method'] == 'polynomial':
            return self.polynomial(value, FrozenPolynomial(rule), rule['strategy'], label)
        self.observe(label, value.data, rule['domain'])
        result = self.polynomial(value, FrozenPolynomial(rule['base']),
            'power', label+'.tanh_base', input_scale=2.**-(rule['doublings']+1))
        for iteration in range(rule['doublings']):
            square = self.multiply(result, result, label+'.square')
            denominator = self.affine(square, 1., 1., label+'.denominator')
            inverse = self.inverse(denominator, rule['reciprocal'], label+f'.inverse{iteration}')
            result = self.affine(self.multiply(result, inverse, label+'.quotient'), 2., label=label+'.double')
        sigmoid = self.affine(result, .5, .5, label+'.sigmoid')
        return self.multiply(value, sigmoid, label+'.activated')


def approximate_forward(config, weights, embeddings, manifest, *, engine=None):
    engine = QwenArithmetic(manifest) if engine is None else engine
    value = Value(embeddings.copy())
    batch, sequence, _ = embeddings.shape
    nq, nkv, dimension = config['num_attention_heads'], config['num_key_value_heads'], config['head_dim']
    nonlinear, blocks, operators = manifest['nonlinear'], engine.blocks, engine.operators
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

