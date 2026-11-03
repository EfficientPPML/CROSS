"""Experimental packed refresh schedule with conditional ideal-value bounds.

Every bound comes from public weights, layout algebra, or a declared
polynomial input domain. These are not bounds on accumulated CKKS error.
Complete encrypted case gates remain required; no plaintext fallback exists.
"""
import math
from fractions import Fraction
from functools import lru_cache
import numpy as np

from encrypted_packed_transformer import PackedTransformer, _power_two
from encrypted_bounded_arithmetic import (BoundedArithmetic, normalized_inverse_bound,
    polynomial_envelope, reciprocal_ratio_bound)


def _upper_float(value):
    result = float(value)
    while Fraction(result) < value:
        result = math.nextafter(result, math.inf)
    return result


def _upper_sqrt(value):
    value = Fraction(value)
    if value < 0:
        raise ValueError('nonnegative public squared norm required')
    if value == 0:
        return 0.
    # Scaling the rational before conversion avoids float(value) underflow
    # even when its square root is well within binary64's normal range.
    exponent = (value.numerator.bit_length()-value.denominator.bit_length())//2
    shift = 2*exponent
    scaled = value/(1 << shift) if shift >= 0 else value*(1 << -shift)
    try:
        result = math.ldexp(math.sqrt(float(scaled)), exponent)
    except OverflowError as error:
        raise ValueError('public norm exceeds floating-point capacity') from error
    while Fraction(result)**2 < value:
        result = math.nextafter(result, math.inf)
        if not math.isfinite(result):
            raise ValueError('public norm exceeds floating-point capacity')
    return result


def _product(*values):
    result = Fraction(1)
    for value in values:
        result *= Fraction(value)
    return _upper_float(result)


def _norm(values):
    return _upper_sqrt(sum(Fraction(float(x))**2 for x in values))


def _matrix_norm_bound(matrix):
    gram = [[sum(x*y for x,y in zip(left,right)) for right in matrix] for left in matrix]
    gershgorin = max(sum(abs(x) for x in row) for row in gram)
    frobenius = sum(x*x for row in matrix for x in row)
    return _upper_sqrt(min(gershgorin, frobenius))


@lru_cache(maxsize=128)
def _affine_bounds(rows, bias, gamma, beta, energy):
    # Public binary-rational products, with outward float/sqrt conversion.
    w = [[Fraction(float(x)) for x in row] for row in rows]
    gamma = [Fraction(float(x)) for x in gamma]
    beta = [Fraction(float(x)) for x in beta]
    a = [[x*g for x,g in zip(row,gamma)] for row in w]
    c = [sum(x*b for x,b in zip(row,beta))+Fraction(float(offset)) for row,offset in zip(w,bias)]
    row_l2 = max(_upper_sqrt(sum(x*x for x in row)) for row in a)
    coordinate = _upper_float(Fraction(energy)*Fraction(row_l2)+max(map(abs,c)))
    energy_out = _upper_float(Fraction(energy)*Fraction(_matrix_norm_bound(a))
                              + Fraction(_upper_sqrt(sum(x*x for x in c))))
    return coordinate, energy_out


def public_affine_bounds(weight, bias, gamma, beta, energy):
    weight = np.asarray(weight, dtype=float)
    bias = np.zeros(weight.shape[0]) if bias is None else np.asarray(bias, dtype=float)
    return _affine_bounds(tuple(map(tuple,weight)), tuple(bias), tuple(gamma), tuple(beta), float(energy))


class BoundedPackedTransformer(PackedTransformer):
    def __init__(self, context, layout, refresher, *, low_memory=True):
        super().__init__(context, layout, low_memory=low_memory)
        self.arithmetic = BoundedArithmetic(context, refresher, low_memory=low_memory)
        self._head_contract = None

    def contract(self, value, bound, reason):
        return self.arithmetic.declare_bound(value, bound, reason)

    def linear(self, value, weight, bias=None, *, input_mask=None):
        weight = np.asarray(weight, dtype=float)
        coordinate = self.arithmetic.public_bound(value)
        energy = getattr(value, '_public_row_l2_bound', _upper_float(Fraction(_upper_sqrt(Fraction(weight.shape[1])))*Fraction(coordinate)))
        mask_bound = 1. if input_mask is None else float(np.max(np.abs(input_mask)))
        coordinate = _upper_float(Fraction(coordinate)*Fraction(mask_bound))
        energy = _upper_float(Fraction(energy)*Fraction(mask_bound))
        affine = getattr(value, '_public_normalization_affine', None)
        if affine is not None and len(affine[0]) == weight.shape[1]:
            gamma, beta, underlying_energy = affine
            if input_mask is None:
                # Keep the known normalization affine inside each public map.
                # This avoids separately maximizing correlated gamma/beta.
                effective = gamma, beta, underlying_energy
            else:
                effective = np.ones(weight.shape[1]), np.zeros(weight.shape[1]), energy
        else:
            effective = np.ones(weight.shape[1]), np.zeros(weight.shape[1]), energy
        bound, output_energy = public_affine_bounds(weight, bias, *effective)
        row_l1 = max(sum(Fraction(abs(float(x))) for x in row) for row in weight)
        triangle = _upper_float(row_l1*Fraction(coordinate) + Fraction(0. if bias is None else float(np.max(np.abs(bias)))))
        bound = min(bound, triangle)
        result = super().linear(value, weight, bias, input_mask=input_mask)
        self.arithmetic.forget_relations(result)
        result = self.contract(result, bound, 'public row-L1/row-L2 matrix bound plus public bias')
        result._public_row_l2_bound = output_energy
        if self._head_contract is not None:
            heads, dim = self._head_contract
            padded = _power_two(dim)
            if weight.shape[0] == heads*padded:
                bounds = []
                for head in range(heads):
                    block = slice(head*padded, head*padded+dim)
                    bounds.append(public_affine_bounds(weight[block], None if bias is None else bias[block], *effective)[1])
                result._public_head_l2_bounds = tuple(bounds)
        return result

    def segmented_sum(self, value, length, stride, *, factor=1., bias=0., group_l1_bound=None):
        if _power_two(length) != length or type(stride) is not int or stride < 1 or self.layout.slots % (length*stride):
            raise ValueError('segmented reduction must partition the packed slots')
        coordinate = self.arithmetic.public_bound(value)
        maximum_group_l1 = _product(length, coordinate)
        group_l1 = maximum_group_l1 if group_l1_bound is None else min(maximum_group_l1, group_l1_bound)
        # Prepare once, before creating rotated aliases. Refreshing every
        # rotated copy separately is unnecessary and can dominate the graph.
        (value,) = self.arithmetic._ensure_capacity((value,), 1)
        result = value
        steps = [stride*(1 << bit) for bit in range((length-1).bit_length())]
        for step in steps:
            result = self.arithmetic.add(result, self.arithmetic.rotate(result, step))
        # Before the anchor mask, a sliding window intersects up to TWO
        # aligned groups. A bound on a signed group sum is insufficient here;
        # the caller must bound the sum of absolute values within each group.
        result = self.contract(result, min(maximum_group_l1, _product(2, group_l1)),
                               'sliding window spans at most two groups with a public absolute-sum bound')
        anchor = ((np.arange(self.layout.slots)//stride) % length == 0).astype(float)
        result = self.arithmetic.weighted_sum([result], [anchor*factor], bias=anchor*bias)
        for step in steps:
            result = self.arithmetic.add(result, self.arithmetic.rotate(result, -step))
        # Broadcast copies have disjoint support. Their magnitudes do not add.
        bound = _upper_float(Fraction(group_l1)*Fraction(float(np.max(np.abs(factor))))
                             + Fraction(float(np.max(np.abs(bias)))))
        return self.contract(result, bound, 'masked aligned reduction followed by disjoint broadcast')

    def normalize(self, value, gamma, epsilon, approximation, *, bias=None, center=False):
        gamma = np.asarray(gamma, dtype=float)
        h = len(gamma)
        if not math.isfinite(epsilon) or epsilon <= 0 or h > self.features:
            raise ValueError('invalid packed normalization contract')
        if center:
            # Both branches must use the same prepared ciphertext; otherwise
            # linear's internal refresh leaves the subtraction with a depleted
            # original and can require refreshing the centered value again.
            (value,) = self.arithmetic._ensure_capacity((value,), 1)
            mean = self.linear(value, np.full((h, h), 1/h))
            value = self.arithmetic.add(value, mean, subtract=True)
        # Use the actual binary64 public mean coefficient, rather than
        # assuming that 1/h is represented exactly (e.g. h=24).
        inverse_mean_weight = Fraction(1)/Fraction(1/h)
        variance_energy = inverse_mean_weight*Fraction(approximation.domain[1])
        value = self.contract(value, _upper_sqrt(variance_energy),
                              'declared variance domain bounds centered/RMS coordinate energy')
        (value,) = self.arithmetic._ensure_capacity((value,), 2)
        squared = self.arithmetic.multiply(value, value)
        variance = self.segmented_sum(squared, self.features, 1, factor=1/h, bias=epsilon,
                                      group_l1_bound=_upper_float(variance_energy))
        del squared
        inverse = self.arithmetic.evaluate(variance, approximation)
        del variance
        normalized = self.arithmetic.multiply(value, inverse)
        ratio = normalized_inverse_bound(approximation)
        normalized_energy = _upper_sqrt(inverse_mean_weight*Fraction(ratio)**2)
        normalized = self.contract(normalized, normalized_energy,
                                    'normalization energy inequality and public polynomial ratio envelope')
        result = self.arithmetic.weighted_sum([normalized], [self.layout.feature(gamma)],
                      bias=0. if bias is None else self.layout.feature(bias))
        result._public_row_l2_bound = _upper_float(Fraction(normalized_energy)*Fraction(float(np.max(np.abs(gamma))))
                                                 + Fraction(0. if bias is None else _norm(bias)))
        result._public_normalization_affine = (tuple(gamma), tuple(np.zeros(h) if bias is None else bias), normalized_energy)
        return result

    def normalize_heads(self, value, gamma, epsilon, approximation, heads):
        gamma = np.asarray(gamma, dtype=float)
        d, padded = len(gamma), _power_two(len(gamma))
        active = self.head_mask(heads, d, include_padding=True)
        inverse_mean_weight = Fraction(1)/Fraction(1/d)
        variance_energy = inverse_mean_weight*Fraction(approximation.domain[1])
        value = self.contract(value, _upper_sqrt(variance_energy),
                              'declared head-variance domain bounds head coordinate energy')
        (value,) = self.arithmetic._ensure_capacity((value,), 2)
        squared = self.arithmetic.multiply(value, value)
        dummy = sum(approximation.domain)/2
        variance = self.segmented_sum(squared, padded, 1, factor=active/d,
            bias=active*epsilon+(1-active)*dummy, group_l1_bound=_upper_float(variance_energy))
        del squared
        inverse = self.arithmetic.evaluate(variance, approximation, output_weights=active)
        del variance
        normalized = self.arithmetic.multiply(value, inverse)
        del inverse
        ratio = normalized_inverse_bound(approximation)
        normalized_energy = _upper_sqrt(inverse_mean_weight*Fraction(ratio)**2)
        normalized = self.contract(normalized, normalized_energy,
                                    'head normalization energy inequality and public polynomial ratio envelope')
        affine = np.zeros(self.features)
        for head in range(heads):
            affine[head*padded:head*padded+d] = gamma
        result = self.arithmetic.weighted_sum([normalized], [self.layout.feature(affine)])
        energy = _product(normalized_energy, float(np.max(np.abs(gamma))))
        result._public_head_l2_bounds = (energy,)*heads
        result._public_row_l2_bound = _upper_sqrt(Fraction(heads)*Fraction(energy)**2)
        return result

    def transpose_pairs(self, value):
        bound = self.arithmetic.public_bound(value)
        (prepared,) = self.arithmetic._ensure_capacity((value,), 1)
        result = self.contract(super().transpose_pairs(prepared), bound, 'public query/key permutation')
        for name in ('_public_row_l2_bound', '_public_head_l2_bounds'):
            if hasattr(value, name): setattr(result, name, getattr(value, name))
        return result

    def rope(self, value, heads, head_dim, theta):
        # Bound the exact binary64 public rotation coefficients. They are
        # close to orthogonal but cos(theta)^2+sin(theta)^2 need not equal 1.
        frequencies = theta ** (-2*np.arange(head_dim//2)/head_dim)
        positions = np.minimum(np.arange(self.layout.width), self.layout.sequence-1)
        angles = positions[:, None]*frequencies[None, :]
        margin = _upper_sqrt(max(Fraction(float(c))**2+Fraction(float(s))**2
                                 for c,s in zip(np.cos(angles).flat, np.sin(angles).flat)))
        head_energy = getattr(value, '_public_head_l2_bounds', None)
        bound = _product(margin, max(head_energy)) if head_energy is not None else _upper_sqrt(
            Fraction(2)*Fraction(self.arithmetic.public_bound(value))**2*Fraction(margin)**2)
        (prepared,) = self.arithmetic._ensure_capacity((value,), 1)
        result = self.contract(super().rope(prepared, heads, head_dim, theta), bound,
                               'public rotary coefficient operator-norm bound')
        if head_energy is not None:
            result._public_head_l2_bounds = tuple(_product(x, margin) for x in head_energy)
        if hasattr(value, '_public_row_l2_bound'):
            result._public_row_l2_bound = _product(value._public_row_l2_bound, margin)
        return result

    def attention(self, query, key, value, *, heads, head_dim, mask, counts, approximations):
        padded = _power_two(head_dim)
        active = self.head_mask(heads, head_dim, include_padding=True)
        mask = mask*active
        q_energy = getattr(query, '_public_head_l2_bounds', (_upper_sqrt(Fraction(head_dim)*Fraction(self.arithmetic.public_bound(query))**2),)*heads)
        k_energy = getattr(key, '_public_head_l2_bounds', (_upper_sqrt(Fraction(head_dim)*Fraction(self.arithmetic.public_bound(key))**2),)*heads)
        product_l1 = max(_upper_float(Fraction(q)*Fraction(k)) for q,k in zip(q_energy,k_energy))
        score_bound = _upper_float(Fraction(product_l1)*Fraction(1/math.sqrt(head_dim)))
        value_bound = self.arithmetic.public_bound(value)
        positive = polynomial_envelope(approximations['exp'])[0] >= 0
        ratio = max(reciprocal_ratio_bound(approximations['reciprocal'], float(1/count))
                    for count in np.unique(counts)) if positive else None
        key, value = self.transpose_pairs(key), self.transpose_pairs(value)
        products = self.arithmetic.multiply(query, key)
        scores = self.segmented_sum(products, padded, 1, factor=mask/math.sqrt(head_dim), group_l1_bound=product_l1)
        del products, key
        scores = self.contract(scores, score_bound, 'Cauchy head-product bound divided by sqrt(actual head width)')
        # Keep mean and centering on one prepared score value. Preparing
        # before sliding sums also uses the tighter aligned-score contract.
        (scores,) = self.arithmetic._ensure_capacity((scores,), 1)
        mean = self.segmented_sum(scores, self.layout.width, self.features, factor=1/counts,
                                   group_l1_bound=_product(float(np.max(counts)), score_bound))
        averaging_gain = max(Fraction(float(count))*Fraction(float(1/count)) for count in np.unique(counts))
        mean = self.contract(mean, _upper_float(Fraction(score_bound)*averaging_gain),
                             'valid-key average with actual binary64 public mean weights')
        centered = self.arithmetic.add(scores, mean, subtract=True)
        exponentials = self.arithmetic.evaluate(centered, approximations['exp'], output_weights=mask)
        del scores, mean, centered
        denominator = self.segmented_sum(exponentials, self.layout.width, self.features,
            factor=1/counts, bias=1-active,
            group_l1_bound=(_upper_float(Fraction(approximations['reciprocal'].domain[1]) /
                                       min(Fraction(float(1/count)) for count in np.unique(counts))) if positive else None))
        inverse = self.arithmetic.evaluate(denominator, approximations['reciprocal'], output_weights=active/counts)
        del denominator
        probabilities = self.arithmetic.multiply(exponentials, inverse)
        del exponentials, inverse
        if positive:
            probabilities = self.contract(probabilities, ratio,
                'positive exponentials and declared denominator imply probability/row-L1 ratio envelope')
        attended = self.arithmetic.multiply(probabilities, value)
        del probabilities, value
        result = self.segmented_sum(attended, self.layout.width, self.features,
                                     group_l1_bound=_product(value_bound, ratio) if positive else None)
        return result if not positive else self.contract(result, _product(value_bound, ratio),
                  'softmax row-L1 envelope times public value bound')

    def evaluate(self, value, model, config, weights, approximations, attention_mask=None):
        heads = config['num_attention_heads']
        dim = config['head_dim'] if model == 'qwen' else config['hidden_size']//heads
        self._head_contract = heads, dim
        if model == 'qwen':
            table = np.asarray(weights['model.embed_tokens.weight'])
            coordinate = float(np.max(np.abs(table)))
            energy = max(_upper_sqrt(sum(Fraction(float(x))**2 for x in row)) for row in table)
        else:
            gamma = np.asarray(weights['bert.embeddings.LayerNorm.weight'])
            beta = np.asarray(weights['bert.embeddings.LayerNorm.bias'])
            normalized_energy = _upper_float(Fraction(_upper_sqrt(Fraction(config['hidden_size'])))*Fraction(float(np.max(np.abs(gamma)))))
            coordinate = _upper_float(Fraction(normalized_energy)+Fraction(float(np.max(np.abs(beta)))))
            energy = _upper_float(Fraction(normalized_energy)+Fraction(_upper_sqrt(sum(Fraction(float(x))**2 for x in beta))))
        value = self.contract(value, coordinate, 'public embedding table or client LayerNorm energy inequality')
        value._public_row_l2_bound = energy
        if model == 'bert':
            value._public_normalization_affine = (tuple(gamma), tuple(beta), _upper_sqrt(Fraction(config['hidden_size'])))
        return super().evaluate(value, model, config, weights, approximations, attention_mask)
