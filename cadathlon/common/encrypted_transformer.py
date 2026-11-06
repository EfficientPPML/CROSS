"""Experimental ciphertext tensor circuits, not yet a validated reference.

Client packing duplicates token activations into a query/key grid. Server
operators act only on ciphertexts; all NumPy arrays here are public weights,
positions and masks. There is no decrypt/refresh-client path. Full depth and
homomorphic bootstrap integration remain prerequisites for reference use.
"""
from dataclasses import dataclass
import math

import numpy as np

from encrypted_polynomial import nonlinear_approximation


@dataclass(frozen=True)
class PairLayout:
    batch: int
    sequence: int
    slots: int

    def __post_init__(self):
        if any(type(n) is not int or n < 1 for n in (self.batch, self.sequence, self.slots)):
            raise ValueError("positive integer packing dimensions required")
        if self.slots % self.period:
            raise ValueError("slot count must be a multiple of the padded query/key grid")

    @property
    def width(self):
        return 1 << (self.sequence - 1).bit_length()

    @property
    def batches(self):
        return 1 << (self.batch - 1).bit_length()

    @property
    def period(self):
        return self.batches * self.width ** 2

    @property
    def rotations(self):
        steps = [1 << bit for bit in range((self.width - 1).bit_length())]
        transposes = [d * (self.width - 1) for d in range(1, self.width)]
        return tuple(sorted({n for step in steps + transposes for n in (step, -step)}))

    def _tile(self, value):
        return np.tile(np.broadcast_to(value, (self.batches, self.width, self.width)).reshape(-1), self.slots // self.period)

    def pack_client_embeddings(self, embeddings):
        """Client-only input packing; no model computation beyond duplication."""
        value = np.asarray(embeddings, dtype=np.float64)
        if value.ndim != 3 or value.shape[:2] != (self.batch, self.sequence) or not np.isfinite(value).all():
            raise ValueError("expected finite (batch, sequence, features) client embeddings")
        batches = np.arange(self.batches) % self.batch
        positions = np.minimum(np.arange(self.width), self.sequence - 1)
        padded = value[batches[:, None], positions[None, :]]
        return [self._tile(padded[:, :, feature, None]) for feature in range(value.shape[-1])]

    def unpack_client_features(self, decoded):
        """Client-only extraction after decryption; selects key-grid column 0."""
        decoded = np.asarray(decoded, dtype=np.float64)
        if decoded.ndim != 2 or decoded.shape[1] != self.slots:
            raise ValueError("expected one full decoded slot vector per feature")
        grid = decoded[:, :self.period].reshape(-1, self.batches, self.width, self.width)
        return grid[:, :self.batch, :self.sequence, 0].transpose(1, 2, 0)

    def attention_mask(self, mask=None, *, causal=False):
        if mask is None:
            mask = np.ones((self.batch, self.sequence), dtype=bool)
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.batch, self.sequence) or not mask.any(axis=1).all():
            raise ValueError("invalid or fully masked attention input")
        padded = np.zeros((self.batches, self.width), dtype=bool)
        padded[:, :self.sequence] = mask[np.arange(self.batches) % self.batch]
        valid = np.broadcast_to(padded[:, None, :], (self.batches, self.width, self.width)).copy()
        if causal:
            valid &= np.arange(self.width)[None, None, :] <= np.arange(self.width)[None, :, None]
        if not valid.any(axis=-1).all():
            raise ValueError("every attention row must have at least one public valid key")
        counts = valid.sum(axis=-1, keepdims=True)
        return self._tile(valid).astype(np.float64), self._tile(counts).astype(np.float64)


class EncryptedTensorOps:
    def __init__(self, evaluator, layout):
        if evaluator.ctx.num_slots != layout.slots:
            raise ValueError("layout differs from evaluator slot count")
        self.evaluator, self.layout = evaluator, layout

    def checkpoint(self, values, label):
        checkpoint = getattr(self.evaluator, 'checkpoint', None)
        return checkpoint(values, label) if checkpoint is not None else values

    def linear(self, features, weight, bias=None, *, input_mask=None):
        weight = np.asarray(weight, dtype=np.float64)
        if weight.ndim != 2 or weight.shape[1] != len(features):
            raise ValueError("public linear weight shape differs from ciphertext features")
        if bias is None:
            bias = np.zeros(weight.shape[0])
        bias = np.asarray(bias, dtype=np.float64)
        if bias.shape != (weight.shape[0],):
            raise ValueError("public bias shape differs from outputs")
        if input_mask is not None:
            input_mask = np.asarray(input_mask, dtype=np.float64)
            if input_mask.shape != (self.layout.slots,):
                raise ValueError("linear input mask must have one public weight per slot")
        prepare = getattr(self.evaluator, '_ensure_capacity', None)
        if prepare is not None:
            # Every output row reuses these same encrypted input features.
            # A depleted input must be bootstrapped once for the matrix, not
            # independently for every weighted_sum row.
            features = prepare(features, 1)
        return [self.evaluator.weighted_sum(features, row if input_mask is None else
                                            [coefficient * input_mask for coefficient in row], bias=offset)
                for row, offset in zip(weight, bias)]

    def residual(self, left, right):
        if len(left) != len(right):
            raise ValueError("residual feature counts differ")
        # Both graph branches end at canonical scale/nsd=1. Real addition
        # aligns their modulus prefixes and rejects incompatible scales.
        return [self.evaluator.add(a, b) for a, b in zip(left, right)]

    def normalize(self, features, weight, epsilon, approximation, *, bias=None, center=False):
        normalized = self.evaluator.normalize(features, approximation, epsilon=epsilon, center=center)
        weight = np.asarray(weight, dtype=np.float64)
        bias = np.zeros(len(features)) if bias is None else np.asarray(bias, dtype=np.float64)
        if weight.shape != (len(features),) or bias.shape != weight.shape:
            raise ValueError("normalization affine weight shape mismatch")
        return [self.evaluator.weighted_sum([value], [gamma], bias=beta)
                for value, gamma, beta in zip(normalized, weight, bias)]

    def transpose_pairs(self, value):
        """Swap query/key axes using public diagonal masks and rotations."""
        width = self.layout.width
        offsets = np.arange(width)[None, :] - np.arange(width)[:, None]
        values, weights = [], []
        for offset in range(-(width - 1), width):
            values.append(self.evaluator.rotate(value, offset * (width - 1)))
            weights.append(self.layout._tile(offsets == offset).astype(np.float64))
        return self.evaluator.weighted_sum(values, weights)

    def sum_keys(self, value, *, factor=1.):
        """Sum keys and duplicate the row total into every key column."""
        result = value
        steps = [1 << bit for bit in range((self.layout.width - 1).bit_length())]
        for step in steps:
            result = self.evaluator.add(result, self.evaluator.rotate(result, step))
        first = self.layout._tile(np.arange(self.layout.width)[None, None, :] == 0)
        result = self.evaluator.weighted_sum([result], [first * factor])
        for step in steps:
            result = self.evaluator.add(result, self.evaluator.rotate(result, -step))
        return result

    def rope(self, features, head_dim, theta):
        if head_dim % 2 or len(features) % head_dim:
            raise ValueError("rotary features require even complete heads")
        positions = np.minimum(np.arange(self.layout.width), self.layout.sequence - 1)[None, :, None]
        output = []
        for base in range(0, len(features), head_dim):
            for index in range(head_dim):
                frequency = theta ** (-2 * (index % (head_dim // 2)) / head_dim)
                angles = positions * frequency
                cos, sin = self.layout._tile(np.cos(angles)), self.layout._tile(np.sin(angles))
                partner = index + head_dim // 2 if index < head_dim // 2 else index - head_dim // 2
                sign = -1 if index < head_dim // 2 else 1
                output.append(self.evaluator.weighted_sum([features[base + index], features[base + partner]], [cos, sign * sin]))
        return output

    def attention(self, query, key, value, *, heads, kv_heads, head_dim, mask, counts, exponential, reciprocal):
        if (len(query) != heads * head_dim or len(key) != kv_heads * head_dim
                or len(value) != len(key) or heads % kv_heads):
            raise ValueError("invalid grouped attention feature dimensions")
        key = [self.transpose_pairs(feature) for feature in key]
        value = [self.transpose_pairs(feature) for feature in value]
        outputs = []
        for head in range(heads):
            key_head = head // (heads // kv_heads)
            products = [self.evaluator.multiply(query[head * head_dim + d], key[key_head * head_dim + d])
                        for d in range(head_dim)]
            scores = self.evaluator.weighted_sum(products, [mask / math.sqrt(head_dim)] * head_dim)
            mean = self.sum_keys(scores, factor=1 / counts)
            centered = self.evaluator.add(scores, mean, subtract=True)
            exponentials = self.evaluator.evaluate(centered, exponential, output_weights=mask)
            denominator_mean = self.sum_keys(exponentials, factor=1 / counts)
            inverse = self.evaluator.evaluate(denominator_mean, reciprocal, output_weights=1 / counts)
            probabilities = self.evaluator.multiply(exponentials, inverse)
            for d in range(head_dim):
                attended = self.evaluator.multiply(probabilities, value[key_head * head_dim + d])
                outputs.append(self.sum_keys(attended))
        return outputs


def bert_ciphertext_forward(ops, features, config, weights, attention_mask, approximations):
    """BERT encoder, pooler and classifier entirely through CKKS operators.

    Returns logits at every query position. The client extracts CLS (position
    zero) after decryption. No input activations or intermediate slots reach
    this function as plaintext; only weights/mask/configuration are public.
    """
    mask, counts = ops.layout.attention_mask(attention_mask)
    h, heads = config['hidden_size'], config['num_attention_heads']
    if h % heads or len(features) != h:
        raise ValueError('invalid BERT feature/head dimensions')
    epsilon = config['layer_norm_eps']

    def linear(values, name):
        return ops.linear(values, weights[name + '.weight'], weights[name + '.bias'])

    def norm(values, name):
        return ops.normalize(values, weights[name + '.weight'], epsilon, approximations['inverse_sqrt'],
                             bias=weights[name + '.bias'], center=True)

    x = list(features)
    for layer in range(config['num_hidden_layers']):
        prefix = f'bert.encoder.layer.{layer}.'
        q, k, v = [linear(x, prefix + 'attention.self.' + name) for name in ('query', 'key', 'value')]
        ops.checkpoint([*q, *k, *v], f'BERT layer {layer}: projections ready')
        attended = ops.attention(q, k, v, heads=heads, kv_heads=heads, head_dim=h // heads,
            mask=mask, counts=counts, exponential=approximations['exp'], reciprocal=approximations['reciprocal'])
        x = norm(ops.residual(x, linear(attended, prefix + 'attention.output.dense')),
                 prefix + 'attention.output.LayerNorm')
        ops.checkpoint(x, f'BERT layer {layer}: attention/normalization ready')
        intermediate = [ops.evaluator.evaluate(value, approximations['gelu'])
                        for value in linear(x, prefix + 'intermediate.dense')]
        ops.checkpoint(intermediate, f'BERT layer {layer}: GELU ready')
        x = norm(ops.residual(x, linear(intermediate, prefix + 'output.dense')), prefix + 'output.LayerNorm')
        del q, k, v, attended, intermediate
        ops.checkpoint(x, f'BERT layer {layer}: output ready')
    # Only CLS reaches BERT's pooler. Zero the other query positions inside
    # this existing linear layer, avoiding an unnecessary tanh domain contract
    # on logits that the model never requests.
    cls_mask = ops.layout._tile(np.arange(ops.layout.width)[None, :, None] == 0)
    pooler = ops.linear(x, weights['bert.pooler.dense.weight'], weights['bert.pooler.dense.bias'], input_mask=cls_mask)
    pooled = [ops.evaluator.evaluate(value, approximations['tanh']) for value in pooler]
    return linear(pooled, 'classifier')


def qwen_ciphertext_forward(ops, features, config, weights, approximations):
    """Qwen3 decoder and vocabulary projection entirely through CKKS operators.

    Returns logits at every query position; client extraction chooses the last
    real input position after decryption. Single-token attention is exactly V,
    so its public-shape specialization omits irrelevant Q/K/softmax work.
    """
    h, heads, kv_heads, d = (config['hidden_size'], config['num_attention_heads'],
                              config['num_key_value_heads'], config['head_dim'])
    if len(features) != h or heads % kv_heads or d % 2:
        raise ValueError('invalid Qwen feature/head dimensions')
    epsilon = config['rms_norm_eps']
    mask, counts = ops.layout.attention_mask(causal=True)

    def linear(values, name):
        return ops.linear(values, weights[name + '.weight'])

    def norm(values, name):
        return ops.normalize(values, weights[name + '.weight'], epsilon, approximations['inverse_sqrt'])

    def norm_heads(values, name):
        return [value for start in range(0, len(values), d) for value in norm(values[start:start + d], name)]

    x = list(features)
    for layer in range(config['num_hidden_layers']):
        prefix = f'model.layers.{layer}.'
        attention = prefix + 'self_attn.'
        z = norm(x, prefix + 'input_layernorm')
        ops.checkpoint(z, f'Qwen layer {layer}: input normalization ready')
        v = linear(z, attention + 'v_proj')
        if ops.layout.sequence == 1:
            attended = [v[(head // (heads // kv_heads)) * d + index]
                        for head in range(heads) for index in range(d)]
        else:
            q = ops.rope(norm_heads(linear(z, attention + 'q_proj'), attention + 'q_norm'), d, config['rope_theta'])
            k = ops.rope(norm_heads(linear(z, attention + 'k_proj'), attention + 'k_norm'), d, config['rope_theta'])
            ops.checkpoint([*q, *k, *v], f'Qwen layer {layer}: projections ready')
            attended = ops.attention(q, k, v, heads=heads, kv_heads=kv_heads, head_dim=d,
                mask=mask, counts=counts, exponential=approximations['exp'], reciprocal=approximations['reciprocal'])
        x = ops.residual(x, linear(attended, attention + 'o_proj'))
        del z, v, attended
        if ops.layout.sequence != 1:
            del q, k
        ops.checkpoint(x, f'Qwen layer {layer}: attention ready')
        z = norm(x, prefix + 'post_attention_layernorm')
        ops.checkpoint(z, f'Qwen layer {layer}: post-attention normalization ready')
        gate = [ops.evaluator.evaluate(value, approximations['silu']) for value in linear(z, prefix + 'mlp.gate_proj')]
        ops.checkpoint(gate, f'Qwen layer {layer}: SiLU ready')
        up = linear(z, prefix + 'mlp.up_proj')
        gated = [ops.evaluator.multiply(left, right) for left, right in zip(gate, up)]
        x = ops.residual(x, linear(gated, prefix + 'mlp.down_proj'))
        del z, gate, up, gated
        ops.checkpoint(x, f'Qwen layer {layer}: MLP ready')
    x = norm(x, 'model.norm')
    ops.checkpoint(x, 'Qwen final normalization ready')
    return ops.linear(x, weights['model.embed_tokens.weight'])


@dataclass(frozen=True)
class _PlannedValue:
    level: int


class _DepthOnlyEvaluator:
    """Static level analysis; values contain no activation data at all."""
    def __init__(self, slots):
        from types import SimpleNamespace
        self.ctx = SimpleNamespace(num_slots=slots)

    def add(self, left, right, subtract=False):
        return _PlannedValue(min(left.level, right.level))

    def rotate(self, value, shift):
        return value

    def weighted_sum(self, values, weights, bias=0.):
        return _PlannedValue(min(value.level for value in values) - 1)

    def multiply(self, left, right):
        return _PlannedValue(min(left.level, right.level) - 1)

    @staticmethod
    def _depth(approximation):
        degree = len(approximation.coefficients) - 1
        return 2 - int(approximation.input_power_basis) + ((degree - 1).bit_length() if degree > 1 else 0)

    def evaluate(self, value, approximation, output_weights=1.):
        return _PlannedValue(value.level - self._depth(approximation))

    def normalize(self, features, approximation, epsilon, center=False):
        remaining = min(value.level for value in features) - self._depth(approximation) - 3 - int(center)
        return [_PlannedValue(remaining) for _ in features]


def plan_ciphertext_depth(model, config, weights, *, batch, sequence, attention_mask=None, approximations):
    """Analyze the exact graph's leveled depth without keys or activations."""
    slots = (1 << (batch - 1).bit_length()) * (1 << (sequence - 1).bit_length()) ** 2
    layout = PairLayout(batch, sequence, slots)
    ops = EncryptedTensorOps(_DepthOnlyEvaluator(slots), layout)
    inputs = [_PlannedValue(0) for _ in range(config['hidden_size'])]
    if model == 'bert':
        outputs = bert_ciphertext_forward(ops, inputs, config, weights, attention_mask, approximations)
    elif model == 'qwen':
        outputs = qwen_ciphertext_forward(ops, inputs, config, weights, approximations)
    else:
        raise ValueError('model must be bert or qwen')
    return {'depth': -min(value.level for value in outputs),
            'minimum_slots': slots, 'rotations': layout.rotations,
            'strategy': 'balanced normalized monomial powers', 'homomorphic_refresh_required': None}


def public_toy_approximations(model):
    """Numerically planned toy contracts; not pretrained-model domain bounds."""
    contracts = {
        'inverse_sqrt': ((.5, 1.5), 2) if model == 'bert' else ((.005, .35), 12),
        'exp': ((-4, 4), 8), 'reciprocal': ((.5, 8), 8),
        'gelu': ((-3, 3), 8), 'silu': ((-2, 2), 2), 'tanh': ((-2, 2), 8),
    }
    if model not in ('bert', 'qwen'):
        raise ValueError('model must be bert or qwen')
    from dataclasses import replace
    result = {name: nonlinear_approximation(name, domain, degree) for name, (domain, degree) in contracts.items()}
    result['silu'] = replace(result['silu'], input_power_basis=True)
    return result
