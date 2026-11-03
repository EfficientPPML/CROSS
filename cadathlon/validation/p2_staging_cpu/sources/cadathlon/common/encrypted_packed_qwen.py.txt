"""Experimental packed single-token Qwen; full encrypted validation pending.

The evaluator accepts only ciphertexts plus public model data. It has no
client callback or decrypt/encrypt path. F32 blocks repeat across all slots.
"""
import math
import time
import numpy as np
from encrypted_polynomial import EncryptedPolynomialEvaluator
from encrypted_rotation import CompiledRotation
from jaxite_word.bsgs import compute_bsgs_params


class MatrixDiagonals:
    def __init__(self, weight, slots, features=32):
        self.weight = np.asarray(weight, dtype=np.float64)
        self.dimension, self.features = slots, features
        if self.weight.ndim != 2:
            raise ValueError('packed linear weights must be a matrix')
        rows, cols = self.weight.shape
        if (not np.isfinite(self.weight).all()
                or min(rows, cols) < 1 or max(rows, cols) > features
                or slots < 2 * features or slots % features):
            raise ValueError('matrix or slot count does not fit packed features')
        self.active = tuple(sorted({d % slots for d in range(-(rows-1), cols)
                                   if np.any(np.diag(self.weight, k=d))}))
        self.n1, self.n2 = compute_bsgs_params(slots, len(self.active), 2.)

    @property
    def rotations(self):
        return ({d % self.n1 for d in self.active} |
                {(d // self.n1)*self.n1 for d in self.active}) - {0}

    def materialize_diagonals(self, indices):
        rows, cols = self.weight.shape
        feature = np.arange(self.dimension) % self.features
        result = {}
        for index in indices:
            displacement = index if index < self.features else index-self.dimension
            source = feature+displacement
            valid = (feature < rows) & (source >= 0) & (source < cols)
            diagonal = np.zeros(self.dimension)
            diagonal[valid] = self.weight[feature[valid], source[valid]]
            result[index] = diagonal
        return result


def validate_packed_config(config, *, batch, sequence_length):
    """Reject unsupported workloads before any expensive client key creation."""
    required = dict(hidden_size=16, intermediate_size=24, num_attention_heads=4,
                    num_key_value_heads=2, head_dim=8, num_hidden_layers=1, vocab_size=32)
    if batch != 1 or sequence_length != 1 or any(config.get(k) != v for k, v in required.items()):
        raise ValueError('experimental F32 packing supports only the one-token public Qwen configuration')
    if (config.get('attention_bias', False) or not config.get('tie_word_embeddings', True)
            or config.get('hidden_act', 'silu') != 'silu'):
        raise ValueError('packed Qwen requires bias-free attention, tied embeddings and SiLU')
    epsilon = config.get('rms_norm_eps')
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('packed RMSNorm requires finite positive epsilon')


def packed_rotation_indices(config, weights, slots):
    """Public BSGS and segmented-normalization key inventory."""
    validate_packed_config(config, batch=1, sequence_length=1)
    prefix = 'model.layers.0.'
    attention, mlp = prefix + 'self_attn.', prefix + 'mlp.'
    projected = weights[attention + 'v_proj.weight']
    expanded = np.repeat(np.asarray(projected).reshape(2, 8, 16), 2, axis=0).reshape(32, 16)
    matrices = [expanded, weights[attention + 'o_proj.weight'], weights[mlp + 'gate_proj.weight'],
                weights[mlp + 'up_proj.weight'], weights[mlp + 'down_proj.weight'],
                weights['model.embed_tokens.weight']]
    rotations = {sign * (1 << bit) % slots for sign in (-1, 1) for bit in range(5)}
    for weight in matrices:
        rotations.update(MatrixDiagonals(weight, slots).rotations)
    return tuple(sorted(rotations))


class SingleTokenFeatureLayout:
    """Client-only codec for a repeated 32-slot feature block."""
    def __init__(self, slots):
        if slots < 64 or slots % 32:
            raise ValueError('F32 packing requires at least two full feature blocks')
        self.slots, self.period = slots, 32

    def pack_client_embeddings(self, embeddings):
        values = np.asarray(embeddings, dtype=np.float64)
        if values.shape != (1, 1, 16) or not np.isfinite(values).all():
            raise ValueError('packed client input must have shape (1, 1, 16) and finite values')
        return [np.tile(np.pad(values.reshape(-1), (0, 16)), self.slots // 32)]

    def unpack_client_features(self, features):
        if len(features) != 1 or np.shape(features[0]) != (self.slots,):
            raise ValueError('packed client output must be one full-slot ciphertext')
        return np.asarray(features[0])[:32].reshape(1, 1, 32)


class CompiledArithmetic(EncryptedPolynomialEvaluator):
    """Reviewed compiled linear/multiplication primitives plus rotations."""
    def __init__(self, context, *, low_memory=False):
        super().__init__(context, strategy='power', compiled_multiply=True,
                         compiled_linear=True, low_memory=low_memory)
        self.rotation = CompiledRotation(context)

    def rotate(self, value, shift):
        if type(shift) is not int:
            raise ValueError('rotation shift must be an integer')
        if shift % self.ctx.num_slots == 0:
            return value
        level = self._level(value)
        self._prepare(level)
        return self._finish(self.rotation.rotate(value, shift % self.ctx.num_slots))

    def _clear_working_caches(self):
        rotation = getattr(self, 'rotation', None)
        if rotation is not None:
            rotation.clear()
        super()._clear_working_caches()


class PackedQwen1:
    def __init__(self, context, *, low_memory=False):
        self.ctx = context
        self.features = 32
        if context.num_slots < 2 * self.features or context.num_slots % self.features:
            raise ValueError('F32 packing requires divisible slot count')
        self.arithmetic = CompiledArithmetic(context, low_memory=low_memory)
        self.phase_results = []

    def public_feature(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or len(values) > self.features or not np.isfinite(values).all():
            raise ValueError('invalid public feature vector')
        return np.tile(np.pad(values, (0, self.features-len(values))), self.ctx.num_slots//self.features)

    def linear(self, value, weight, bias=None):
        source = MatrixDiagonals(weight, self.ctx.num_slots, self.features)
        if bias is not None and np.shape(bias) != (source.weight.shape[0],):
            raise ValueError('packed linear bias shape differs from output rows')
        if not source.active:
            return self.arithmetic.weighted_sum([value], [0.],
                bias=0. if bias is None else self.public_feature(bias))
        level = self.arithmetic._level(value)
        if level < 1:
            raise ValueError('insufficient packed-linear level')
        self.arithmetic._prepare(level)
        divisor = math.prod(self.ctx._param_cache.q_moduli_at_level(level)[-2:])
        operation = self.ctx.bsgs_matvec[level, source.dimension, source.n1, source.n2]
        started = time.monotonic()
        try:
            operation.preprocess(source, pt_scale=self.arithmetic.scale*divisor/value._ckks_scale,
                                 active_diagonal_indices=source.active, memory_bounded=True, n_jobs=2)
            result = operation.matvec(value)
            result.polynomial.block_until_ready()
        finally:
            operation._release()
        if bias is not None:
            constant = self.arithmetic._encode_public(self.public_feature(bias), level-1, result._ckks_scale)
            result = self.ctx.he_add[level-1].add_plain(result, constant)
        self.phase_results.append({'operation':'packed-linear', 'shape':list(np.shape(weight)),
                                   'input_level':level, 'output_level':self.arithmetic._level(result),
                                   'elapsed_s':time.monotonic()-started})
        return self.arithmetic._finish(result)

    def reduce_features(self, value, *, factor=1., bias=0.):
        # Each block may differ. Anchor before broadcasting prevents cross-block
        # leakage from the unmasked rotate/add butterfly intermediates.
        result = value
        for bit in range(5):
            result = self.arithmetic.add(result, self.arithmetic.rotate(result, 1 << bit))
        anchor = (np.arange(self.ctx.num_slots) % self.features == 0).astype(float)
        result = self.arithmetic.weighted_sum([result], [anchor*factor], bias=anchor*bias)
        for bit in range(5):
            result = self.arithmetic.add(result, self.arithmetic.rotate(result, -(1 << bit)))
        return result

    def normalize(self, value, gamma, epsilon, inverse_sqrt):
        squares = self.arithmetic.multiply(value, value)
        variance = self.reduce_features(squares, factor=1/len(gamma), bias=epsilon)
        del squares
        inverse = self.arithmetic.evaluate(variance, inverse_sqrt)
        del variance
        normalized = self.arithmetic.multiply(value, inverse)
        return self.arithmetic.weighted_sum([normalized], [self.public_feature(gamma)])

    def evaluate(self, value, config, weights, approximations):
        validate_packed_config(config, batch=1, sequence_length=1)
        heads, kv_heads, dim = config['num_attention_heads'], config['num_key_value_heads'], config['head_dim']
        if heads*dim > self.features or heads % kv_heads:
            raise ValueError('invalid packed head layout')
        prefix = 'model.layers.0.'
        attention = prefix+'self_attn.'
        mlp = prefix+'mlp.'
        def norm(x, name):
            return self.normalize(x, weights[name+'.weight'], config['rms_norm_eps'], approximations['inverse_sqrt'])
        def residual(x, y):
            return self.arithmetic.add(x, y)
        hidden = value
        z = norm(hidden, prefix+'input_layernorm')
        self.arithmetic.checkpoint(z, 'packed input normalization ready')
        projected = weights[attention+'v_proj.weight']
        # S1 GQA expansion is public row duplication inside this projection.
        expanded = np.concatenate([projected[(head//(heads//kv_heads))*dim:(head//(heads//kv_heads)+1)*dim]
                                   for head in range(heads)])
        attended = self.linear(z, expanded)
        hidden = residual(hidden, self.linear(attended, weights[attention+'o_proj.weight']))
        self.arithmetic.checkpoint(hidden, 'packed attention ready')
        del z, attended
        z = norm(hidden, prefix+'post_attention_layernorm')
        self.arithmetic.checkpoint(z, 'packed post-attention normalization ready')
        gate = self.arithmetic.evaluate(self.linear(z, weights[mlp+'gate_proj.weight']), approximations['silu'],
                                       output_weights=self.public_feature(np.ones(config['intermediate_size'])))
        self.arithmetic.checkpoint(gate, 'packed SiLU ready')
        up = self.linear(z, weights[mlp+'up_proj.weight'])
        hidden = residual(hidden, self.linear(self.arithmetic.multiply(gate, up), weights[mlp+'down_proj.weight']))
        self.arithmetic.checkpoint(hidden, 'packed MLP ready')
        del z, gate, up
        hidden = norm(hidden, 'model.norm')
        self.arithmetic.checkpoint(hidden, 'packed final normalization ready')
        return self.linear(hidden, weights['model.embed_tokens.weight'])
