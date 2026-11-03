"""Experimental ciphertext-only packed toy transformer graphs.

Activations use (batch, query, key, feature) slots, with public zero padding.
Client codecs are separate from the keyless arithmetic. Polynomial domains
are explicit toy contracts, not guarantees for pretrained checkpoints.
Full encrypted graph validation is required before any reference claim.
"""
from dataclasses import dataclass
from collections import deque
from contextlib import closing
import math
import numpy as np

from encrypted_packed_qwen import CompiledArithmetic, MatrixDiagonals


DEFAULT_EXTRA_PAYLOAD_BYTES = 256 * 1024**2
DEFAULT_MAX_PENDING = 8



def validate_rotation_strategy(strategy):
    if type(strategy) is not str or strategy not in ('direct','chained'):
        raise ValueError('rotation_strategy must be exactly direct or chained')
    return strategy


@dataclass(frozen=True)
class LinearRotationPlan:
    strategy: str
    baby_indices: tuple
    baby_steps: tuple
    giant_indices: tuple
    giant_steps: tuple
    final_giant: int
    required_keys: tuple
    nonzero_rotation_calls: int


def linear_rotation_plan(source, *, strategy='direct'):
    """Public actual gaps; sparse matrices need not share the dense3-key plan."""
    validate_rotation_strategy(strategy)
    babies=tuple(sorted({d % source.n1 for d in source.active}-{0}))
    giants=tuple(sorted({d // source.n1 * source.n1 for d in source.active},reverse=strategy=='chained'))
    if strategy=='direct':
        baby_steps=babies;giant_steps=giants;final=0
    else:
        baby_steps=tuple(index-previous for previous,index in zip((0,*babies),babies))
        giant_steps=tuple(previous-current for previous,current in zip(giants,giants[1:]))
        final=min(giants) if giants else 0
    shifts=(*baby_steps,*giant_steps,final)
    keys=tuple(sorted({shift % source.dimension for shift in shifts if shift % source.dimension}))
    calls=sum(bool(shift % source.dimension) for shift in shifts)
    return LinearRotationPlan(strategy,babies,baby_steps,giants,giant_steps,final,keys,calls)


def require_chained_rotation_keys(context, required):
    """Check initialized public key/map membership without generating/reading keys."""
    from collections.abc import Mapping
    cache=getattr(context,'_param_cache',None)
    if cache is None or context.secret_key is not None or cache.secret_key is not None or context.parameters.get('secret_key') is not None:
        raise ValueError('chained linear requires an initialized keyless context')
    if not required:return
    if getattr(cache,'per_level_rotation_keys',False):
        raise ValueError('chained linear requires supplied reusable public rotation keys')
    keys=getattr(cache,'raw_rotation_keys',None);maps=getattr(cache,'coef_maps',None)
    if not isinstance(keys,Mapping) or not isinstance(maps,Mapping):
        raise ValueError('chained linear requires public rotation-key/map inventories')
    for key in required:
        entry=keys.get(key)
        if (not isinstance(entry,Mapping) or entry.get('a') is None or entry.get('b') is None
                or maps.get(key) is None):
            raise ValueError(f'chained linear lacks supplied public key/map for rotation{key}')


def pending_limit(output_payload_bytes, groups, *, extra_payload_bytes=DEFAULT_EXTRA_PAYLOAD_BYTES,
                  max_pending=DEFAULT_MAX_PENDING):
    """Count one existing singleton payload plus a bounded extra live queue."""
    for name, value in [('groups', groups), ('max_pending', max_pending)]:
        if type(value) is not int or value < 1:
            raise ValueError(name + ' must be a positive integer')
    if type(extra_payload_bytes) is not int or extra_payload_bytes < 0:
        raise ValueError('extra payload budget must be a nonnegative integer')
    if output_payload_bytes is None:
        # Symbolic plaintext/depth planners have no physical ciphertext shape.
        return 1
    if type(output_payload_bytes) is not int or output_payload_bytes < 1:
        raise ValueError('output payload size must be a positive integer')
    return min(groups, max_pending, 1 + extra_payload_bytes // output_payload_bytes)


def output_payload_bytes(graph, value):
    """Exact public shape/dtype size after the weighted sum's one rescale."""
    payload = getattr(value, 'polynomial', None)
    cache = getattr(graph.ctx, '_param_cache', None)
    if payload is None or cache is None:
        return None
    level = graph.arithmetic._level(value)
    if level < 1:
        raise ValueError('insufficient packed-linear level')
    shape = tuple(payload.shape)
    if len(shape) < 2 or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError('invalid public ciphertext payload shape')
    output_limbs = cache.num_q_at_level(level - 1)
    if type(output_limbs) is not int or output_limbs < 1:
        raise ValueError('invalid output modulus count')
    return math.prod(shape[:-1]) * output_limbs * np.dtype(payload.dtype).itemsize


def giant_chunks(groups, make_inner, limit):
    """Yield in original order, releasing consumed queued wrappers promptly."""
    if type(limit) is not int or limit < 1:
        raise ValueError('positive pending limit required')
    iterator = iter(groups)
    while True:
        pending = deque()
        item = None
        try:
            for _ in range(limit):
                try:
                    giant, indices = next(iterator)
                except StopIteration:
                    break
                pending.append((giant, make_inner(giant, indices)))
            if not pending:
                return
            while pending:
                item = pending.popleft()
                yield item
                # The consumer also releases its inner/contribution locals.
                # Neither side may pin the preceding chunk during a new fill.
                item = None
        finally:
            item = None
            pending.clear()



def _power_two(n):
    if type(n) is not int or n < 1:
        raise ValueError('positive integer dimension required')
    return 1 << (n - 1).bit_length()


def packed_feature_width(model, config):
    if model not in ('bert', 'qwen'):
        raise ValueError('packed model must be bert or qwen')
    h, heads = config['hidden_size'], config['num_attention_heads']
    d = config['head_dim'] if model == 'qwen' else h // heads
    if model == 'bert' and h % heads:
        raise ValueError('BERT hidden width must divide its heads')
    output = config['vocab_size'] if model == 'qwen' else config['num_labels']
    return _power_two(max(h, config['intermediate_size'], output, heads * _power_two(d)))


@dataclass(frozen=True)
class PackedPairLayout:
    batch: int
    sequence: int
    features: int
    slots: int

    def __post_init__(self):
        b, s, f = _power_two(self.batch), _power_two(self.sequence), _power_two(self.features)
        if f != self.features or type(self.slots) is not int or self.slots < max(2*f, b*s*s*f) or self.slots % (b*s*s*f):
            raise ValueError('packed pair period must fit and divide the full slot count')

    @property
    def width(self):
        return _power_two(self.sequence)

    @property
    def padded_batch(self):
        return _power_two(self.batch)

    @property
    def period(self):
        return self.padded_batch * self.width**2 * self.features

    @property
    def shape(self):
        return self.padded_batch, self.width, self.width, self.features

    def tile(self, values):
        values = np.asarray(values)
        return np.tile(np.broadcast_to(values, self.shape).reshape(-1), self.slots // self.period)

    def feature(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or len(values) > self.features or not np.isfinite(values).all():
            raise ValueError('finite public feature vector must fit the packed width')
        return self.tile(np.pad(values, (0, self.features-len(values))))

    def pack_client_embeddings(self, embeddings):
        values = np.asarray(embeddings, dtype=np.float64)
        if (values.ndim != 3 or values.shape[:2] != (self.batch, self.sequence)
                or values.shape[2] > self.features or not np.isfinite(values).all()):
            raise ValueError('client embedding shape differs from packed layout')
        out = np.zeros(self.shape)
        rows = values[np.minimum(np.arange(self.padded_batch), self.batch-1)[:, None],
                      np.minimum(np.arange(self.width), self.sequence-1)[None, :]]
        out[..., :values.shape[2]] = rows[:, :, None, :]
        return [np.tile(out.reshape(-1), self.slots // self.period)]

    def unpack_client_features(self, features, output_features=None):
        if len(features) != 1 or np.shape(features[0]) != (self.slots,):
            raise ValueError('packed client output must contain one full slot vector')
        output_features = self.features if output_features is None else output_features
        if type(output_features) is not int or not 1 <= output_features <= self.features:
            raise ValueError('invalid packed output feature count')
        return np.asarray(features[0])[:self.period].reshape(self.shape)[:self.batch, :self.sequence, 0, :output_features]

    def attention_mask(self, attention_mask=None, *, causal=False):
        public = np.ones((self.batch, self.sequence), bool) if attention_mask is None else np.asarray(attention_mask, dtype=bool)
        if public.shape != (self.batch, self.sequence):
            raise ValueError('attention mask shape differs from packed layout')
        valid = np.zeros((self.padded_batch, self.width, self.width, 1), bool)
        valid[:, :, :self.sequence, 0] = public[np.minimum(np.arange(self.padded_batch), self.batch-1), None, :]
        if causal:
            query = np.minimum(np.arange(self.width), self.sequence-1)
            valid &= (np.arange(self.width)[None, :] <= query[:, None])[None, :, :, None]
        counts = valid.sum(axis=2, keepdims=True)
        if np.any(counts == 0):
            raise ValueError('fully masked attention rows are unsupported')
        return self.tile(valid).astype(float), self.tile(counts).astype(float)


class PackedTransformer:
    def __init__(self, context, layout, *, low_memory=False, preserve_dispatch_cache=False):
        if context.num_slots != layout.slots:
            raise ValueError('packed layout differs from initialized context slots')
        self.ctx, self.layout, self.features = context, layout, layout.features
        self.arithmetic = CompiledArithmetic(context, low_memory=low_memory,
            preserve_dispatch_cache=preserve_dispatch_cache)

    def linear(self, value, weight, bias=None, *, input_mask=None):
        """One-level sparse BSGS with one active rotation key at a time."""
        strategy=validate_rotation_strategy(getattr(self,'linear_rotation_strategy','direct'))
        source = MatrixDiagonals(weight, self.layout.slots, self.features)
        plan=linear_rotation_plan(source,strategy=strategy)
        if bias is not None and np.shape(bias) != (source.weight.shape[0],):
            raise ValueError('packed linear bias differs from output rows')
        if input_mask is not None:
            input_mask = np.asarray(input_mask, dtype=np.float64)
            if input_mask.shape != (self.layout.slots,) or not np.isfinite(input_mask).all():
                raise ValueError('packed linear mask must be a finite full slot vector')
        bias_slots = 0. if bias is None else self.layout.feature(bias)
        if strategy=='chained':
            if getattr(self.arithmetic,'refresher',None) is not None:
                raise ValueError('chained linear requires static capacity without implicit refresh')
            require_chained_rotation_keys(self.ctx,plan.required_keys)
        if not source.active:
            return self.arithmetic.weighted_sum([value], [0.], bias=bias_slots)
        (value,) = self.arithmetic._ensure_capacity((value,), 1)
        count_bytes = output_payload_bytes(self, value)
        babies = {0: value}
        previous=value
        for index,step in zip(plan.baby_indices,plan.baby_steps):
            babies[index]=self.arithmetic.rotate(previous if strategy=='chained' else value,step)
            if strategy=='chained':previous=babies[index]
        del previous
        groups = {}
        for diagonal in source.active:
            giant = diagonal // source.n1 * source.n1
            groups.setdefault(giant, []).append(diagonal)
        limit = pending_limit(count_bytes, len(groups),
            extra_payload_bytes=getattr(self, 'linear_extra_payload_bytes', DEFAULT_EXTRA_PAYLOAD_BYTES),
            max_pending=getattr(self, 'linear_max_pending', DEFAULT_MAX_PENDING))

        def make_inner(giant, indices):
            diagonals = source.materialize_diagonals(indices)
            values, weights = [], []
            for index in indices:
                diagonal = diagonals[index]
                if input_mask is not None:
                    # The mask belongs to the source coordinate of each
                    # diagonal, including masks that vary within a block.
                    diagonal = diagonal * np.roll(input_mask, -index)
                values.append(babies[index % source.n1])
                weights.append(np.roll(diagonal, giant))
            return self.arithmetic.weighted_sum(values, weights)

        result = None
        ordered=((giant,groups[giant]) for giant in plan.giant_indices)
        previous_giant=None
        with closing(giant_chunks(ordered, make_inner, limit)) as chunks:
            for giant, inner in chunks:
                if strategy=='chained':
                    if result is None:
                        result=inner
                    else:
                        rotated=self.arithmetic.rotate(result,previous_giant-giant)
                        result=self.arithmetic.add(rotated,inner)
                        del rotated
                    previous_giant=giant
                    del inner
                else:
                    contribution=self.arithmetic.rotate(inner,giant)
                    del inner
                    result=contribution if result is None else self.arithmetic.add(result,contribution)
                    del contribution
        if strategy=='chained':result=self.arithmetic.rotate(result,plan.final_giant)
        if bias is not None:
            level = self.arithmetic._level(result)
            self.arithmetic._prepare(level)
            constant = self.arithmetic._encode_public(bias_slots, level, result._ckks_scale)
            result = self.ctx.he_add[level].add_plain(result, constant)
        return self.arithmetic._finish(result)


    def segmented_sum(self, value, length, stride, *, factor=1., bias=0.):
        if _power_two(length) != length or type(stride) is not int or stride < 1 or self.layout.slots % (length*stride):
            raise ValueError('segmented reduction must partition the packed slots')
        result = value
        steps = [stride * (1 << bit) for bit in range((length-1).bit_length())]
        for step in steps:
            result = self.arithmetic.add(result, self.arithmetic.rotate(result, step))
        anchor = ((np.arange(self.layout.slots) // stride) % length == 0).astype(float)
        result = self.arithmetic.weighted_sum([result], [anchor * factor], bias=anchor * bias)
        for step in steps:
            result = self.arithmetic.add(result, self.arithmetic.rotate(result, -step))
        return result

    def normalize(self, value, gamma, epsilon, approximation, *, bias=None, center=False):
        gamma = np.asarray(gamma, dtype=float)
        h = len(gamma)
        if not math.isfinite(epsilon) or epsilon <= 0 or h > self.features:
            raise ValueError('invalid packed normalization contract')
        if center:
            # Mean only reaches actual hidden rows. Subtracting a mean from
            # zero padding would otherwise contaminate the variance.
            mean = self.linear(value, np.full((h, h), 1 / h))
            value = self.arithmetic.add(value, mean, subtract=True)
        squared = self.arithmetic.multiply(value, value)
        variance = self.segmented_sum(squared, self.features, 1, factor=1/h, bias=epsilon)
        del squared
        inverse = self.arithmetic.evaluate(variance, approximation)
        del variance
        normalized = self.arithmetic.multiply(value, inverse)
        return self.arithmetic.weighted_sum([normalized], [self.layout.feature(gamma)],
                    bias=0. if bias is None else self.layout.feature(bias))

    def head_mask(self, heads, head_dim, *, include_padding=False):
        padded = _power_two(head_dim)
        if heads * padded > self.features:
            raise ValueError('attention heads exceed the packed feature width')
        mask = np.zeros(self.features)
        for head in range(heads):
            mask[head*padded:head*padded+(padded if include_padding else head_dim)] = 1.
        return self.layout.feature(mask)

    def normalize_heads(self, value, gamma, epsilon, approximation, heads):
        gamma = np.asarray(gamma, dtype=float)
        d, padded = len(gamma), _power_two(len(gamma))
        active = self.head_mask(heads, d, include_padding=True)
        squared = self.arithmetic.multiply(value, value)
        # Inactive packed head groups receive a declared in-domain input,
        # not epsilon (which lies outside the inverse-sqrt contract).
        dummy = sum(approximation.domain) / 2
        variance = self.segmented_sum(squared, padded, 1, factor=active/d,
                                      bias=active*epsilon + (1-active)*dummy)
        del squared
        inverse = self.arithmetic.evaluate(variance, approximation, output_weights=active)
        del variance
        normalized = self.arithmetic.multiply(value, inverse)
        del inverse
        affine = np.zeros(self.features)
        for head in range(heads):
            affine[head*padded:head*padded+d] = gamma
        return self.arithmetic.weighted_sum([normalized], [self.layout.feature(affine)])

    def transpose_pairs(self, value):
        width = self.layout.width
        offset_grid = np.arange(width)[None, :] - np.arange(width)[:, None]
        values, weights = [], []
        for offset in range(-(width-1), width):
            values.append(self.arithmetic.rotate(value, offset * (width-1) * self.features))
            weights.append(self.layout.tile((offset_grid == offset)[None, :, :, None]).astype(float))
        return self.arithmetic.weighted_sum(values, weights)

    def rope(self, value, heads, head_dim, theta):
        if head_dim % 2 or not math.isfinite(theta) or theta <= 0:
            raise ValueError('RoPE requires even head width and positive theta')
        padded = _power_two(head_dim)
        index = np.arange(self.features)
        within = index % padded
        valid = (index < heads*padded) & (within < head_dim)
        frequencies = theta ** (-2 * (within % (head_dim//2)) / head_dim)
        positions = np.minimum(np.arange(self.layout.width), self.layout.sequence-1)
        angles = positions[None, :, None, None] * frequencies[None, None, None, :]
        cosine = self.layout.tile(np.cos(angles) * valid)
        sine = np.sin(angles)
        first = self.layout.tile(-sine * (valid & (within < head_dim//2)))
        second = self.layout.tile(sine * (valid & (within >= head_dim//2)))
        return self.arithmetic.weighted_sum([value, self.arithmetic.rotate(value, head_dim//2),
            self.arithmetic.rotate(value, -head_dim//2)], [cosine, first, second])

    def attention(self, query, key, value, *, heads, head_dim, mask, counts, approximations):
        padded = _power_two(head_dim)
        active = self.head_mask(heads, head_dim, include_padding=True)
        mask = mask * active
        key, value = self.transpose_pairs(key), self.transpose_pairs(value)
        products = self.arithmetic.multiply(query, key)
        scores = self.segmented_sum(products, padded, 1, factor=mask / math.sqrt(head_dim))
        del products, key
        mean = self.segmented_sum(scores, self.layout.width, self.features, factor=1/counts)
        centered = self.arithmetic.add(scores, mean, subtract=True)
        exponentials = self.arithmetic.evaluate(centered, approximations['exp'], output_weights=mask)
        del scores, mean, centered
        denominator = self.segmented_sum(exponentials, self.layout.width, self.features,
                                          factor=1/counts, bias=1-active)
        inverse = self.arithmetic.evaluate(denominator, approximations['reciprocal'], output_weights=active/counts)
        del denominator
        probabilities = self.arithmetic.multiply(exponentials, inverse)
        del exponentials, inverse
        attended = self.arithmetic.multiply(probabilities, value)
        del probabilities, value
        return self.segmented_sum(attended, self.layout.width, self.features)

    @staticmethod
    def head_projection(weight, bias, heads, head_dim, *, source_heads=None):
        weight = np.asarray(weight, dtype=float)
        source_heads = heads if source_heads is None else source_heads
        if weight.shape[0] != source_heads*head_dim or heads % source_heads:
            raise ValueError('invalid grouped head projection rows')
        padded = _power_two(head_dim)
        out = np.zeros((heads*padded, weight.shape[1]))
        offset = np.zeros(heads*padded) if bias is not None else None
        for head in range(heads):
            source = head // (heads//source_heads)
            out[head*padded:head*padded+head_dim] = weight[source*head_dim:(source+1)*head_dim]
            if bias is not None:
                offset[head*padded:head*padded+head_dim] = bias[source*head_dim:(source+1)*head_dim]
        return out, offset

    @staticmethod
    def head_output(weight, heads, head_dim):
        weight = np.asarray(weight, dtype=float)
        if weight.shape[1] != heads*head_dim:
            raise ValueError('attention output columns differ from heads')
        padded = _power_two(head_dim)
        result = np.zeros((weight.shape[0], heads*padded))
        for head in range(heads):
            result[:, head*padded:head*padded+head_dim] = weight[:, head*head_dim:(head+1)*head_dim]
        return result

    def evaluate(self, value, model, config, weights, approximations, attention_mask=None):
        if packed_feature_width(model, config) != self.features:
            raise ValueError('model feature width differs from the packed layout')
        return (self._bert(value, config, weights, approximations, attention_mask) if model == 'bert'
                else self._qwen(value, config, weights, approximations))

    def _bert(self, value, config, weights, approximations, attention_mask):
        h, heads = config['hidden_size'], config['num_attention_heads']
        d, epsilon = h//heads, config['layer_norm_eps']
        mask, counts = self.layout.attention_mask(attention_mask)
        def linear(x, name, **kw):
            return self.linear(x, weights[name+'.weight'], weights[name+'.bias'], **kw)
        def norm(x, name):
            return self.normalize(x, weights[name+'.weight'], epsilon, approximations['inverse_sqrt'],
                                  bias=weights[name+'.bias'], center=True)
        x = value
        for layer in range(config['num_hidden_layers']):
            prefix = f'bert.encoder.layer.{layer}.'
            projections = []
            for name in ('query', 'key', 'value'):
                stem = prefix+'attention.self.'+name
                w, b = self.head_projection(weights[stem+'.weight'], weights[stem+'.bias'], heads, d)
                projections.append(self.linear(x, w, b))
            attended = self.attention(*projections, heads=heads, head_dim=d,
                                       mask=mask, counts=counts, approximations=approximations)
            name = prefix+'attention.output.dense'
            branch = self.linear(attended, self.head_output(weights[name+'.weight'], heads, d), weights[name+'.bias'])
            x = norm(self.arithmetic.add(x, branch), prefix+'attention.output.LayerNorm')
            del projections, attended, branch
            self.arithmetic.checkpoint(x, f'packed BERT layer {layer}: attention/normalization ready')
            middle = self.arithmetic.evaluate(linear(x, prefix+'intermediate.dense'), approximations['gelu'],
                      output_weights=self.layout.feature(np.ones(config['intermediate_size'])))
            x = norm(self.arithmetic.add(x, linear(middle, prefix+'output.dense')), prefix+'output.LayerNorm')
            del middle
            self.arithmetic.checkpoint(x, f'packed BERT layer {layer}: output ready')
        cls_mask = self.layout.tile((np.arange(self.layout.width) == 0)[None, :, None, None])
        pooler = linear(x, 'bert.pooler.dense', input_mask=cls_mask)
        pooled = self.arithmetic.evaluate(pooler, approximations['tanh'],
                                          output_weights=self.layout.feature(np.ones(h)))
        return linear(pooled, 'classifier')

    def _qwen(self, value, config, weights, approximations):
        h, heads, kv, d = (config['hidden_size'], config['num_attention_heads'],
                           config['num_key_value_heads'], config['head_dim'])
        epsilon = config['rms_norm_eps']
        mask, counts = self.layout.attention_mask(causal=True)
        def linear(x, name):
            return self.linear(x, weights[name+'.weight'])
        def norm(x, name):
            return self.normalize(x, weights[name+'.weight'], epsilon, approximations['inverse_sqrt'])
        def project(x, name, source_heads):
            w, _ = self.head_projection(weights[name+'.weight'], None, heads, d, source_heads=source_heads)
            return self.linear(x, w)
        x = value
        for layer in range(config['num_hidden_layers']):
            prefix = f'model.layers.{layer}.'
            attention = prefix+'self_attn.'
            z = norm(x, prefix+'input_layernorm')
            v = project(z, attention+'v_proj', kv)
            if self.layout.sequence == 1:
                attended = v
            else:
                q = project(z, attention+'q_proj', heads)
                k = project(z, attention+'k_proj', kv)
                q = self.rope(self.normalize_heads(q, weights[attention+'q_norm.weight'], epsilon,
                              approximations['inverse_sqrt'], heads), heads, d, config['rope_theta'])
                k = self.rope(self.normalize_heads(k, weights[attention+'k_norm.weight'], epsilon,
                              approximations['inverse_sqrt'], heads), heads, d, config['rope_theta'])
                attended = self.attention(q, k, v, heads=heads, head_dim=d, mask=mask, counts=counts,
                                           approximations=approximations)
                del q, k
            branch = self.linear(attended, self.head_output(weights[attention+'o_proj.weight'], heads, d))
            x = self.arithmetic.add(x, branch)
            del z, v, attended, branch
            self.arithmetic.checkpoint(x, f'packed Qwen layer {layer}: attention ready')
            z = norm(x, prefix+'post_attention_layernorm')
            gate = self.arithmetic.evaluate(linear(z, prefix+'mlp.gate_proj'), approximations['silu'],
                        output_weights=self.layout.feature(np.ones(config['intermediate_size'])))
            up = linear(z, prefix+'mlp.up_proj')
            gated = self.arithmetic.multiply(gate, up)
            x = self.arithmetic.add(x, linear(gated, prefix+'mlp.down_proj'))
            del z, gate, up, gated
            self.arithmetic.checkpoint(x, f'packed Qwen layer {layer}: MLP ready')
        return self.linear(norm(x, 'model.norm'), weights['model.embed_tokens.weight'])


def plan_packed_transformer(model, config, weights, *, batch, sequence, slots=65536, attention_mask=None):
    """Exact graph depth/rotation inventory without keys or activation values."""
    from types import SimpleNamespace
    from encrypted_transformer import _DepthOnlyEvaluator, _PlannedValue, public_toy_approximations

    features = packed_feature_width(model, config)
    layout = PackedPairLayout(batch, sequence, features, slots)
    rotations = set()

    class Arithmetic(_DepthOnlyEvaluator):
        def rotate(self, value, shift):
            if shift % slots:
                rotations.add(shift % slots)
            return value

        def checkpoint(self, value, label):
            return value

    class Planner(PackedTransformer):
        def __init__(self):
            self.layout, self.features = layout, features
            self.ctx = SimpleNamespace(num_slots=slots)
            self.arithmetic = Arithmetic(slots)

        def linear(self, value, weight, bias=None, *, input_mask=None):
            source = MatrixDiagonals(weight, slots, features)
            rotations.update(source.rotations)
            return _PlannedValue(value.level-1)

    result = Planner().evaluate(_PlannedValue(0), model, config, weights,
                                 public_toy_approximations(model), attention_mask)
    return dict(depth=-result.level, features=features, minimum_slots=layout.period,
                rotations=tuple(sorted(rotations)), strategy='packed pair-feature graph; sequential sparse BSGS')
