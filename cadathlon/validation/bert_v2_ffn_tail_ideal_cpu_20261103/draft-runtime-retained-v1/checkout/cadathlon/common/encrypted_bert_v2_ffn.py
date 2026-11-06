"""Complete isolated layer0 FFN/residual/MLP norm over one valid row tile.

Two explicit full-slot refreshes use caller-assumed B128 input and the frozen
GELU64 B16 envelope. No decoded ranges, codec, key generation or implicit
refresh is used. Whole-chain HE/domain accuracy remains unvalidated.
"""
from collections.abc import Mapping
from types import MappingProxyType
import hashlib
import math
import numpy as np

from encrypted_bert_v2_gelu import BertV2Gelu, Gelu64Contract
from encrypted_fullslot_refresh_binding import FullSlotRefreshBinding, typed
from encrypted_scale_planning import weighted_sum_scale, drop_divisor

TENSORS = {
    'intermediate.weight': ((3072, 768), '9a0c8aa79b2bac48fc35f37b6695d253080f7eedd556b70b65411aab9c319fd3'),
    'intermediate.bias': ((3072,), 'b279856af79c545cdb2f914e02e9a953204152319f160e484ebfcc8d6b825774'),
    'output.weight': ((768, 3072), '754a718ad071ca9ec27790aee1b82dc9db41cc3ef90c6c73a90ada8668f6fa0d'),
    'output.bias': ((768,), '744629ad72bdc326f22cd92c6429610c06ed41dd3a2fedd307d3f01f861638b9'),
}


def freeze_weights(arrays):
    if not isinstance(arrays, Mapping) or set(arrays) != set(TENSORS):
        raise ValueError('all four pinned full-width FFN tensors required')
    result = {}
    for name, (shape, expected) in TENSORS.items():
        a = np.asarray(arrays[name])
        if a.dtype.kind not in 'fiu' or a.shape != shape or not np.isfinite(a).all():
            raise ValueError('invalid FFN tensor '+name)
        with np.errstate(over='ignore', invalid='ignore'):
            raw = a.astype('<f4')
            lift = raw.astype('<f8')
        if (not np.isfinite(raw).all() or not np.array_equal(a, lift)
                or hashlib.sha256(raw.tobytes()).hexdigest() != expected):
            raise ValueError('FFN tensor differs from exact pinned FP32 lift: '+name)
        result[name] = np.frombuffer(lift.tobytes(), dtype='<f8').reshape(shape)
    return MappingProxyType(result)


class BertV2FFNTile:
    def __init__(self, arithmetic, linear, refresher, arrays, manifest_bytes, gamma, beta,
                 *, original_positions, progress=None):
        from encrypted_chained_streaming_linear import ChainedStreamingLinear
        from encrypted_bert_v2_row_norm import V2RowLayerNorm
        if not isinstance(linear, ChainedStreamingLinear):
            raise TypeError('promoted chained streaming linear required')
        if (type(original_positions) is not tuple or not 1 <= len(original_positions) <= 16
                or any(type(x) is not int or not 0 <= x < 128 for x in original_positions)
                or tuple(sorted(set(original_positions))) != original_positions):
            raise ValueError('one ordered compact valid query tile with original positions required')
        if progress is not None and not callable(progress):
            raise TypeError('progress must be callable or None')
        self.arithmetic, self.linear, self.refresher = arithmetic, linear, refresher
        self.model = freeze_weights(arrays)
        self.original_positions, self.progress = original_positions, progress
        self.refresh_binding = FullSlotRefreshBinding(arithmetic, refresher)
        self.gelu = BertV2Gelu(arithmetic, Gelu64Contract(manifest_bytes), progress=self._callback)
        self.normalizer = V2RowLayerNorm(arithmetic, manifest_bytes, gamma, beta,
                                        site='0.mlp', epsilon=1e-12, progress=self._callback)
        normalizer = self.normalizer
        self._active = False
        self._source = self._source_state = None
        self._retained = []
        self.last_run = None
        self._norm_progress = normalizer.progress
        self._bound = (arithmetic, linear, refresher, normalizer, self.model, self.refresh_binding,
                       self.gelu, progress, linear.arithmetic, linear.layout)
        self._positions = original_positions
        self._weights = self._weight_state()
        self.required_keys = tuple(sorted(set(linear.required_rotation_indices(self.model['intermediate.weight'], linear.layout))
                                  | set(linear.required_rotation_indices(self.model['output.weight'], linear.layout))))
        self._required_keys = self.required_keys
        self._guard()

    def _weight_state(self):
        return tuple((n, id(v), v.dtype.str, v.shape, v.strides, v.flags.writeable) for n, v in self.model.items())

    def _fingerprint(self, value): return self.gelu._metadata._value_fingerprint(value)

    def _guard(self):
        now = (self.arithmetic, self.linear, self.refresher, self.normalizer, self.model,
               self.refresh_binding, self.gelu, self.progress, self.linear.arithmetic, self.linear.layout)
        if any(a is not b for a, b in zip(now, self._bound)):
            raise ValueError('construction-bound FFN owner changed')
        if (typed(self.original_positions) != typed(self._positions) or self._weight_state() != self._weights
                or typed(self.required_keys) != typed(self._required_keys)
                or self.normalizer.progress is not self._norm_progress):
            raise ValueError('public FFN model/position/key/callback contract changed')
        ctx = self.arithmetic.ctx
        if (self.linear.ctx is not ctx or self.linear.arithmetic.ctx is not ctx
                or self.normalizer.arithmetic is not self.arithmetic
                or self.normalizer.contract.site != '0.mlp'
                or typed(self.linear.layout.slots) != (int, 65536)
                or typed(self.linear.layout.features) != (int, 4096)
                or self.arithmetic.refresher is not None or self.linear.arithmetic.refresher is not None
                or typed(self.linear.arithmetic.scale) != typed(self.arithmetic.scale)):
            raise ValueError('one static borrowed full-slot context required')
        self.refresh_binding.check(); self.gelu._guard(); self.normalizer._guard()
        from encrypted_chained_streaming_linear import require_chained_rotation_keys
        require_chained_rotation_keys(ctx, self.required_keys)
        if self._source is not None and self._fingerprint(self._source) != self._source_state:
            raise ValueError('FFN original input changed')
        if any(self._fingerprint(v) != state for v, state in self._retained):
            raise ValueError('FFN retained residual changed')

    def _callback(self, label, value, counts):
        self._guard()
        state = self._fingerprint(value)
        if self.progress is not None: self.progress('FFN '+label, value, dict(counts))
        self._guard()
        if self._fingerprint(value) != state:
            raise ValueError('FFN callback changed result metadata or payload')

    def _note(self, label, value, stages):
        self._guard()
        self.gelu._metadata._validate_ciphertext(value, minimum_levels=0, canonical=False)
        for name in tuple(vars(value)):
            if name.startswith('_public_'): delattr(value, name)
        self._callback(label, value, {})
        stages.append(dict(stage=label, level=self.arithmetic._level(value), scale=value._ckks_scale))
        return value

    def _linear(self, value, name, stages):
        level = self.arithmetic._level(value)
        expected = weighted_sum_scale([value._ckks_scale], self.arithmetic.scale,
                    drop_divisor(self.arithmetic.ctx._param_cache.q_moduli_at_level(level), level))['output_scale']
        result = self.linear.linear_tile(value, self.model[name+'.weight'], self.model[name+'.bias'],
                                       valid_rows=len(self.original_positions))
        if self.arithmetic._level(result) != level-1 or typed(result._ckks_scale) != typed(expected):
            raise ValueError('dense output differs from actual scale/level plan')
        return self._note(name+' dense', result, stages)

    def evaluate_tile(self, value):
        self._guard()
        if self._active: raise ValueError('FFN tile cannot be reentered')
        self.gelu._metadata._validate_ciphertext(value, minimum_levels=2, canonical=False)
        self._active = True
        self._source, self._source_state = value, self._fingerprint(value)
        self.last_run = None
        stages = []
        try:
            residual = self._note('input B128 refresh', self.refresh_binding.refresh(value, public_bound=128.), stages)
            self._retained.append((residual, self._fingerprint(residual)))
            middle = self._linear(residual, 'intermediate', stages)
            activated = self._note('GELU64', self.gelu.evaluate_tile(middle), stages)
            refreshed = self._note('GELU B16 refresh', self.refresh_binding.refresh(activated, public_bound=16.), stages)
            output = self._linear(refreshed, 'output', stages)
            combined = self.arithmetic.add(output, residual)
            if self.arithmetic._level(combined) != 24 or typed(combined._ckks_scale) != typed(output._ckks_scale):
                raise ValueError('residual addition changed actual first-operand metadata')
            combined = self._note('residual', combined, stages)
            self._retained.clear()
            result = self.normalizer.normalize_tile(combined, valid_rows=len(self.original_positions),
                                                   original_positions=self.original_positions)
            final = self._note('MLP normalization', result.output, stages)
            if self.arithmetic._level(final) != 13:
                raise ValueError('full FFN tail must finish at level13')
            self.last_run = dict(actual_metadata=stages, explicit_refresh_bounds=[128.,16.],
                                original_positions=self.original_positions,
                                range_verified=False, encrypted_error_verified=False)
            return result
        finally:
            self._retained.clear()
            self._source = self._source_state = None
            self._active = False
