"""Experimental distinct-row linear tiles using genuine CKKS BSGS arithmetic.

Each physical feature block holds a different client row. Unused features and
unused rows are zero; there is no periodic-row replication. The evaluator uses
only ciphertexts, public matrix/bias/mask data and an initialized keyless context.
It borrows the context and does not encrypt, decrypt, generate keys or refresh.

This is a component for streamed full-width projections, not a complete BERT or
Qwen implementation. The client codec is backend-free; crypto imports are lazy.
"""
from dataclasses import dataclass
from types import SimpleNamespace
import math
import numpy as np


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name + ' must be a positive integer')
    return value


def _finite_real_array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
        raise ValueError(name + ' must contain finite real numbers')
    result = array.astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise ValueError(name + ' exceeds float64 range')
    return result


@dataclass(frozen=True)
class RowTileLayout:
    """Public layout for disjoint rows; F4096 fits BERT widths768 and3072.

    Layout metadata is public. Exact logical row count is required to unpack or
    evaluate a stream. Partial final tiles have zero inactive rows, including
    their affine bias; those rows never become additional logical inputs.
    """
    slots: int
    features: int = 4096

    def __post_init__(self):
        for name in ('slots', 'features'):
            value = _positive_integer(getattr(self, name), name)
            if value & (value - 1):
                raise ValueError(name + ' must be a power of two')
        if self.slots < 2 * self.features or self.slots % self.features:
            raise ValueError('slots must contain at least two complete feature blocks')

    @property
    def rows_per_tile(self):
        return self.slots // self.features

    def _width(self, width):
        if _positive_integer(width, 'width') > self.features:
            raise ValueError('width exceeds the feature block')
        return width

    def _valid_rows(self, valid_rows):
        if _positive_integer(valid_rows, 'valid_rows') > self.rows_per_tile:
            raise ValueError('valid_rows exceeds tile capacity')
        return valid_rows

    def tile_count(self, rows):
        return (_positive_integer(rows, 'rows') + self.rows_per_tile - 1) // self.rows_per_tile

    def linear_contract(self, *, rows, input_width, output_width):
        """Exact public shape/order contract for a linear ciphertext stream."""
        count = self.tile_count(rows)
        self._width(input_width); self._width(output_width)
        return {'schema': 'distinct-row-linear-v1', 'slots': self.slots,
            'features': self.features, 'rows': rows, 'input_width': input_width,
            'output_width': output_width, 'rows_per_tile': self.rows_per_tile,
            'tile_count': count, 'last_tile_rows': rows-(count-1)*self.rows_per_tile,
            'row_order': 'contiguous-input-order', 'feature_padding': 'zero',
            'inactive_rows': 'zero-including-bias'}

    def feature(self, values, *, valid_rows):
        """Public feature constants, zero outside active rows/features."""
        values = _finite_real_array(values, 'feature constants')
        if values.ndim != 1:
            raise ValueError('feature constants must be a vector')
        self._width(len(values)); self._valid_rows(valid_rows)
        result = np.zeros((self.rows_per_tile, self.features), dtype=np.float64)
        result[:valid_rows, :len(values)] = values
        return result.reshape(self.slots)

    def input_mask(self, width, *, valid_rows):
        return self.feature(np.ones(self._width(width)), valid_rows=valid_rows)

    def pack_client_rows(self, rows):
        """Yield full-slot client plaintexts, one tile at a time.

        This method does no encryption. The caller must keep the supplied rows
        unchanged while consuming the iterator. No input row is repeated.
        """
        rows = _finite_real_array(rows, 'client rows')
        if rows.ndim != 2 or rows.shape[0] < 1:
            raise ValueError('client rows must be a nonempty rank-two array')
        self._width(rows.shape[1])
        for start in range(0, len(rows), self.rows_per_tile):
            block = rows[start:start + self.rows_per_tile]
            tile = np.zeros((self.rows_per_tile, self.features), dtype=np.float64)
            tile[:len(block), :rows.shape[1]] = block
            yield tile.reshape(self.slots)

    def unpack_client_rows(self, tiles, *, rows, width, padding_atol=None):
        """Decode logical rows; reject missing/extra tiles and optional bad padding.

        Decoded CKKS padding is approximate. A caller may supply a nonnegative
        absolute error tolerance; this never changes logical row selection.
        """
        count = self.tile_count(rows); self._width(width)
        if padding_atol is not None and (isinstance(padding_atol, (bool, np.bool_))
                or not isinstance(padding_atol, (int, float))
                or not math.isfinite(padding_atol) or padding_atol < 0):
            raise ValueError('padding_atol must be a finite nonnegative real number')
        iterator = iter(tiles); output = np.empty((rows, width), dtype=np.float64)
        for index in range(count):
            try:
                raw = next(iterator)
            except StopIteration as error:
                raise ValueError('missing client output tile') from error
            tile = _finite_real_array(raw, 'client output tile')
            if tile.shape != (self.slots,):
                raise ValueError('client output tile must contain every physical slot')
            tile = tile.reshape(self.rows_per_tile, self.features)
            start = index * self.rows_per_tile; valid = min(self.rows_per_tile, rows-start)
            if padding_atol is not None:
                if (np.any(np.abs(tile[:valid, width:]) > padding_atol)
                        or np.any(np.abs(tile[valid:]) > padding_atol)):
                    raise ValueError('decoded tile padding exceeds the declared tolerance')
            output[start:start+valid] = tile[:valid, :width]
        try:
            next(iterator)
        except StopIteration:
            return output
        raise ValueError('extra client output tile')


class StreamingLinear:
    """Borrow a secure keyless context and evaluate one row tile at a time.

    Uses the existing sparse MatrixDiagonals/CompiledArithmetic implementation
    via PackedTransformer.linear. Each map consumes one real rescale level.
    Masks are public source-coordinate weights; row/feature padding is masked
    even if incoming ciphertext padding is hostile. Output feature padding and
    inactive rows are zero in the ideal circuit, up to ordinary CKKS error.

    The default direct strategy retains one key per requested rotation. The
    optional chained strategy shares baby-step gaps and accumulates giant
    steps in descending order. Its public key inventory must be generated with
    the same strategy. It consumes the same one level and has a different
    key-switch noise history; a numerical tolerance requires encrypted testing.
    Chained evaluation requires static capacity and never refreshes implicitly.
    """
    def __init__(self, context, layout, *, low_memory=True, preserve_dispatch_cache=False, rotation_strategy='direct'):
        from encrypted_packed_transformer import validate_rotation_strategy
        self.rotation_strategy=validate_rotation_strategy(rotation_strategy)
        if not isinstance(layout, RowTileLayout):
            raise TypeError('expected RowTileLayout')
        if type(low_memory) is not bool or type(preserve_dispatch_cache) is not bool:
            raise TypeError('memory options must be explicit booleans')
        if context.num_slots != layout.slots:
            raise ValueError('context slots differ from the row layout')
        if context.degree != 2 * layout.slots:
            raise ValueError('row layout must cover every physical CKKS slot')
        cache = getattr(context, '_param_cache', None)
        if cache is not None and cache.batch != 1:
            raise ValueError('row tiles require a batch-one evaluator context')
        from encrypted_packed_qwen import CompiledArithmetic
        self.ctx, self.layout = context, layout
        # Constructor validates initialization, CD2, actual secure Q/P and all
        # three possible secret-key locations before creating arithmetic.
        self.arithmetic = CompiledArithmetic(context, low_memory=low_memory,
            preserve_dispatch_cache=preserve_dispatch_cache)

    @staticmethod
    def required_rotation_indices(weight, layout, *, rotation_strategy='direct'):
        """Exact public rotations for the selected strategy, including sparse gaps."""
        if not isinstance(layout, RowTileLayout):
            raise TypeError('expected RowTileLayout')
        from encrypted_packed_qwen import MatrixDiagonals
        from encrypted_packed_transformer import linear_rotation_plan,validate_rotation_strategy
        validate_rotation_strategy(rotation_strategy)
        weight = _finite_real_array(weight, 'public weight')
        return linear_rotation_plan(MatrixDiagonals(weight,layout.slots,layout.features),strategy=rotation_strategy).required_keys

    def _validate_ciphertext(self, value):
        from jaxite_word import Polynomial
        from support import require_secure_execution
        if (self.ctx.secret_key is not None or self.ctx._param_cache.secret_key is not None
                or self.ctx.parameters.get('secret_key') is not None):
            raise ValueError('streamed evaluator must remain keyless')
        if not isinstance(value, Polynomial):
            raise TypeError('linear input must be a canonical Polynomial ciphertext')
        if (value.batch != 1 or value.num_elements != 2
                or value.degree != 2 * self.layout.slots):
            raise ValueError('row tiles require one two-component ciphertext with the declared ring degree')
        require_secure_execution(self.ctx, value)
        if self.arithmetic._level(value) < 1:
            raise ValueError('streamed linear requires one remaining rescale level')

    def linear_tile(self, value, weight, bias=None, *, valid_rows, input_mask=None):
        self.layout._valid_rows(valid_rows)
        weight = _finite_real_array(weight, 'public weight')
        if weight.ndim != 2:
            raise ValueError('public weight must be a matrix')
        self.layout._width(weight.shape[0]); self.layout._width(weight.shape[1])
        if bias is not None:
            bias = _finite_real_array(bias, 'public bias')
            if bias.shape != (weight.shape[0],):
                raise ValueError('public bias differs from the output width')
        mask = self.layout.input_mask(weight.shape[1], valid_rows=valid_rows)
        if input_mask is not None:
            input_mask = _finite_real_array(input_mask, 'public source mask')
            if input_mask.shape != (self.layout.slots,):
                raise ValueError('public source mask must contain every physical slot')
            mask *= input_mask
        self._validate_ciphertext(value)
        from encrypted_packed_transformer import PackedTransformer
        # Per-call bias view avoids changing shared layout state. It restores
        # zeros in inactive rows as part of the existing public bias addition.
        active_layout = SimpleNamespace(slots=self.layout.slots,
            feature=lambda values: self.layout.feature(values, valid_rows=valid_rows))
        graph = SimpleNamespace(ctx=self.ctx, layout=active_layout,
            features=self.layout.features, arithmetic=self.arithmetic,
            linear_rotation_strategy=getattr(self,'rotation_strategy','direct'))
        return PackedTransformer.linear(graph, value, weight, bias, input_mask=mask)

    def iter_evaluate(self, ciphertexts, weight, bias=None, *, rows):
        """Yield outputs without retaining the stream; consume to verify its size.

        A partial consumer cannot establish that no trailing input tiles exist.
        Complete consumption rejects both missing and extra ciphertext tiles.
        No client codec is called by this evaluator.
        """
        count = self.layout.tile_count(rows); iterator = iter(ciphertexts)
        for index in range(count):
            try:
                value = next(iterator)
            except StopIteration as error:
                raise ValueError('missing ciphertext tile') from error
            valid = min(self.layout.rows_per_tile, rows-index*self.layout.rows_per_tile)
            yield self.linear_tile(value, weight, bias, valid_rows=valid)
            del value
        try:
            next(iterator)
        except StopIteration:
            return
        raise ValueError('extra ciphertext tile')

    def clear(self):
        """Release arithmetic bindings/caches; the caller still owns the context."""
        self.arithmetic._clear_working_caches()
