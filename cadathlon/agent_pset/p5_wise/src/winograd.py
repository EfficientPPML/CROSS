"""Execute a native WISE WinogradReady schedule with CROSS-compatible arithmetic.

Input ciphertexts are ordered by tile position, then ciphertext block, exactly
as upstream WISE. A schedule's slot count must match the physical CKKS slots;
the caller is responsible for generating its padded/tiled client layout.
"""
from types import SimpleNamespace
import numpy as np
from schedule import Schedule, execute


def signed_sum(values, coefficients, arithmetic):
    """Winograd A/B matrices use exact integer +/-1; consume no HE levels."""
    coefficients = np.asarray(coefficients)
    if not np.isin(coefficients, (-1, 0, 1)).all():
        raise ValueError("WISE Winograd A/B coefficients must be -1, 0 or 1")
    result = None
    for value, coefficient in zip(values, coefficients, strict=True):
        if coefficient == 0:
            continue
        if result is None:
            result = value if coefficient == 1 else arithmetic.add(
                arithmetic.add(value, value, subtract=True), value, subtract=True)
        else:
            result = arithmetic.add(result, value, subtract=coefficient == -1)
    return arithmetic.add(values[0], values[0], subtract=True) if result is None else result


def rotations(layer, slots):
    result = {int(i) % slots for i in np.asarray(layer.i2c_offset).ravel()}
    result.update(int(i) % slots for i in layer.global_rots)
    result.update(slots >> (i + 1) for i in range(layer.output_rotations))
    return sorted(result - {0})


def execute_winograd(layer, inputs, slots, arithmetic):
    """Native 4 -> 16 -> 4 Winograd transforms, not a dense Toeplitz rewrite."""
    tile_out = layer.Ht * layer.Wt
    tile_in = (layer.Ht + layer.R - 1) * (layer.Wt + layer.S - 1)
    if len(inputs) != tile_out * layer.T_cols:
        raise ValueError("input does not match WISE's tile-major ciphertext layout")
    if len(layer.bias) != tile_out * layer.T_rows:
        raise ValueError("WISE output bias does not match tile-major layout")
    routed = [[None] * tile_in for _ in range(layer.T_cols)]
    for tile in range(tile_out):
        for block in range(layer.T_cols):
            value = inputs[tile * layer.T_cols + block]
            for j in range(tile_out):
                target = int(layer.i2c_target[tile, j])
                offset = int(layer.i2c_offset[tile, j]) % slots
                if routed[block][target] is not None:
                    raise ValueError("overlapping WISE Winograd routing targets")
                routed[block][target] = arithmetic.rotate(value, offset) if offset else value
    if any(v is None for row in routed for v in row):
        raise ValueError("incomplete WISE Winograd routing")
    transformed = [[signed_sum(routed[block], layer.B_kron[i], arithmetic)
                    for block in range(layer.T_cols)] for i in range(tile_in)]
    products = []
    for i in range(tile_in):
        linear = SimpleNamespace(mat=layer.mats[i], T_rows=layer.T_rows, T_cols=layer.T_cols,
                                 bias=np.zeros((layer.T_rows, slots)), output_rotations=0)
        schedule = Schedule(slots, [("linear", linear)], layer.T_cols * slots, layer.T_rows * slots, {})
        products.append(execute(schedule, transformed[i], arithmetic))
    outputs = []
    for tile in range(tile_out):
        for block in range(layer.T_rows):
            result = signed_sum([p[block] for p in products], layer.A_kron[tile], arithmetic)
            bias = np.asarray(layer.bias[len(outputs)])
            result = arithmetic.add_bias(result, bias) if hasattr(arithmetic, "add_bias") else result + bias
            for i in range(layer.output_rotations):
                result = arithmetic.add(result, arithmetic.rotate(result, slots >> (i + 1)))
            outputs.append(result)
    return outputs
