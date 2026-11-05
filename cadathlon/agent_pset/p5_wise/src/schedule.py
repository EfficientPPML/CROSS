"""WISE-generated NN schedules, with a backend-neutral BSGS interpreter.

The planner calls upstream packers and preserves their diagonal coefficients,
baby/giant partitions, ciphertext blocks, bias and output-fold schedule. It does
not replace a WISE plan with CROSS's separate Mapping planner.
"""
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import numpy as np

from workloads import parameters
from wise_source import load_wise


@dataclass
class Schedule:
    slots: int
    layers: list
    input_size: int
    output_size: int
    provenance: dict
    input_indices: np.ndarray | None = None
    packed_input_size: int | None = None

    @property
    def input_ciphertexts(self):
        return ((self.packed_input_size or self.input_size) + self.slots - 1) // self.slots

    @property
    def depth(self):
        return len(self.layers)

    def rotations(self, strategy="bsgs"):
        rotations = set()
        for kind, layer in self.layers:
            if kind in ("square", "herpn"):
                continue
            if kind == "winograd":
                from winograd import rotations as winograd_rotations
                rotations.update(winograd_rotations(layer, self.slots))
                continue
            for row in layer.mat:
                for block in row:
                    for d in block.diags:
                        rotations.update((d.bs, d.gs) if strategy == "bsgs" else ((d.bs + d.gs) % self.slots,))
            rotations.update(self.slots >> (i + 1) for i in range(layer.output_rotations))
        return sorted(int(r) for r in rotations if r % self.slots)

    def identity(self):
        h = hashlib.sha256()
        h.update(json.dumps({"slots": self.slots, "input": self.input_size, "output": self.output_size,
                             "wise": self.provenance}, sort_keys=True).encode())
        if self.input_indices is not None:
            h.update(np.asarray(self.input_indices, dtype="<i8").tobytes())
            h.update(str(self.packed_input_size).encode())
        for kind, layer in self.layers:
            h.update(kind.encode())
            if kind == "square":
                continue
            if kind == "herpn":
                h.update(np.asarray(layer.a0, dtype="<f8").tobytes())
                h.update(np.asarray(layer.a1, dtype="<f8").tobytes())
                continue
            h.update(str((layer.T_rows, layer.T_cols, layer.output_rotations)).encode())
            h.update(np.asarray(layer.bias, dtype="<f8").tobytes())
            if kind == "winograd":
                for name in ("R", "S", "Ht", "Wt", "i2c_target", "i2c_offset", "A_kron", "B_kron"):
                    h.update(np.asarray(getattr(layer, name), dtype="<i8").tobytes())
            for matrix in (layer.mats if kind == "winograd" else [layer.mat]):
                for row in matrix:
                    for block in row:
                        h.update(str((block.bx, block.by)).encode())
                        for d in block.diags:
                            h.update(str((d.bs, d.gs)).encode())
                            h.update(np.asarray(d.data, dtype="<f8").tobytes())
        return h.hexdigest()

    def summary(self):
        return {"sha256": self.identity(), "slots": self.slots, "depth": self.depth,
                "input_size": self.input_size, "output_size": self.output_size,
                "rotation_keys": self.rotations(), "wise": self.provenance,
                "layers": [{"operation": kind, **({} if kind in ("square", "herpn") else {
                    "input_ciphertexts": layer.T_cols * (4 if kind == "winograd" else 1),
                    "output_ciphertexts": layer.T_rows * (4 if kind == "winograd" else 1),
                    "plaintext_products": layer.num_diags * (16 if kind == "winograd" else 1),
                    "output_folds": layer.output_rotations})}
                    for kind, layer in self.layers]}


def build_schedule(case, slots, wise_root=None, *, convolution="toeplitz"):
    if type(slots) is not int or slots < 2 or slots & (slots - 1):
        raise ValueError("slots must be a power of two >= 2")
    wise, provenance = load_wise(wise_root)
    if convolution not in ("toeplitz", "winograd", "tiled-toeplitz"):
        raise ValueError("unknown synthetic convolution schedule")
    if case["network"] == "cnn" and convolution == "winograd":
        from winograd_network import build_cnn
        return build_cnn(case, slots, wise, provenance)
    if case["network"] == "cnn" and convolution == "tiled-toeplitz":
        from winograd_network import build_tiled_cnn
        return build_tiled_cnn(case, slots, wise, provenance)
    p = parameters(case)
    batch = case["batch"]
    def order(c, h=1, w=1):
        return wise.Order(C=c, H=h, W=w, batchsize=(batch, 1, 1))
    def linear(transform):
        return "linear", transform.pack(slots)
    input_indices = None
    if case["network"] == "mlp":
        source, middle, target = order(case["inputs"]), order(case["hidden"]), order(case["classes"])
        if batch > 1:
            # Adjacent slots carry distinct images of the same feature. The
            # first dense transform can reuse each weight diagonal across B.
            # The final WISE transform restores ordinary batch-major outputs.
            source = wise.Order(C=case["inputs"], H=1, W=1, batchsize=(1, 1, batch))
            middle = wise.Order(C=case["hidden"], H=1, W=1, batchsize=(1, 1, batch))
            input_indices = np.asarray(source.perm)
        layers = [linear(wise.pack_fc(source, middle, p["w1"], p["b1"])), ("square", None),
                  linear(wise.pack_fc(middle, target, p["w2"], p["b2"]))]
        input_size = batch * case["inputs"]
    elif case["network"] == "cnn":
        side, channels = case["side"], case["channels"]
        source, conv = order(1, side, side), order(channels, side, side)
        pooled, target = order(channels, side // 2, side // 2), order(case["classes"])
        layers = [linear(wise.pack_conv2d_toeplitz(source, conv, 1, p["kernel"], bn_b=p["b1"])),
                  ("square", None),
                  linear(wise.pack_avgpool2d(conv, pooled, size=2, stride=2, padding=0)),
                  linear(wise.pack_fc(pooled, target, p["w2"], p["b2"]))]
        input_size = batch * side * side
    else:
        raise ValueError("unknown network")
    return Schedule(slots, layers, input_size, batch * case["classes"], provenance,
                    input_indices, input_size if input_indices is not None else None)


def pack_inputs(schedule, x):
    flat = np.asarray(x, dtype=np.float64).ravel()
    if flat.size != schedule.input_size or not np.isfinite(flat).all():
        raise ValueError("wrong input size or nonfinite client input")
    if schedule.input_indices is not None:
        result = np.zeros(schedule.input_ciphertexts * schedule.slots)
        result[schedule.input_indices] = flat
        return result.reshape(-1, schedule.slots)
    return np.pad(flat, (0, (-flat.size) % schedule.slots)).reshape(-1, schedule.slots)


class NumpyArithmetic:
    """Plaintext schedule verification only; never used for encrypted inference."""
    @staticmethod
    def rotate(x, shift): return np.roll(x, -shift)
    @staticmethod
    def add(a, b, *, subtract=False): return a - b if subtract else a + b
    @staticmethod
    def weighted_sum(values, weights, *, bias=0.):
        return sum(v * w for v, w in zip(values, weights)) + bias
    @staticmethod
    def multiply(a, b): return a * b


def execute(schedule, inputs, arithmetic, *, strategy="bsgs", progress=None):
    if strategy not in ("bsgs", "direct"):
        raise ValueError("strategy must be bsgs or direct")
    values = list(inputs)
    if len(values) != schedule.input_ciphertexts:
        raise ValueError("wrong number of input ciphertexts")
    for index, (kind, layer) in enumerate(schedule.layers):
        if kind == "square":
            values = [arithmetic.multiply(v, v) for v in values]
        elif kind == "herpn":
            if len(values) != len(layer.a0) or len(values) != len(layer.a1):
                raise ValueError("HerPN coefficient/ciphertext count mismatch")
            values = [(arithmetic.quadratic(v, a1, a0) if hasattr(arithmetic, "quadratic")
                       else v*v + a1*v + a0) for v, a1, a0 in zip(values, layer.a1, layer.a0, strict=True)]
        elif kind == "winograd":
            from winograd import execute_winograd
            values = execute_winograd(layer, values, schedule.slots, arithmetic)
        else:
            if len(values) != layer.T_cols:
                raise ValueError("WISE layer input count mismatch")
            babies = {}
            outputs = []
            for row in layer.mat:
                groups = defaultdict(list)
                for block in row:
                    for diag in block.diags:
                        baby = diag.bs if strategy == "bsgs" else (diag.bs + diag.gs) % schedule.slots
                        giant = diag.gs if strategy == "bsgs" else 0
                        key = (block.by, baby)
                        if key not in babies:
                            babies[key] = arithmetic.rotate(values[block.by], int(baby)) if baby else values[block.by]
                        weight = np.asarray(diag.data)
                        if strategy == "direct":
                            weight = np.roll(weight, -diag.gs)
                        groups[giant].append((babies[key], weight))
                result = None
                for giant, terms in sorted(groups.items()):
                    term = arithmetic.weighted_sum([v for v, _ in terms], [w for _, w in terms])
                    if giant:
                        term = arithmetic.rotate(term, int(giant))
                    result = term if result is None else arithmetic.add(result, term)
                if result is None:
                    result = arithmetic.weighted_sum([values[0]], [np.zeros(schedule.slots)])
                # WISE puts bias before the output fold. Encode/add at the
                # resulting ciphertext's real scale; this consumes no level.
                bias = np.asarray(layer.bias[len(outputs)])
                if hasattr(arithmetic, "add_bias"):
                    result = arithmetic.add_bias(result, bias)
                else:
                    result = result + bias
                for i in range(layer.output_rotations):
                    result = arithmetic.add(result, arithmetic.rotate(result, schedule.slots >> (i + 1)))
                outputs.append(result)
            values = outputs
        if progress is not None:
            progress(index, kind, values)
    return values
