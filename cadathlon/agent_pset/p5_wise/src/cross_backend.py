"""Execute WISE's public schedules with keyless CROSS CKKS primitives."""
from collections import OrderedDict
import hashlib
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from support import cross_root
cross_root()


def profile(depth):
    from jaxite_word import he_params, util
    from jaxite_word.composite_prime_gen import composite_prime_gen
    dnum = 4
    if type(depth) is not int or depth < 1:
        raise ValueError("depth must be a positive integer")
    # The longer pretrained chain exhausts the 21-bit prime neighborhood at
    # larger rings. Wider composite groups keep every prime distinct without
    # reducing the circuit depth or bypassing the Q*P security bound.
    scale_bits, word_bits, aux_bits = (42, 24, 23) if depth <= 4 else (46, 31, 31)
    first_bits = scale_bits+1 if depth <= 4 else 60
    for degree in (8192, 16384, 32768, 65536):
        try:
            q = composite_prime_gen(2, 2 * (depth + 1), first_bits, scale_bits, 2 * degree, word_bits)
            p = util.generate_p_towers(q, dnum, degree, aux_bits=aux_bits)
        except ValueError:
            continue
        if he_params.qp_is_secure(q, p, degree, he_params.SecurityTarget(128, "classical")):
            return {"degree": degree, "depth": depth, "dnum": dnum, "q": q, "p": p,
                    "scale_bits": scale_bits, "first_bits": first_bits,
                    "register_word_bits": word_bits, "aux_bits": aux_bits,
                    "security_bits": 128, "security_model": "classical"}
    raise ValueError("No supported secure degree for the requested circuit")


def create_contexts(spec, rotations):
    import jax
    from jaxite_word import CKKSContext, key_gen
    jax.config.update("jax_enable_x64", True)
    q, p, degree, dnum = spec["q"], spec["p"], spec["degree"], spec["dnum"]
    keys = key_gen.gen_pke_pair(q, [], degree)
    evaluation = key_gen.gen_evaluation_key(keys["secret_key"], q=q, P=p, dnum=dnum)
    rotation_keys = {r: key_gen.gen_rotation_key(keys["secret_key"], q, p, r, dnum=dnum)[r]
                     for r in rotations}
    scale = float(q[-2]) * q[-1]
    params = dict(degree=degree, num_slots=degree // 2, scaling_factor=scale, output_scale=scale,
                  q_towers=q, p_towers=p, composite_degree=2, degree_layout=(64, degree // 64),
                  p=spec["scale_bits"], CKKS_M_FACTOR=1, max_bits_in_word=spec["first_bits"], noise_scale_degree=1)
    client = CKKSContext(dict(params, **keys))
    server = CKKSContext(dict(params, evaluation_key=[
        jax.numpy.asarray(evaluation[name], dtype="uint32").transpose(0, 2, 1) for name in ("a", "b")]))
    server.program_initialization(rotations, dnum=dnum, r=64, c=degree // 64, batch=1,
                                  pregenerated_rotation_keys=rotation_keys)
    return client, server


def make_arithmetic(server, cache_bytes=256 * 1024 ** 2, *, linear_chunk=0, rotation_cache_bytes=0):
    from encrypted_polynomial import EncryptedPolynomialEvaluator
    from encrypted_rotation import CompiledRotation
    from encrypted_compilation import CompiledTensorProduct

    class Arithmetic(EncryptedPolynomialEvaluator):
        def __init__(self):
            super().__init__(server, compiled_linear=True, compiled_multiply=True)
            self.rotator = CompiledRotation(server)
            if type(rotation_cache_bytes) is not int or rotation_cache_bytes < 0:
                raise ValueError("rotation cache budget must be a nonnegative integer")
            if rotation_cache_bytes:
                from rotation_cache import CachedRotation
                self.rotator = CachedRotation(server, rotation_cache_bytes)
            self.tensor_product = CompiledTensorProduct(server)
            self.public_cache = OrderedDict()
            self.cached_bytes = 0
            self.peak_cached_bytes = 0
            if type(linear_chunk) is not int or not 0 <= linear_chunk <= 16:
                raise ValueError("linear fusion chunk must be an integer in 0..16")
            from linear_fusion import FusedPlaintextSum
            self.linear_fusion = FusedPlaintextSum(server, linear_chunk) if linear_chunk else None

        def weighted_sum(self, values, weights, *, bias=0.):
            if self.linear_fusion is None:
                return super().weighted_sum(values, weights, bias=bias)
            return self.linear_fusion.weighted_sum(self, values, weights, bias=bias)

        def rotate(self, value, shift):
            return value if shift % self.ctx.num_slots == 0 else self.rotator.rotate(value, shift)

        def _encode_public(self, slots, level, scale):
            array = np.asarray(slots, dtype="<f8")
            key = (level, float(scale), array.shape, hashlib.sha256(array.tobytes()).digest())
            if key in self.public_cache:
                self.public_cache.move_to_end(key)
                return self.public_cache[key]
            value = super()._encode_public(slots, level, scale)
            size = value.polynomial.nbytes
            while self.public_cache and self.cached_bytes + size > cache_bytes:
                _, removed = self.public_cache.popitem(last=False)
                self.cached_bytes -= removed.polynomial.nbytes
            if size <= cache_bytes:
                self.public_cache[key] = value
                self.cached_bytes += size
                self.peak_cached_bytes = max(self.peak_cached_bytes, self.cached_bytes)
            return value

        def add_bias(self, value, bias):
            if not np.any(bias):
                return value
            level = self._level(value)
            plaintext = self._encode_public(bias, level, value._ckks_scale)
            return self.ctx.he_add[level].add_plain(value, plaintext)

        def quadratic(self, value, a1, a0):
            """WISE HerPN x²+a1*x+a0 with one genuine composite rescale."""
            level = self._level(value)
            if level < 1:
                raise ValueError("HerPN needs one remaining multiplication level")
            square = self.tensor_product.multiply(value, value)
            linear_weight = self._encode_public(a1, level, value._ckks_scale)
            linear = self._plaintext_multiply.multiply(value, linear_weight, level)
            result = self.ctx.he_add[level].add(square, linear)
            constant = self._encode_public(a0, level, square._ckks_scale)
            result = self.ctx.he_add[level].add_plain(result, constant)
            result = self._rescale.rescale(result, level, level - 1)
            result._ckks_nsd = 1
            return result

    return Arithmetic()
