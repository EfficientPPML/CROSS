"""Unoptimized, real CROSS baseline; no alternative model or small test ring.

One image per ciphertext, one ciphertext per chip. This is the serving shape
the CROSS submission to the FHE benchmarking suite uses today, and it leaves
the ring mostly empty: the derived plan carries 8192 slots and the network
uses 784 of them.

Construction is lazy on purpose. The benchmark harness runs encryption,
evaluation and decryption as three separate processes, and only the middle one
needs the compiled `Mapping`. Building it in the client processes too would
add minutes to every stage and would not measure anything real, so the client
path builds just the codec context and the `Mapping` is built on first
`execute`.
"""


class BaselineService:
    """Encrypt / execute / decrypt for a batch, one image per ciphertext."""

    def __init__(self, model, input_shape, batch, devices, keys=None):
        from jaxite_word import nn, packing
        program = nn.vectorize(model, input_shape, warn=False)
        self.packed = packing.pack(program, packing.PackingPolicy(lazy_constants=True))
        self.batch = batch
        self.devices = tuple(devices)
        self.lanes = len(self.devices)
        self.keys = keys
        self._ctx = None
        self._mapping = None

    # -- client side: the codec only, no compiled program ------------------
    def _context(self):
        if self._ctx is None:
            import mapping as mapping_mod
            from ckks_ctx import CKKSContext
            self._ctx = CKKSContext(mapping_mod.ring_runtime_parameters(
                self.packed.ring_config, keys=self.keys))
        return self._ctx

    # -- server side: the compiled Mapping ---------------------------------
    def _compiled(self):
        if self._mapping is None:
            from jaxite_word import Mapping
            self._mapping = Mapping(self.packed, global_batch=self.lanes,
                                    devices=self.devices, keys=self.keys)
        return self._mapping

    def encrypt(self, images):
        ctx = self._context()
        scale = float(self.packed.ring_config.scaling_factor)
        rows = [row for row in images]
        if len(rows) != self.batch:
            raise ValueError(f"expected {self.batch} images, got {len(rows)}")
        return [ctx.encrypt_slots(self.packed.pack(row), scale=scale) for row in rows]

    @staticmethod
    def _stack(window):
        """Concatenate batch-1 ciphertexts into one batch-N ciphertext.

        Public API only: `to_array` exposes the canonical rank-5 payload and
        `Polynomial.from_array` wraps one back, so the tracked CKKS scale has
        to be carried across by hand.
        """
        import jax.numpy as jnp
        import numpy as np
        from polynomial import Polynomial

        head = window[0]
        payload = np.concatenate([np.asarray(ct.to_array()) for ct in window], axis=0)
        rows, columns = head.degree_layout
        stacked = Polynomial.from_array(
            jnp.asarray(payload, dtype=jnp.uint32),
            {"batch": payload.shape[0], "num_elements": payload.shape[1],
             "degree": rows * columns, "num_moduli": payload.shape[4],
             "precision": head.precision, "degree_layout": (rows, columns)},
            {"moduli": list(head.get_moduli())})
        for attribute in ("_ckks_scale", "_ckks_nsd"):
            if hasattr(head, attribute):
                setattr(stacked, attribute, getattr(head, attribute))
        return stacked

    def execute(self, bundle):
        mapping = self._compiled()
        outputs = []
        for start in range(0, len(bundle), self.lanes):
            window = list(bundle[start:start + self.lanes])
            live = len(window)
            window += [window[-1]] * (self.lanes - live)
            result = mapping.execute(self._stack(window))
            # batch_slice is the public way back to one ciphertext per image
            outputs.extend(result.batch_slice(lane) for lane in range(live))
        return outputs

    def decrypt(self, bundle):
        import numpy as np
        ctx = self._context()
        rows = []
        for ct in bundle:
            scale = getattr(ct, "_ckks_scale", None)
            slots = np.asarray(ctx.decrypt_slots(ct, scale=scale))
            rows.append(self.packed.unpack(np.real(slots)))
        return np.asarray(rows)

    def release(self):
        if self._mapping is not None:
            self._mapping.release()
            self._mapping = None
        self._ctx = None


def build_service(model, input_shape, *, batch, devices, keys=None):
    return BaselineService(model, input_shape, batch=batch, devices=devices, keys=keys)
