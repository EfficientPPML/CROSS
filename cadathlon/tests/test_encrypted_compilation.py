"""Compiled code must not keep retired public-control device arrays alive."""
import gc
from pathlib import Path
import sys
import weakref

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
from encrypted_compilation import ExplicitArrayCompiler


def test_code_reuse_survives_global_cache_clear_without_retaining_control_buffers():
    import jax
    import jax.numpy as jnp

    compiler = ExplicitArrayCompiler()
    payload = jnp.arange(8, dtype=jnp.float32)

    def bind(offset):
        controls = jnp.full((8,), offset, dtype=jnp.float32)
        reference = weakref.ref(controls)
        binding = compiler.bind(lambda data: data + controls, (payload,),
                                signature='public-add', device=jax.devices()[0])
        return binding, reference

    for offset in (1., 3., -2.):
        binding, controls_reference = bind(offset)
        actual = binding(payload)
        np.testing.assert_array_equal(actual, np.arange(8, dtype=np.float32) + offset)
        del binding, actual
        jax.clear_caches()
        gc.collect()
        assert controls_reference() is None
    assert compiler.compile_count == 1
