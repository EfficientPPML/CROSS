"""Registry lifecycle counterexample; tiny insecure arithmetic only."""
from pathlib import Path
import sys,json,gc,weakref
HERE=Path(__file__).resolve().parent;ROOT=HERE/'checkout';sys.path[:0]=[str(ROOT),str(ROOT/'cadathlon/common')]
import jax,numpy as np
from jaxite_word import Polynomial
import polynomial
jax.config.update('jax_enable_x64',True)
shape=dict(batch=1,num_elements=2,degree=256,num_moduli=2,precision=32,degree_layout=(16,16))
source=Polynomial(shape,{'moduli':[12289,40961]})
source.polynomial=jax.numpy.ones(source.shape,dtype='uint32')
output=Polynomial(dict(shape,num_moduli=1),{'moduli':[12289]})
ref=weakref.ref(output.ntt_ctx)
def lower(x):return output._clone_with_payload(x.polynomial[...,:1])
compiled=jax.jit(lower)
first=compiled(source);jax.block_until_ready(first.polynomial)
keys_before=len(polynomial._POLYNOMIAL_REGISTRY)
polynomial._POLYNOMIAL_REGISTRY.clear()
try:
 second=compiled(source);jax.block_until_ready(second.polynomial);error=None
except Exception as e:error=type(e).__name__+': '+str(e)
report={'scope':__doc__,'registry_entries_before_clear':keys_before,'cached_polynomial_output_after_registry_clear_error':error,'expected_counterexample':error is not None}
print(json.dumps(report));(HERE/'pytree_registry.json').write_text(json.dumps(report,indent=2)+'\n')
