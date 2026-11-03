"""CPU-only abstract-shape bounds for proposed feature batching; no HE execution.

The summed SSA outputs bound the explicitly represented logical NTT arrays if
all outputs are retained. This is intentionally loose. Compiler-internal dot
workspaces, allocator fragmentation, public tables and other live tensors are
not included, so this is NOT an HBM capacity guarantee or a latency estimate.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
if os.environ['JAX_PLATFORMS'] != 'cpu':
    raise RuntimeError('Abstract planning requires JAX_PLATFORMS=cpu')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import jax
import jax.numpy as jnp
from jaxite_word import he_params, util
from jaxite_word.composite_prime_gen import composite_prime_gen
from jaxite_word import finite_field, ntt_mm

jax.config.update('jax_enable_x64', True)
N, R, C, Q = 131072, 256, 512, 84
q = composite_prime_gen(2, Q, 61, 56, 2*N, 31)
sources = [Path(__file__), ROOT/'jaxite_word/hemul.py', ROOT/'jaxite_word/ntt_mm.py',
           ROOT/'jaxite_word/util.py', ROOT/'jaxite_word/finite_field.py',
           ROOT/'jaxite_word/bconv.py']
hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}


def bytes_of(aval):
    return math.prod(aval.shape) * aval.dtype.itemsize if hasattr(aval, 'shape') else 0


def inventory(program):
    sizes = []
    for equation in program.eqns:
        sizes.extend(bytes_of(value.aval) for value in equation.outvars)
        for parameter in equation.params.values():
            children = parameter if isinstance(parameter, (list, tuple)) else (parameter,)
            for child in children:
                child = getattr(child, 'jaxpr', child)
                if hasattr(child, 'eqns'):
                    sizes.extend(inventory(child))
    return sizes


def ntt_logical_bound(batch, moduli):
    # Skip allocation-heavy constructor: the core NTT implementation needs
    # only the three explicit public transform operands and small FF controls.
    operator = object.__new__(ntt_mm.NTTCiphertextBarrettContext)
    operator.ff_ctx = finite_field.BarrettContext(moduli)
    count = len(moduli)
    def forward(payload, first, middle, last):
        operator.ntt_bat_tf_step1 = first
        operator.ntt_tf_step2 = middle
        operator.ntt_bat_tf_step3 = last
        return ntt_mm.NTTCiphertextContextBase.ntt(operator, payload)
    shapes = ((batch, R, C, count), (R, 4, R, 4, count),
              (R, C, count), (C, 4, C, 4, count))
    types = (jnp.uint32, jnp.uint8, jnp.uint32, jnp.uint8)
    closed = jax.make_jaxpr(forward)(*[jax.ShapeDtypeStruct(s, t) for s, t in zip(shapes, types)])
    sizes = inventory(closed.jaxpr)
    return {'flattened_batch': batch, 'limbs': count,
            'largest_logical_temporary_bytes': max(sizes),
            'sum_all_ssa_output_bytes_upper_bound': sum(sizes),
            'input_payload_bytes': batch*N*count*4,
            'one_direction_public_table_bytes': 16*(R*R+C*C)*count + 4*N*count}


report = {'scope': __doc__, 'degree': N, 'q_count': Q, 'profiles': {},
          'batch_contract': 'One feature per Polynomial batch lane, identical key/context/level/scale/nsd; plaintext coefficients broadcast from batch one.',
          'no_extra_rotation_keys': True, 'jax_version': jax.__version__, 'source_sha256': hashes}
for dnum in (3, 8):
    p = util.generate_p_towers(q, dnum, N, aux_bits=30)
    he_params.assert_p_coverage(q, p, dnum)
    secure = he_params.qp_is_secure(q, p, N, he_params.SecurityTarget(128, 'classical'))
    assert secure
    alpha = math.ceil(Q/dnum)
    parts = math.ceil(Q/alpha)
    # Largest complement uses the smallest final partition. The no-rescale
    # bootstrap tensor product is the conservative top-Q primitive here.
    last_partition = Q - alpha*(parts-1)
    complement = Q + len(p) - last_partition
    profile = {'p_count': len(p), 'parts': parts, 'alpha': alpha, 'actual_qp_secure': secure,
               'raw_key_pair_bytes': 2*parts*(Q+len(p))*N*4,
               'formatted_uint64_key_pair_bytes': 2*parts*(Q+len(p))*N*8,
               'max_complement_limbs': complement, 'batches': {}}
    for batch in (1, 2, 4):
        # NTT consumers are the original third component, one complemented
        # partition, and two P components during modulus down. All are exact
        # core NTT calls over abstract arrays; no key tables are allocated.
        cases = [('original_q', batch, q),
                 ('partition_complement', batch, (q+p)[-complement:]),
                 ('moddown_p_two_components', 2*batch, p)]
        profile['batches'][str(batch)] = {
            'ciphertext_payload_bytes': batch*2*N*Q*4,
            'two_inputs_plus_output_payload_bytes': 3*batch*2*N*Q*4,
            'ntt_shapes': {name: ntt_logical_bound(flat_batch, moduli) for name, flat_batch, moduli in cases}}
    report['profiles'][str(dnum)] = profile
report['source_unchanged'] = hashes == {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
assert report['source_unchanged']
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--report', type=Path, default=Path(__file__).with_suffix('.json'))
args = parser.parse_args()
args.report.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
