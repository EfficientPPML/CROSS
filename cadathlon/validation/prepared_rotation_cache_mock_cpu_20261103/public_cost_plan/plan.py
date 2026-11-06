"""Source/count-only planning. No crypto, model, backend or payload imports."""
from pathlib import Path
import hashlib, json, math

HERE=Path(__file__).resolve().parent
BASE=Path('/tmp/bert-server-compute-stage4-successor-v2-20261103')
COMMON=BASE/'joined/checkout/cadathlon/common'
FILES=[COMMON/name for name in ('encrypted_rotation.py','encrypted_polynomial.py',
    'encrypted_packed_qwen.py','encrypted_packed_transformer.py',
    'encrypted_chained_streaming_linear.py','encrypted_shallow_score_routing.py',
    'encrypted_factored_score_routing.py','encrypted_bert_attention_tile.py',
    'encrypted_compilation.py','encrypted_ntt.py')]
FILES += [BASE/'runtime/jaxite_word/he_params.py',
          BASE/'runtime/stage/key_contexts.py',BASE/'joined/encrypted_bert_layer_attention.py']
AGG=Path('/tmp/bert-full-model-immutable-masks-20261103/source-bound-v1/report.json')
TEST=AGG.parent.parent/'test_full_model_owner.py'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ct(q):return 2*131072*q*4
def prepared(q):
    digits=math.ceil(q/13)
    return 2*digits*131072*(q+14)*4+131072*4

def run():
    before={str(p):sha(p) for p in FILES+[AGG,TEST]}
    total=906829; tiles=8; layers=12
    route=26*128*2*tiles*layers
    attention=4328*2*tiles*layers
    rows=[]
    for q in (100,98,96,94,50,48):
        rows.append(dict(q=q,ciphertext_bytes=ct(q),prepared_pair_and_map_bytes=prepared(q),
                         two_entry_cache_bytes=2*prepared(q),
                         hoisted_digits_bytes=math.ceil(q/13)*131072*(q+14)*4))
    result=dict(scope='Public formula/source analysis; no new model/HE/backend execution or timing claim',
        source_hashes=before,baseline_K128_rotations=total,
        attention_rotations=attention,attention_fraction=attention/total,
        routing_internal_rotations=route,routing_fraction=route/total,
        attention_one_query_tile=dict(score_rotations=4328,value_mix_rotations=4328,
            ciphertext_products=256,routing_plaintext_terms=7168,
            route_maps_per_direction=128,route_internal_rotations_per_map=26),
        dense=dict(square_768=dict(diagonals=1535,giant_groups=24,rotations=86),
                   rectangular_768_3072=dict(diagonals=3839,giant_groups=60,rotations=122),
                   chained_key_union=[1,64,61760,64064]),
        bounded_cache=dict(max_entries=2,suggested_byte_budget=2<<30,
            full_Q100_pair_and_map_bytes=prepared(100),
            scope='Reuses completed explicit key operands only; no rotation/rescale/order change',
            repeated_dense_baby_calls=63,repeated_dense_baby_key=1,
            caveat='A single direct QK projection mostly uses distinct keys; this does not explain its wall time'),
        qkv_shared_babies=dict(saved_rotations_per_layer=16*63,
            saved_rotations_all_layers=16*63*12,fraction=16*63*12/total,
            extra_retained_Q_bytes_layer0=8*ct(98),extra_retained_Q_bytes_later=8*ct(50),
            one_max_level_64_ciphertext_baby_bank_bytes=64*ct(100)),
        reused_key_value_alignments=dict(saved_rotations_per_layer=240*7,
            saved_rotations_all_layers=240*7*12,fraction=240*7*12/total,
            full_aligned_cache_bytes_layer0=240*ct(98),
            full_aligned_cache_bytes_later=240*ct(50)),
        separate_exact_order_hoisting_proposal=dict(saved_decompositions=24*128*2*8*12,
            rotation_count_unchanged=total,
            warning='Existing QP hoisted API cannot be substituted without checking ModDown placement and exact residue parity'),
        levels=rows,
        notes=['Counts exclude bootstrap internals; baseline used identity refresh in its ideal test.',
               'Device transfer, public encoding, NTT construction and JAX tracing costs are not measured here.',
               'Prepared cache byte budget excludes transient conversions, controls, ciphertexts and compiler buffers.',
               'Mutable key backing cannot use an identity-only content cache. Authenticated readonly mapping lifecycle must be bound.',
               'One-level weighted sums encode every non-scalar diagonal; caching all square-map plaintexts at Q100 costs about75GiB.'])
    after={str(p):sha(p) for p in FILES+[AGG,TEST]}
    result['source_hashes_after']=after;result['source_files_unchanged']=before==after
    result['passed']=before==after
    (HERE/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','attention_fraction','routing_fraction','bounded_cache')}))
    return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(run())
