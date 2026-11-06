"""Public geometry/scale/key plan only; no arithmetic backend imports."""
from pathlib import Path
from types import SimpleNamespace
import json,sys,math
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'checkout/cadathlon/common'))
from encrypted_qwen_head_routing import QwenHeadLayout,QwenHeadRouting
from encrypted_scale_planning import weighted_sum_scale,multiply_scale,drop_divisor

def derive():
    profile=next(x for x in json.loads((P/'profile_audit.json').read_bytes())['profiles'] if x['name']=='depth49_p60_d8')
    q=tuple(profile['q']);delta=profile['canonical_scale']
    records=[]
    for size in (1,8,16,19,32):
        layout=QwenHeadLayout(tuple(range(size)))
        keys=QwenHeadRouting.required_rotation_indices(SimpleNamespace(layout=layout))
        tiles=[]
        for t in range(layout.tile_count):
            rows=layout.tile_rows(t)
            rotations=sum(rows-int(k%16<rows)+7+15+int(k!=0) for k in range(size))
            tiles.append(dict(query_tile=t,valid_rows=rows,query_positions=list(layout.tile_positions(t)),
                score_ciphertext_products=size,score_weighted_sums=2*size,score_rotations=rotations,
                score_additions=8*size-1,score_depth_below_minimum_input=3))
        records.append(dict(sequence=size,tiles=tiles,required_component_rotation_keys=list(keys),
            raw_rotation_pair_bytes=len(keys)*profile['rotation_pair_bytes']))
    source=math.nextafter(delta,math.inf)
    rope=weighted_sum_scale([source]*3,delta,drop_divisor(q,26))['output_scale']
    gqa=weighted_sum_scale([rope]*9,delta,drop_divisor(q,25))['output_scale']
    broadcast=weighted_sum_scale([gqa]*16,delta,drop_divisor(q,24))['output_scale']
    product=multiply_scale(rope,broadcast,drop_divisor(q,23))
    scores=weighted_sum_scale([product]*16,delta,drop_divisor(q,22))['output_scale']
    return dict(scope='Public geometry/actual binary64 metadata only. No keygen, backend or HE.',
        layout=dict(slots=65536,row_width=4096,query_heads=16,kv_heads=8,head_dim=128,key_stride=32),
        methods=dict(expand_kv=dict(depth=1,rotations=8,weighted_sums=1),rope=dict(depth=1,rotations=2,weighted_sums=1)),
        workloads=records,scale_example=dict(input26=source,rope25=rope,expanded_key24=gqa,
            broadcast_key23=broadcast,products22=product,scores21=scores),
        key_scope='Component only; no projection/RMSNorm/softmax/PV/bootstrap union. Reusable inventory deliberately includes all16 row-rotation offsets even for shorter layouts.',
        memory_scope='Raw public key payload count only, excludes relin/PK, controls, copies, compiler and ciphertext memory.')

if __name__=='__main__':
    (P/'public_plan.json').write_text(json.dumps(derive(),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(keys_S32=len(derive()['workloads'][-1]['required_component_rotation_keys']),scale_example=derive()['scale_example'])))
