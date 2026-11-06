"""Public PV key/count/actual metadata plan; no crypto or model backend."""
from pathlib import Path
import hashlib,json,math,sys,tarfile
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'checkout/cadathlon/common'))
from encrypted_qwen_head_routing import QwenHeadLayout
from encrypted_qwen_attention_routing import QwenAttentionRouting
from encrypted_scale_planning import weighted_sum_scale,multiply_scale,drop_divisor

def predecessor():
    pin=json.loads((P/'predecessor/pin.json').read_bytes())
    report=json.loads((P/'predecessor/report.json').read_bytes())
    digest=lambda raw:hashlib.sha256(raw).hexdigest()
    archive=P/'predecessor/sources.tar.gz'
    assert digest(archive.read_bytes())==pin['archive_sha256']==report['source_archive_sha256']
    assert report['passed'] is True and report['tests']==dict(tests=72,errors=0,failures=0,skipped=0)
    assert report['source_hashes_before']==report['source_hashes_after']
    with tarfile.open(archive) as t:
        members=t.getmembers()
        assert len(members)==len({m.name for m in members})==len(report['source_hashes_before'])
        assert all(m.isfile() for m in members)
        assert {m.name:digest(t.extractfile(m).read()) for m in members}==report['source_hashes_before']
    head='checkout/cadathlon/common/encrypted_qwen_head_routing.py'
    assert digest((P/head).read_bytes())==pin['runtime_sha256']==report['source_hashes_before'][head]
    assert digest((P/'test_head_routing.py').read_bytes())==report['source_hashes_before']['test_head_routing.py']
    return pin

def derive():
    inherited=predecessor()
    profile=next(x for x in json.loads((P/'profile_audit.json').read_bytes())['profiles'] if x['name']=='depth49_p60_d8')
    q=tuple(profile['q']);delta=profile['canonical_scale'];records=[]
    for size in (1,8,16,19,32):
        layout=QwenHeadLayout(tuple(range(size)))
        # Only pure key inventory uses this uninitialized view; no context or
        # primitive is constructed by the public planner.
        view=object.__new__(QwenAttentionRouting);view.layout=layout
        keys=view.required_rotation_indices();tiles=[]
        for tile in range(layout.tile_count):
            rows=layout.tile_rows(tile)
            tiles.append(dict(query_tile=tile,valid_rows=rows,
                query_positions=list(layout.tile_positions(tile)),ciphertext_products=size,
                weighted_sums=2*size+1,additions=8*size-1,
                rotations=sum(rows-int(k%16<rows)+7+15+int(k!=0) for k in range(size)),
                depth_below_minimum_input=3,reserved_output_levels=2))
        records.append(dict(sequence=size,tiles=tiles,required_component_rotation_keys=list(keys),
            raw_rotation_key_payload_bytes=len(keys)*profile['rotation_pair_bytes']))
    probability25=math.nextafter(delta,math.inf);value24=delta
    probability_anchors24=weighted_sum_scale([probability25]*16,delta,drop_divisor(q,25))['output_scale']
    value_broadcast23=weighted_sum_scale([value24]*16,delta,drop_divisor(q,24))['output_scale']
    product22=multiply_scale(probability_anchors24,value_broadcast23,drop_divisor(q,23))
    output21=weighted_sum_scale([product22],delta,drop_divisor(q,22))['output_scale']
    return dict(scope='Public head/PV geometry and actual binary64 metadata only; no backend, model or HE.',
        predecessor=inherited,probability_fixture='Explicit public stand-in derived from prior score output; no encrypted softmax.',
        layout=dict(slots=65536,row_width=4096,query_heads=16,kv_heads=8,head_dim=128,key_stride=32),
        workloads=records,scale_example=dict(probabilities25=probability25,expanded_values24=value24,
            probability_anchors24=probability_anchors24,value_broadcast23=value_broadcast23,
            products22=product22,attended_output21=output21),
        key_scope='Combined head/RoPE/score/PV component only; excludes projections, RMSNorm, softmax and refresh.',
        memory_scope='Raw public rotation-key payload only; excludes relin/PK, controls, copies, compiler and ciphertexts.')

if __name__=='__main__':
    report=derive();(P/'public_plan.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    assert not any(n in sys.modules for n in ('jax','jaxlib','torch','transformers','safetensors'))
    print(json.dumps(dict(keys_S32=len(report['workloads'][-1]['required_component_rotation_keys']),scale_example=report['scale_example'])))
