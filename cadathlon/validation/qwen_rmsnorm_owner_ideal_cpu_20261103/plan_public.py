"""Public exact circuit/scale/key inventory for the isolated RMSNorm owner."""
from pathlib import Path
import hashlib,json,math,sys
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'checkout/cadathlon/common'))
from encrypted_qwen_rmsnorm import QwenRMSContract,rms_scale_plan,DELTA
def derive():
    raw=(P/'inputs/candidate_manifest.json').read_bytes();meta=(P/'site_metadata.json').read_bytes()
    audit=json.loads((P/'inputs/profile_audit.json').read_bytes());profile=next(p for p in audit['profiles'] if p['name']=='depth49_p60_d8')
    records=[]
    for site in sorted(json.loads(meta)):
        c=QwenRMSContract(raw,meta,site);bits=c.width.bit_length()-1
        scales=tuple(math.nextafter(DELTA,math.inf if i%2 else -math.inf) for i in range(len(c.events)))
        plan=rms_scale_plan(profile['q'],c,input_scale=math.nextafter(DELTA,math.inf),refresh_scales=scales)
        keys=sorted({shift%65536 for b in range(bits) for shift in (1<<b,-(1<<b))})
        union=sorted(set(keys)|set(audit['bootstrap_rotation_indices'])) if c.events else keys
        records.append(dict(site=site,width=c.width,groups=c.groups,domain=c.domain,method=c.method,
            degree=c.degree,iterations=c.iterations,events=c.events,gamma_sha256=c.gamma_sha256,
            rotations=2*bits,additions=2*bits,
            ciphertext_products=c.degree+1 if c.method=='polynomial' else 3*c.iterations-1,
            weighted_sums=c.degree+3 if c.method=='polynomial' else c.iterations+2,
            norm_rotation_keys=keys,component_and_refresh_key_union=union,
            raw_rotation_key_payload_bytes=len(union)*profile['rotation_pair_bytes'],scale_example=plan))
    assert len(records)==113 and sum(len(r['events']) for r in records)==260
    return dict(scope='Public fixed-circuit/actual binary64 metadata with explicitly assumed identity-refresh adjacent scales; no HE error proof.',
        manifest_sha256=hashlib.sha256(raw).hexdigest(),sites=records,site_count=113,
        total_inverse_refresh_events=260,degree512_retained_basis_raw_bytes=7511998464,
        raw_input_bound_max=32768,normalized_output_bound_max=8192,
        refresh_scale_scope='Example values only; runtime always consumes actual returned metadata and replays the same operations.',
        key_scope='Per-site normalization and required bootstrap union only; not full Qwen key inventory.')
if __name__=='__main__':
    r=derive();(P/'public_plan.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    assert not any(n in sys.modules for n in ('jax','jaxlib','torch','transformers','safetensors'))
    print(json.dumps(dict(sites=113,refresh_events=260,max_keys=max(len(x['component_and_refresh_key_union']) for x in r['sites']))))
