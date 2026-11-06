"""Portable verification of public evidence; does not execute model arithmetic."""
from pathlib import Path
import hashlib,json,math,tarfile,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parent
def verify(out):
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    r=json.loads((out/'report.json').read_bytes());p=json.loads((out/'public_plan.json').read_bytes())
    assert r['passed'] is True and r['encrypted_execution'] is False
    assert all(r[n] is True for n in ('cleanup_completed','terminal_scope_completed','termination_scope_completed','source_files_unchanged','predecessor_verified','zero_audited_parameter_input_reads'))
    assert r['source_hashes_before']==r['source_hashes_after']
    assert sha(out/'sources.tar.gz')==r['source_archive_sha256'] and sha(out/'public_plan.json')==r['public_plan_sha256']
    with tarfile.open(out/'sources.tar.gz')as t:
        ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==len(r['source_hashes_before']) and all(m.isfile() for m in ms)
        data={m.name:t.extractfile(m).read() for m in ms}
    assert {n:hashlib.sha256(b).hexdigest() for n,b in data.items()}==r['source_hashes_before']
    inventory=json.loads(data['public_inventory.json']);assert set(inventory['sources'])==set(data)
    suites=ET.parse(out/'tests.xml').getroot().findall('testsuite');tests={n:sum(int(s.get(n,0)) for s in suites) for n in ('tests','failures','errors','skipped')}
    assert tests==r['tests']==dict(tests=inventory['expected_tests'],failures=0,errors=0,skipped=0)
    assert len(r['children'])==2 and all(c['direct_returncode']==0 and all(c[n] is True for n in ('direct_child_reaped','group_empty_verified','cleanup_completed')) for c in r['children'])
    for name in ('tests-audit.json','plan-audit.json'):
        a=json.loads((out/name).read_bytes());assert a['passed'] is True and a['denied_attempts']==a['backend_modules']==[]
    manifest=json.loads(data['inputs/candidate_manifest.json']);sites=json.loads(data['inputs/site_metadata.json'])
    keys=json.loads(data['inputs/actual_key_plan.json'])
    assert p['passed'] is True and p['reference']['executed'] is False and len(p['layers'])==28 and len(p['workloads'])==32
    assert p['resources']['rotation_indices']==keys['combined_keys'] and p['resources']['rotation_keys']==166
    assert p['resources']['raw_rotation_store_bytes']==158746017792 and p['resources']['all38_final_ciphertext_bytes']==532*1024**2
    inverse=sum(len(sites[f'{layer}.{kind}.inverse_sqrt']['refresh_events']) for layer in range(28) for kind in ('input','q','k','post'));assert inverse==260
    silu=sum(manifest['nonlinear'][f'{layer}.silu'].get('doublings',0) for layer in range(28));assert silu==145
    soft_factor=sum(3+2*manifest['softmax'][str(layer)]['squarings'] for layer in range(28))
    for s,w in enumerate(p['workloads'],1):
        tiles=(s+15)//16;c=w['refresh_counts']
        assert w['sequence']==s and w['tiles']==tiles and c==dict(rms_inverse=260*tiles,silu=145*tiles,attention_outer=149*tiles,tail_outer=56*tiles,interlayer=27*tiles,final_input=1,softmax=(s-1)*soft_factor,total=(260+145+149+56+27)*tiles+1+(s-1)*soft_factor)
        assert w['last_tile']==(s-1)//16 and w['last_local_row']==(s-1)%16 and w['last_position']==3*(s-1)
        assert w['last_row_rotation']==4096*((s-1)%16) and (not w['last_row_rotation'] or w['last_row_rotation'] in keys['combined_keys'])
        assert [e['input_level'] for e in w['entry_events']]==[4]*27+[3]
        assert w['entry_events'][-1]['bound']==2048.
        assert [e['output_level'] for e in w['final_stages']]==[3,3,25,7,6]
    assert p['workloads'][-1]['refresh_counts']['total']==12931
    assert p['final_norm']['degree']==64 and p['final_norm']['output_level']==7 and p['final_norm']['inverse_refresh_count']==0
    b=p['vocabulary']['blocks'];assert len(b)==38 and [(x['start'],x['stop']) for x in b]==[(s,min(s+4096,151936)) for s in range(0,151936,4096)]
    assert all(x['physical_row']==0 and x['output_level']==6 for x in b)
    assert len(p['final_scale_examples'])==9 and all(math.isfinite(x['actual_all38_L6_scale']) and x['actual_all38_L6_scale']>0 for x in p['final_scale_examples'])
    print('VERIFIED PUBLIC ONLY:28 layers,32 schedules,38 full-vocabulary blocks,166 keys,114 checks; no model/HE trajectory.')
    return r
if __name__=='__main__':verify(P/'public-v1')
