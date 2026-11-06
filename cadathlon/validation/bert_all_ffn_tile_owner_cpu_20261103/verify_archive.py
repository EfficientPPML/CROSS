"""Offline source/pin/ideal-case verification; no model, arrays or HE loaded."""
from pathlib import Path
import hashlib,json,math,tarfile
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read_tar(path,expected):
    with tarfile.open(path) as t:
        ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==len(expected)
        assert all(m.isfile() and not m.name.startswith('/') and '..' not in Path(m.name).parts for m in ms)
        data={m.name:t.extractfile(m).read() for m in ms}
    assert {n:hashlib.sha256(b).hexdigest() for n,b in data.items()}==expected
    return data

def main():
    for name,digest in json.loads((ROOT/'artifact_sha256.json').read_text()).items():assert sha(ROOT/name)==digest,name
    report=json.loads((ROOT/'source-bound-v1/report.json').read_text())
    assert report['passed'] and report['test_returncode']==0 and report['child_reaped']
    assert report['original_sources_unchanged'] and report['executed_snapshot_unchanged'] and report['assets_unchanged']
    sources=report['source_hashes_before'];assert sources==report['original_hashes_after']
    assert sha(ROOT/'source-bound-v1/sources.tar.gz')==report['source_archive_sha256']
    data=read_tar(ROOT/'source-bound-v1/sources.tar.gz',sources)
    manifest=json.loads(data['candidate_manifest.json']);pins=json.loads(data['ffn_tensor_pins.json']);certificate=json.loads(data['gelu_refresh_certificate.json'])
    assert pins['passed'] and pins['tensors']==48 and set(pins['sites'])=={str(x) for x in range(12)}
    assert certificate['passed'] and certificate['manifest_sha256']==hashlib.sha256(data['candidate_manifest.json']).hexdigest()
    fixture_contract=json.loads(data['fixture_contract.json']);prep=json.loads((ROOT/'native-fixtures-v1/fixture_preparation.json').read_text())
    assert prep['passed'] and prep['source_files_unchanged'] and prep['source_archive_verified'] and prep['asset_files_unchanged'] and not prep['cleanup_errors']
    assert prep['source_hashes_before']==prep['source_hashes_after']
    assert sha(ROOT/'native-fixtures-v1/sources.tar.gz')==prep['source_archive_sha256']
    read_tar(ROOT/'native-fixtures-v1/sources.tar.gz',prep['source_hashes_before'])
    assert sha(ROOT/'native-fixtures-v1/fixture_preparation.json')==fixture_contract['preparation_report_sha256']
    assert prep['fixture_npz_sha256']==fixture_contract['fixture_npz_sha256']==report['assets_after']['native_fixture_sha256']
    assert len(prep['arrays'])==26
    assert len(report['cases'])==24
    slots=tiles=refreshes=0;worst=0.;bounds=[];levels=[]
    for layer in range(12):
        gelu=manifest['gelu'][str(layer)]['polynomial'];norm=manifest['norms'][f'{layer}.mlp']
        gd=(gelu['degree']-1).bit_length()*2+2
        nd=(norm['polynomial']['degree']-1).bit_length()*(1 if norm['method']=='power' else 2)+7
        bound=float(certificate['sites'][str(layer)]['refresh_bound']);bounds.append(bound);levels.append(24-nd)
        for K in (15,128):
            c=report['cases'][f'{layer}/K{K}'];t=math.ceil(K/16)
            assert c['passed'] and c['layer']==layer and c['K']==K and c['tiles']==t
            assert c['physical_slots_checked']==65536*t and c['actual_loop_vs_independent_polynomial_gate']==1e-8
            assert all(v<1e-8 for v in c['max_abs_errors'].values())
            assert c['identity_refresh_calls']==2*t and not c['encrypted_or_refresh_error_claim']
            obs=c['domain_observations'];glo,ghi=gelu['domain'];nlo,nhi=norm['polynomial']['domain']
            assert glo<=obs['gelu_input_min']<=obs['gelu_input_max']<=ghi
            assert nlo<=obs['mlp_variance_min']<=obs['mlp_variance_max']<=nhi
            assert obs['input_max_abs']<128 and obs['gelu_max_abs']<bound
            for m in c['metadata']:
                assert m['layer']==layer and m['explicit_refresh_bounds']==[128.,bound]
                assert m['output_level']==24-nd>=2 and m['normalization_method']==norm['method']
                assert [x['level'] for x in m['actual_metadata']]==[25,24,24-gd,25,24,24,24-nd]
                assert not m['range_verified'] and not m['encrypted_error_verified']
            plan=c['public_operation_plan']
            assert c['actual_primitive_counts']=={name:t*count for name,count in plan['per_tile_primitive_calls'].items()}
            assert c['public_plaintext_products']==t*plan['per_tile_public_plaintext_terms']
            slots+=c['physical_slots_checked'];tiles+=t;refreshes+=c['identity_refresh_calls'];worst=max(worst,*c['max_abs_errors'].values())
    assert tiles==108 and refreshes==216 and slots==7077888
    assert bounds==[16.,64.,32.,32.,32.,32.,16.,64.,64.,64.,64.,16.]
    print(json.dumps(dict(passed=True,source_files=len(sources),layers=12,cases=24,tiles=tiles,physical_slots_checked_per_stage=slots,identity_refresh_calls=refreshes,worst_actual_loop_vs_polynomial=worst,final_levels=levels,scope='Offline archive and ideal-result consistency only; no HE or model inputs loaded.')))
if __name__=='__main__':main()
