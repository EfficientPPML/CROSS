"""Offline evidence integrity and independent all24 norm schedule accounting."""
from pathlib import Path
import hashlib,json,math,tarfile
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    for n,h in json.loads((HERE/'artifact_sha256.json').read_text()).items():assert sha(HERE/n)==h,n
    d=HERE/'measured';r=json.loads((d/'report.json').read_text())
    assert r['passed'] and r['returncode']==0 and r['child_reaped']
    assert r['source_files_unchanged'] and r['snapshot_unchanged'] and not r['encrypted_execution']
    assert r['external_checkpoint_unchanged'] and r['external_checkpoint_sha256_before']==r['external_checkpoint_sha256_after']=='e1f8085ef673c9eb8a1a93a84270742e8aa0ccbceadef01eef15e26c44b46034'
    before=r['source_hashes_before'];assert before==r['source_hashes_after'] and len(before)==121
    assert sha(d/'sources.tar.gz')==r['source_archive_sha256']
    with tarfile.open(d/'sources.tar.gz') as t:
        members=t.getmembers();assert len(members)==len({m.name for m in members})==121 and all(m.isfile() for m in members)
        contents={m.name:t.extractfile(m).read() for m in members}
    assert {n:hashlib.sha256(v).hexdigest() for n,v in contents.items()}==before
    assert sha(HERE/'source/encrypted_bert_all_row_norm.py')==before['checkout/cadathlon/common/encrypted_bert_all_row_norm.py']
    manifest=json.loads(contents['candidate_manifest.json']);cases=r['cases'];assert len(cases)==288
    assert r['physical_slots_checked']==sum(c['slots_checked'] for c in cases)==18874368
    assert r['max_abs_vs_independent_chebval']==max(c['max_abs_vs_independent_chebval'] for c in cases)
    schedules=0
    for layer in range(12):
        for role in ('attention','mlp'):
            site=f'{layer}.{role}';rule=manifest['norms'][site];degree=rule['polynomial']['degree'];method=rule['method']
            pd=2+math.ceil(math.log2(degree))*(1 if method=='power' else 2);depth=pd+5;schedules+=50-depth-2
            rows=[c for c in cases if c['site']==site];assert len(rows)==12
            assert len({(c['valid_rows'],c['level_choice'],c['source_scale']) for c in rows})==12
            assert {c['valid_rows'] for c in rows}=={1,16} and {c['level_choice'] for c in rows}=={'minimum','normal'}
            assert len({c['source_scale'] for c in rows})==3
            for c in rows:
                assert c['degree']==degree and c['method']==method
                assert c['slots_checked']==65536 and math.isfinite(c['max_abs_vs_independent_chebval']) and 0<=c['max_abs_vs_independent_chebval']<c['gate']==1e-8
                L=c['source_level'];p=c['plan'];assert p['consumed_levels']==depth and p['polynomial_depth']==pd
                assert L==(depth+2 if c['level_choice']=='minimum' else (20 if role=='attention' else 24))
                assert p['levels']==dict(mean=L-1,centered=L-1,squares=L-2,variance=L-3,inverse=L-3-pd,normalized=L-4-pd,output=L-5-pd)
                assert p['levels']['output']>=2 and p['reserved_output_levels']==2
                assert c['actual_counts']['rotate']==42 and c['actual_counts']['multiply']==degree+1
                assert c['actual_counts']['weighted_sum']==(5 if method=='power' else degree+4)
                lo,hi=rule['polynomial']['domain'];v=c['variance_observed'];assert lo<=v[0]<=v[1]<=hi
                for stage in p['weighted_sum_stages']:
                    first=stage['terms'][0]
                    assert stage['output_scale']==first['output_scale'] and stage['output_level']==stage['input_level']-1
                    for term in stage['terms']:
                        expected=term['input_scale']*((term['target_scale']*term['divisor'])/term['input_scale'])/term['divisor']
                        assert term['output_scale']==expected
    assert schedules==816
    terminal=json.loads((HERE/'terminal.json').read_text())
    assert terminal['child_reaped_by_collector'] and terminal['child_returncode']==0 and not terminal['outer_tool_exit_or_reap_claim']
    assert '338 passed in 10.28s' in (d/'tests.log').read_text()
    print('Verified121 sources,288 ideal full-slot cases (18,874,368 slots),816 schedules; no HE/full-model claim.')
if __name__=='__main__':main()
