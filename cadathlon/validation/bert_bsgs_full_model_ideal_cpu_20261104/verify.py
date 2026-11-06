"""Portable full12 ideal evidence verifier; no runtime/backend imports."""
from pathlib import Path
import argparse,hashlib,json,math,tarfile,xml.etree.ElementTree as ET

COUNTS={15:dict(rotate=81845,multiply=7008,pt_products=517267,rescales=37398),
    128:dict(rotate=538189,multiply=59236,pt_products=2058877,rescales=269701)}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_bytes())
def finite(value):return type(value)in(int,float)and math.isfinite(value)
def members(path):
    with tarfile.open(path)as tar:
        rows=tar.getmembers();assert len(rows)==len({m.name for m in rows})
        assert all(m.isfile()and not m.name.startswith('/')and '..'not in Path(m.name).parts for m in rows)
        return {m.name:hashlib.sha256(tar.extractfile(m).read()).hexdigest()for m in rows}
def verify(root,current=None):
    p=Path(root);manifest=read(p/'artifact_hashes.json')
    assert {str(x.relative_to(p)):sha(x)for x in p.rglob('*')if x.is_file()and x.name!='artifact_hashes.json'}==manifest
    r=read(p/'report.json');h=r['source_hashes_before'];review=read(p/'root-review/terminal.json')
    for n in ('passed','complete','source_files_unchanged','source_archive_verified','asset_files_unchanged','all_direct_children_reaped','all_owned_groups_empty','cleanup_completed','termination_scope_completed','terminal_scope_completed'):assert r[n]is True
    assert r['encrypted_execution']is False and r['model_forward']is False and r['raw_arrays_archived']is False
    assert r['only_initial_embeddings_supplied']is True
    assert len(h)==150 and h==r['source_hashes_after']==r['snapshot_source_hashes_after']==members(p/'sources.tar.gz')
    assert sha(p/'sources.tar.gz')==r['source_archive_sha256']==review['source_archive_sha256']=='c7b79e973d0e0688070c29c68557321c29fdadb6e13b18418ac03ddbae341b55'
    assert sha(p/'report.json')==review['report_sha256']=='8a9589ce238fe3e936603b80e1b2402c5f7cb1ae25d48efd0bcb61d9fce9746b'
    assert r['asset_hashes_before']==r['asset_hashes_after']and review['assets_rehashed']is True
    assert r['tests']==dict(tests=82,errors=0,failures=0,skipped=0)
    cases=list(ET.parse(p/'tests.xml').iter('testcase'));assert len(cases)==82 and all(c.find(n)is None for c in cases for n in ('error','failure','skipped'))
    assert r['original_tests_selected']==33 and r['new_fixture_metadata_tests_expected']==2 and r['new_collector_tests_expected']==47 and r['legacy_collector_tests_excluded']==13
    assert review['outer_session']==2171 and review['outer_exit_code']==0 and review['outer_reaped']and review['producer_pid_and_owned_group_absent']
    process=r['process'];assert process['direct_returncode']==0 and not process['cleanup_errors']and not process['signals_sent']and not r['cleanup_errors']
    assert all(process[n]is True for n in ('direct_child_reaped','group_empty_verified','cleanup_completed'))
    for K,refresh,domains in ((15,136,302),(128,1067,2346)):
        row=read(p/f'joined-K{K}.json');T=(K+15)//16
        assert sha(p/f'joined-K{K}.json')==r['case_file_sha256'][str(K)]
        assert row['K']==K and row['passed']and row['all_connections_are_actual_output_objects']and row['native_prediction_agrees']
        assert row['ideal_gate']==1e-7 and row['native_logit_gate']=='abs(error)<=0.02+0.01*abs(native)'
        assert row['physical_stage_slots_checked']==(24*T+2)*65536 and row['output_level']==2
        assert row['primitive_counts']==COUNTS[K]and row['public_operation_plan']['counts']==COUNTS[K]
        assert row['identity_refresh_calls']==row['public_operation_plan']['identity_refreshes']==refresh
        stages=row['stage_errors'];assert len(stages)==24 and {(s['layer'],s['phase'])for s in stages}=={(i,n)for i in range(12)for n in ('attention','block')}
        for s in stages:assert s['physical_slots']==T*65536 and finite(s['max_abs'])and 0<=s['max_abs']<=1e-7
        for n in ('pooler_all_slot_max_abs','logits_all_slot_max_abs'):assert finite(row[n])and 0<=row[n]<=1e-7
        for name,count in (('actual_domain_observations',domains),('independent_domain_observations',156)):
            assert len(row[name])==count
            for d in row[name]:
                a,b=d['domain'];lo,hi=d['minimum'],d['maximum'];assert all(finite(v)for v in (a,b,lo,hi))and a<=lo<=hi<=b and type(d['count'])is int and d['count']>0
                if name.startswith('actual'):assert d['passed']is True
                else:assert all(type(d[n])is int and d[n]==0 for n in ('below','above','nonfinite'))
        assert row['actual_metadata']['explicit_direct_ciphertext_connections']and row['actual_metadata']['original_positions']==list(range(K))
    if current:assert all(sha(Path(current)/n)==v for n,v in h.items())
    return dict(passed=True,sources=150,tests=82,cases=[15,128],current_checked=current is not None,scope='full12 ideal trajectories only; no HE/ranked/performance claim')
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('root');a.add_argument('--current');v=a.parse_args();print(json.dumps(verify(v.root,v.current),sort_keys=True))
