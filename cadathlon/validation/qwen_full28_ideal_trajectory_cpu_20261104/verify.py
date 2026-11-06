"""Portable exact evidence/metric verifier; no checkpoint/input/body reads."""
from pathlib import Path
import hashlib,json,math,tarfile
P=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def numeric_rows(rows,*,count=None,gate=1e-8):
    if count is not None:assert len(rows)==count
    for row in rows:
        assert row['gate']==gate
        if row['passed']:
            assert row['finite']is True and row['shape_valid']is True
            assert type(row['max_abs_error'])is float and math.isfinite(row['max_abs_error']) and row['max_abs_error']<=gate

def verify(root=P):
    hashes=json.loads((root/'artifact_hashes.json').read_bytes())
    assert {n:sha(root/n)for n in hashes}==hashes
    report=json.loads((root/'actual/report.json').read_bytes());before=report['source_hashes_before']
    assert len(before)==101 and before==report['source_hashes_after']==report['execution_source_hashes_before']==report['execution_source_hashes_after']
    for key in ('complete','cleanup_completed','termination_scope_completed','terminal_scope_completed','source_files_unchanged','execution_files_unchanged'):assert report[key]is True,key
    assert sha(root/'actual/sources.tar.gz')==report['source_archive_sha256']
    with tarfile.open(root/'actual/sources.tar.gz')as archive:
        members=archive.getmembers();assert len(members)==len({m.name for m in members})==101 and all(m.isfile()for m in members)
        bodies={m.name:archive.extractfile(m).read()for m in members}
        assert {n:hashlib.sha256(b).hexdigest()for n,b in bodies.items()}==before
    predecessor=json.loads(bodies['runtime_predecessor.json']);assert len(predecessor['source_hashes'])==87
    assert all(before[n]==h for n,h in predecessor['source_hashes'].items())
    candidate=json.loads(bodies['inputs/candidate_manifest.json']);requirement=json.loads(bodies['trajectory_requirement.json'])
    assert hashlib.sha256(bodies['inputs/candidate_manifest.json']).hexdigest()==requirement['candidate_sha256']=='7f6478a6bc710f3d2a370198e075e6ba8b0e55c144d981bca0eed59eaf176333'
    assert len(report['children'])==len(report['fixtures'])==2
    worker_pass=[];summary=[]
    for i,name in enumerate(('S1','S8')):
        worker=json.loads((root/f'actual/{name}/report.json').read_bytes());entry=report['fixtures'][i];child=report['children'][i]
        assert entry['index']==worker['fixture_index']==i and entry['report_sha256']==sha(root/f'actual/{name}/report.json')
        assert worker['source_hashes_before']==worker['source_hashes_after']==before
        assert worker['asset_sha256_before']==worker['asset_sha256_after']==requirement['assets']['sha256']
        assert worker['fixture']==requirement['fixtures'][i]
        assert worker['calibration_prerequisite']['passed']is True
        assert worker['input_provenance']['token_file_sha256']==requirement['calibration']['tokens_sha256']
        assert worker['cleanup_completed']is True and worker['source_files_unchanged']is True and worker['assets_unchanged']is True
        assert all(child[k]is True for k in ('direct_child_reaped','group_empty_verified','cleanup_completed'))
        assert child['direct_returncode']==entry['exit_code']==(0 if worker['passed']else 1)
        worker_pass.append(worker['passed'])
        if worker['passed']:
            assert worker['complete']is True and worker['termination_scope_completed']is True
            assert worker['coverage']==dict(passed=True,operators=421,blocks=28,logit_blocks=38,logits=151936,physical_metric_slots=487*65536,domains=worker['coverage']['domains'])
            obs=worker['owner_observations'];numeric_rows(list(obs['operators'].values()),count=421);numeric_rows(list(obs['blocks'].values()),count=28);numeric_rows(list(obs['logits'].values()),count=38)
            assert all(r['passed'] for kind in ('operators','blocks','logits')for r in obs[kind].values())
            assert all(r['count']==65536 for kind in ('operators','blocks','logits')for r in obs[kind].values())
            assert len(obs['domains'])==worker['coverage']['domains']
            for name,domain in obs['domains'].items():
                assert domain['below']==domain['above']==domain['nonfinite']==0
                assert math.isfinite(domain['minimum'])and math.isfinite(domain['maximum'])
                assert domain['domain'][0]<=domain['minimum']<=domain['maximum']<=domain['domain'][1]
            comparison=worker['frozen_reference_comparison'];numeric_rows([comparison['logits']]);numeric_rows(list(comparison['blocks'].values()),count=28);numeric_rows(list(comparison['operators'].values()),count=421)
            assert all(r['passed']for r in [comparison['logits'],*comparison['blocks'].values(),*comparison['operators'].values()])
            for key in ('independent_reference_domains','frozen_reference_domains'):
                assert worker[key] and all(r['below']==r['above']==r['nonfinite']==0 for r in worker[key].values())
            logits=worker['logit_metrics'];assert logits['passed']and logits['finite']and logits['all_logits']==151936 and logits['physical_slots_checked']==38*65536 and logits['max_abs_error']<=1e-8
            assert len(worker['actual_logit_metadata'])==38 and all(r['level']==6 and math.isfinite(r['scale'])and r['scale']>0 for r in worker['actual_logit_metadata'])
            assert worker['refresh_count']==worker['expected_refresh_count']==(638 if i==0 else 3270)
            summary.append(dict(fixture=i,passed=True,operators=421,blocks=28,logits=151936,worst_owner_error=max(r['max_abs_error']for k in ('operators','blocks','logits')for r in obs[k].values())))
        else:
            assert worker['complete']is False and worker['failure_detail']
            if i==0:
                error=worker['failure_detail'];assert error['type']=='DomainFailure' and error['label']=='final.inverse_sqrt'
                assert error['record']['domain']==candidate['nonlinear']['final.inverse_sqrt']['domain']
                assert error['record']['above']==1 and error['record']['below']==error['record']['nonfinite']==0
                assert error['record']['minimum']>error['record']['domain'][1]
                assert 'owner_observations'not in worker
            summary.append(dict(fixture=i,passed=False,failure=worker['failure_detail']))
    assert report['passed']==all(worker_pass)
    print(json.dumps(dict(verified=True,sources=101,aggregate_passed=report['passed'],fixtures=summary),allow_nan=False))
if __name__=='__main__':verify()
