"""Portable public fixture proof verification; no backend or payload access."""
from pathlib import Path
import argparse,hashlib,io,json,os,tarfile
HERE=Path(__file__).resolve().parent
LAB=Path('/tmp/bert-normalized-round-gain4-ready-20261104')
REPORT='aca358a292ca8cc867241e09abec18889b20b31ccc4111b4fa66b2830ff271ad'
WORKER='1b914350ab11ee8ae9fc83cb9e27dec03d2b37a5372ce8177ae7add453ecca82'
ARCHIVE='3966436b4ed2cd717c4ee50ddb652f0500fdfe03bbbdbc2f1c916af9f4b232cb'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(path):return json.loads(path.read_bytes())
def verify(current=False):
    hashes=read(HERE/'artifact_hashes.json');actual={p.relative_to(HERE).as_posix()for p in HERE.rglob('*')if p.is_file()and p.name!='artifact_hashes.json'and '__pycache__'not in p.parts}
    assert set(hashes)==actual and all(sha((HERE/n).read_bytes())==h for n,h in hashes.items())
    assert sha((HERE/'report.json').read_bytes())==REPORT and sha((HERE/'worker/worker_report.json').read_bytes())==WORKER
    r=read(HERE/'report.json');w=read(HERE/'worker/worker_report.json')
    assert all(r[n]is True for n in ['passed','execute','worker_reaped','worker_group_empty_verified','all_direct_children_reaped','all_owned_groups_empty','process_cleanup_completed','source_files_unchanged','source_archive_verified','terminal_scope_completed','termination_scope_completed','workspace_retained'])
    assert all(w[n]is True for n in ['passed','phase_complete','loader_cleanup_completed','source_files_unchanged','terminal_scope_completed','termination_scope_completed'])
    assert r['stage']==w['stage']=='fixture'and r['worker_exit_code']==0 and type(r['worker_exit_code'])is int
    assert r['cleanup_errors']==[]and not r.get('error')and not w.get('error')and w['forbidden_attempts']==[]
    expected=r['source_hashes_start'];assert len(expected)==163 and expected==r['source_hashes_end']==w['source_hashes_start']==w['source_hashes_end']==read(HERE/'source_manifest.json')
    blob=(HERE/'sources.tar.gz').read_bytes();assert sha(blob)==ARCHIVE==r['source_archive_sha256']
    with tarfile.open(fileobj=io.BytesIO(blob))as tar:
        members=tar.getmembers();assert len(members)==len({m.name for m in members})==163
        assert all(m.isfile()and m.size<=8<<20 and not Path(m.name).is_absolute()and '..'not in Path(m.name).parts for m in members)and sum(m.size for m in members)<=16<<20
        sources={m.name:tar.extractfile(m).read()for m in members}
    assert {n:sha(b)for n,b in sources.items()}==expected
    assert w['completed_ciphertexts']==2 and type(w['completed_ciphertexts'])is int
    assert r['worker_report_sha256']==WORKER and r['readback']==w['readback']and r['request_contract']==w['request_contract']
    rb=r['readback'];assert rb['tile_count']==2 and len(rb['manifest']['tiles'])==2 and rb['manifest']['role']=='request'
    assert [x['index']for x in rb['manifest']['tiles']]==[0,1]
    independent=read(HERE/'independent_readback.json');assert independent['passed']is True and independent['input_readback']==rb and independent['output_readback']is None
    assert independent['report_sha256']==REPORT and independent['source_archive_sha256']==ARCHIVE and independent['forbidden_attempts']==[]and independent['backend_modules']==[]
    observation=read(HERE/'terminal_observation.json');assert observation['authoritative_reaped']is True and observation['authoritative_tool_exit_code']==0 and observation['both_producers_absent']is True
    assert len(r['children'])==1
    child=r['children'][0];assert all(child[n]is True for n in ['direct_child_reaped','group_empty_verified','cleanup_completed'])and child['direct_returncode']==0 and child['cleanup_errors']==[]
    if current:
        for n,h in expected.items():assert sha((LAB/n).read_bytes())==sha((LAB/'actual-fixture-v1/execution_source'/n).read_bytes())==h
        assert sha((LAB/'actual-fixture-v1/report.json').read_bytes())==REPORT
        assert not any(Path('/proc',str(pid)).exists()for pid in [2586890,2586897])
        st=Path(r['workspace']).lstat();assert [st.st_dev,st.st_ino]==r['workspace_identity']
    return dict(passed=True,phase='fixture_only',sources=163,encrypted_inputs=2,raw_ciphertext_bytes=209715200,current=current,numerical_validation=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--current',action='store_true');args=p.parse_args();print(json.dumps(verify(args.current),sort_keys=True))
