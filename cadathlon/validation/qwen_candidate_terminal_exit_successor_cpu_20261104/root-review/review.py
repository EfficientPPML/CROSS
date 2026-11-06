"""Read-only verification of candidate terminal-status141 synthetic preflight."""
from pathlib import Path
import hashlib,json,os,tarfile,xml.etree.ElementTree as ET
P=Path('/tmp/qwen-expanded-candidate-checker-terminal-exit-preflight-v1-20261104')
LAB=Path('/tmp/qwen-expanded-candidate-checker-terminal-exit-successor-20261104')
OUT=Path('/tmp/qwen-candidate-terminal-preflight-root-review-20261104')
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_bytes())
def inventory(root):
    result={}
    for p in root.rglob('*'):
        assert not p.is_symlink()
        if p.is_file():result[p.relative_to(root).as_posix()]=sha(p.read_bytes())
        else:assert p.is_dir()
    return result
raw=(P/'source_manifest.json').read_bytes();expected=json.loads(raw)
assert sha(raw)=='45cd92473d9ba558288651d59d4f089c7fb9ba37650802848cad24687beaa437' and len(expected)==141
assert raw==Path('/tmp/qwen-expanded-candidate-checker-terminal-exit-sources-v1-20261104.json').read_bytes()
assert inventory(LAB)==inventory(P/'execution_source')==expected
pins={'report.json':'c847b64d3cada0add4071fafe5c6ce79f2b70b0db6da7abcde41adf48f853ab9',
      'worker/report.json':'316b887ae7acdb1c00db66e927371542cfffcab3358e620a0a4b7d478f28a155',
      'sources.tar.gz':'04edc1012ef8d503b878979c21cdeb997c2dd6e098a9407e8d59d9328ddb6c12'}
assert all(sha((P/n).read_bytes())==h for n,h in pins.items())
r=read(P/'report.json');w=read(P/'worker/report.json');owner=read(P/'terminal_observation.json')
for row in (r,w):
    assert all(row.get(k)is True for k in ('passed','complete','cleanup_completed','signal_restoration_completed','source_files_unchanged',
        'terminal_after_body_cleanup_and_signal_restoration_attempts','acceptance_requires_independent_exit0_reap_and_exact_report_hash'))
    assert row['source_hashes_start']==row['source_hashes_end']==expected and row['source_manifest_sha256']==sha(raw)
    assert row['audit_denials']==[] and 'error_type'not in row and row['bootstrap_origins']=={'__main__':'entry.py'}
assert r['worker']==w and r['worker_report_sha256']==pins['worker/report.json']
assert all(r.get(k)is False for k in ('actual_candidate_stage','model_forward','coefficient_selection'))
assert type(r['child_returncode'])is int and r['child_returncode']==0 and r['child_reaped']is True and r['child_group_absent']is True
assert w['tests']==781 and w['pytest_returncode']==w['failures']==w['errors']==w['skipped']==0
backends={'torch','transformers','jax','jaxlib','jaxite_word','tensorflow','tokenizers','safetensors','flax','torch_xla'}
assert len(w['installed_import_denials'])==10 and set(w['installed_import_denials'])==backends
tree=ET.parse(P/'worker/tests.xml');cases=tree.findall('.//testcase')
assert len(cases)==781 and not any(tree.findall('.//'+n)for n in ('failure','error','skipped'))
assert sum('test_terminal_exit' in c.attrib.get('classname','')for c in cases)==43
archive=P/'sources.tar.gz';assert r['source_archive']['sha256']==pins['sources.tar.gz'] and r['source_archive']['bytes']==archive.stat().st_size
with tarfile.open(archive)as tar:
    members=tar.getmembers();assert len(members)==len({m.name for m in members})==141
    assert sum(m.size for m in members)<=96<<20
    assert all(m.isfile()and m.name in expected and 0<m.size<=(64<<20 if m.name=='candidate_proof/selector-assessments.json'else 8<<20)for m in members)
    assert {m.name:sha(tar.extractfile(m).read())for m in members}==expected
parent=read(Path('/tmp/qwen-expanded-candidate-checker-ready-sources-v2-20261104.json'));assert len(parent)==139
assert {n for n,h in parent.items()if expected[n]!=h}=={'entry.py','preflight_expectations.json'}
assert set(expected)-set(parent)=={'TERMINAL_SUCCESSOR.md','test_terminal_exit.py'}
assert read(LAB/'execution_authorization.json')=={'execute':False}
peer=Path('/tmp/qwen-candidate-terminal-contract-security-review-20261104/final-delta-review.json')
assert sha(peer.read_bytes())=='dd90d2c7f56941eb2f8d04596686c2e39ef0810b6426689699b49a37184b23c9'
assert read(peer)['source_manifest_sha256']==sha(raw)
assert owner['passed']is True and owner['session']==29701 and owner['exit_code']==0 and owner['reaped']is True
assert owner['authoritative_tool_chunk']=='797e97'
assert owner['report_sha256']==pins['report.json'] and owner['worker_report_sha256']==pins['worker/report.json']
assert owner['source_manifest_sha256']==sha(raw) and owner['source_archive_sha256']==pins['sources.tar.gz']
observed=[{k:x[k]for k in ('pid','pgid','ppid')}for x in owner['observed_owned_processes']]
assert observed==[dict(pid=2988850,pgid=2988850,ppid=16918),dict(pid=2988851,pgid=2988851,ppid=2988850)]
assert r['pid']==2988850 and r['child_pid']==r['child_pgid']==w['pid']==2988851
for row in observed:
    for kind,func in (('pid',os.kill),('pgid',os.killpg)):
        try:func(row[kind],0)
        except ProcessLookupError:pass
        else:raise AssertionError('Observed owned process/group remains')
report=dict(schema='qwen-candidate-terminal-preflight-independent-root-review-v1',passed=True,
    scope='Source/synthetic/terminal proof only; no actual candidate outcome, model or private body read',
    source_count=141,synthetic_tests=781,installed_import_denials=10,terminal_boundary_tests_within_781=43,
    source_manifest_sha256=sha(raw),source_archive_sha256=pins['sources.tar.gz'],report_sha256=pins['report.json'],
    worker_report_sha256=pins['worker/report.json'],current_execution_start_end_archive_equal=True,
    parent139_unchanged=137,numerical_six_protocol_and_candidate_unchanged=True,
    security_final_delta_review_sha256=sha(peer.read_bytes()),owner_terminal_sha256=sha((P/'terminal_observation.json').read_bytes()),
    owner_session=29701,owner_exit_code=0,owner_reaped=True,observed_processes=observed,
    independently_verified_pids_and_pgids_absent=True,wrapper_pid=None,wrapper_scope='PPID16918 is tool infrastructure, not owned or retired by this proof',
    scientific_statuses=[0,1],infrastructure_status=70,actual_grant_issued=False,old_producer_results_reclassified=False,
    review_script_sha256=sha(Path(__file__).read_bytes()))
OUT.mkdir(mode=0o700)
with (OUT/'review.json').open('x')as f:json.dump(report,f,indent=2);f.write('\n')
with (OUT/'review.py').open('xb')as f:f.write(Path(__file__).read_bytes())
print(json.dumps(dict(passed=True,review=str(OUT/'review.json'),sha256=sha((OUT/'review.json').read_bytes()),sources=141,tests=781)))
