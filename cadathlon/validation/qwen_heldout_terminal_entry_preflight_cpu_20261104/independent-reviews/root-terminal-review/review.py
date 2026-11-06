"""Independent readback of completed heldout terminal-status171 preflight."""
from pathlib import Path
import hashlib,json,os,tarfile,xml.etree.ElementTree as ET
P=Path('/tmp/qwen-heldout1088-runtime-preflight-v5-20261104')
LAB=Path('/tmp/qwen-heldout1088-runtime-v5-20261104')
OUT=Path('/tmp/qwen-heldout-terminal-preflight-root-review-20261104')
OWNER=Path('/tmp/qwen-heldout1088-runtime-preflight-v5-owner-terminal-20261104.json')
LIVE=Path('/tmp/qwen-heldout1088-runtime-preflight-v5-owner-live-20261104.json')
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
assert len(expected)==171 and sha(raw)=='9985aad6205af33a429fba3802bac415397973418e568ea3414d61197f03deeb'
assert raw==Path('/tmp/qwen-heldout1088-runtime-sources-v5-20261104.json').read_bytes()
assert inventory(LAB)==inventory(P/'execution_source')==expected
pins={'report.json':'c4b4a7df248fe3b24d4c9d8e0f8cf4c99fcfb6bdaeae8e25aa9c136cff2ab31f',
      'worker/report.json':'e3a449af0a13a070e4bda85436de1b0dfa7a51b6ecaa0080a561b9d8ee00dfd2',
      'sources.tar.gz':'417f2c15b85f0f3bd17d071a6a6a4bddd734e3fee1e37d812e1bd1d901b2d7c6'}
assert all(sha((P/n).read_bytes())==h for n,h in pins.items())
r=read(P/'report.json');w=read(P/'worker/report.json');owner=read(OWNER);live=read(LIVE)
for row in (r,w):
    assert all(row.get(k)is True for k in ('passed','complete','cleanup_completed','signal_restoration_completed','source_files_unchanged',
        'terminal_after_body_cleanup_and_signal_restoration_attempts','acceptance_requires_independent_exit0_reap_and_exact_report_hash'))
    assert row['source_hashes_start']==row['source_hashes_end']==expected and row['source_manifest_sha256']==sha(raw)
    assert row['audit_denials']==[] and 'error_type'not in row and row['bootstrap_origins']=={'__main__':'heldout_entry.py'}
    assert row['model_forward']is False and row['coefficient_selection']is False
assert r['worker']==w and r['worker_report_sha256']==pins['worker/report.json']
assert r['actual_heldout_stage']is False and w['validation_inputs_read']is False
assert type(r['child_returncode'])is int and r['child_returncode']==0 and r['child_reaped']is True and r['child_group_absent']is True
assert w['tests']==563 and w['pytest_returncode']==w['failures']==w['errors']==w['skipped']==0
backends={'torch','transformers','jax','jaxlib','jaxite_word','tensorflow','tokenizers','safetensors','flax','torch_xla'}
assert len(w['installed_import_denials'])==10 and set(w['installed_import_denials'])==backends
tree=ET.parse(P/'worker/tests.xml');cases=tree.findall('.//testcase')
assert len(cases)==563 and not any(tree.findall('.//'+n)for n in ('failure','error','skipped'))
assert sum('test_heldout_terminal_exit' in c.attrib.get('classname','')for c in cases)==85
archive=P/'sources.tar.gz';assert r['source_archive']['sha256']==pins['sources.tar.gz'] and r['source_archive']['bytes']==archive.stat().st_size
with tarfile.open(archive)as tar:
    members=tar.getmembers();assert len(members)==len({m.name for m in members})==171
    assert sum(m.size for m in members)<=96<<20
    assert all(m.isfile()and m.name in expected and 0<m.size<=(64<<20 if m.name=='candidate_proof/selector-assessments.json'else 8<<20)for m in members)
    assert {m.name:sha(tar.extractfile(m).read())for m in members}==expected
parent=read(LAB/'parent139_sources.json');assert len(parent)==139 and all(expected[n]==h for n,h in parent.items())
old=read(Path('/tmp/qwen-heldout1088-runtime-sources-v2-20261104.json'));assert len(old)==167
assert {n for n,h in old.items()if expected[n]!=h}=={'heldout_entry.py','heldout_preflight_expectations.json','test_heldout_no_grant.py'}
assert set(expected)-set(old)=={'TERMINAL_ENTRY_DESIGN.md','heldout_terminal_origin.json','shared_terminal_exit_probe.py.txt','test_heldout_terminal_exit.py'}
assert read(LAB/'execution_authorization.json')=={'execute':False}
source_review=Path('/tmp/qwen-heldout-runtime-root-review-20261104/review-v5.json')
assert sha(source_review.read_bytes())=='f36975b0d41be2245316eb171ae559a89ff47df709b29dcd22ddafd03d8f7a52'
assert read(source_review)['passed']is True and read(source_review)['source_manifest_sha256']==sha(raw)
peer=Path('/tmp/qwen-heldout-terminal-exit-security-review-20261104/review.json')
assert sha(peer.read_bytes())=='943c27733483b1c407d167e3e17b76d2b1b8a2865a624502de368337f028c274'
assert owner['session']==live['session']==97252 and owner['exit_code']==0 and owner['reaped']is True
assert owner['report_sha256']==pins['report.json'] and owner['worker_report_sha256']==pins['worker/report.json']
assert owner['source_manifest_sha256']==sha(raw) and owner['source_archive_sha256']==pins['sources.tar.gz']
observed=[{k:x[k]for k in ('pid','pgid','ppid')}for x in live['processes']]
assert observed==[dict(pid=2992574,pgid=2992574,ppid=16918),dict(pid=2992576,pgid=2992576,ppid=2992574)]
assert owner['collector']==dict(pid=2992574,pgid=2992574,absent=True) and owner['worker']==dict(pid=2992576,pgid=2992576,absent=True)
assert r['pid']==2992574 and r['child_pid']==r['child_pgid']==w['pid']==2992576
for row in observed:
    for kind,func in (('pid',os.kill),('pgid',os.killpg)):
        try:func(row[kind],0)
        except ProcessLookupError:pass
        else:raise AssertionError('Observed owned process/group remains')
report=dict(schema='qwen-heldout-terminal-preflight-independent-root-review-v1',passed=True,
    scope='Source/synthetic/terminal proof only; no actual candidate outcome, model or private body read',
    source_count=171,synthetic_tests=563,installed_import_denials=10,terminal_boundary_tests_within_563=85,
    source_manifest_sha256=sha(raw),source_archive_sha256=pins['sources.tar.gz'],report_sha256=pins['report.json'],
    worker_report_sha256=pins['worker/report.json'],current_execution_start_end_archive_equal=True,
    parent167_unchanged=164,parent139_unchanged=True,numerical_six_protocol_and_candidate_unchanged=True,
    source_review_sha256=sha(source_review.read_bytes()),security_source_review_sha256=sha(peer.read_bytes()),
    owner_terminal_sha256=sha(OWNER.read_bytes()),live_process_observation_sha256=sha(LIVE.read_bytes()),
    owner_session=97252,owner_exit_code=0,owner_reaped=True,observed_processes=observed,
    independently_verified_pids_and_pgids_absent=True,separate_wrapper_observed=False,
    wrapper_scope='No separate wrapper retirement claim; PPID16918 is tool infrastructure',
    scientific_statuses=[0,1],infrastructure_status=70,actual_grant_issued=False,old_producer_results_reclassified=False,
    review_script_sha256=sha(Path(__file__).read_bytes()))
OUT.mkdir(mode=0o700)
with (OUT/'review.json').open('x')as f:json.dump(report,f,indent=2);f.write('\n')
with (OUT/'review.py').open('xb')as f:f.write(Path(__file__).read_bytes())
print(json.dumps(dict(passed=True,review=str(OUT/'review.json'),sha256=sha((OUT/'review.json').read_bytes()),sources=171,tests=563)))
