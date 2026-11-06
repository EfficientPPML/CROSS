"""Independent read-only verification of the completed heldout167 preflight."""
from pathlib import Path
import hashlib,json,os,tarfile,xml.etree.ElementTree as ET
P=Path('/tmp/qwen-heldout1088-runtime-preflight-v2-20261104')
LAB=Path('/tmp/qwen-heldout1088-runtime-v2-20261104')
OUT=Path('/tmp/qwen-heldout-runtime-root-preflight-review-20261104')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(path):return json.loads(path.read_bytes())
def inventory(root):
    rows={}
    for p in root.rglob('*'):
        assert not p.is_symlink()
        if p.is_file():rows[p.relative_to(root).as_posix()]=sha(p.read_bytes())
        else:assert p.is_dir()
    return rows
manifest=(P/'source_manifest.json').read_bytes();expected=json.loads(manifest)
assert sha(manifest)=='cce2630028708f41e79614bb367039fd146fc924471bfd674ade1737f9c6017e' and len(expected)==167
assert manifest==Path('/tmp/qwen-heldout1088-runtime-sources-v2-20261104.json').read_bytes()
assert sha((P/'report.json').read_bytes())=='1575f36e4e647d35f5685a626c8b9fb6ba4eb530ed08ab4aa8d3615cc42ffa81'
r=read(P/'report.json');w=read(P/'worker/report.json');outer=read(P/'terminal_observation.json')
assert r['worker']==w and r['worker_report_sha256']==sha((P/'worker/report.json').read_bytes())
for row in (r,w):
    assert all(row.get(k)is True for k in ('passed','complete','cleanup_completed','signal_restoration_completed','source_files_unchanged',
        'terminal_after_body_cleanup_and_signal_restoration_attempts','acceptance_requires_independent_exit0_reap_and_exact_report_hash'))
    assert row['source_hashes_start']==row['source_hashes_end']==expected and row['source_manifest_sha256']==sha(manifest)
    assert row['audit_denials']==[] and 'error_type'not in row
    assert row['model_forward']is False and row['coefficient_selection']is False
    assert row['bootstrap_origins']=={'__main__':'heldout_entry.py'}
assert r['actual_heldout_stage']is False and w['validation_inputs_read']is False
assert type(r['child_returncode'])is int and r['child_returncode']==0 and r['child_reaped']is True and r['child_group_absent']is True
assert w['tests']==478 and w['pytest_returncode']==w['failures']==w['errors']==w['skipped']==0
assert set(w['installed_import_denials'])=={'torch','transformers','jax','jaxlib','jaxite_word','tensorflow','tokenizers','safetensors','flax','torch_xla'}
x=ET.parse(P/'worker/tests.xml');assert len(x.findall('.//testcase'))==478
assert not any(x.findall('.//'+k)for k in ('error','failure','skipped'))
assert inventory(LAB)==inventory(P/'execution_source')==expected
archive=(P/'sources.tar.gz').read_bytes()
assert sha(archive)==r['source_archive']['sha256']=='6cab10a38f4074bea8089d50d3544c6e462c79bb309cb44fc5c408ce3dd4d991'
assert len(archive)==r['source_archive']['bytes']
with tarfile.open(P/'sources.tar.gz')as tar:
    members=tar.getmembers()
    assert len(members)==len({m.name for m in members})==167 and sum(m.size for m in members)<=96<<20
    assert all(m.isfile()and m.name in expected and 0<m.size<=(64<<20 if m.name=='candidate_proof/selector-assessments.json'else 8<<20)for m in members)
    assert {m.name:sha(tar.extractfile(m).read())for m in members}==expected
parent=read(Path('/tmp/qwen-expanded-candidate-checker-ready-sources-v2-20261104.json'))
assert len(parent)==139 and read(LAB/'parent139_sources.json')==parent and all(expected[n]==h for n,h in parent.items())
assert read(LAB/'execution_authorization.json')=={'execute':False}
assert outer['passed']is True and outer['tool_session']==62369 and outer['tool_exit_code']==0 and outer['tool_reaped']is True
assert outer['tool_terminal_chunk']=='47d970' and outer['process_observation_tool_chunk']=='6f9d12'
assert outer['report_sha256']==sha((P/'report.json').read_bytes()) and outer['source_manifest_sha256']==sha(manifest) and outer['source_archive_sha256']==sha(archive)
observed=outer['observed_processes']
assert observed=={'collector':{'pid':2938023,'pgid':2938023,'ppid':16918},'worker':{'pid':2938141,'pgid':2938141,'ppid':2938023}}
assert r['pid']==observed['collector']['pid'] and r['child_pid']==r['child_pgid']==w['pid']==observed['worker']['pid']
for row in observed.values():
    for kind,func in (('pid',os.kill),('pgid',os.killpg)):
        try:func(row[kind],0)
        except ProcessLookupError:pass
        else:raise AssertionError('Observed owned process/group remains')
peer=Path('/tmp/qwen-heldout-runtime-root-review-20261104/review-v2.json')
assert sha(peer.read_bytes())=='ba55a6423276dc7f2c39ed73ebdaa5431fe0ee2955d47c93b032cbbdde2f1506' and read(peer)['passed']is True
report=dict(schema='qwen-heldout-preflight-independent-root-review-v1',passed=True,
    scope='Public/source/terminal proof review only; no model, candidate outcome, private journal or validation-body read',
    source_manifest_sha256=sha(manifest),source_archive_sha256=sha(archive),report_sha256=sha((P/'report.json').read_bytes()),
    worker_report_sha256=sha((P/'worker/report.json').read_bytes()),sources=167,tests=478,installed_import_denials=10,
    current_execution_start_end_archive_equal=True,parent139_unchanged=True,exact_numerical_six_unchanged=True,
    source_delta_review_sha256=sha(peer.read_bytes()),owner_terminal_sha256=sha((P/'terminal_observation.json').read_bytes()),
    owner_session=62369,owner_tool_exit_code=0,owner_tool_reaped=True,observed_processes=observed,independently_verified_owned_pids_and_pgids_absent=True,
    wrapper_pid=None,wrapper_scope='Unavailable; no claim about PPID16918 ownership or retirement',
    actual_execution_grant_issued=False,actual_model_or_validation_body_read=False,
    review_script_sha256=sha(Path(__file__).read_bytes()))
OUT.mkdir(mode=0o700)
with (OUT/'review.json').open('x')as f:json.dump(report,f,indent=2);f.write('\n')
with (OUT/'review.py').open('xb')as f:f.write(Path(__file__).read_bytes())
print(json.dumps(dict(passed=True,review=str(OUT/'review.json'),sha256=sha((OUT/'review.json').read_bytes()),sources=167,tests=478)))
