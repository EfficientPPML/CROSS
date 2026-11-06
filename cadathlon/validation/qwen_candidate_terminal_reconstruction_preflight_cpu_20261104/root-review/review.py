"""Independent public/source review of the completed synthetic44 preflight."""
from pathlib import Path
import hashlib,json,os,tarfile,xml.etree.ElementTree as ET

P=Path('/tmp/qwen-candidate-terminal-reconstruction-preflight-v2-20261104')
LAB=Path('/tmp/qwen-candidate-terminal-reconstruction-v2-20261104')
OUT=Path('/tmp/qwen-reconstruction-root-preflight-review-20261104')
OWNER=Path('/tmp/qwen-candidate-terminal-reconstruction-preflight-v2-terminal-20261104.json')
LIVE=Path('/tmp/qwen-candidate-terminal-reconstruction-preflight-v2-owner-20261104.json')
PEER=Path('/tmp/qwen-terminal-reconstruction-security-review-20261104/review.json')
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
assert len(expected)==44 and sha(manifest)=='c0a00954ae17a8262df609a872b3081eee83e441fd9f8ad4187387a64c0d8439'
assert manifest==Path('/tmp/qwen-candidate-terminal-reconstruction-sources-v2-20261104.json').read_bytes()
assert inventory(LAB)==inventory(P/'execution_source')==expected
pins={'report.json':'5abbe4a12ff0bc20ea1099192448a94f27cd9482d07b28d029b6abfc3e35c5e1',
      'worker/report.json':'160a50d4e09ac291361b85163441dc6f004ebd62b7fab9e1603bf22b932577c1',
      'sources.tar.gz':'e7495280691dc62c9f91c0575932586ed86ecfcbfc3bddd26300038243f076a9'}
assert all(sha((P/n).read_bytes())==h for n,h in pins.items())
r=read(P/'report.json');w=read(P/'worker/report.json');owner=read(OWNER);live=read(LIVE)
for row in (r,w):
    assert all(row.get(k)is True for k in ('passed','complete','cleanup_completed','signal_restoration_completed',
        'outer_termination_completed','terminal_after_body_cleanup_and_signal_restoration_attempts','requires_independent_outer_receipt'))
    assert row['source_manifest_sha256']==sha(manifest) and row['source_hashes_start']==row['source_hashes_end']==expected
    assert row['audit_denials']==[] and 'error_type'not in row
    assert all(row.get(k)is False for k in ('actual_reconstruction','model_forward','validation_read'))
assert r['schema']=='qwen-reconstruction-collector-v1' and w['schema']=='qwen-reconstruction-worker-v1'
assert r['worker']==w and r['worker_report_sha256']==pins['worker/report.json']
assert r['execution_source_hashes_start']==r['execution_source_hashes_end']==expected
assert r['pid']==2940641 and w['pid']==2940642
c=r['child']
assert c['pid']==c['process_group']==2940642 and type(c['direct_returncode'])is int and c['direct_returncode']==0
assert all(c[k]is True for k in ('direct_child_reaped','group_empty_verified','cleanup_completed'))
assert c['cleanup_errors']==c['signals_sent']==[]
assert w['private_read_attempted']is False and w['private_read_confirmed']is False and w['private_read_state']=='not_attempted'
assert w['tests']==139 and w['failures']==w['errors']==w['skipped']==0
tree=ET.parse(P/'worker/tests.xml');cases=tree.findall('.//testcase')
assert len(cases)==139 and not any(tree.findall('.//'+n)for n in ('failure','error','skipped'))
backend_cases={x.attrib['name'] for x in cases if x.attrib['name'].startswith('test_actual_backend_import_denied_before_module_body[')}
backends={'jax','jaxlib','jaxite_word','tensorflow','flax','torch_xla','torch','transformers','safetensors','tokenizers','numpy'}
assert backend_cases=={'test_actual_backend_import_denied_before_module_body['+n+']'for n in backends}
assert r['source_archive']['sha256']==pins['sources.tar.gz'] and r['source_archive']['bytes']==(P/'sources.tar.gz').stat().st_size
with tarfile.open(P/'sources.tar.gz')as tar:
    members=tar.getmembers()
    assert len(members)==len({m.name for m in members})==44
    assert sum(m.size for m in members)<=16<<20
    assert all(m.isfile()and m.name in expected and 0<m.size<=8<<20 for m in members)
    assert {m.name:sha(tar.extractfile(m).read())for m in members}==expected
producer=(LAB/'producer139_sources.json').read_bytes();producer_map=json.loads(producer)
assert len(producer_map)==139 and sha(producer)=='ccd5834b83b9dfec2d3ea988fb395a61ec4b05ce3f2f54621f29200cbbb8e06a'
source_pins=read(LAB/'reviewed_source_pins.json');assert len(source_pins)==13
assert all(expected['sources/'+n]==h==producer_map[n]for n,h in source_pins.items())
assert read(LAB/'execution_authorization.json')=={'execute':False}
assert read(LAB/'preflight_expectations.json')=={'tests':139,'backend_import_denials':11,'actual_execution':False}
assert sha(PEER.read_bytes())=='7021e884c6185ab19825e10cf912c694b69b61c8af0925ef53ca71a2c973c95f'
peer=read(PEER);assert peer['concrete_unresolved_blockers']==[] and peer['all44_current_match_manifest']is True
assert all(expected[n]==h for n,h in peer['reviewed_sources'].items())
assert owner['schema']=='qwen-reconstruction-preflight-owner-terminal-v1'
assert owner['session_id']==live['authoritative_session']==99452 and owner['authoritative_exit_code']==0 and owner['authoritative_reaped']is True
assert owner['source_manifest_sha256']==sha(manifest) and owner['source_archive_sha256']==pins['sources.tar.gz']
assert owner['report_sha256']==pins['report.json'] and owner['worker_report_sha256']==pins['worker/report.json']
observed={n:{'pid':pid,'pgid':pgid}for n,pid,pgid in [('wrapper',2940640,2940640),('collector',2940641,2940640),('worker',2940642,2940642)]}
assert {(x['pid'],x['pgid'])for x in live['observed_processes']}=={(x['pid'],x['pgid'])for x in observed.values()}
for name,pair in observed.items():
    assert owner[name]==dict(pair,absent=True)
    try:os.kill(pair['pid'],0)
    except ProcessLookupError:pass
    else:raise AssertionError('Observed process remains')
for pgid in {x['pgid']for x in observed.values()}:
    try:os.killpg(pgid,0)
    except ProcessLookupError:pass
    else:raise AssertionError('Observed process group remains')
report=dict(schema='qwen-reconstruction-preflight-independent-root-review-v1',passed=True,
    scope='Public source, synthetic proof and terminal readback only; no actual candidate outcome or private body read',
    source_manifest_sha256=sha(manifest),source_archive_sha256=pins['sources.tar.gz'],report_sha256=pins['report.json'],
    worker_report_sha256=pins['worker/report.json'],source_count=44,synthetic_tests=139,
    backend_denial_cases_within_139=11,current_execution_start_end_archive_equal=True,reviewed_producer_sources_exact=13,
    producer139_manifest_sha256=sha(producer),security_review_sha256=sha(PEER.read_bytes()),
    owner_terminal_sha256=sha(OWNER.read_bytes()),live_process_observation_sha256=sha(LIVE.read_bytes()),
    owner_session=99452,owner_observed_exit_code=0,owner_observed_reaped=True,observed_processes=observed,
    independently_verified_pids_and_pgids_absent=True,actual_grant_issued=False,actual_private_or_model_bodies_read=False,
    limitations=['Recorded metrics and argmax values are checked; unavailable raw arrays are not recomputed.',
        'Private inventory digest would bind verifier readback, not a retroactive producer per-file commitment.',
        'Path resolution can inspect metadata; no OS confinement or hostile same-user protection is established.',
        'Actual reconstruction requires producer terminal evidence and a separate root grant.'],
    review_script_sha256=sha(Path(__file__).read_bytes()))
OUT.mkdir(mode=0o700)
with (OUT/'review.json').open('x')as f:json.dump(report,f,indent=2);f.write('\n')
with (OUT/'review.py').open('xb')as f:f.write(Path(__file__).read_bytes())
print(json.dumps(dict(passed=True,review=str(OUT/'review.json'),sha256=sha((OUT/'review.json').read_bytes()),sources=44,tests=139)))
