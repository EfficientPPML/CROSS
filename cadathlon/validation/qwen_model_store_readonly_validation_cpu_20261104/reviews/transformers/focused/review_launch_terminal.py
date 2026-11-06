"""Public terminal and proposed-launch readback only. No candidate execution.

All supplied store records are treated as public metadata. No described store
path is stat'ed or opened. Capacity values are root's recorded observation.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import xml.etree.ElementTree as ET

OUT=Path('/tmp/qwen-readonly-focused-transformers-review-20261104')
PROOF=Path('/tmp/qwen-model-store-readonly-focused-proof-20261104/preparation-v1')
LAUNCH=Path('/tmp/qwen-model-store-readonly-root-launch-20261104')
SOURCE=Path('/tmp/qwen-model-store-readonly-v2-20261104')
sha=lambda raw:hashlib.sha256(raw).hexdigest()
canon=lambda v:(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
read=lambda p:json.loads(p.read_bytes())


def archived(path):
    with tarfile.open(path,'r:gz')as archive:
        members=archive.getmembers()
        assert len(members)==len({m.name for m in members})and all(m.isfile()for m in members)
        return {m.name:sha(archive.extractfile(m).read())for m in members}


handoff=read(PROOF/'handoff.json');owner=read(PROOF/'owner-terminal.json')
assert sha((PROOF/'handoff.json').read_bytes())=='e33fb6e12cf5c44754e7ba655c1b719aada1334dd98a62d2cbfdbfe777ffb5fd'
assert sha((PROOF/'owner-terminal.json').read_bytes())=='bc252201ce44537fc04cab38ed5dddc864c6b5be749e7e3a410c66920f9af9c2'
for name,h in handoff['files'].items():assert sha((PROOF/name).read_bytes())==h
expected=read(PROOF/'sources-before.json')
assert len(expected)==54 and read(PROOF/'sources-after.json')==read(PROOF/'execution-after.json')==expected
assert archived(PROOF/'sources.tar.gz')==expected
for base in (Path(handoff['source_root']),PROOF/'execution'):
    current={p.relative_to(base).as_posix():sha(p.read_bytes())for p in base.rglob('*')if p.is_file()}
    assert current==expected
config=read(PROOF/'run_config.json');report=read(PROOF/'report.json');audit=read(PROOF/'worker-audit.json')
xml=ET.parse(PROOF/'tests.xml').getroot()
counts={name:len(list(xml.iter(tag)))for name,tag in [('tests','testcase'),('failures','failure'),('errors','error'),('skipped','skipped')]}
assert counts==dict(tests=62,failures=0,errors=0,skipped=0)==report['tests']==owner['tests']
assert [dict(classname=n.attrib['classname'],name=n.attrib['name'])for n in xml.iter('testcase')]==config['selected_xml_cases']
assert audit['collected_nodes']==config['selected_nodes']and audit['denied']==audit['loaded_backend']==[]
assert audit['runtime_sources_before']==audit['runtime_sources_after']==config['runtime_source_expectations']
assert audit['runtime_module_origins']=={n:dict(file=p,spec=p)for n,p in config['runtime_module_origins'].items()}
assert report['passed']is True and report['child_returncode']==0 and report['child_reaped']is True and report['child_group_absent']is True
assert owner['session_id']==20688 and owner['authoritative_wait_tool_chunk']=='751336'and owner['tool_exit_code']==0 and owner['tool_session_reaped']is True
observed=owner['observed_pids'];assert audit['pid']==observed['worker_pid']and audit['pgid']==observed['worker_pgid']
identities={(observed['outer_pid'],observed['outer_pgid']),(observed['worker_pid'],observed['worker_pgid'])}
assert len(owner['child_terminals'])==15
for row in owner['child_terminals']:
    raw=(PROOF/'test-child-terminals'/row['file']).read_bytes();assert sha(raw)==row['sha256']
    value=json.loads(raw);assert value['reaped']is True and value['process_group_absent']is True
    assert (value['pid'],value['pgid'],value['exit_code'])==(row['pid'],row['pgid'],row['exit_code'])
    identities.add((row['pid'],row['pgid']))
absence=[]
for pid,pgid in sorted(identities):
    try:os.kill(pid,0)
    except ProcessLookupError:pass
    else:raise AssertionError(('recorded PID is present',pid))
    try:os.killpg(pgid,0)
    except ProcessLookupError:pass
    else:raise AssertionError(('recorded group is present',pgid))
    absence.append(dict(pid=pid,pgid=pgid,pid_absent=True,group_absent=True))
terminal=dict(schema='qwen-readonly-focused-independent-terminal-readback-v1',passed=True,
    source_count=54,runtime_count=42,tests=counts,report_sha256=sha((PROOF/'report.json').read_bytes()),
    owner_receipt_sha256=sha((PROOF/'owner-terminal.json').read_bytes()),
    handoff_sha256=sha((PROOF/'handoff.json').read_bytes()),source_manifest_sha256=sha((PROOF/'sources-before.json').read_bytes()),
    source_archive_sha256=sha((PROOF/'sources.tar.gz').read_bytes()),zero_backend_denials=True,
    current_execution_start_end_archive_equal=True,recorded_process_absence=absence,
    authoritative_reap='CNN-owned20688/tool751336 exit0, authenticated owner receipt; reviewer did not own that handle',
    independent_live_observation_claimed=False,tests_rerun=False,candidate_imported=False,
    actual_model_or_body_integrity_admitted=False,scope='Public proof/source/receipt readback only; runtime recording seams unchanged')
(OUT/'terminal-readback.json').write_bytes(canon(terminal))

pins={'launch-proposal.json':'d28ce307babd2bf66754b3e657620e5262006ab88dfa606f0ab5dc087fbfc571',
      'grant.json':'eafbba1d2e463302c9d8a0305a038c204faa3ceaa531ce36b4cbd70c8f80d9f7',
      'prelaunch-review.json':'3db70ff1012b02ac928ebed6be5dd3efbdb1f0b6156fff6f4784e1608f895475',
      'resource-review.json':'f29b1b7cb20b1964c63ddc4a91594847756a17d3cda3a88296b8c27d61796736'}
for name,h in pins.items():
    raw=(LAUNCH/name).read_bytes();assert sha(raw)==h and canon(json.loads(raw))==raw
proposal=read(LAUNCH/'launch-proposal.json');grant=read(LAUNCH/'grant.json')
review=read(LAUNCH/'prelaunch-review.json');resources=read(LAUNCH/'resource-review.json')
map_path=Path('/tmp/qwen-model-store-readonly-sources-v2-20261104.json')
raw_map=map_path.read_bytes();runtime=json.loads(raw_map)
assert len(runtime)==42 and sha(raw_map)=='b73e669b0cc2bc34d7930f2a8a4c620be719fe663dfa6ae7b12df7cae22c2bc0'and canon(runtime)==raw_map
assert {p.relative_to(SOURCE).as_posix():sha(p.read_bytes())for p in SOURCE.rglob('*')if p.is_file()}==runtime
actual_archive=Path('/tmp/qwen-model-store-readonly-source-handoff-v2-20261104/sources.tar.gz')
assert sha(actual_archive.read_bytes())==proposal['source_archive_sha256']=='06a7ae13b6df2adb9808af69d7906eec1e86ed811c419da9c1d064479fe95a25'
assert archived(actual_archive)==runtime
policy=read(SOURCE/'model_readonly_policy.json');dep=read(SOURCE/'model_dependencies.json')
stores_path=Path('/tmp/qwen-model-store-metadata-expectations-20261104/expected_stores.json')
store_raw=stores_path.read_bytes();stores=json.loads(store_raw)
assert sha(store_raw)=='c3023d3782b3aa3b9b9607f07d291162383281a59b86be71fb9cb185fbcf8604'and canon(stores)==store_raw
assert len(stores['roots'])==3 and len(stores['files'])==198
assert sorted(r['path']for r in stores['roots'].values())==sorted(policy['bootstrap_borrowed_roots'])
catalog=read(SOURCE/'unadmitted_aggregate_manifest.json')['catalog']
physical={(r['family'],r['descriptor']['file']):r['descriptor']for r in catalog['matrices'].values()}
assert len(physical)==197
for row in stores['files']:
    if row['family']=='gamma':
        assert row['file_sha256']==catalog['gamma_container_sha256']and row['raw_sha256']is None
    else:
        d=physical[(row['family'],row['name'])];assert row['file_sha256']==d['file_sha256']and row['raw_sha256']==d['sha256']
        size=8
        for dimension in d['shape']:size*=dimension
        assert size<row['state']['size']<=size+16384
extent=sum(r['state']['size']for r in stores['files']if r['family']!='gamma')
assert extent==4767900288 and extent+policy['memory_margin_bytes']==5304771200
assert resources['matrix_extent_bytes']==extent and resources['memory_floor_bytes']==extent+policy['memory_margin_bytes']
assert resources['available_memory_bytes']>=resources['memory_floor_bytes']
assert resources['available_evidence_bytes']>=policy['evidence_reserve_bytes']==resources['evidence_reserve_bytes']
assert resources['planning_body_fds']==policy['planning_body_fds']==398 and resources['runtime_fd_allowance']==policy['runtime_fd_allowance']==32
assert resources['soft_nofile']>=max(policy['minimum_soft_nofile'],resources['root_open_fd_baseline']+398+32)
assert resources['actual_process_rechecks_own_baseline']is True

constants=dict(schema='qwen-model-store-readonly-grant-v1',role='validate_existing_model_stores',execute=True,
    model_id='ee3daf8b07141134466d02a6e1c141365ac4e0f0117d62f887c416a050b04bfb',
    aggregate_manifest_sha256='18189e5f2f2de3a4939c6afca19d79822465d917caf06c4f6e36668bd37c1f55',
    catalog_sha256='6faa4630169f055b853a62a64088952e2ead48dc2ace5b61b6aa438c0106a0c0',
    cores=[120,121,122,123],host_numeric_threads=1,minimum_soft_nofile=512,
    memory_margin_bytes=512<<20,evidence_reserve_bytes=128<<20,parameter_body_read_only=True,
    new_export_authorized=False,model_execution_authorized=False,key_or_ciphertext_access_authorized=False,
    token_or_calibration_access_authorized=False,selection_or_revision_authorized=False)
extras=dict(source_root=str(SOURCE),source_manifest_sha256=sha(raw_map),source_archive_sha256=sha(actual_archive.read_bytes()),
    expected_stores=dict(path=str(stores_path),sha256=sha(store_raw)),
    prelaunch_review=dict(path=str(LAUNCH/'prelaunch-review.json'),sha256=pins['prelaunch-review.json']),
    evidence_output='/tmp/qwen-model-store-readonly-actual-v1-20261104',
    dependency_policy_sha256=runtime['model_dependencies.json'],readonly_policy_sha256=runtime['model_readonly_policy.json'])
assert canon(grant)==canon(constants|extras)
assert review['passed']is True and review['actual_model_integrity_admitted']is False
assert review['source_archive_members_verified']is True and review['source_manifest_sha256']==sha(raw_map)
assert review['source_archive_sha256']==sha(actual_archive.read_bytes())and review['expected_stores_sha256']==sha(store_raw)
for row in review['reviews'].values():assert sha(Path(row['path']).read_bytes())==row['sha256']
assert review['resource_review']==dict(path=str(LAUNCH/'resource-review.json'),sha256=pins['resource-review.json'])
environment=dict(PATH='/usr/bin:/bin',HOME='/home/jianming_gatech',LANG='C.UTF-8',TZ='UTC',**dep['environment'])
argv=['env','-i',*[k+'='+v for k,v in environment.items()],'taskset','-c','120-123',dep['interpreter'],
    '-I','-B',str(SOURCE/'model_entry.py'),'--role','validate_existing_model_stores',
    '--manifest',str(map_path),'--manifest-sha256',sha(raw_map),'--grant',str(LAUNCH/'grant.json'),
    '--grant-sha256',pins['grant.json'],'--output',extras['evidence_output']]
assert proposal['argv']==argv and proposal['launch_authorized']is False and proposal['direct_process_owner']=='root'
assert proposal['script_sha256']==sha(Path('/tmp/qwen-focused-root-review-20261104/prepare_actual_readonly_launch.py').read_bytes())
leaf=PurePosixPath(extras['evidence_output'])
blocked=[PurePosixPath(str(SOURCE)),PurePosixPath(str(map_path)),PurePosixPath(str(LAUNCH)),
         PurePosixPath(str(stores_path)),*map(PurePosixPath,policy['protected_paths']+policy['bootstrap_borrowed_roots'])]
assert all(leaf!=p and leaf not in p.parents and p not in leaf.parents for p in blocked)
for path,h in dep['numpy_sources'].items():assert sha(Path(path).read_bytes())==h

result=dict(schema='qwen-readonly-actual-launch-peer-source-review-v1',passed=True,
    source_only_ready_for_exact_root_owned_proposal=True,launch_authorization_issued=False,
    proposal_pins=pins,source_manifest_sha256=sha(raw_map),source_archive_sha256=sha(actual_archive.read_bytes()),
    expected_store_sha256=sha(store_raw),expected_store_count=198,matrix_extent_bytes=extent,
    policy_memory_floor_bytes=extent+policy['memory_margin_bytes'],planning_body_fds=398,runtime_fd_allowance=32,
    resources_independently_reobserved=False,resource_claim_source=str(LAUNCH/'resource-review.json'),
    exact_argv_and_closed_grant_match=True,current42_archive_members_match=True,
    findings=[
      'Bootstrap authenticates canonical42 source map/retained bytes before local imports; grant/model/catalog/expected-store/resource/dependency gates precede body-enabled reader acquisition.',
      'The grant selects only readonly existing197 matrices/one gamma container, with198logical and113gamma validation; all key/model-forward/token/calibration/export authority remains false.',
      'Original held roots/files are checked against c302 metadata before hashing/mmap; output records are reconstructed from live FDs and compared to expectations before borrowed lease retirement.',
      'Final publication/close/audit restoration remains under terminal70; report presence cannot replace root direct wait/reap/group absence and final observed-store/source/archive review.',
      'Resource note is a root time-point observation; actual process must recheck exact120-123/interpreter/thread environment, own FD baseline, memory floor and evidence reserve before opening stores.'
    ],unresolved_concrete_source_blockers=[],
    limitations=['Focused62 positive traversal uses recorded admission/resource/audit/reader seams and does not prove the entire real NumPy-under-ReadScope/body-validation combination succeeds.',
                 'c302 is metadata expectations only; actual ModelAdmission still requires genuine readonly validation plus independent terminal/root proof.',
                 'Python audit and four NumPy Python hashes/three origins are not an OS sandbox or complete binary dependency closure.'],
    candidate_imported=False,tests_executed=False,borrowed_store_stat=False,borrowed_body_read=False,
    current_capacity_probe_performed=False,actual_launch_performed=False,
    review_script_sha256=sha(Path(__file__).read_bytes()))
(OUT/'actual-launch-review.json').write_bytes(canon(result))
print(json.dumps(dict(passed=True,terminal_readback_sha256=sha((OUT/'terminal-readback.json').read_bytes()),actual_launch_review_sha256=sha((OUT/'actual-launch-review.json').read_bytes()))))
