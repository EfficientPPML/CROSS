"""Bounded public failed-proof readback; no candidate imports or test replay."""
import ast
import hashlib
import json
import os
from pathlib import Path
import stat
import tarfile
import time
import xml.etree.ElementTree as ET
P=Path('/tmp/qwen-server-snapshot-integration-focused-proof-20261104/preparation-v1')
O=Path(__file__).parent
H=lambda raw:hashlib.sha256(raw).hexdigest()
pins={}
def read(name):
    path=P/name;info=path.lstat();assert stat.S_ISREG(info.st_mode)
    raw=path.read_bytes();pins[name]=H(raw);return raw
handoff=json.loads(read('handoff.json'));config=json.loads(read('run_config.json'))
assert pins['handoff.json']=='ad7b69f004da28a838094861bd384bc8759556951991252bff0f6691a86dedfe'
for name,digest in handoff['files'].items():assert H(read(name))==digest,name
expected=json.loads(read('sources-before.json'))
assert len(expected)==404 and json.loads(read('sources-after.json'))==json.loads(read('execution-after.json'))==expected
def inventory(root):
    result={}
    for path in root.rglob('*'):
        assert not path.is_symlink()
        if path.is_file():result[path.relative_to(root).as_posix()]=H(path.read_bytes())
    return result
assert inventory(Path(config['source_root']))==inventory(P/'execution')==expected
with tarfile.open(P/'sources.tar.gz')as archive:
    members=archive.getmembers();assert len(members)==len({m.name for m in members})==404
    assert all(m.isfile()and not m.name.startswith('/')and '..'not in Path(m.name).parts for m in members)
    assert {m.name:H(archive.extractfile(m).read())for m in members}==expected
runtime={n[8:]:v for n,v in expected.items()if n.startswith('runtime/')}
assert len(runtime)==391 and runtime==inventory(Path('/tmp/qwen-server-snapshot-runtime-v2-20261104'))
audit=json.loads(read('worker-audit.json'));live=json.loads(read('live-observation.json'))
assert audit['runtime_sources_before']==audit['runtime_sources_after']==config['runtime_source_expectations']==runtime
assert audit['denied']==audit['loaded_backend']==[]and audit['numpy_imported']is False and audit['runtime_module_origins']=={}
assert audit['collected_nodes']==config['selected_nodes']and audit['forbidden_roots']==config['forbidden_roots']
assert audit['pytest_exit_code']==1 and type(audit['pytest_exit_code'])is int and audit['affinity']==[112,113,114,115]
assert audit['snapshot_integration_recording_only']is True and audit['scientific_admission_tested']is False
assert audit['pid']==audit['pgid']==live['worker_pid']==live['worker_pgid']==646144
assert live['outer_pid']==live['outer_pgid']==646142
xml=ET.fromstring(read('tests.xml'));cases=list(xml.iter('testcase'))
counts={k:len(list(xml.iter(v)))for k,v in [('tests','testcase'),('failures','failure'),('errors','error'),('skipped','skipped')]}
assert counts==dict(tests=36,failures=25,errors=0,skipped=0)
assert [dict(classname=c.attrib['classname'],name=c.attrib['name'])for c in cases]==config['selected_xml_cases']
ordered=[n.split('[',1)[1][:-1]for n in config['selected_nodes']]
failed={c.attrib['name'].split('[',1)[1][:-1]for c in cases if c.find('failure')is not None}
assert failed==set(ordered[:25])
log=read('tests.log').decode();assert '25 failed, 11 passed in 34.09s'in log
for name in('report.json','owned-pytest-metadata.json','test-child-terminals.json'):assert not (P/name).exists()
metadata={};records={};failed_logs={};aliases={};total=0
base=P/'pytest-owned';assert stat.S_ISDIR(base.lstat().st_mode)
for item in os.scandir(base):
    path=Path(item.path);info=item.stat(follow_symlinks=False)
    assert info.st_uid==os.geteuid()
    if stat.S_ISLNK(info.st_mode):
        assert path.name.startswith('test_snapshot_integration_')and path.name.endswith('current')
        target=os.readlink(path);assert len(target)<=4096 and '..'not in Path(target).parts
        destination=Path(target)if Path(target).is_absolute()else path.parent/target
        assert destination.parent==base and destination.name.startswith(path.name[:-7])
        target_info=destination.lstat();assert stat.S_ISDIR(target_info.st_mode)and target_info.st_uid==os.geteuid()
        assert os.readlink(path)==target
        aliases[path.relative_to(P).as_posix()]=dict(target=target,followed=False)
        continue
    assert stat.S_ISDIR(info.st_mode)and path.name.startswith('test_snapshot_integration_')
    files={}
    for child in os.scandir(path):
        cpath=Path(child.path);ci=child.stat(follow_symlinks=False)
        assert stat.S_ISREG(ci.st_mode)and ci.st_uid==os.geteuid()
        assert child.name in('child.log','parent-terminal.json','child-terminal.json','result.json','tiny-public-proof.json')
        total+=ci.st_size;assert total<=16<<20
        raw=cpath.read_bytes();assert len(raw)==ci.st_size
        files[child.name]=raw;metadata[cpath.relative_to(P).as_posix()]=dict(bytes=len(raw),sha256=H(raw))
    parent=json.loads(files['parent-terminal.json']);scenario=parent['scenario'];assert scenario in ordered and scenario not in records
    assert parent['schema']=='qwen-snapshot-integration-owned-child-v1'and parent['child_result_read']is False
    pid=parent['pid'];assert type(pid)is int and pid>0 and type(parent['owned_session_pgid'])is int and parent['owned_session_pgid']==pid
    assert parent['observed_pgid']is None or(type(parent['observed_pgid'])is int and parent['observed_pgid']==pid)
    assert parent['reaped']is True and parent['group_absent']is True
    assert type(parent['exit_code'])is int and parent['exit_code']==int(scenario in failed)
    row=dict(parent=parent,directory=path.relative_to(P).as_posix(),log_sha256=H(files['child.log']))
    if scenario in failed:
        assert set(files)=={'parent-terminal.json','child.log'}
        text=files['child.log'].decode()
        assert 'self.full_boundary,self.capacity'in text and 'control_callbacks=tuple(_function_binding(fn)'in text
        assert text.rstrip().endswith('TypeError: original source-fixed Python callback required')
        failed_logs[scenario]=dict(path=str(path/'child.log'),sha256=H(files['child.log']),traceback=text)
        row['positive_result_absent']=True
    else:
        terminal=json.loads(files['child-terminal.json']);result=json.loads(files['result.json'])
        assert terminal['scenario']==result['scenario']==scenario
        assert terminal['pid']==result['pid']==result['pgid']==terminal['child_reported_pgid']==pid
        assert terminal['exit_code']==0 and terminal['reaped']is True and terminal['group_absent']is True
        for name in('assertions_completed','original_main_restored','temporary_names_removed','profile_restored','declared_exports_restored','runtime_sources_unchanged','definition_only_entry'):assert result[name]is True
        for name in('genuine_cli_bootstrap','genuine_backend_imported','scientific_admission','capacity_measurement','model_key_ciphertext_body_reads','actual_server_execution'):assert result[name]is False
        assert result['backend_denials']==[]and result['source_count']==391
        assert result['entry_dispatch_excluded_sha256']==config['excluded_entry_dispatch_ast_sha256']
        allowed={'child.log','parent-terminal.json','child-terminal.json','result.json'}
        if scenario.startswith('process_'):
            allowed.add('tiny-public-proof.json');assert len(files['tiny-public-proof.json'])<=1024
            assert result['counters']['ids']==result['counters']['digest']==result['counters']['ready_ids']==1
        assert set(files)==allowed
        row.update(terminal=terminal,result=result)
    records[scenario]=row
assert len(records)==36 and len(metadata)==100 and len(aliases)==15 and len(failed_logs)==25
assert len({r['parent']['pid']for r in records.values()})==36
assert not list(os.scandir(P/'tmp'))
# Inspect the two actual source expressions, rather than importing/replaying.
fixture=ast.parse((P/'execution/integration_fixture.py').read_bytes())
cap=next(n for n in ast.walk(fixture)if isinstance(n,ast.FunctionDef)and n.name=='capture_runtime')
call=next(n for n in ast.walk(cap)if isinstance(n,ast.Call)and ast.unparse(n.func)=='self.process._runtime')
assert ast.unparse(call.args[4])=='self.full_boundary'and ast.unparse(call.args[5])=='self.capacity'
runtime_tree=ast.parse((P/'execution/runtime/qwen_server_process.py').read_bytes())
binding=next(n for n in runtime_tree.body if isinstance(n,ast.FunctionDef)and n.name=='_function_binding')
assert ast.unparse(binding.body[0].test)=='type(fn) is not FunctionType'
pids={646142,646144}|{r['parent']['pid']for r in records.values()}
absence=[]
for kind,fn in(('pid',os.kill),('pgid',os.killpg)):
    for value in sorted(pids):
        try:fn(value,0)
        except ProcessLookupError:absence.append(dict(kind=kind,id=value,absent=True))
        else:raise AssertionError('named owned process/group remains present')
def write(name,value):
    raw=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode();(O/name).write_bytes(raw);return H(raw)
absence_pin=write('process-absence.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),scope='independent post-exit signal0 only; no live/wait authority',checks=absence))
record_pin=write('owned-public-records.json',dict(records=[records[n]for n in ordered],regular_files=metadata,unfollowed_aliases=aliases))
failed_pin=write('failed-child-logs.json',failed_logs)
report=dict(schema='qwen-snapshot-integration-failed-terminal-peer-review-v1',verification_passed=True,run_passed=False,
    proof_root=str(P),source_count=404,runtime_count=391,all_source_current_execution_start_end_archive_equal=True,
    tests=counts,passed_case_count=11,passed_groups=dict(recorded_process=6,terminal_function=5),exact_ordered_XML_and_collection=True,
    early_parent_records=36,failed_children=25,positive_result_terminal_pairs=11,regular_owned_files=100,unfollowed_aliases=15,
    worker_backend_audit_empty=True,report_absent=True,positive_aggregate_inspection_absent=True,
    diagnosis='All25 direct cases pass bound self.full_boundary/self.capacity into actual _runtime, whose _function_binding requires exact FunctionType. The first rejected bound source_check callback prevents those intended assertions from running. Six process cases supply plain nested callbacks and five terminal cases do not use this fixture.',
    secondary_failure='The success-only outer inspector requires144 child artifacts; only94 child artifacts plus6 tiny fixtures exist after25 early setup failures, so it raises before aggregate metadata/records/report publication. Original early parent records, logs, XML, worker audit and source end maps remain.',
    prior_review_limitation='The earlier source-only acceptance did not catch the MethodType versus FunctionType fixture mismatch. That historical receipt remains unchanged; this failure does not establish a runtime391 defect.',
    failed_child_state_limit='No positive result exists for25 failed children. Do not infer their completion/restoration flags or intended mutation coverage. Early parent records supply their observed owned identities and wait/reap result.',
    root_reported_owner=dict(launch_chunk='4c43f9',session=31389,terminal_chunk='915588',exit_code=1,reaped=True,independent_live_chunk='300315'),
    reviewer_waited=False,reviewer_live_observation=False,process_absence_sha256=absence_pin,owned_public_records_sha256=record_pin,failed_logs_sha256=failed_pin,proof_hashes=pins,
    rerun=False,candidate_imports=False,backend_or_capacity_probe=False,borrowed_body_reads=False,actual_authority=False)
pin=write('review.json',report)
print(json.dumps({'review_sha256':pin,'tests':counts,'regular':100,'aliases':15,'processes_and_groups_absent':len(pids)},sort_keys=True))
