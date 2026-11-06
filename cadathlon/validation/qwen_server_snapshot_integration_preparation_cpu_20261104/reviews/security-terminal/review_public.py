"""Independent public terminal readback only; never imports/tests candidate."""
import hashlib,json,os,stat,tarfile,time
from pathlib import Path
import xml.etree.ElementTree as ET
P=Path('/tmp/qwen-server-snapshot-integration-focused-proof-20261104/preparation-v2')
O=Path(__file__).parent;H=lambda b:hashlib.sha256(b).hexdigest();pins={}
def read(name):
    p=P/name;assert stat.S_ISREG(p.lstat().st_mode)
    raw=p.read_bytes();pins[name]=H(raw);return raw
h=json.loads(read('handoff.json'));c=json.loads(read('run_config.json'));r=json.loads(read('report.json'))
a=json.loads(read('worker-audit.json'));live=json.loads(read('live-observation.json'))
assert pins['handoff.json']=='50ffb8a07c5d63d69ffd60e8e40a789f272075851427f7326eee5bf9b3af6555'
assert pins['report.json']=='98c2642ab0cdc5dd5e7871c3d55bd6b8ff7b1da3e7b76e6e3de29db06f56afbb'
for name,digest in h['files'].items():assert H(read(name))==digest,name
raw=read('sources-before.json');m=json.loads(raw)
assert raw==read('sources-after.json')==read('execution-after.json')
assert raw==(json.dumps(m,sort_keys=True,separators=(',',':'))+'\n').encode()
assert len(m)==405 and H(raw)==r['source_manifest_sha256']==c['source_manifest_sha256']
def inventory(root):
    out={}
    for p in root.rglob('*'):
        assert not p.is_symlink()
        if p.is_file():out[p.relative_to(root).as_posix()]=H(p.read_bytes())
    return out
assert inventory(Path(c['source_root']))==inventory(P/'execution')==m
with tarfile.open(P/'sources.tar.gz')as t:
    members=t.getmembers();assert len(members)==len({x.name for x in members})==405
    assert all(x.isfile()and not x.name.startswith('/')and '..'not in Path(x.name).parts for x in members)
    assert {x.name:H(t.extractfile(x).read())for x in members}==m
runtime={n[8:]:v for n,v in m.items()if n.startswith('runtime/')};assert len(runtime)==391
assert runtime==a['runtime_sources_before']==a['runtime_sources_after']==c['runtime_source_expectations']
assert runtime==inventory(Path('/tmp/qwen-server-snapshot-runtime-v2-20261104'))
assert r['passed']is True and r['sources_unchanged']is True and r['child_returncode']==0 and type(r['child_returncode'])is int
assert r['child_reaped']is True and r['child_group_absent']is True
assert a['denied']==a['loaded_backend']==[]and a['numpy_imported']is False and a['runtime_module_origins']=={}
assert a['pytest_exit_code']==0 and type(a['pytest_exit_code'])is int and a['affinity']==[112,113,114,115]
assert a['forbidden_roots']==c['forbidden_roots']and a['collected_nodes']==r['selected_nodes']==c['selected_nodes']
assert a['snapshot_integration_recording_only']is True and a['scientific_admission_tested']is False
assert a['pid']==a['pgid']==live['worker_pid']==live['worker_pgid']==665002
assert live['outer_pid']==live['outer_pgid']==664985
xml=ET.fromstring(read('tests.xml'));cases=list(xml.iter('testcase'))
counts={k:len(list(xml.iter(v)))for k,v in [('tests','testcase'),('failures','failure'),('errors','error'),('skipped','skipped')]}
assert counts==r['tests']==dict(tests=36,failures=0,errors=0,skipped=0)
assert [dict(classname=x.attrib['classname'],name=x.attrib['name'])for x in cases]==c['selected_xml_cases']==r['selected_xml_cases']
read('tests.log')
metadata=json.loads(read('owned-pytest-metadata.json'));records=json.loads(read('test-child-terminals.json'))
assert records==r['child_records']and len(records)==36
assert len(metadata)==r['retained_owned_files']==165
assert sum(v['kind']=='regular'for v in metadata.values())==150 and sum(v['kind']=='owned_pytest_alias'for v in metadata.values())==15
aliases={};regular={};observed=set();total=0
for leaf in(P/'tmp',P/'pytest-owned'):
    assert stat.S_ISDIR(leaf.lstat().st_mode)
    def walk(directory):
        global total
        for item in os.scandir(directory):
            p=Path(item.path);name=p.relative_to(P).as_posix();info=item.stat(follow_symlinks=False)
            assert info.st_uid==os.geteuid()
            if stat.S_ISLNK(info.st_mode):
                row=metadata[name];assert row['kind']=='owned_pytest_alias'and p.parent==P/'pytest-owned'
                assert p.name.startswith('test_snapshot_integration_')and p.name.endswith('current')
                target=os.readlink(p);assert target==row['target']and len(target)<=4096 and '..'not in Path(target).parts
                dest=Path(target)if Path(target).is_absolute()else p.parent/target
                assert dest==P/row['resolved_target']and dest.parent==p.parent
                ti=dest.lstat();assert stat.S_ISDIR(ti.st_mode)and ti.st_uid==os.geteuid()
                assert os.readlink(p)==target;after=p.lstat()
                assert (after.st_dev,after.st_ino,after.st_mode,after.st_uid)==(info.st_dev,info.st_ino,info.st_mode,info.st_uid)
                aliases[name]=dict(target=target,followed=False);observed.add(name)
            elif stat.S_ISDIR(info.st_mode):walk(p)
            else:
                assert stat.S_ISREG(info.st_mode);row=metadata[name]
                assert row['kind']=='regular'and row['bytes']==info.st_size
                raw=p.read_bytes();assert H(raw)==row['sha256'];total+=len(raw);assert total<=16<<20
                regular[name]=row;observed.add(name)
    walk(leaf)
assert observed==set(metadata)
ordered=[n.split('[',1)[1][:-1]for n in c['selected_nodes']]
assert [row['result']['scenario']for row in records]==ordered
pids={664985,665002};covered=set();counters={};process_fixtures={}
for row in records:
    early=row['parent_terminal'];terminal=row['terminal'];result=row['result'];scenario=result['scenario']
    for key,value in(('parent_terminal_path',early),('terminal_path',terminal),('result_path',result)):
        name=row[key];assert name in regular and json.loads((P/name).read_bytes())==value;covered.add(name)
    assert row['log_path']in regular and regular[row['log_path']]['sha256']==row['log_sha256'];covered.add(row['log_path'])
    assert regular[row['parent_terminal_path']]['sha256']==row['parent_terminal_sha256']
    assert early['schema']=='qwen-snapshot-integration-owned-child-v1'and early['child_result_read']is False
    assert early['scenario']==terminal['scenario']==scenario
    pid=result['pid'];assert type(pid)is int and pid>0 and type(result['pgid'])is int and result['pgid']==pid
    assert pid not in pids;pids.add(pid)
    for owner in(early,terminal):
        assert type(owner['pid'])is int and owner['pid']==pid and type(owner['exit_code'])is int and owner['exit_code']==0
        assert owner['reaped']is True and owner['group_absent']is True
        assert owner['observed_pgid']is None or(type(owner['observed_pgid'])is int and owner['observed_pgid']==pid)
    assert early['observed_pgid']==terminal['observed_pgid']and early['owned_session_pgid']==terminal['child_reported_pgid']==pid
    assert terminal['scope']=='synthetic retained-runtime integration only'
    for name in('assertions_completed','original_main_restored','temporary_names_removed','profile_restored','declared_exports_restored','runtime_sources_unchanged','definition_only_entry'):assert result[name]is True
    for name in('genuine_cli_bootstrap','genuine_backend_imported','scientific_admission','capacity_measurement','model_key_ciphertext_body_reads','actual_server_execution'):assert result[name]is False
    assert result['backend_denials']==[]and result['source_count']==391
    assert result['entry_dispatch_excluded_sha256']==c['excluded_entry_dispatch_ast_sha256']
    counters[scenario]=result['counters']
    if not scenario.startswith('terminal_'):
        assert result['counters']['replacement']==0
        assert result['counters']['ids']==result['counters']['digest']==result['counters']['ready_ids']==1
    if scenario.startswith('process_'):
        name=(Path(row['result_path']).parent/'tiny-public-proof.json').as_posix();assert name in regular and regular[name]['bytes']<=1024
        value=json.loads((P/name).read_bytes())
        assert value=={'synthetic_public_observation':'changed after hot check'if scenario=='process_public_proof_changed'else True}
        process_fixtures[scenario]=dict(path=name,**regular[name]);covered.add(name)
    if scenario=='process_default_capacity_denial':
        assert all(result['counters'][n]==0 for n in('core_load','stage','evidence'))and 'core.install'not in result['events']
    if scenario=='process_clean':
        events=result['events'];assert result['counters']['guard_retire']==1
        assert events.index('loader.context.retired')<events.index('evidence.close')<events.index('audit.close')<events.index('shield.exit')<events.index('guard.retire')
    if scenario.startswith('terminal_'):
        assert result['detail']['terminal_status']==(0 if scenario=='terminal_zero'else 1 if scenario=='terminal_one'else 70)
assert len(pids)==38 and len(process_fixtures)==6 and covered==set(regular)
for flag in('actual_client_role_execution','actual_device_initialized','actual_execution_authorized','actual_key_generation','actual_proofs_present','backend_context_construction','encrypted_execution','full_scientific_admission_tested','genuine_capacity_measured','genuine_cli_bootstrap','model_execution','numpy_imported','observed_child_module_origin_table'):assert r[flag]is False
assert r['definition_only_entry']is True and r['snapshot_integration_recording_only']is True
assert r['retained_child_artifacts']==144 and r['owned_public_fixture_files']==6
old=Path('/tmp/qwen-server-snapshot-integration-failed-security-review-20261104/review.json')
assert H(old.read_bytes())=='afea0689bb3a0072b05d4e77b50e9140c40a00aac1b89a5307d8e9c09b69849d'
oldreview=json.loads(old.read_bytes());oldproof=Path(oldreview['proof_root'])
for name,digest in oldreview['proof_hashes'].items():assert H((oldproof/name).read_bytes())==digest
assert not(oldproof/'report.json').exists()
absence=[]
for kind,fn in(('pid',os.kill),('pgid',os.killpg)):
    for value in sorted(pids):
        try:fn(value,0)
        except ProcessLookupError:absence.append(dict(kind=kind,id=value,absent=True))
        else:raise AssertionError('named owned process/group remains present')
def write(name,value):
    raw=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode();(O/name).write_bytes(raw);return H(raw)
absence_pin=write('process-absence.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),scope='independent post-exit signal0; no live/wait authority',checks=absence))
record_pin=write('owned-public-record-pins.json',dict(regular_files=regular,unfollowed_aliases=aliases,process_fixtures=process_fixtures,counters=counters))
review=dict(schema='qwen-server-snapshot-integration-terminal-peer-review-v2',verification_passed=True,run_passed=True,proof_root=str(P),source_count=405,runtime_count=391,tests=counts,
    source_current_execution_start_end_archive_equal=True,exact_ordered_collection_and_XML=True,early_parent_records=36,positive_child_records=36,regular_child_artifacts=144,tiny_owned_public_fixtures=6,unfollowed_aliases=15,
    worker_backend_audit_empty=True,all_child_restoration_flags=True,all_counted_replacement_calls_zero=True,derived_ids_digest_once_per_nonterminal_case=True,default_capacity_denial_before_core=True,
    owner_scope='Root owns launch/wait/reap. Reviewer independently rechecks public records and post-exit absence only.',root_reported_owner=dict(launch_chunk='5005f4',session=46330,terminal_chunk='98ea2c',exit_code=0,reaped=True,independent_live_chunk='fbfe6b'),
    reviewer_waited=False,reviewer_live_observation=False,process_absence_sha256=absence_pin,owned_public_record_pins_sha256=record_pin,proof_hashes=pins,
    historical_failure=dict(review_sha256=H(old.read_bytes()),tests=36,failures=25,passed=11,report_absent=True,preserved=True,scope='Common bound-method fixture setup mismatch; no intended direct mutation coverage in that failed run.'),
    scope='Completed36 synthetic retained runtime/control/backend constructor and Finalizer cases with explicit scientific/admission/core/audit/stage/evidence seams, source-owned temporary numeric names, and definition-only __main__ fixture. No genuine CLI/native/model/HE/capacity/scientific authority.',
    rerun=False,candidate_imports=False,backend_or_capacity_probes=False,borrowed_body_reads=False,actual_authority=False)
pin=write('review.json',review)
print(json.dumps({'review_sha256':pin,'tests':counts,'regular':150,'aliases':15,'pid_and_group_count':38},sort_keys=True))
