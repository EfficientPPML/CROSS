"""Independent bounded public readback; no candidate imports or fixture bodies."""
import hashlib,json,os,stat,tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
proof=Path('/tmp/qwen-full28-device-focused-proof-20261104/preparation-v4')
source=Path('/tmp/qwen-full28-device-focused-tests-v3-20261105')
out=Path('/tmp/qwen-full28-device460-terminal-cnn-review-20261105')
sha=lambda raw:hashlib.sha256(raw).hexdigest()
canon=lambda value:(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def read(path,limit=2<<20):
    before=path.lstat();assert stat.S_ISREG(before.st_mode)and before.st_uid==os.geteuid()and 0<=before.st_size<=limit
    state=lambda v:(v.st_dev,v.st_ino,v.st_size,v.st_mode,v.st_uid,v.st_mtime_ns,v.st_ctime_ns)
    fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        assert state(os.fstat(fd))==state(before);blocks=[];size=0
        while size<=limit:
            block=os.read(fd,min(65536,limit+1-size))
            if not block:break
            blocks.append(block);size+=len(block)
        assert size==before.st_size and state(os.fstat(fd))==state(before)==state(path.lstat())
        return b''.join(blocks)
    finally:os.close(fd)
def inv(root):
    result={}
    for p in sorted(root.rglob('*')):
        info=p.lstat();assert not stat.S_ISLNK(info.st_mode)
        if stat.S_ISREG(info.st_mode):result[p.relative_to(root).as_posix()]=sha(read(p,64<<20))
        else:assert stat.S_ISDIR(info.st_mode)
    return result
public_names=['sources-before.json','sources-after.json','execution-after.json','worker-audit.json',
    'worker-early.json','worker-terminal.json','worker-owner-terminal.json','collector-terminal.json',
    'live-observation.json','tests.xml','tests.log','run_config.json','handoff.json','launch-proposal.json',
    'executed_driver.py','worker_driver.py','inspect_children.py']
raws={name:read(proof/name)for name in public_names}
public_pins={name:dict(path=str(proof/name),sha256=sha(raw),bytes=len(raw))for name,raw in raws.items()}
config=json.loads(raws['run_config.json']);handoff=json.loads(raws['handoff.json'])
assert sha(raws['handoff.json'])=='3a0b3e9ef11be7cf46cb3d3e0a877ec5a09367e1a1502dd8b3935a18da5c5c55'
for name,key in [('run_config.json','config_sha256'),('executed_driver.py','driver_sha256'),('worker_driver.py','worker_sha256'),('inspect_children.py','inspector_sha256'),('launch-proposal.json','proposal_sha256')]:
    assert sha(raws[name])==handoff[key]
expected=json.loads(raws['sources-before.json'])
assert len(expected)==460 and sha(raws['sources-before.json'])==config['source_manifest_sha256']==handoff['source_manifest_sha256']
assert inv(source)==inv(proof/'execution')==expected
assert json.loads(raws['sources-after.json'])==json.loads(raws['execution-after.json'])==expected
archive_bytes=read(proof/'sources.tar.gz',16<<20)
assert sha(archive_bytes)==config['source_archive_sha256']==handoff['source_archive_sha256']
public_pins['sources.tar.gz']=dict(path=str(proof/'sources.tar.gz'),sha256=sha(archive_bytes),bytes=len(archive_bytes))
with tarfile.open(proof/'sources.tar.gz')as archive:
    members=archive.getmembers();assert len(members)==len({m.name for m in members})==460 and all(m.isfile()for m in members)
    assert {m.name:sha(archive.extractfile(m).read())for m in members}==expected
runtime=json.loads(read(source/'runtime_sources.json'))
assert len(runtime)==403 and {n[8:]:h for n,h in expected.items()if n.startswith('runtime/')}==runtime
assert config['runtime_source_expectations']==runtime
selection=json.loads(read(source/'selected_tests.json'))
assert selection['cases']==config['selected_cases']and len(selection['cases'])==59
xml=ET.fromstring(raws['tests.xml']);cases=list(xml.iter('testcase'))
counts={key:len(list(xml.iter(tag)))for key,tag in [('tests','testcase'),('failures','failure'),('errors','error'),('skipped','skipped')]}
assert counts==dict(tests=59,failures=0,errors=0,skipped=0)
assert [dict(classname=c.attrib['classname'],name=c.attrib['name'])for c in cases]==config['selected_xml_cases']
xml_failed={case for case,node in zip(config['selected_cases'],cases)if node.find('failure')is not None}
audit=json.loads(raws['worker-audit.json']);worker=json.loads(raws['worker-owner-terminal.json'])
collector=json.loads(raws['collector-terminal.json']);live=json.loads(raws['live-observation.json'])
early=json.loads(raws['worker-early.json']);terminal=json.loads(raws['worker-terminal.json'])
assert audit['collected_nodes']==config['selected_nodes']and audit['pytest_exit_code']==audit['planned_exit_code']==0
assert audit['denied']==audit['loaded_backend']==[]and audit['numpy_imported']is False and audit['runtime_module_origins']=={}
assert audit['runtime_sources_before']==audit['runtime_sources_after']==runtime
assert audit['affinity']==[112,113,114,115]and audit['primary_type']is None and audit['cleanup_types']==[]
assert worker['exit_code']==0 and worker['direct_wait_completed']is True and worker['child_reaped']is True
assert worker['pid_absent']is True and worker['group_absent']is True and worker['primary_type']is None and worker['cleanup_types']==[]
assert collector['error_type']is None and collector['planned_exit_code']==0 and collector['normal_main_return']is True
assert collector['pid']==collector['pgid']==worker['parent_pid']==worker['parent_pgid']==live['outer_pid']==live['outer_pgid']==922320
assert worker['worker_pid']==worker['observed_pgid']==worker['owned_session_pgid']==live['worker_pid']==live['worker_pgid']==922321
assert collector['start_ticks']==worker['parent_start_ticks']==live['outer_start_ticks']==20418229
assert worker['start_ticks']==live['worker_start_ticks']==20418255
for row in (early,terminal,audit):
    assert type(row['pid'])is int and row['pid']==row['pgid']==922321 and row['start_ticks']==20418255
assert early['config_sha256']==sha(raws['run_config.json'])and early['source_manifest_sha256']==config['source_manifest_sha256']
assert terminal['normal_main_return']is True and terminal['planned_exit_code']==terminal['pytest_exit_code']==0
records={};owned_pins={};aliases={};fixture_metadata={};failures={};total=0
base=proof/'pytest-owned';plan=config['artifact_plan']
assert stat.S_ISDIR(base.lstat().st_mode)
for child in sorted(base.iterdir()):
    info=child.lstat();assert info.st_uid==os.geteuid()
    if stat.S_ISLNK(info.st_mode):
        assert child.name.startswith('test_device_integration_')and child.name.endswith('current')
        target=os.readlink(child);assert 0<len(target)<=4096 and '\\'not in target
        lexical=Path(target);assert str(lexical)==target and '..'not in lexical.parts and '.'not in lexical.parts
        if lexical.is_absolute():named=lexical
        else:
            assert target not in('.', '..')and '/'not in target and lexical.name==target
            named=child.parent/target
        assert 0<len(named.name)<=255 and named.parent==base and named!=child
        target_info=named.lstat();assert stat.S_ISDIR(target_info.st_mode)and target_info.st_uid==os.geteuid()
        prefix=child.name[:-len('current')];assert named.name.startswith(prefix)and named.name[len(prefix):].isdigit()
        after=child.lstat();assert(info.st_dev,info.st_ino,info.st_mode,info.st_uid)==(after.st_dev,after.st_ino,after.st_mode,after.st_uid)
        assert os.readlink(child)==target
        aliases[child.name]=dict(target=target,followed=False,target_metadata_only=True);continue
    assert stat.S_ISDIR(info.st_mode)
    data={name:read(child/name,plan['per_record_limit_bytes'])for name in ('parent-terminal.json','child-terminal.json','child.log')}
    parent=json.loads(data['parent-terminal.json']);term=json.loads(data['child-terminal.json'])
    case=parent['case'];assert case in config['selected_cases']and case not in records and term['case']==case
    assert parent['status']=='terminal'and parent['direct_wait_completed']is True and parent['reaped']is True and parent['group_absent']is True
    assert parent['primary_type']is None and type(parent['exit_code'])is int and parent['exit_code']in(0,70)
    assert type(term['pid'])is int and term['pid']>0 and type(term['pgid'])is int
    assert term['pid']==term['pgid']==parent['pid']==parent['owned_session_pgid']
    assert parent['observed_pgid']is None or(type(parent['observed_pgid'])is int and parent['observed_pgid']==term['pgid'])
    assert type(term['starttime_ticks'])is int and term['starttime_ticks']>0
    assert parent['starttime_ticks']is None or parent['starttime_ticks']==term['starttime_ticks']
    try:(child/'result.json').lstat()
    except FileNotFoundError:result=None
    else:
        data['result.json']=read(child/'result.json',plan['per_record_limit_bytes']);result=json.loads(data['result.json'])
    if parent['exit_code']==0:
        assert case not in xml_failed and result is not None and result['passed']is True and result['case']==case
        assert term['normal_return']is True and term['status']=='normal_return'and parent['child_report_read']is True
        assert result['pid']==result['pgid']==term['pid']and result['starttime_ticks']==term['starttime_ticks']
        assert term['result_sha256']==parent['result_sha256']==sha(data['result.json'])
        assert parent['child_terminal_sha256']==sha(data['child-terminal.json'])
        assert result['backend_denials']==term['backend_denials']==[]
        for field in ('definition_only_entry','original_main_restored','temporary_names_removed','profile_restored','declared_exports_restored','source_unchanged'):assert result[field]is True
        for field in ('genuine_cli_bootstrap','model_key_ciphertext_body_reads','native_backend_imported','capacity_measured','actual_execution_authorized'):assert result[field]is False
    else:raise AssertionError('successful closure requires every original child normal zero/result')
    allowed=set(plan['fixture_files'][case]);seen=set();dirs=[]
    def walk(folder,prefix=''):
        for path in sorted(folder.iterdir()):
            rel=prefix+path.name;s=path.lstat();assert s.st_uid==os.geteuid()and not stat.S_ISLNK(s.st_mode)
            if stat.S_ISDIR(s.st_mode):
                # Fixture creates these empty owned roots even when no file is
                # selected. They are metadata only, never body-read candidates.
                assert rel in ('outputs','public-memory')or any(n.startswith(rel+'/')for n in allowed)
                dirs.append(rel);walk(path,rel+'/');continue
            assert stat.S_ISREG(s.st_mode)
            if not prefix and rel in data:continue
            assert rel in allowed
            seen.add(rel)
            fixture_metadata[path.relative_to(proof).as_posix()]=dict(bytes=s.st_size,device=s.st_dev,inode=s.st_ino,
                mode=stat.S_IMODE(s.st_mode),body_read=False,body_copied=False)
    walk(child)
    if result is not None:
        assert seen==allowed
        declared=parent['fixture_metadata'];assert declared['body_read']is False and declared['body_copied']is False
        assert set(declared['files'])==seen and declared['directories']==sorted(dirs)
        for name in seen:
            actual=fixture_metadata[(child/name).relative_to(proof).as_posix()]
            assert all(declared['files'][name][k]==actual[k]for k in ('bytes','device','inode','mode'))
    for name,raw in data.items():
        total+=len(raw);assert total<=plan['total_record_limit_bytes']
        owned_pins[(child/name).relative_to(proof).as_posix()]=dict(sha256=sha(raw),bytes=len(raw))
    records[case]=dict(parent=parent,child=term,result_present=result is not None,log_sha256=sha(data['child.log']),
        fixture_files=sorted(seen),fixture_directories=sorted(dirs),readiness=False)
assert set(records)==set(config['selected_cases'])and len(records)==59 and len(failures)==0 and len(owned_pins)==236
assert sum(row['result_present']for row in records.values())==59
assert not list((proof/'tmp').iterdir())
for name in ('report.json','child-artifact-review.json'):
    raws[name]=read(proof/name)
    public_pins[name]=dict(path=str(proof/name),sha256=sha(raws[name]),bytes=len(raws[name]))
report=json.loads(raws['report.json']);artifacts=json.loads(raws['child-artifact-review.json'])
assert report['passed']is True and report['tests']==counts and report['source_count']==460 and report['runtime_count']==403
assert report['sources_unchanged']is True and report['source_manifest_sha256']==config['source_manifest_sha256']
assert report['source_archive_sha256']==config['source_archive_sha256']and report['artifact_review_sha256']==sha(raws['child-artifact-review.json'])
assert report['validated_children']==artifacts['validated_children']==59
assert artifacts['passed']is True and artifacts['regular_files']==236 and artifacts['fixture_metadata_files']==14
assert artifacts['aliases_followed']is False and artifacts['fixture_bodies_read']is False
assert len(fixture_metadata)==14
root_tool=Path('/tmp/qwen-full28-device-focused-root-harness-review-20261105/tool-terminal460.json')
assert sha(read(root_tool))=='d4b7da940d8ae501639c149bad36d5d67381f9cff6a3bb8e52043618e6842136'
tool=json.loads(read(root_tool));assert tool['exit_code']==0 and tool['session_id']==2862 and tool['terminal_tool']=='5aa837'
assert tool['terminal_observed_by_root']is True
identities=[dict(pid=922320,pgid=922320,role='collector',start_ticks=20418229),
    dict(pid=922321,pgid=922321,role='worker',start_ticks=20418255)]+[
    dict(pid=row['child']['pid'],pgid=row['child']['pgid'],role=case,start_ticks=row['child']['starttime_ticks'])for case,row in records.items()]
assert len({row['pid']for row in identities})==61
for row in identities:
    for field,call in [('pid',os.kill),('pgid',os.killpg)]:
        try:call(row[field],0)
        except ProcessLookupError:row[field+'_absent']=True
        else:raise AssertionError('recorded process/group still present')
assert inv(source)==inv(proof/'execution')==expected
assert all(read(proof/name)==raw for name,raw in raws.items())
for relative,pin in owned_pins.items():assert sha(read(proof/relative))==pin['sha256']
categories={name:sum(row['category']==name for row in failures.values())for name in sorted({row['category']for row in failures.values()})}
review=dict(schema='full28-device460-independent-terminal-review-v1',reviewer='cnn_performance',
    public_terminal_review_passed=True,tests_passed=True,actual_ready=False,source_count=460,runtime_count=403,
    source_manifest_sha256=config['source_manifest_sha256'],source_archive_sha256=config['source_archive_sha256'],
    source_current_execution_start_end_archive_exact=True,counts=counts,passed_cases=59,failed_cases=0,
    worker_normal_exit_code=0,collector_parent_reported_exit_code=0,collector_error_type=None,
    root_owner=dict(session=2862,launch_tool='9ba0f2',terminal_tool='5aa837',authority='root owns authoritative direct exit/reap; this reviewer only independently checks public joins and absence'),
    child_normal_zero_results=59,child_early_only_exit70=0,parent_direct_wait_records=59,
    retained_regular_records=len(owned_pins),fixture_files_metadata_only=len(fixture_metadata),unfollowed_aliases=len(aliases),
    pid_group_pairs_independently_absent=61,failure_locations=categories,
    worker_backend_audit_empty=True,failed_child_backend_denials_not_promoted_to_clean=True,
    collector_report_verified=True,recording_only_completed_proof=True,
    limitations=['Only named synthetic recording seams and source-owned fixtures were exercised; no runtime promotion.',
        'No candidate imports, collection, tests, backend/native probes, fixture body reads or rerun.',
        'No genuine scientific admission, actual authority, device capacity, learned model or HE claim.'])
root_review=Path('/tmp/qwen-full28-device-focused-root-harness-review-20261105/success460-terminal-review.json')
assert sha(read(root_review))=='fe2a893fa7249b755f1bd595fa4d240571ee5a87a499859db4aece114cc63af2'
review['root_success_terminal_review']=dict(path=str(root_review),sha256=sha(read(root_review)))
review['root_tool_terminal']=dict(path=str(root_tool),sha256=sha(read(root_tool)))
review['report_sha256']=sha(raws['report.json'])
assert not out.exists();out.mkdir()
for name,value in [('review.json',review),('public-pins.json',public_pins),('owned-record-pins.json',owned_pins),
    ('owned-records.json',records),('failure-locations.json',failures),('fixture-metadata.json',fixture_metadata),
    ('aliases.json',aliases),('process-absence.json',identities)]:
    (out/name).write_bytes(canon(value))
(out/'review_public.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(review,sort_keys=True,indent=2));print('review_sha256',sha((out/'review.json').read_bytes()))
