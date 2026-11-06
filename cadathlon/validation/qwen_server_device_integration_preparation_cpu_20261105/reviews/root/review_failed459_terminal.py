"""Bounded review of the original failed459 run; never reads fixture bodies."""
from pathlib import Path
import datetime, hashlib, io, json, os, stat, tarfile
import xml.etree.ElementTree as ET

HERE=Path(__file__).parent
ROOT=Path('/tmp/qwen-full28-device-focused-proof-20261104/preparation-v3')
SOURCE=Path('/tmp/qwen-full28-device-focused-tests-v2-20261105')
pins={}
def sha(raw):return hashlib.sha256(raw).hexdigest()
def state(s):return(s.st_dev,s.st_ino,s.st_mode,s.st_uid,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def read(p,limit=2<<20):
    before=p.lstat();assert stat.S_ISREG(before.st_mode)and before.st_uid==os.geteuid()and before.st_size<=limit
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC)
    try:
        assert state(before)==state(os.fstat(fd));parts=[];total=0
        while total<=limit:
            block=os.read(fd,min(65536,limit+1-total))
            if not block:break
            parts.append(block);total+=len(block)
        assert total==before.st_size and state(before)==state(os.fstat(fd))==state(p.lstat())
    finally:os.close(fd)
    raw=b''.join(parts);pins[str(p)]=dict(bytes=len(raw),sha256=sha(raw));return raw
def doc(p):return json.loads(read(p))
def gone(pid):
    for f in(os.kill,os.killpg):
        try:f(pid,0)
        except ProcessLookupError:continue
        raise AssertionError('process or group still exists: '+str(pid))
def inventory(root):
    result={}
    def walk(p):
        assert stat.S_ISDIR(p.lstat().st_mode)
        for child in p.iterdir():
            if stat.S_ISDIR(child.lstat().st_mode):walk(child)
            else:result[child.relative_to(root).as_posix()]=sha(read(child,8<<20))
    walk(root);return result

tool=doc(HERE/'tool-terminal459.json')
assert tool['session_id']==20753 and tool['exit_code']==70 and tool['terminal_tool']=='d24664'and tool['terminal_observed_by_root']is True
gone(901567);gone(901568)
config_raw=read(ROOT/'run_config.json');assert sha(config_raw)=='a73571443c2c19c840fcdfdf4ae5f690e9ebdc8cfb7738bf468542eb52653e87'
config=json.loads(config_raw);expected=doc(ROOT/'sources-before.json')
assert len(expected)==459 and pins[str(ROOT/'sources-before.json')]['sha256']=='66126ceea29081ca2e1cb7781cc4bee4804e4c3faced8d72e7f4e4b85ebec318'
assert inventory(SOURCE)==inventory(ROOT/'execution')==doc(ROOT/'sources-after.json')==doc(ROOT/'execution-after.json')==expected
archive=read(ROOT/'sources.tar.gz',8<<20);assert sha(archive)=='b45917e53042999c6d5bf72a53cb14a637f18c5fd9f3d7fb46335d1261856442'
with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz')as t:
    members=t.getmembers();assert len(members)==len({m.name for m in members})==459 and set(m.name for m in members)==set(expected)
    for m in members:
        assert m.isfile()and m.size<=8<<20 and sha(t.extractfile(m).read())==expected[m.name]
for name,hash_value in [('executed_driver.py','06054b368504260ebf8c6689be2612721968667dbe4825960b05e0e860d8e1e5'),('worker_driver.py','e2f304bfceb6bcd2bfaaaa57c2e3c78f10ca419390c3b0ef57a8236ad7027543'),('inspect_children.py','26288c3a11dbe7e2578920b070bad4cf281e85a877934af39bd97bec7597b0cf')]:
    assert sha(read(ROOT/name))==hash_value
assert sha(read(HERE/'live459.json'))=='f0df32f9c08a39d7b904705c6f66aca3e2969058949d53324754ffb31002a1e2'
live=doc(ROOT/'live-observation.json');collector=doc(ROOT/'collector-terminal.json');owner=doc(ROOT/'worker-owner-terminal.json');worker=doc(ROOT/'worker-terminal.json');audit=doc(ROOT/'worker-audit.json');early=doc(ROOT/'worker-early.json')
assert collector['pid']==collector['pgid']==live['outer_pid']==live['outer_pgid']==901567 and collector['start_ticks']==live['outer_start_ticks']==20158353
assert collector['planned_exit_code']==70 and collector['normal_main_return']is False and collector['error_type']=='FileNotFoundError'
assert owner['worker_pid']==owner['owned_session_pgid']==live['worker_pid']==901568 and owner['parent_pid']==owner['parent_pgid']==901567
assert owner['start_ticks']==live['worker_start_ticks']==20158379 and owner['parent_start_ticks']==20158353
assert owner['exit_code']==1 and owner['primary_type']is None and owner['cleanup_types']==[]
assert all(owner[n]is True for n in('direct_wait_completed','child_reaped','pid_absent','group_absent'))
for row in(worker,audit,early):assert row['pid']==row['pgid']==901568 and row['start_ticks']==20158379
assert worker['normal_main_return']is True and worker['pytest_exit_code']==worker['planned_exit_code']==1
assert audit['pytest_exit_code']==audit['planned_exit_code']==1 and audit['primary_type']is None
assert audit['denied']==audit['loaded_backend']==audit['cleanup_types']==[]and audit['numpy_imported']is False
assert audit['collected_nodes']==config['selected_nodes']and len(audit['collected_nodes'])==59
assert audit['runtime_sources_before']==audit['runtime_sources_after']==config['runtime_source_expectations']and len(audit['runtime_sources_before'])==403
xml=ET.fromstring(read(ROOT/'tests.xml'));cases=list(xml.iter('testcase'));bad={c.attrib['name'][len('test_device_integration['):-1]for c in cases if c.find('failure')is not None}
assert len(cases)==59 and len(bad)==16 and not list(xml.iter('error'))and not list(xml.iter('skipped'))
assert [dict(classname=c.attrib['classname'],name=c.attrib['name'])for c in cases]==config['selected_xml_cases']
read(ROOT/'tests.log');plan=config['artifact_plan'];base=ROOT/'pytest-owned';children={};fixtures={};aliases={};regular_count=0;total_bytes=0
for p in sorted(base.iterdir()):
    info=p.lstat()
    if stat.S_ISLNK(info.st_mode):
        target=os.readlink(p);lexical=Path(target)
        assert p.name.startswith('test_device_integration_')and p.name.endswith('current')and 0<len(target)<=4096 and str(lexical)==target and '\\'not in target and '..'not in lexical.parts and '.'not in lexical.parts
        if not lexical.is_absolute():assert lexical.name==target and '/'not in target
        named=lexical if lexical.is_absolute()else base/lexical
        assert named.parent==base and named!=p and stat.S_ISDIR(named.lstat().st_mode)
        prefix=p.name[:-len('current')];assert named.name.startswith(prefix)and named.name[len(prefix):].isdigit()
        assert state(info)==state(p.lstat())and os.readlink(p)==target
        aliases[p.name]=dict(target=target,followed=False);continue
    assert stat.S_ISDIR(info.st_mode)
    parent=doc(p/'parent-terminal.json');terminal=doc(p/'child-terminal.json');log=read(p/'child.log')
    case=parent['case'];assert case in config['selected_cases']and case not in children and terminal['case']==case
    pid=parent['pid'];assert type(pid)is int and pid>0 and terminal['pid']==terminal['pgid']==parent['owned_session_pgid']==pid
    assert parent['observed_pgid']is None or parent['observed_pgid']==pid
    assert parent['starttime_ticks']is None or parent['starttime_ticks']==terminal['starttime_ticks']
    assert type(terminal['starttime_ticks'])is int and terminal['starttime_ticks']>0
    assert all(parent[n]is True for n in('direct_wait_completed','reaped','group_absent'))and parent['primary_type']is None
    names={'parent-terminal.json','child-terminal.json','child.log'}
    if case in bad:
        assert parent['exit_code']==70 and parent['child_report_read']is False and terminal['normal_return']is False and terminal['status']=='entered'
        assert not os.path.lexists(p/'result.json')
    else:
        result_raw=read(p/'result.json');result=json.loads(result_raw);names.add('result.json')
        assert parent['exit_code']==0 and parent['child_report_read']is True and terminal['normal_return']is True and terminal['status']=='normal_return'
        assert result['case']==case and result['pid']==result['pgid']==pid and result['starttime_ticks']==terminal['starttime_ticks']and result['passed']is True
        assert parent['result_sha256']==terminal['result_sha256']==sha(result_raw)and parent['child_terminal_sha256']==pins[str(p/'child-terminal.json')]['sha256']
        assert result['backend_denials']==terminal['backend_denials']==[]
        assert all(result[n]is True for n in('definition_only_entry','original_main_restored','temporary_names_removed','profile_restored','declared_exports_restored','source_unchanged'))
        assert all(result[n]is False for n in('genuine_cli_bootstrap','model_key_ciphertext_body_reads','native_backend_imported','capacity_measured','actual_execution_authorized'))
    gone(pid);observed={};directories=[];seen_records=set()
    def walk(directory,prefix=''):
        for item in directory.iterdir():
            s=item.lstat();relative=prefix+item.name
            assert s.st_uid==os.geteuid()and not stat.S_ISLNK(s.st_mode)
            if stat.S_ISDIR(s.st_mode):directories.append(relative);walk(item,relative+'/')
            elif relative in names:
                assert prefix==''and stat.S_ISREG(s.st_mode);seen_records.add(relative)
            else:
                assert relative in plan['fixture_files'][case]and stat.S_ISREG(s.st_mode)
                observed[relative]=dict(bytes=s.st_size,device=s.st_dev,inode=s.st_ino,mode=stat.S_IMODE(s.st_mode))
    walk(p);assert seen_records==names
    if case not in bad:
        declared=parent['fixture_metadata'];assert declared['body_read']is False and declared['body_copied']is False
        assert set(observed)==set(plan['fixture_files'][case])==set(declared['files'])and sorted(directories)==declared['directories']
        for name,row in observed.items():assert all(declared['files'][name][key]==value for key,value in row.items())
    for name,row in observed.items():fixtures[(p/name).relative_to(ROOT).as_posix()]=dict(**row,body_read=False)
    child_pins={name:pins[str(p/name)]for name in names};regular_count+=len(names);total_bytes+=sum(v['bytes']for v in child_pins.values());assert total_bytes<=plan['total_record_limit_bytes']
    children[case]=dict(pid=pid,pgid=pid,starttime_ticks=terminal['starttime_ticks'],exit_code=parent['exit_code'],original_wait_complete=True,pid_group_absent=True,
        passed=case not in bad,records=child_pins,fixture_metadata=observed,directories=sorted(directories),failure_tail=None if case not in bad else log.decode(errors='replace').splitlines()[-3:])
assert len(children)==len({r['pid']for r in children.values()})==59 and set(children)==set(config['selected_cases'])
assert regular_count==220 and len(fixtures)==6 and not{row['pid']for row in children.values()}&{901567,901568}
gone(901567);gone(901568)
assert inventory(SOURCE)==inventory(ROOT/'execution')==expected
assert not os.path.lexists(ROOT/'report.json')and not os.path.lexists(ROOT/'child-artifact-review.json')and not list((ROOT/'tmp').iterdir())
result=dict(schema='full28-focused459-root-failed-terminal-v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),tool_terminal=tool,
    source_count=459,runtime_count=403,current_execution_archive_exact=True,tests=59,passed=43,failed=16,errors=0,skipped=0,
    pytest_seconds=float(next(xml.iter('testsuite')).attrib['time']),collector_failure='success-only inspector requested an absent result.json after child failure; no aggregate pass report exists',
    observed_failure_groups={'engine_parameter_owner_reference':3,'audit_sticky_denial':6,'run_numpy_import_denial':6,'evidence_dirfd_write_denial':1},
    group_labels_are_observed_locations_not_root_causes=True,children=children,regular_child_records=220,fixture_files_metadata_only=6,fixtures=fixtures,aliases=aliases,
    all61_original_pids_groups_absent=True,fixture_bodies_read=False,fixture_bodies_copied=False,aliases_followed=False,public_pins=pins,
    scope='failed recording test suite only; no genuine backend/capacity/model/key/ciphertext body or HE proof; frozen source and failure history preserved')
destination=HERE/'failed459-terminal-review.json';assert not destination.exists()
payload=(json.dumps(result,sort_keys=True,indent=2)+'\n').encode();destination.write_bytes(payload)
print(json.dumps({'path':str(destination),'sha256':sha(payload),'passed':43,'failed':16,'records':regular_count,'fixture_metadata_files':len(fixtures),'all61_absent':True}))
