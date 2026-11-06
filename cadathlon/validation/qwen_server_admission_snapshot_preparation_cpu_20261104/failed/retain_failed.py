"""Bounded public failed-run readback. Does not import or rerun any source."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tarfile
import xml.etree.ElementTree as ET

ROOT=Path('/tmp/qwen-server-admission-snapshot-focused-proof-20261104/preparation-v1')
OUT=Path('/tmp/qwen-snapshot-failed-run-retention-20261104')
def sha(b):return hashlib.sha256(b).hexdigest()
def encoded(v):return(json.dumps(v,sort_keys=True,indent=2)+'\n').encode()
def compact(v):return(json.dumps(v,sort_keys=True,separators=(',',':'))+'\n').encode()
def emit(name,value):(OUT/name).write_bytes(encoded(value))
def state(s):return dict(dev=s.st_dev,inode=s.st_ino,mode=s.st_mode,uid=s.st_uid,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def regular(path,cap):
    before=path.lstat();assert stat.S_ISREG(before.st_mode)and before.st_size<=cap
    fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
    try:
        assert state(os.fstat(fd))==state(before)
        chunks=[];total=0
        while True:
            data=os.read(fd,min(65536,cap+1-total))
            if not data:break
            chunks.append(data);total+=len(data);assert total<=cap
        assert state(os.fstat(fd))==state(before)
    finally:os.close(fd)
    assert state(path.lstat())==state(before)
    return b''.join(chunks)
def inventory(root):
    paths=sorted(root.rglob('*'));assert not any(p.is_symlink()for p in paths)
    return{p.relative_to(root).as_posix():sha(regular(p,1<<24))for p in paths if p.is_file()}

config=json.loads(regular(ROOT/'run_config.json',1<<20));source=Path(config['source_root'])
manifest=regular(ROOT/'sources-before.json',1<<20);expected=json.loads(manifest)
assert len(expected)==37 and compact(expected)==manifest
assert inventory(source)==inventory(ROOT/'execution')==expected
for name in('sources-after.json','execution-after.json'):assert regular(ROOT/name,1<<20)==manifest
with tarfile.open(ROOT/'sources.tar.gz','r:gz')as archive:
    members=archive.getmembers();assert len(members)==len({m.name for m in members})==37
    assert all(m.isfile()and not Path(m.name).is_absolute()and'..'not in Path(m.name).parts for m in members)
    assert{m.name:sha(archive.extractfile(m).read())for m in members}==expected
assert sha(manifest)==config['source_manifest_sha256']
assert sha(regular(ROOT/'sources.tar.gz',1<<24))==config['source_archive_sha256']

whitelist=('README.md','before_executed_driver.py','before_worker_driver.py','driver_provenance.json',
    'executed_driver.py','executed_driver.py.patch','handoff.json','launch-proposal.json','run_config.json',
    'source-handoff.json','sources-before.json','sources-after.json','execution-after.json','sources.tar.gz',
    'worker_driver.py','worker_driver.py.patch','live-observation.json','tests.log','tests.xml','worker-audit.json')
copied={}
for name in whitelist:
    raw=regular(ROOT/name,1<<24);destination=OUT/'evidence'/name;destination.parent.mkdir(exist_ok=True)
    destination.write_bytes(raw);copied[name]=sha(raw)
handoff=json.loads((OUT/'evidence/handoff.json').read_bytes())
assert all(copied[n]==v for n,v in handoff['files'].items())
xml=ET.fromstring((OUT/'evidence/tests.xml').read_bytes());cases=list(xml.iter('testcase'))
counts={name:len(list(xml.iter(tag)))for name,tag in(('tests','testcase'),('failures','failure'),('errors','error'),('skipped','skipped'))}
assert counts==dict(tests=18,failures=18,errors=0,skipped=0)
assert[dict(classname=c.attrib['classname'],name=c.attrib['name'])for c in cases]==config['selected_xml_cases']
audit=json.loads((OUT/'evidence/worker-audit.json').read_bytes())
assert audit['collected_nodes']==config['selected_nodes']
assert audit['denied']==audit['loaded_backend']==[]and audit['numpy_imported']is False
assert audit['runtime_sources_before']==audit['runtime_sources_after']==audit['runtime_module_origins']=={}
assert type(audit['pytest_exit_code'])is int and audit['pytest_exit_code']==1
assert audit['affinity']==[112,113,114,115]

owned=ROOT/'pytest-owned';owned_state=owned.lstat()
assert stat.S_ISDIR(owned_state.st_mode)and owned_state.st_uid==os.geteuid()
child_rows=[];names=set();total=0
for case in cases:
    failure=''.join(case.find('failure').itertext())
    match=re.search(r"tmp_path = PosixPath\('([^']+)'\)",failure);assert match
    path=Path(match.group(1));assert path.parent==owned and path.name not in names
    names.add(path.name);s=path.lstat();assert stat.S_ISDIR(s.st_mode)and s.st_uid==os.geteuid()
    assert os.listdir(path)==['child.log']
    log=regular(path/'child.log',1<<16);total+=len(log);assert total<=1<<20
    text=log.decode();assert 'line 63, in main' in text and 'PublicContract' in text and 'ValueError: public source state changed' in text
    assert '>       assert code==0' in failure and 'E       assert 1 == 0' in failure
    target=OUT/'child-logs'/path.name/'child.log';target.parent.mkdir(parents=True);target.write_bytes(log)
    child_rows.append(dict(case=case.attrib['name'],owned_directory=path.name,log_sha256=sha(log),
        observed_exit_code_in_pytest_failure=1,constructor_error='ValueError: public source state changed',
        failure_before_guard=True,result_present=False,terminal_present=False,
        pid=None,pgid=None,independent_child_retirement_observed=False,
        wait_scope='The selected test calls wait before the failing code==0 assertion; this review did not own child handles.'))
aliases=[]
entries=os.listdir(owned);assert len(entries)<=64
for name in sorted(entries):
    path=owned/name;s=path.lstat()
    if stat.S_ISDIR(s.st_mode):assert name in names;continue
    assert stat.S_ISLNK(s.st_mode)and name.startswith('test_snapshot_guard_')and name.endswith('current')
    assert s.st_uid==os.geteuid()
    target_raw=os.readlink(path);assert len(target_raw)<=4096
    lexical=Path(target_raw)if Path(target_raw).is_absolute()else owned/target_raw
    assert '..'not in lexical.parts and lexical.parent==owned and lexical.name in names
    assert str(lexical)==str(owned/lexical.name)
    # Target was already inspected through its own explicit XML-derived name.
    # Never stat/open/traverse the symlink itself or resolve through it.
    assert state(path.lstat())==state(s)and os.readlink(path)==target_raw
    aliases.append(dict(name=name,target=target_raw,link_state=state(s),followed=False))
assert len(names)==18 and len(aliases)==17
assert state(owned.lstat())==state(owned_state)
tmp=ROOT/'tmp';assert stat.S_ISDIR(tmp.lstat().st_mode)and os.listdir(tmp)==[]
missing=('report.json','owned-pytest-metadata.json','test-child-terminals.json')
assert all(not os.path.lexists(ROOT/n)for n in missing)
emit('child-failures.json',child_rows);emit('owned-aliases.json',aliases)

live=json.loads((OUT/'evidence/live-observation.json').read_bytes());absence={}
for role in('outer','worker'):
    pid=live[role+'_pid'];pgid=live[role+'_pgid'];assert type(pid)is int and type(pgid)is int and pid==pgid
    if role=='worker':assert audit['pid']==pid and audit['pgid']==pgid
    for field,value,operation in(('pid',pid,os.kill),('pgid',pgid,os.killpg)):
        try:operation(value,0)
        except ProcessLookupError:absent=True
        else:absent=False
        absence[role+'_'+field]={'value':value,'absent_now':absent}
assert all(v['absent_now']for v in absence.values())
emit('process-absence.json',dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),observations=absence,
    identities_from='source-generated live-observation.json cross-bound with worker-audit.json',
    reviewer_had_live_observation=False,reviewer_owned_wait=False,child_identities_not_recorded=True))
assert inventory(source)==inventory(ROOT/'execution')==expected
assert all(sha(regular(ROOT/n,1<<24))==v for n,v in copied.items())
emit('review.json',dict(schema='qwen-snapshot-failed-terminal-security-review-v1',evidence_review_passed=True,run_passed=False,
    source_count=37,sources_current_execution_archive_before_after_equal=True,source_manifest_sha256=sha(manifest),
    source_archive_sha256=copied['sources.tar.gz'],xml=counts,ordered_collection_and_xml_equal=True,
    backend_denials=[],loaded_backend=[],numpy_imported=False,child_logs_retained=18,aliases_recorded_without_following=17,
    all_failures='PublicContract.__post_init__ rejects synthetic marker bytes at snapshot_child.py line63, before snapshot capture or guard exercise',
    absent_output_names=list(missing),
    outer_secondary_failure='Root reported inspect_owned missing18 child terminal records; no final report was emitted. Selected collector source requires those records.',
    root_reported_owner={'session':97824,'launch_tool':'32b53a','terminal_tool':'9b71cb','exit_code':1,'reaped':True},
    reviewer_direct_owner=False,reviewer_independent_live=False,observed_outer_worker_absent_now=True,
    child_pid_pgid_receipts_available=False,child_group_absence_not_proved=True,
    limits=['18 failures provide no snapshot guard success/failure coverage.',
        'Child logs/pytest code1 do not fabricate absent PID/PGID or restoration/terminal receipts.',
        'Worker audit is the completed worker report; missing child terminal audits are not inferred.',
        'Original source-only harness review remains historical and is not rewritten as runtime validation.',
        'No candidate import, test rerun, new probe, capacity access or borrowed body read by reviewer.']))
readme='''# Failed snapshot37/18 run — bounded public retention

All18 tests failed before guard exercise: PublicContract construction rejected
the synthetic marker bytes. Sources37/current/execution/archive/start/end are
unchanged. The worker recorded exact18 collection, exit1, no backend attempts
or NumPy modules. This is a failed fixture run, not guard validation.

Only the20 explicitly listed original evidence/source files and18 XML-derived
owned child.log paths are copied. Seventeen pytest aliases are recorded as link
metadata without following them; two validator cases share a truncated prefix.
No result.json or child-terminal.json exists. Child PIDs/PGIDs and individual
group retirement therefore remain unobserved. No identifiers were invented.
The root-reported owner session97824 terminal9b71cb was exit1/reaped. This
review independently observed the recorded outer/worker PID+PGID absent after
exit, and did not own their launch/wait or observe them live. The collector's
secondary missing-terminal failure left no final report.

Source/header/metadata review only; no rerun, candidate import, device/capacity
probe or private/body read. Frozen original evidence is unchanged. Root/author
owner receipts, if supplied later, remain separate evidence.
'''
(OUT/'README.md').write_text(readme)
index={p.relative_to(OUT).as_posix():sha(p.read_bytes())for p in sorted(OUT.rglob('*'))if p.is_file()and p.name!='files.json'}
emit('files.json',index)
print(json.dumps({'review':str(OUT/'review.json'),'review_sha256':sha((OUT/'review.json').read_bytes()),
    'index_sha256':sha((OUT/'files.json').read_bytes()),'indexed_files':len(index),'xml':counts,'process_absence':absence},sort_keys=True))
