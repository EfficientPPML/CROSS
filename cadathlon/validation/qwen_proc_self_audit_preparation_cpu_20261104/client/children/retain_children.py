"""Copy only the root-authorized96 terminal child artifacts, never source trees."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat

OUT=Path(__file__).parent
PLAN=Path('/tmp/qwen-client-kernel-child-artifact-plan-20261104/selection-plan.json')
PROOF=Path('/tmp/qwen-client-kernel-focused-proof-20261104/preparation-v2')
TEMP=Path('/tmp/pytest-of-jianming_gatech')
def h(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def state(info):return dict(device=info.st_dev,inode=info.st_ino,size=info.st_size,mtime_ns=info.st_mtime_ns,ctime_ns=info.st_ctime_ns,mode=stat.S_IMODE(info.st_mode),uid=info.st_uid)
def exact_dir(path):
    info=path.lstat()
    if not stat.S_ISDIR(info.st_mode)or info.st_uid!=os.geteuid():raise ValueError('original owned ordinary artifact directory required')
    return state(info)
def bounded(path,maximum):
    parent=exact_dir(path.parent);before=path.lstat()
    if not stat.S_ISREG(before.st_mode)or before.st_uid!=os.geteuid()or not 0<=before.st_size<=maximum:
        raise ValueError('bounded original regular artifact required')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC)
    try:
        opened=os.fstat(fd)
        if state(opened)!=state(before):raise ValueError('original artifact identity differs')
        chunks=[];size=0
        while size<=maximum:
            block=os.read(fd,min(65536,maximum+1-size))
            if not block:break
            chunks.append(block);size+=len(block)
        raw=b''.join(chunks)
        if state(os.fstat(fd))!=state(before)or state(path.lstat())!=state(before)or len(raw)!=before.st_size:
            raise ValueError('artifact changed during bounded read')
        after=exact_dir(path.parent)
        if (after['device'],after['inode'],after['uid'])!=(parent['device'],parent['inode'],parent['uid']):
            raise ValueError('artifact parent replaced')
        return raw,state(before)
    finally:os.close(fd)
def load(raw):
    def pairs(items):
        out={}
        for key,value in items:
            if key in out:raise ValueError('duplicate artifact JSON key')
            out[key]=value
        return out
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite artifact JSON')))
def absent(pid):
    try:os.kill(pid,0)
    except ProcessLookupError:return True
    return False
def group_absent(pgid):
    try:os.killpg(pgid,0)
    except ProcessLookupError:return True
    return False

plan_raw=PLAN.read_bytes();assert h(plan_raw)=='0f86efd83313c60696f9622af265219bccdd755157cc122f8d0b0264e415f650'
plan=load(plan_raw);cases=plan['cases'];assert len(cases)==24
report_raw=(PROOF/'report.json').read_bytes();report=load(report_raw)
assert report['passed']is True and report['tests']==dict(tests=24,failures=0,errors=0,skipped=0)
assert report['selected_nodes']==[r['node']for r in cases]
assert report['source_manifest_sha256']=='208049d311bc26bb8db8ce71c610c6c484dfbec064e1b26bbbe5f51483a1ec40'
assert report['source_archive_sha256']=='9ed75e3c56eaf743a34f591d6978af69394de93a8ea806a26bb5fb9b98d4a617'
live_raw=(PROOF/'live-observation.json').read_bytes();live=load(live_raw)
temp_state=exact_dir(TEMP)
# Closed one-level discovery only; never follow current links or enter sources.
candidates=[];first=cases[0]
with os.scandir(TEMP)as entries:
    bases=sorted((Path(item.path)for item in entries if re.fullmatch(r'pytest-[0-9]+',item.name)
        and item.is_dir(follow_symlinks=False)),key=lambda p:int(p.name.split('-')[1]))
for base in bases:
    directory=base/first['relative_directory']
    try:
        exact_dir(base);exact_dir(directory)
        raw,_=bounded(directory/'receipt.json',65536)
    except FileNotFoundError:continue
    candidate=load(raw)
    if (candidate.get('schema')=='qwen-client-own-proc-probe-v1'
            and candidate.get('case')==first['case']and candidate.get('baseline')is False
            and candidate.get('source_manifest_sha256')==first['source_manifest_sha256']):
        candidates.append(base)
if len(candidates)!=1:raise ValueError('one unique authorized client24 artifact base required')
base=candidates[0];base_state=exact_dir(base)
destination=OUT/'child-artifacts'
if destination.exists():raise FileExistsError('fresh artifact retention required')
origins={};manifest={};summary=[];retained={};pids=set()
for row in cases:
    directory=base/row['relative_directory'];directory_state=exact_dir(directory)
    bundle={};states={}
    for name,limit in row['exact_files'].items():
        raw,observed=bounded(directory/name,limit['maximum_bytes'])
        if not raw and not limit['allow_empty']:raise ValueError('nonempty JSON receipt required')
        bundle[name]=raw;states[name]=observed
    receipt=load(bundle['receipt.json']);owner=load(bundle['owner.json'])
    assert receipt['schema']=='qwen-client-own-proc-probe-v1'
    assert receipt['case']==row['case']and receipt['baseline']is row['baseline']
    assert receipt['source_manifest_sha256']==owner['source_manifest_sha256']==row['source_manifest_sha256']
    assert receipt['passed']is receipt['original_open_restored']is receipt['audit_inactive']is True
    assert receipt['forbidden_loaded_modules']==[]
    assert receipt['recording_admission']is True and receipt['real_role_admission']is False
    assert receipt['own_proc_metadata_only']is True and receipt['actual_capacity_check']is False
    assert receipt['backend_imported']is False and receipt['body_content_recorded']is False
    pid=receipt['pid'];pgid=receipt['pgid']
    assert type(pid)is int and pid>0 and type(pgid)is int and pgid==pid
    assert type(owner['pid'])is int and owner['pid']==pid and type(owner['pgid'])is int and owner['pgid']==pgid
    assert owner['pgid_observed_by_parent']is None or(type(owner['pgid_observed_by_parent'])is int and owner['pgid_observed_by_parent']==pgid)
    assert type(owner['returncode'])is int and owner['returncode']==0
    assert owner['waited']is owner['group_absent']is owner['synthetic_scope']is True
    assert owner['normal_source_readiness']is False and owner['receipt_sha256']==h(bundle['receipt.json'])
    assert pid not in pids and absent(pid)and group_absent(pgid);pids.add(pid)
    denied=row['baseline']or row['case'].split(':')[0]in('flag','spelling')
    assert receipt['expected_denial']is denied
    if denied:
        assert receipt['observation']['read_completed']is False and receipt['observation']['error_type']=='PermissionError'
        assert receipt['audit_denials']and receipt['cleanup_error_type']=='PermissionError'
        if row['baseline']:assert receipt['audit_denials']==['open alias']
    else:
        assert receipt['observation']=={'read_completed':True,'nonempty':True}
        assert receipt['audit_denials']==[]and receipt['cleanup_error_type']is None
    for name,raw in bundle.items():
        relative=row['destination_directory']+'/'+name
        retained[relative]=raw
        manifest[relative]=h(raw)
        origins[relative]=dict(original_path=str(directory/name),sha256=h(raw),bytes=len(raw),
            original_state=states[name],original_directory_state=directory_state,index=row['index'],node=row['node'])
    summary.append(dict(index=row['index'],node=row['node'],case=row['case'],baseline=row['baseline'],
        pid=pid,pgid=pgid,current_pid_absent=True,current_group_absent=True,
        source_manifest_sha256=row['source_manifest_sha256'],receipt_sha256=h(bundle['receipt.json']),
        owner_sha256=h(bundle['owner.json']),expected_denial=denied))
assert len(retained)==96 and len(summary)==len(pids)==24
destination.mkdir(mode=0o700)
for relative,raw in retained.items():
    target=OUT/relative;target.parent.mkdir(mode=0o700,exist_ok=True)
    with target.open('xb')as stream:stream.write(raw)
    os.chmod(target,0o600)
    assert h(target.read_bytes())==manifest[relative]
# Final original identity/hash readback touches only the same96 exact files.
for relative,row in origins.items():
    raw,observed=bounded(Path(row['original_path']),max(row['bytes'],1))
    assert h(raw)==row['sha256']and observed==row['original_state']
assert exact_dir(base)==base_state and exact_dir(TEMP)==temp_state
map_raw=canonical(manifest);origin_raw=canonical(origins)
(OUT/'files.json').write_bytes(map_raw);(OUT/'origins.json').write_bytes(origin_raw)
receipt=dict(schema='qwen-client-kernel-child-retention-v1',passed=True,
    source_plan_sha256=h(plan_raw),proof_report_sha256=h(report_raw),proof_live_receipt_sha256=h(live_raw),
    original_base=str(base),original_base_state=base_state,files=96,children=24,
    complete_case_owner_source_bindings=True,original_before_after_copy_equal=True,
    files_sha256=h(map_raw),origins_sha256=h(origin_raw),child_records=summary,
    outer_tool_terminal=dict(owner='root',session_id=68374,terminal_chunk='170b97',exit_code=0,reaped=True,
        authority='parent supplied; this retention script does not re-poll or own that session'),
    root_independent_live_observation=False,
    current_parent_pid_absent=absent(live['worker_pid']),current_parent_group_absent=group_absent(live['worker_pgid']),
    script_sha256=h(Path(__file__).read_bytes()),
    scope=dict(terminal_public_artifact_copy_only=True,tests_rerun=False,metadata_probe_rerun=False,
        source_tree_or_owned_subtree_traversal=False,aliases_followed=False,old464_source_read=False,
        private_model_key_CT_token_body_reads=False,backend_context=False,actual_role=False))
(OUT/'retention-review.json').write_bytes(canonical(receipt))
handoff=dict(schema='qwen-client-kernel-child-retention-handoff-v1',root=str(OUT),copied_files=96,
    copied_total_bytes=sum(len(v)for v in retained.values()),children=24,
    files_sha256=h(map_raw),origins_sha256=h(origin_raw),retention_review_sha256=h(canonical(receipt)),
    original_base=str(base),source_plan_sha256=h(plan_raw),complete=True,rerun=False)
(OUT/'handoff.json').write_bytes(canonical(handoff))
print(json.dumps(handoff,sort_keys=True))
