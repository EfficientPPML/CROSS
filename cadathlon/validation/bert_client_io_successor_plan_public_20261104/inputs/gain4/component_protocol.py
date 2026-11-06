"""Public source, terminal and file bindings for one isolated routing proof.

No key payload, ciphertext payload, native model or backend is opened by a
terminal gate. A completed externally pinned parent receipt authorizes the
next role; worker reports alone never do so.
"""
from pathlib import Path
import hashlib
import io
import json
import os
import stat
import sys
import tarfile
import component_contract as contract
import component_key_gate as key_gate
from component_transport import _CloseState, _stream

HERE=Path(__file__).resolve().parent
KEY_REPORT='022cbdf687273e765668184f32cb7cd501d371c4220916e1c8f02574cf3a7f08'
STAGES=('fixture','server','validate')
MANIFEST_SCHEMA='cross-normalized-gain4-artifacts-v1'
TERMINAL_SCHEMA='cross-normalized-gain4-stage-terminal-v1'
PREDECESSOR_REPORT='7931359e105b8bc6004883e9f4b74405fd23357997722b71e2af833cf35c8059'
RETIRED_PARENT_SOURCES=('public_plan/inputs/full_key_plan.json',)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def require_ready():
    value=key_gate.parse(key_gate.read(HERE/'execution_authorization.json'))
    if type(value)is not dict or set(value)!={'ready'} or value['ready']is not True:
        raise ValueError('actual phase remains pending exact source-bound review')


def relative(name):
    if (type(name)is not str or not name or Path(name).is_absolute() or '..' in Path(name).parts
            or Path(name).as_posix()!=name or name.startswith('./')):
        raise ValueError('canonical relative source path required')
    return name


def sources():
    inventory=key_gate.parse(key_gate.read(HERE/'source_inventory.json',1<<20))
    if type(inventory)is not list or not inventory or len(inventory)!=len(set(inventory)):
        raise ValueError('fixed unique execution source inventory required')
    return {relative(n):(HERE/n).read_bytes() for n in inventory}


def hashes(raw):return {n:sha(value) for n,value in raw.items()}


def check_predecessor():
    raw=key_gate.read(HERE/'routing_predecessor/report.json')
    if sha(raw)!=PREDECESSOR_REPORT:raise ValueError('exact disabled predecessor proof required')
    report=key_gate.parse(raw);before=report['source_hashes_start']
    if (report['passed']is not True or report['execute']is not False or len(before)!=147
            or before!=report['source_hashes_end']):raise ValueError('completed unchanged disabled predecessor required')
    archive=key_gate.read(HERE/'routing_predecessor/sources.tar.gz',16<<20)
    if sha(archive)!=report['source_archive_sha256']:raise ValueError('disabled predecessor archive differs')
    key_gate.verify_source_archive(archive,before)
    overlays=key_gate.parse(key_gate.read(HERE/'gain4_component_overlays.json'))
    if type(overlays)is not dict or not overlays or not set(overlays)<=set(before):
        raise ValueError('explicit predecessor source overlays required')
    for name,old in before.items():
        if name in RETIRED_PARENT_SOURCES:
            if (HERE/name).exists():raise ValueError('retired duplicate key plan must remain outside execution closure')
            continue
        expected=overlays.get(name,old);key_gate.sha_field(expected)
        if sha((HERE/relative(name)).read_bytes())!=expected:raise ValueError('unreviewed predecessor source drift: '+name)
    return dict(report_sha256=PREDECESSOR_REPORT,archive_sha256=sha(archive),original_sources=147,overlays=sorted(overlays))


def check_sources(expected):
    if type(expected)is not dict or not expected or hashes(sources())!=expected:
        raise ValueError('exact reviewed execution source closure differs')
    check_predecessor()
    overlays=key_gate.parse(key_gate.read(HERE/'gain4_component_overlays.json'))
    for name,row in json.loads((HERE/'runtime_source_origins.json').read_bytes()).items():
        if sha((HERE/relative(name)).read_bytes())!=overlays.get(name,row['sha256']):
            raise ValueError('copied runtime changed: '+name)
    for name,row in json.loads((HERE/'input_sources.json').read_bytes()).items():
        if sha((HERE/relative(name)).read_bytes())!=overlays.get(name,row['sha256']):
            raise ValueError('frozen owner/input changed: '+name)
    profile=(HERE/'runtime/stage/server_inputs/p60_profile_and_refresh.json').read_bytes()
    plan=(HERE/'runtime/stage/server_inputs/twelve_layer_plan.json').read_bytes()
    if sha(profile)!=contract.PROFILE_SHA or sha(plan)!=contract.KEY_PLAN_SHA:
        raise ValueError('fixed public profile/key plan required')
    check_gain4_proof()
    contract.numerical_plan()
    return profile,plan


def check_gain4_proof():
    origin=key_gate.parse(key_gate.read(HERE/'gain4_component_origin.json'))
    raw=key_gate.read(HERE/'gain4_ideal/report.json')
    if sha(raw)!=origin['gain4_report_sha256']:raise ValueError('frozen gain4 ideal report differs')
    report=key_gate.parse(raw);pins=report['source_hashes_before']
    if len(pins)!=131 or pins!=report['source_hashes_after']or pins!=report['execution_source_hashes_after']or any(report.get(n)is not True for n in ('passed','complete','cleanup_completed','terminal_scope_completed','termination_scope_completed')):raise ValueError('completed gain4 ideal proof required')
    archive=key_gate.read(HERE/'gain4_ideal/sources.tar.gz',16<<20)
    if sha(archive)!=origin['gain4_archive_sha256']or sha(archive)!=report['source_archive_sha256']:raise ValueError('gain4 source archive differs')
    key_gate.verify_source_archive(archive,pins)
    for name,pin in origin['new_gain4_files'].items():
        if sha((HERE/relative(name)).read_bytes())!=pin:raise ValueError('gain4 measured runtime/fixture changed: '+name)
    return True


def imported_sources(expected):
    names={}
    for directory in (HERE/'common',HERE/'runtime/stage',HERE/'runtime/jaxite_word',HERE):
        names.update({p.stem:p for p in directory.glob('*.py')})
    names['jaxite_word']=HERE/'runtime/jaxite_word/__init__.py'
    found={}
    for name,module in tuple(sys.modules.items()):
        filename=getattr(module,'__file__',None)
        if not filename:continue
        path=Path(filename).resolve();wanted=names.get(name,names.get(name.rsplit('.',1)[-1]) if name.startswith('jaxite_word.') else None)
        if wanted is not None and path!=wanted:
            raise ValueError('mixed runtime import: '+name)
        if HERE in path.parents and path.suffix=='.py':
            n=path.relative_to(HERE).as_posix();value=sha(path.read_bytes())
            if expected.get(n)!=value:raise ValueError('import outside reviewed closure: '+name)
            found[name]=dict(path=n,sha256=value)
    return found


def close(fd, state=None):
    (_CloseState() if state is None else state).close(fd)


def write(path,value,*,replace=True):
    """Publish canonical metadata, with explicit FD ownership on all failures."""
    path=Path(path);raw=contract.canonical(value);state=_CloseState()
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        initial=os.fstat(fd);temporary=path.name+'.tmp' if replace else path.name
        item=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=fd)
        try:
            with _stream(item,'wb',state) as stream:
                stream.write(raw);stream.flush();os.fsync(item)
        finally:close(item,state)
        current=path.parent.lstat()
        if not stat.S_ISDIR(current.st_mode) or (current.st_dev,current.st_ino)!=(initial.st_dev,initial.st_ino):
            raise ValueError('publication parent changed; artifacts retained')
        if replace:os.replace(temporary,path.name,src_dir_fd=fd,dst_dir_fd=fd)
        os.fsync(fd)
    finally:close(fd,state)


def identity(path):
    value=Path(path).lstat()
    if not stat.S_ISDIR(value.st_mode) or stat.S_IMODE(value.st_mode)!=0o700:
        raise ValueError('original private-mode proof IO directory required')
    return [value.st_dev,value.st_ino]


def check_identity(path,expected):
    if (type(expected)is not list or len(expected)!=2 or any(type(v)is not int for v in expected)
            or identity(path)!=expected):raise ValueError('proof IO directory identity changed')


def validate_artifact_manifest(value,*,request,role,input_digest=None):
    contract.validate_contract(request)
    if role not in ('request','response'):raise ValueError('artifact role required')
    count=2 if role=='request' else 54
    if (type(value)is not dict or set(value)!= {'schema','contract','role','input_digest','tiles'}
            or value['schema']!=MANIFEST_SCHEMA or contract.canonical(value['contract'])!=contract.canonical(request)
            or value['role']!=role or type(value['tiles'])is not list or len(value['tiles'])!=count):
        raise ValueError('exact complete ordered artifact manifest required')
    key_gate.sha_field(value['input_digest'])
    if role=='response' and value['input_digest']!=input_digest:
        raise ValueError('response belongs to different encrypted fixtures')
    for i,row in enumerate(value['tiles']):
        if (type(row)is not dict or set(row)!= {'index','manifest_sha256'} or type(row['index'])is not int or row['index']!=i):
            raise ValueError('ordered complete unique artifact identities required')
        key_gate.sha_field(row['manifest_sha256'])
    return value


def readback(workspace,request,*,role,input_digest=None):
    """Read every authenticated payload; callers perform terminal gates first."""
    from component_transport import _validate_ciphertext,manifest_digest,request_digest
    if role not in ('request','response'):raise ValueError('artifact role required')
    folder=Path(workspace)/('inputs' if role=='request' else 'outputs')
    count=2 if role=='request' else 54
    state=_CloseState()
    fd=os.open(folder,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        before=os.fstat(fd)
        if set(os.listdir(fd))!={'manifest.json'}|{f'tile{i}' for i in range(count)}:
            raise ValueError('exact complete ciphertext directory inventory required')
        from component_transport import _read_regular
        raw=_read_regular('manifest.json',1<<20,directory_fd=fd,_close_state=state)
        value=validate_artifact_manifest(key_gate.parse(raw),request=request,role=role,input_digest=input_digest)
        if raw!=contract.canonical(value):raise ValueError('canonical artifact manifest required')
        manifests=[];plan=contract.numerical_plan()
        for i,row in enumerate(value['tiles']):
            scale=None if role=='request' else plan['checkpoints'][i]['scale']
            payload,actual,binding,manifest=_validate_ciphertext(folder/f'tile{i}',contract=request,role=role,
                tile_index=i,expected_scale=scale,input_digest=None if role=='request' else input_digest,state=state)
            del payload
            if manifest_digest(manifest)!=row['manifest_sha256']:raise ValueError('ciphertext manifest changed')
            manifests.append(manifest)
        if role=='request' and request_digest(manifests,request)!=value['input_digest']:
            raise ValueError('complete ordered fixture digest differs')
        now=folder.lstat()
        if not stat.S_ISDIR(now.st_mode) or (now.st_dev,now.st_ino)!=(before.st_dev,before.st_ino):
            raise ValueError('artifact directory replaced during readback')
        return dict(manifest=value,manifest_sha256=sha(raw),tile_count=count,input_digest=value['input_digest'])
    finally:close(fd,state)


def terminal_gate(evidence,expected_sha,*,stage,expected_sources,workspace,workspace_identity,request):
    """Authenticate a prior completed parent receipt without any private read."""
    key_gate.sha_field(expected_sha)
    if stage not in STAGES:raise ValueError('known prior phase required')
    evidence=Path(evidence);raw=key_gate.read(evidence/'report.json')
    if sha(raw)!=expected_sha:raise ValueError('externally pinned terminal report differs')
    r=key_gate.parse(raw)
    if (type(r)is not dict or r.get('schema')!=TERMINAL_SCHEMA or r.get('stage')!=stage
            or any(r.get(k)is not True for k in ('passed','execute','worker_reaped','worker_group_empty_verified','all_direct_children_reaped',
                'all_owned_groups_empty','process_cleanup_completed','source_files_unchanged','source_archive_verified','terminal_scope_completed','termination_scope_completed'))
            or type(r.get('worker_exit_code'))is not int or r['worker_exit_code']!=0 or r.get('error') or r.get('cleanup_errors')
            or r.get('workspace')!=str(workspace) or r.get('workspace_identity')!=workspace_identity
            or r.get('source_hashes_start')!=expected_sources or r.get('source_hashes_end')!=expected_sources
            or contract.canonical(r.get('request_contract'))!=contract.canonical(request)):
        raise ValueError('completed unchanged terminal phase receipt required')
    check_identity(workspace,workspace_identity)
    archive=key_gate.read(evidence/'sources.tar.gz',16<<20)
    if sha(archive)!=key_gate.sha_field(r.get('source_archive_sha256')):raise ValueError('terminal source archive differs')
    key_gate.verify_source_archive(archive,expected_sources)
    worker_raw=key_gate.read(evidence/'worker/worker_report.json')
    if sha(worker_raw)!=key_gate.sha_field(r.get('worker_report_sha256')):raise ValueError('terminal worker linkage differs')
    w=key_gate.parse(worker_raw)
    if (type(w)is not dict or w.get('stage')!=stage or
            any(w.get(k)is not True for k in ('passed','phase_complete','loader_cleanup_completed','source_files_unchanged','terminal_scope_completed','termination_scope_completed'))
            or w.get('error') or w.get('forbidden_attempts')!=[] or w.get('source_hashes_start')!=expected_sources
            or w.get('source_hashes_end')!=expected_sources or w.get('workspace_identity')!=workspace_identity
            or contract.canonical(w.get('request_contract'))!=contract.canonical(request)
            or w.get('readback')!=r.get('readback')):
        raise ValueError('completed phase worker proof differs')
    readback=w.get('readback')
    if stage in ('fixture','server'):
        role='request' if stage=='fixture' else 'response'
        if type(readback)is not dict or readback.get('tile_count')!=(2 if role=='request' else 54):
            raise ValueError('all phase artifacts must complete before next role')
        validate_artifact_manifest(readback.get('manifest'),request=request,role=role,input_digest=readback.get('input_digest'))
        if readback.get('manifest_sha256')!=sha(contract.canonical(readback['manifest'])):
            raise ValueError('phase artifact manifest digest differs')
    return r


def authenticate_keys(evidence,workspace,*,profile,plan):
    keys=key_gate.authenticate(evidence,KEY_REPORT,workspace,profile_bytes=profile,plan_bytes=plan)
    if keys['context_id']!=contract.CONTEXT_ID:raise ValueError('existing key context differs')
    return keys
