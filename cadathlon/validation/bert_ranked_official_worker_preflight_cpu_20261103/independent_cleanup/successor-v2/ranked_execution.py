"""Public fresh-attempt ownership and exact copied entrypoint bindings.

This is a preparation adapter, not a replacement official timing harness.
Existing retained fixture/reference/key workspaces never become owned here.
"""
from pathlib import Path
import hashlib,importlib.util,json,os,stat,sys,uuid
import key_gate
from ranked_case_gate import RankedCase
from ranked_stage_gate import case_binding
HERE=Path(__file__).resolve().parent
NAMES={1:'small',2:'medium',3:'large'}
SERVER_FIELDS={'size','batch_size','model_id','model_revision','public_keys','server_data','upload','download','server_report'}
QUARANTINE=[]


def outside(path,bases):
    path=Path(path).absolute();bases=tuple(Path(base).absolute() for base in bases)
    # Public producer paths are already absolute. Do not stat/resolve retained
    #private roots merely to check a new output. Resolve only the new candidate
    #after lexical containment rejects known protected descendants.
    if any(path==base or base in path.parents for base in bases):
        raise ValueError('new attempt/evidence must be outside retained prerequisites')
    resolved=path.resolve()
    if any(resolved==base or base in resolved.parents for base in bases):
        raise ValueError('new attempt/evidence must be outside retained prerequisites')
    return path


def protected(case):
    return (case.fixture_workspace.path,case.reference_workspace.path)


def initialize_attempt(root,case):
    """Called before fresh setup, never implicitly by an online-stage worker."""
    if type(case)is not RankedCase:raise ValueError('authenticated ranked case required')
    root=outside(root,protected(case));root.mkdir(mode=0o700,parents=False,exist_ok=False)
    info=root.lstat();body=dict(schema='cross-bert-fresh-ranked-attempt-v1',mode='official-fresh-key',
        attempt_id=uuid.uuid4().hex,case_binding=case_binding(case),root=str(root),root_identity=[info.st_dev,info.st_ino],
        keys_must_be_generated_after_attempt_creation=True,retained_key_mode=False)
    raw=key_gate.canonical(body);path=root/'.ranked_attempt.json'
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    try:
        view=memoryview(raw)
        while view:
            n=os.write(fd,view)
            if n<=0:raise OSError('short attempt publication')
            view=view[n:]
    finally:
        primary=sys.exception()
        try:os.close(fd)
        except BaseException as error:
            QUARANTINE.append({'descriptor':fd,'path':str(path),'close_state':'uncertain; never retry'})
            if primary is None:raise
            primary.add_note('attempt publication descriptor close: '+type(error).__name__)
    return key_gate.digest(raw)


def authenticate_attempt(root,expected_sha,case):
    root=outside(root,protected(case));raw=key_gate.read(root/'.ranked_attempt.json',1<<20)
    if key_gate.digest(raw)!=key_gate.sha_field(expected_sha):raise ValueError('externally pinned attempt differs')
    body=key_gate.parse(raw);info=root.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode)!=0o700:raise ValueError('owned private staged root required')
    nonce=body.get('attempt_id') if type(body)is dict else None
    expected=dict(schema='cross-bert-fresh-ranked-attempt-v1',mode='official-fresh-key',attempt_id=nonce,
        case_binding=case_binding(case),root=str(root),root_identity=[info.st_dev,info.st_ino],
        keys_must_be_generated_after_attempt_creation=True,retained_key_mode=False)
    if (type(nonce)is not str or len(nonce)!=32 or any(c not in '0123456789abcdef' for c in nonce)
            or raw!=key_gate.canonical(expected)):raise ValueError('fresh attempt case/identity/schema differs')
    return body


def paths(root,case):
    root=outside(root,protected(case));io=root/'io'/NAMES[case.size];inter=io/'intermediate'
    result=dict(size=case.size,batch_size=case.count,model_id='google-bert/bert-base-cased-finetuned-mrpc',
        model_revision=case.model_revision,public_keys=io/'public_keys',client_private=io/'client_private',
        server_data=root/'io/server_data',test_input=root/'datasets'/NAMES[case.size]/'intermediate/test_input.txt',
        preprocessed_input=inter/'client_preprocessed_input',upload=io/'ciphertexts_upload',download=io/'ciphertexts_download',
        decrypted_results=inter/'decrypted_results.jsonl',predictions=io/'encrypted_model_predictions.txt',
        server_report=io/'server_reported_steps.json')
    for path in (v for v in result.values() if isinstance(v,Path)):
        for candidate in (path,*path.parents):
            if candidate==root:break
            try:info=candidate.lstat()
            except FileNotFoundError:continue
            if stat.S_ISLNK(info.st_mode):raise ValueError('fresh staged paths cannot traverse symbolic links')
    return result


def require_fresh(paths,stage):
    names={1:('public_keys','client_private'),2:('server_data',),3:('preprocessed_input','upload','download','decrypted_results','predictions','server_report'),
        4:('download','decrypted_results','predictions','server_report'),5:('decrypted_results','predictions')}[stage]
    for name in names:
        p=paths[name]
        if p.exists() or p.is_symlink():raise FileExistsError('new stage artifact already exists')
    private=paths['client_private']
    if stage==3 and any((private/name).exists() or (private/name).is_symlink() for name in ('requests.json','completed.json')):
        raise FileExistsError('fresh official request ledger required')


def adapter():
    """Import the exact copied preprocess/postprocess bodies, no runtime hook."""
    origin=key_gate.parse((HERE/'worker_copy_origins.json').read_bytes())
    base=HERE/'upstream_adapter'
    for name in ('benchmark','p2_runtime','stages'):
        path=base/(name+'.py');rel=path.relative_to(HERE).as_posix()
        if key_gate.digest(path.read_bytes())!=origin[rel]['sha256']:raise ValueError('copied upstream adapter differs')
        module_name=name if name!='stages' else '_ranked_upstream_stages'
        old=sys.modules.get(module_name)
        if old is not None:
            if Path(old.__file__).resolve()!=path:raise ValueError('mixed upstream adapter import')
            continue
        spec=importlib.util.spec_from_file_location(module_name,path);module=importlib.util.module_from_spec(spec)
        sys.modules[module_name]=module
        try:spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name,None);raise
    return sys.modules['_ranked_upstream_stages']


class HeldDirectory:
    """Borrow identity only. Failed close is quarantined and never retried."""
    def __init__(self,path):
        self.path=Path(path);self.fd=None;self.uncertain=None
        fd=os.open(self.path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW);self.fd=fd
        try:
            info=os.fstat(fd);self.identity=(info.st_dev,info.st_ino);self.check()
        except BaseException as primary:
            try:self.close()
            except BaseException as error:primary.add_note('borrowed directory cleanup: '+type(error).__name__)
            raise
    def check(self):
        if self.fd is None or self.uncertain is not None:raise ValueError('borrowed directory is closed or uncertain')
        visible=self.path.lstat();held=os.fstat(self.fd)
        if not stat.S_ISDIR(visible.st_mode) or (visible.st_dev,visible.st_ino)!=self.identity or (held.st_dev,held.st_ino)!=self.identity:
            raise ValueError('borrowed fresh-attempt directory changed')
    def close(self):
        fd,self.fd=self.fd,None
        if fd is not None:
            try:os.close(fd)
            except BaseException:
                self.uncertain=fd;QUARANTINE.append(self);raise
