"""Exact default-body parity and synthetic isolated receipt ownership checks."""
from pathlib import Path
from contextlib import contextmanager,nullcontext
from types import SimpleNamespace,ModuleType
import ast,copy,hashlib,io,json,os,stat,sys,tarfile
import numpy as np
import pytest
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'runtime/stage'),str(HERE/'runtime')]


def node(name,path):return next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef)and n.name==name)
def original_node(name,path):
    raw=(HERE/'baseline'/Path(path).name).read_bytes()
    return next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef)and n.name==name)

def test_default_decrypt_body_is_exact_original_ast():
    new=copy.deepcopy(node('client_decrypt_decode',HERE/'runtime/stage/file_stages.py'))
    old=original_node('client_decrypt_decode','runtime/stage/file_stages.py')
    assert new.args.kwonlyargs[0].arg=='completion_directory'and new.args.kw_defaults[0].value is None
    new.args.kwonlyargs=[];new.args.kw_defaults=[]
    first=new.body.pop(0);assert isinstance(first,ast.If)and ast.unparse(first.test)=='completion_directory is not None'
    final_with=new.body[-2];assert isinstance(final_with,ast.With)
    decision=final_with.body[0];assert isinstance(decision,ast.If)and ast.unparse(decision.test)=='completion_directory is None'
    final_with.body=decision.body
    assert ast.dump(new,include_attributes=False)==ast.dump(old,include_attributes=False)

def compile_lifecycle(monkeypatch):
    # Execute only actual helper/receipt definitions with synthetic ordinary I/O.
    raw=(HERE/'runtime/stage/request_lifecycle.py').read_text();tree=ast.parse(raw)
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in ('completion_path','record_completed')]
    @contextmanager
    def new_file(path,raw):
        fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        try:os.write(fd,raw)
        finally:os.close(fd)
        yield
    canonical=lambda value:(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode()
    namespace=dict(Path=Path,stat=stat,hashlib=hashlib,new_file=new_file,canonical=canonical,digest=lambda x:hashlib.sha256(canonical(x)).hexdigest(),_sha=lambda x:None)
    exec(compile(ast.Module(body=selected,type_ignores=[]),'<actual completion definitions>','exec'),namespace)
    return namespace

@pytest.mark.parametrize('separate',[False,True])
def test_receipt_destination_and_original_request_unchanged(tmp_path,monkeypatch,separate):
    n=compile_lifecycle(monkeypatch);private=tmp_path/'keys';private.mkdir(mode=0o700);owned=tmp_path/'result';owned.mkdir(mode=0o700)
    original=private/'requests.json';original.write_bytes(b'{"synthetic":true}\n');original.chmod(0o600);before=original.read_bytes()
    batch=dict(public={'context_id':'c'},requests=[{'contract':{'request_id':'0'*32}}])
    n['_paths']=lambda paths:{'upload':tmp_path/'upload','client_requests':original}
    def observe(path,kind,nonces):return dict(device=1,inode=2,sha256=hashlib.sha256(n['canonical'](batch)if kind=='upload'else before).hexdigest())
    n['_observe']=observe
    kwargs={'completion_directory':owned}if separate else{}
    result=n['record_completed']({'client_private':private},context_id='c',batch=batch,**kwargs)
    path=(owned if separate else private)/'completed.json'
    assert path.exists()and stat.S_IMODE(path.stat().st_mode)==0o600
    assert (private/'completed.json').exists()is(not separate)
    assert original.read_bytes()==before and result['objects']['client_requests']['sha256']==hashlib.sha256(before).hexdigest()
    assert not (tmp_path/'upload').exists() # no retirement or upload writes

@pytest.mark.parametrize('kind',['public','symlink','original','missing','file'])
def test_bad_explicit_completion_root(kind,tmp_path,monkeypatch):
    n=compile_lifecycle(monkeypatch);private=tmp_path/'keys';private.mkdir(mode=0o700);root=tmp_path/'results'
    if kind=='public':root.mkdir(mode=0o755);root.chmod(0o755)
    elif kind=='symlink':root.symlink_to(private,target_is_directory=True)
    elif kind=='original':root=private
    elif kind=='file':root.write_text('file')
    with pytest.raises((ValueError,FileNotFoundError)):n['completion_path']({'client_private':private},root)

@pytest.mark.parametrize('separate',[False,True])
def test_actual_handler_threads_only_optional_receipt(tmp_path,monkeypatch,separate):
    events=[];private=tmp_path/'keys';private.mkdir(mode=0o700);owned=tmp_path/'result';owned.mkdir(mode=0o700)
    batch=dict(public={'context_id':'c'},requests=[dict(contract={},input_digest='i')])
    response=dict(responses=[dict(request_id='n',scale=1.,response_manifest_sha256='m')])
    download=tmp_path/'download';download.mkdir();(download/'batch.json').write_text('{}');(download/'n').mkdir()
    paths=dict(client_private=private,public_keys=tmp_path/'public',upload=tmp_path/'upload',download=download,decrypted_results=owned/'decoded')
    decoded=np.zeros(65536);decoded[:2]=[.2,.7]
    client=SimpleNamespace(public_manifest={'context_id':'c'},ctx=SimpleNamespace(decrypt_slots=lambda value:decoded))
    def completion_path(p,r):events.append('completion-path');return r/'completed.json'
    def record(p,**kw):events.append(('record',kw))
    monkeypatch.setitem(sys.modules,'key_contexts',SimpleNamespace(load_client=lambda *a,**kw:nullcontext(client)))
    monkeypatch.setitem(sys.modules,'request_lifecycle',SimpleNamespace(record_completed=record,completion_path=completion_path))
    monkeypatch.setitem(sys.modules,'support',SimpleNamespace(require_secure_execution=lambda *a:None))
    def records(*a):events.append('private-records');return [SimpleNamespace(target_idx=9)],'input'
    def read_manifest(path):
        return dict(targets=[9])if path.name=='requests.json'else response if path.parent==download else batch
    @contextmanager
    def new_file(path,raw):events.append(('result',raw));yield
    batches=SimpleNamespace(validate_private_requests=lambda *a:None,validate_public_batch=lambda *a,**kw:batch,validate_response_batch=lambda *a,**kw:response)
    namespace=dict(Path=Path,np=np,_paths=lambda p:1,_inputs=lambda:({},'candidate'),_records=records,read_manifest=read_manifest,batches=batches,
        _scale_checker=lambda *a:None,load_ciphertext=lambda *a,**kw:(object(),{}),manifest_digest=lambda m:'m',new_file=new_file,canonical=lambda r:json.dumps(r).encode(),digest=lambda b:'b')
    fn=node('client_decrypt_decode',HERE/'runtime/stage/file_stages.py');exec(compile(ast.Module(body=[fn],type_ignores=[]),'<actual client handler>','exec'),namespace)
    result=namespace['client_decrypt_decode'](paths,**({'completion_directory':owned}if separate else{}))
    assert result=={'sample_count':1,'public_batch_sha256':'b'}
    record_call=next(x for x in events if type(x)is tuple and x[0]=='record')[1]
    assert record_call==dict(context_id='c',batch=batch,**({'completion_directory':owned}if separate else{}))
    if separate:assert events.index('completion-path')<events.index('private-records')
    else:assert 'completion-path'not in events
