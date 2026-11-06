import ast,sys,time,os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
HERE=Path(__file__).parent
@contextmanager
def scope(*a,**kw):yield

def load(file,name,ns):
    tree=ast.parse((HERE/'sources'/file).read_bytes());n=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
    exec(compile(ast.Module(body=[n],type_ignores=[]),str(HERE/'sources'/file),'exec'),ns)
    return ns[name]

@pytest.mark.parametrize('kind',[KeyboardInterrupt,SystemExit])
def test_worker_later_cleanup_ordinary_failure_does_not_replace_prior_cancellation(kind):
    cancel=kind('close cancellation');ordinary=OSError('later source read');expected={'x':'a'*64};counter=0
    envelope={'request':{'stage':3,'staged_root':'/public/owned'},'phase':'client_preprocess_input'}
    class Held:
        def __init__(self,path):pass
        def check(self):pass
        def close(self):raise cancel
    def hashes(e):
        nonlocal counter
        counter+=1
        if counter==2:raise ordinary
        return expected
    ns=dict(Path=Path,time=time,os=NS(umask=lambda n:0o077,environ={'JAX_PLATFORMS':'cpu'},getpid=lambda:123),write=lambda *a:None,hashes=hashes,
      key_gate=NS(parse=lambda b:expected if b==b'manifest' else envelope,read=lambda p,*a:b'manifest' if p=='manifest' else b'request',digest=lambda b:'d'*64,sha_field=lambda x:x),
      authenticate=lambda *a:(None,None,None,None,None,None,{}),protocol=NS(binding=lambda *a:{},declared_phases=lambda p:{}),execution=NS(HeldDirectory=Held),
      local_modules=lambda e:{},execute=lambda *a:None,terminal_scope=scope,cleanup_signals=scope)
    with pytest.raises(kind)as seen:load('bridge_worker.py','run',ns)('request','d'*64,'output','manifest')
    assert seen.value is cancel


def test_harness_child_environment_restored_if_directory_creation_fails(tmp_path,monkeypatch):
    root=tmp_path/'attempt';root.mkdir();target=tmp_path/'evidence';expected={'x':'a'*64};primary=OSError('mkdir failure');original=lambda:{}
    process=NS(child_environment_overrides=original)
    execution=NS(outside=lambda path,bases:Path(path),protected=lambda case:())
    monkeypatch.setitem(sys.modules,'ranked_execution',execution)
    monkeypatch.setitem(sys.modules,'terminal',NS(terminal_scope=scope))
    monkeypatch.setitem(sys.modules,'cleanup',NS(cleanup_signals=scope))
    mkdir=Path.mkdir
    def failing(p,*a,**kw):
        if p.name=='harness-process':raise primary
        return mkdir(p,*a,**kw)
    monkeypatch.setattr(Path,'mkdir',failing)
    config={'source_hashes':expected,'evidence_root':str(tmp_path/'phases')};state={'next_phase':0}
    envelope={'request':{'fixture_evidence':'/public/fixture','reference_evidence':'/public/reference'},'authorization':{'size0_evidence':'/public/size0','storage_path':'/public/grant'}}
    ns=dict(Path=Path,os=os,process=process,common=NS(NAMES=('client_key_generation',),write=lambda *a:None,hashes=lambda x:x,sources=lambda:expected),
        integration=NS(invocation=lambda p:(config,state),envelope_for=lambda *a:envelope),worker=NS(public_case=lambda p:(None,object())),
        protocol=NS(authorize=lambda *a,**kw:None),bridge_terminal=NS(finish=lambda p,f,w:f()))
    try:
        with pytest.raises(OSError)as seen:load('bridge_harness.py','run_prepared',ns)(root,target)
        assert seen.value is primary
        assert process.child_environment_overrides is original
    finally:process.child_environment_overrides=original
