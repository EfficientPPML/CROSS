"""Sticky genuine-body rethrows through actual cleanup helpers; synthetic only."""
from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace
import ast,os,sys,weakref
import pytest
import cleanup
from cleanup_origin import mark,is_cleanup,protect_body,select
from component_protocol import Finalizer
from terminal import terminal_scope
from finish import finish
from test_role_lifetimes import fake_loader
import key_contexts as loaders
ROOT=Path(__file__).resolve().parent

def throwing(error):
 def call(*a,**k):raise error
 return call

@contextmanager
def identity():yield

@pytest.mark.parametrize('kind',[KeyboardInterrupt,SystemExit])
@pytest.mark.parametrize('genuine',[False,True])
def test_finalizer_sticky_identity_is_not_retagged(kind,genuine):
 first=ValueError('body')if genuine else mark(OSError('earlier close'));later=kind('later close')
 final=Finalizer(first);final.attempt('same object',throwing(first));final.attempt('later',throwing(later))
 expected=first if genuine else later
 with pytest.raises(type(expected))as actual:final.raise_failure()
 assert actual.value is expected and is_cleanup(first)is not genuine

@pytest.mark.parametrize('family',['transport','keys'])
@pytest.mark.parametrize('kind',[KeyboardInterrupt,SystemExit])
@pytest.mark.parametrize('genuine',[False,True])
def test_reused_close_state_does_not_confuse_sticky_body_with_cleanup(family,kind,genuine):
 if family=='transport':from component_transport import _CloseState
 else:from client_keys import _CloseState
 state=_CloseState();first=ValueError('body')if genuine else mark(OSError('close'));later=kind('close');seen=[]
 expected=first if genuine else later
 with pytest.raises(type(expected))as actual:
  try:raise first
  finally:
   state.cleanup('sticky',throwing(first));seen.append('sticky attempted')
   state.cleanup('later',throwing(later));seen.append('later attempted')
 assert actual.value is expected and seen[0]=='sticky attempted'and is_cleanup(first)is not genuine
 assert len(state.errors)==2

@pytest.mark.parametrize('scope',['cleanup_signals','termination_signals'])
@pytest.mark.parametrize('kind',[KeyboardInterrupt,SystemExit])
@pytest.mark.parametrize('genuine',[False,True])
def test_signal_restoration_sticky_identity(monkeypatch,scope,kind,genuine):
 first=ValueError('body')if genuine else mark(OSError('earlier close'));later=kind('later restore');seen=[]
 monkeypatch.setattr(cleanup,'_capture_mask',lambda errors:cleanup._MISSING)
 def restore(old,mask,errors):
  cleanup._attempt(errors,'sticky',throwing(first));seen.append('sticky')
  cleanup._attempt(errors,'later',throwing(later));seen.append('later')
 monkeypatch.setattr(cleanup,'_restore',restore)
 expected=first if genuine else later
 with pytest.raises(type(expected))as actual:
  with getattr(cleanup,scope)():raise first
 assert actual.value is expected and seen==['sticky','later']and is_cleanup(first)is not genuine

@pytest.mark.parametrize('kind',[KeyboardInterrupt,SystemExit])
def test_terminal_prior_and_failure_publication_keep_genuine_body(tmp_path,kind):
 first=ValueError('body');later=kind('publication');final=Finalizer(first);report={'passed':True};writes=[]
 @contextmanager
 def shield():
  try:yield
  finally:
   final.attempt('sticky',throwing(first));final.attempt('later',throwing(later));final.raise_failure()
 def write(path,value):writes.append(value['passed'])
 with pytest.raises(ValueError)as actual:
  with terminal_scope(report,tmp_path/'x',write,shield,lambda:(first,None)):raise first
 assert actual.value is first and writes==[False]and not is_cleanup(first)

@pytest.mark.parametrize('kind',[KeyboardInterrupt,SystemExit])
def test_finish_sticky_readback_before_cancellation(tmp_path,kind):
 first=ValueError('body');later=kind('republication');events=[]
 def read(path):
  # This independently owned cleanup state sees the active body exception.
  from component_transport import _CloseState
  _CloseState().cleanup('sticky readback close',throwing(first));events.append('read attempted');return b'{}'
 def write(path,value):events.append('write attempted');raise later
 with pytest.raises(ValueError)as actual:finish(tmp_path/'x',throwing(first),write,read,termination=identity,shield=identity)
 assert actual.value is first and events==['read attempted','write attempted']and not is_cleanup(first)

@pytest.mark.parametrize('server',[False,True])
@pytest.mark.parametrize('kind',[KeyboardInterrupt,SystemExit])
def test_loader_sticky_transfers_preserve_body_and_quarantine(fake_loader,server,kind):
 c=fake_loader;first=ValueError('body');later=kind('second barrier');events=[]
 class Pending:
  def __init__(self,error):self.error=error
  def block_until_ready(self):events.append(self.error);raise self.error
 cm=loaders.load_server(c.public,profile_bytes=b'',plan_bytes=b'')if server else loaders.load_client(c.public,c.private,profile_bytes=b'',plan_bytes=b'')
 with pytest.raises(ValueError)as actual:
  with cm as handle:
   handle.ctx._param_cache._formatted_rotation_keys={'a':Pending(first),'b':Pending(later)}
   raise first
 assert actual.value is first and events==[first,later]and not is_cleanup(first)
 assert c.quarantine and all(not a._mmap.closed for a in c.quarantine[0]['public_mappings'])and not loaders._ACTIVE.locked()

@pytest.mark.parametrize('kind',[KeyboardInterrupt,SystemExit])
def test_actual_arithmetic_finally_body_sticky_barriers(kind):
 # Execute the exact production except/finally AST over reachable fake arrays.
 # The numerical try body is deliberately replaced; no crypto import or work.
 source=ROOT/'component_arithmetic.py';tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='exercise')
 scope=next(n for n in fn.body if isinstance(n,ast.Try));scope.body=ast.parse('raise first').body
 wrapper=ast.FunctionDef(name='run',args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),body=[scope],decorator_list=[])
 first=ValueError('body');later=kind('barrier');events=[]
 class Pending:
  def __init__(self,error):self.polynomial=self;self.error=error
  def block_until_ready(self):events.append(self.error);raise self.error
 a,b=Pending(first),Pending(later);ledger=SimpleNamespace(live={1:weakref.ref(a),2:weakref.ref(b)});quarantine=[]
 namespace=dict(first=first,primary=None,entered=True,ctx=object(),initial_state=object(),graph=None,refresher=None,provider=None,ledger=ledger,value=object(),_FAILURE_QUARANTINE=quarantine,protect_body=protect_body,select=select,mark=mark)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[])),str(source),'exec'),namespace)
 with pytest.raises(ValueError)as actual:namespace['run']()
 assert actual.value is first and events==[first,later]and len(quarantine)==2 and not is_cleanup(first)

@pytest.mark.parametrize('entry',['source_guard.py','component_worker.py','collect.py'])
def test_preimport_read_sticky_body_never_calls_unauthenticated_helper(tmp_path,monkeypatch,entry):
 tree=ast.parse((ROOT/entry).read_text());name='read'if entry=='source_guard.py'else'_initial_read'
 fn=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name==name)
 namespace=dict(os=os,stat=__import__('stat'),_UNCERTAIN=[])
 exec(compile(ast.Module(body=[fn],type_ignores=[]),str(ROOT/entry),'exec'),namespace)
 p=tmp_path/'source';p.write_bytes(b'public');first=ValueError('source body');close=os.close
 def sticky(fd):close(fd);raise first
 with monkeypatch.context()as patch:
  patch.setattr(os,'read',throwing(first));patch.setattr(os,'close',sticky)
  with pytest.raises(ValueError)as actual:namespace[name](p,1024)
 assert actual.value is first and not is_cleanup(first)
 final=Finalizer(first);final.attempt('later',throwing(KeyboardInterrupt('close')))
 with pytest.raises(ValueError)as actual:final.raise_failure()
 assert actual.value is first
