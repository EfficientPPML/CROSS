"""Actual receipt-observation AST; synthetic descriptors/files only."""
from pathlib import Path
from types import SimpleNamespace
import ast,hashlib,os,stat,pytest
HERE=Path(__file__).resolve().parent

def install_close_state(ns):
 ns['sys']=__import__('sys');ns['_IO_FAILURE_QUARANTINE']=[]
 tree=ast.parse((HERE/'runtime/stage/client_keys.py').read_bytes())
 selected=[n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='_CloseState']
 assert len(selected)==1
 exec(compile(ast.Module(body=selected,type_ignores=[]),'<actual shared cleanup state>','exec'),ns)

@pytest.fixture
def env(tmp_path):
 tree=ast.parse((HERE/'runtime/stage/request_lifecycle.py').read_bytes());names={'_identity','_file_state','_tree','_observe'}
 fakeos=SimpleNamespace(**{n:getattr(os,n)for n in dir(os)})
 ns=dict(os=fakeos,stat=stat,hashlib=hashlib);install_close_state(ns)
 exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names],type_ignores=[]),'<actual receipt observation>','exec'),ns)
 root=tmp_path/'download';root.mkdir(mode=0o700);raw=b'{"public":"synthetic"}\n';(root/'batch.json').write_bytes(raw)
 nonce='0'*32;sample=root/nonce;sample.mkdir()
 for n in ('bundle.json','transport.json','ct_0.npz'):(sample/n).write_bytes(b'ordinary synthetic bytes, no ciphertext')
 direct=tmp_path/'result';direct.write_bytes(raw)
 roles={};opened=[];closed=[];real_open=os.open;real_close=os.close
 def opening(path,flags,*a,**kw):
  fd=real_open(path,flags,*a,**kw);role='observation'if str(path)in (str(root),str(direct),'batch.json')else 'tree';roles[fd]=role;opened.append((fd,role));return fd
 def closing(fd):closed.append((fd,roles.get(fd)));return real_close(fd)
 fakeos.open=opening;fakeos.close=closing
 return SimpleNamespace(ns=ns,os=fakeos,root=root,direct=direct,nonce=nonce,raw=raw,roles=roles,opened=opened,closed=closed,real_close=real_close)

def run(e,branch):return e.ns['_observe'](e.root if branch=='directory'else e.direct,'download'if branch=='directory'else 'decrypted_results',[e.nonce])

@pytest.mark.parametrize('branch',['file','directory'])
def test_never_dup_or_fdopen_and_all_descriptors_closed(env,branch):
 e=env;called=[]
 def forbidden(*a,**k):called.append(1);raise RuntimeError('fdopen handoff forbidden')
 e.os.dup=e.os.fdopen=forbidden;result=run(e,branch)
 assert not called and result['sha256']==hashlib.sha256(e.raw).hexdigest()
 assert e.opened==e.closed if branch=='file'else sorted(e.opened)==sorted(e.closed)

@pytest.mark.parametrize('branch',['file','directory'])
@pytest.mark.parametrize('kind',[ValueError,KeyboardInterrupt,SystemExit])
def test_body_primary_survives_all_owned_close_errors(env,branch,kind):
 e=env;primary=kind('read body');seen=[]
 def read(fd,size):raise primary
 def close(fd):
  e.real_close(fd);e.closed.append((fd,e.roles[fd]))
  if e.roles[fd]=='observation':seen.append(fd);raise OSError('close after body')
 e.os.read=read;e.os.close=close
 with pytest.raises(kind)as got:run(e,branch)
 assert got.value is primary and len(seen)==(2 if branch=='directory'else 1)
 assert sorted(e.opened)==sorted(e.closed)

@pytest.mark.parametrize('kind',[KeyboardInterrupt,SystemExit])
def test_cleanup_only_cancellation_after_ordinary_close(env,kind):
 e=env;ordinary=OSError('metadata close');cancel=kind('directory close');count=0
 def close(fd):
  nonlocal count;e.real_close(fd);e.closed.append((fd,e.roles[fd]))
  if e.roles[fd]=='observation':
   count+=1;raise ordinary if count==1 else cancel
 e.os.close=close
 with pytest.raises(kind)as got:run(e,'directory')
 assert got.value is cancel and count==2 and sorted(e.opened)==sorted(e.closed)
 assert any('metadata close'in note for note in cancel.__notes__)

@pytest.mark.parametrize('branch',['file','directory'])
def test_read_chunks_and_hash_stay_exact(env,branch):
 e=env;actual=os.read;e.os.read=lambda fd,n:actual(fd,min(n,3))
 assert run(e,branch)['sha256']==hashlib.sha256(e.raw).hexdigest()

def test_batch_replacement_before_open_is_rejected(env):
 e=env;original=e.os.open
 def opening(path,*a,**kw):
  if path=='batch.json':(e.root/'batch.json').rename(e.root/'old');(e.root/'batch.json').write_bytes(e.raw)
  return original(path,*a,**kw)
 e.os.open=opening
 with pytest.raises(ValueError,match='replaced before read'):run(e,'directory')
 assert sorted(e.opened)==sorted(e.closed)

@pytest.mark.parametrize('branch',['file','directory'])
def test_mutation_during_descriptor_read_rejected(env,branch):
 e=env;real=os.read;done=False;path=e.direct if branch=='file'else e.root/'batch.json'
 def read(fd,n):
  nonlocal done;b=real(fd,n)
  if not done:done=True;path.write_bytes(e.raw+b'changed')
  return b
 e.os.read=read
 with pytest.raises(ValueError,match='changed'):run(e,branch)
 assert sorted(e.opened)==sorted(e.closed)

@pytest.mark.parametrize('branch',['file','directory'])
def test_oversized_metadata_rejected_without_read(env,branch):
 e=env;p=e.direct if branch=='file'else e.root/'batch.json'
 with p.open('wb')as f:f.truncate((16<<20)+1)
 e.os.read=lambda *a:(_ for _ in ()).throw(AssertionError('oversized body read'))
 with pytest.raises(ValueError,match='bounded'):run(e,branch)
 assert sorted(e.opened)==sorted(e.closed)

def test_schema_and_retirement_definitions_unchanged():
 old=ast.parse((HERE/'baseline/request_lifecycle.py').read_bytes());new=ast.parse((HERE/'runtime/stage/request_lifecycle.py').read_bytes())
 for name in ('_paths','_identity','_file_state','retire_completed'):
  a=next(n for n in old.body if isinstance(n,ast.FunctionDef)and n.name==name);b=next(n for n in new.body if isinstance(n,ast.FunctionDef)and n.name==name)
  assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)

from pathlib import Path
from types import SimpleNamespace
import ast,hashlib,os,stat,pytest

@pytest.mark.parametrize('kind',['upload','download'])
@pytest.mark.parametrize('failure',[ValueError,KeyboardInterrupt,SystemExit])
def test_tree_body_primary_survives_nested_close(tmp_path,kind,failure):
 raw=(HERE/'runtime/stage/request_lifecycle.py').read_bytes();tree=ast.parse(raw);fake=SimpleNamespace(**{n:getattr(os,n)for n in dir(os)})
 ns=dict(os=fake,stat=stat,hashlib=hashlib);install_close_state(ns)
 exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in ('_identity','_file_state','_tree')],type_ignores=[]),'<actual tree>','exec'),ns)
 root=tmp_path/kind;root.mkdir();(root/'batch.json').write_bytes(b'{}');nonce='0'*32;sample=root/nonce;sample.mkdir()
 dirs=[sample]
 if kind=='upload':
  dirs=[]
  for i in range(8):p=sample/('tile'+str(i));p.mkdir();dirs.append(p)
 for p in dirs:
  for n in ('bundle.json','transport.json','ct_0.npz'):(p/n).write_bytes(b'ordinary synthetic metadata')
 primary=failure('body');closed=[]
 def observe(name,*a,**kw):
  if name=='ct_0.npz':raise primary
  return os.stat(name,*a,**kw)
 def close(fd):os.close(fd);closed.append(fd);raise OSError('nested close')
 fake.stat=observe;fake.close=close;fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY)
 try:
  with pytest.raises(failure)as got:ns['_tree'](fd,kind,[nonce])
  assert got.value is primary
 finally:os.close(fd)
 assert len(closed)==(2 if kind=='upload'else 1)

@pytest.mark.parametrize('failure',[KeyboardInterrupt,SystemExit])
def test_tree_cleanup_only_cancellation_stops_capture_and_closes_parent(env,failure):
 e=env;sample=e.root/e.nonce
 for p in sample.iterdir():p.unlink()
 for i in range(8):
  tile=sample/('tile'+str(i));tile.mkdir()
  for n in ('bundle.json','transport.json','ct_0.npz'):(tile/n).write_bytes(b'synthetic')
 primary=failure('sample cancellation');count=0
 def close(fd):
  nonlocal count;e.real_close(fd);e.closed.append((fd,e.roles[fd]));count+=1
  if count==1:raise OSError('tile close')
  raise primary
 e.os.close=close;fd=os.open(e.root,os.O_RDONLY|os.O_DIRECTORY)
 try:
  with pytest.raises(failure)as got:e.ns['_tree'](fd,'upload',[e.nonce])
  assert got.value is primary and count==2
 finally:os.close(fd)
 assert sorted(e.opened)==sorted(e.closed)

@pytest.mark.parametrize('branch',['file','directory'])
def test_successful_observation_schema_matches_original(env,branch):
 e=env;old=ast.parse((HERE/'baseline/request_lifecycle.py').read_bytes());ns=dict(os=os,stat=stat,hashlib=hashlib)
 def read(fd,name,maximum):
  child=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
  try:return os.read(child,maximum+1)
  finally:os.close(child)
 ns['_read']=read
 exec(compile(ast.Module(body=[n for n in old.body if isinstance(n,ast.FunctionDef)and n.name in ('_identity','_file_state','_tree','_observe')],type_ignores=[]),'<exact original observation>','exec'),ns)
 args=(e.root if branch=='directory'else e.direct,'download'if branch=='directory'else 'decrypted_results',[e.nonce])
 assert ns['_observe'](*args)==run(e,branch)

from pathlib import Path
from types import SimpleNamespace
import ast,hashlib,os,stat,pytest
@pytest.mark.parametrize('kind',[KeyboardInterrupt,SystemExit])
def test_tree_cleanup_then_observe_cancellation(kind,tmp_path):
 t=ast.parse((HERE/'runtime/stage/request_lifecycle.py').read_bytes());fake=SimpleNamespace(**{n:getattr(os,n)for n in dir(os)});ns=dict(os=fake,stat=stat,hashlib=hashlib);install_close_state(ns)
 exec(compile(ast.Module(body=[n for n in t.body if isinstance(n,ast.FunctionDef)and n.name in ('_identity','_file_state','_tree','_observe')],type_ignores=[]),'<actual nested receipt>','exec'),ns)
 root=tmp_path/'download';root.mkdir();(root/'batch.json').write_bytes(b'{}');nonce='0'*32;d=root/nonce;d.mkdir()
 for name in ('bundle.json','transport.json','ct_0.npz'):(d/name).write_bytes(b'synthetic')
 calls=[];cancel=kind('outer cleanup cancellation')
 def close(fd):
  os.close(fd);calls.append(fd)
  if len(calls)==1:raise OSError('inner cleanup')
  raise cancel
 fake.close=close
 with pytest.raises(kind)as got:ns['_observe'](root,'download',[nonce])
 assert got.value is cancel and len(calls)==2
