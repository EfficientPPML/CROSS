"""Authorized authentic74-public-matrix key audit; no native/HE/private reads."""
import argparse,hashlib,io,json,os,stat,struct,sys,tarfile,time
from pathlib import Path
import numpy as np
from public_functions import planners,fixed_keys,pins
HERE=Path(__file__).parent.resolve()
MODEL=Path('/tmp/cross-bert-public-model-b51668856dab42a3846ba1b7c7a71d53')
CHECKPOINT=Path('/home/jianming_gatech/cross_validation_assets/bert/f150c1d609d1e50dd5e2e5408661cfac8339277c/model.safetensors')
MODEL_REPORT='5b9589b51828d066706c4c46a6404d8c52fcd67a2c9aa9f3f57c73df8caeefe0'
KEY_REPORT='022cbdf687273e765668184f32cb7cd501d371c4220916e1c8f02574cf3a7f08'
KEY_ARCHIVE='083851cf0667d2bb500514b5599028312c7d387236347ce3de3b585c3dcd2a29'
def digest(raw):return hashlib.sha256(raw).hexdigest()
def state(s):return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def close(fd,primary=None):
 try:os.close(fd)
 except BaseException as error:
  if primary is None:raise
  primary.add_note('public descriptor close: '+type(error).__name__)
def read(path,maximum=8<<20,*,directory_fd=None):
 before=os.stat(path,dir_fd=directory_fd,follow_symlinks=False)
 fd=os.open(path,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW,dir_fd=directory_fd)
 try:
  opened=os.fstat(fd)
  if not stat.S_ISREG(opened.st_mode)or state(opened)!=state(before)or not 0<opened.st_size<=maximum:raise ValueError('bounded original public file required')
  chunks=[];size=0
  while True:
   block=os.read(fd,min(1<<20,maximum+1-size))
   if not block:break
   chunks.append(block);size+=len(block)
   if size>maximum:raise ValueError('public size limit')
  if state(os.fstat(fd))!=state(opened)or state(os.stat(path,dir_fd=directory_fd,follow_symlinks=False))!=state(opened):raise ValueError('public file changed')
  return b''.join(chunks)
 finally:close(fd,sys.exception())
def hash_fd(fd):
 os.lseek(fd,0,os.SEEK_SET);h=hashlib.sha256()
 while chunk:=os.read(fd,1<<20):h.update(chunk)
 return h.hexdigest()
def exact_read(fd,n):
 out=[];size=0
 while size<n:
  chunk=os.read(fd,n-size)
  if not chunk:raise ValueError('short public checkpoint')
  out.append(chunk);size+=len(chunk)
 return b''.join(out)
def source_hashes():
 paths=[p for p in HERE.rglob('*')if p.is_file()and(p.relative_to(HERE).parts[0]=='inputs'or p.parent==HERE and p.suffix in('.py','.json','.md'))]
 return {str(p.relative_to(HERE)):digest(p.read_bytes())for p in sorted(paths)}
def write(path,value):path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
def authenticate_public():
 keyroot=HERE/'inputs/actual_stage1';modelroot=HERE/'inputs/actual_stage2'
 keyreport=json.loads(read(keyroot/'report.json'));assert digest(read(keyroot/'report.json'))==KEY_REPORT and keyreport['passed']and keyreport['actual_key_generation']and keyreport['all_processes_reaped']
 public=json.loads(read(keyroot/'public_manifest.json'));worker=json.loads(read(keyroot/'worker/worker_report.json'))
 assert worker['passed']and worker['actual_key_generation_completed']and worker['public_manifest']==public and keyreport['context_id']==public['context_id']
 sys.path[:0]=[str(HERE/'inputs/gates')]
 import key_gate,model_gate
 # Preserve the exact reviewed gate conditions; tighten only the borrowed read
 #utility to a single FD with before/opened/after identity and bounded bytes.
 key_gate.read=read
 stub=type('Pins',(),pins())();sys.modules['server_public_model']=stub
 result=model_gate.authenticate(modelroot,MODEL_REPORT,MODEL,keys=dict(context_id=public['context_id'],key_report_sha256=KEY_REPORT,key_source_archive_sha256=KEY_ARCHIVE))
 manifest=json.loads(read(modelroot/'model_manifest.json'))
 return public,result,manifest

def audit(output):
 output=Path(output).resolve();output.mkdir(exist_ok=False);started=time.monotonic();before=source_hashes()
 report=dict(passed=False,scope='Authentic public stage2/checkpoint matrix/key planning only; no model/native/HE/key/CT/private body reads',pid=os.getpid(),source_hashes_before=before,matrices=[],no_payloads_archived=True)
 write(output/'report.json',report)
 cachefd=checkpointfd=None
 try:
  with tarfile.open(output/'sources.tar.gz','w:gz')as t:
   for n in before:t.add(HERE/n,arcname=n)
  report['source_archive_sha256']=digest((output/'sources.tar.gz').read_bytes())
  public,model,manifest=authenticate_public();bound=pins();MatrixDiagonals,planner=planners()
  selected={n:shape for n,shape in bound['shapes'].items()if len(shape)==2};assert len(selected)==74
  cache=Path(model['server_data'])/public['context_id'];cache_state=cache.lstat();cachefd=os.open(cache,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
  assert (os.fstat(cachefd).st_dev,os.fstat(cachefd).st_ino)==(cache_state.st_dev,cache_state.st_ino)
  checkpointfd=os.open(CHECKPOINT,os.O_RDONLY|os.O_NOFOLLOW);checkpoint_state=os.fstat(checkpointfd)
  assert stat.S_ISREG(checkpoint_state.st_mode)
  report['checkpoint_sha256_before']=hash_fd(checkpointfd);assert report['checkpoint_sha256_before']==bound['CHECKPOINT_SHA256']
  os.lseek(checkpointfd,0,os.SEEK_SET);header_size=struct.unpack('<Q',exact_read(checkpointfd,8))[0];assert 0<header_size<1<<20
  header=json.loads(exact_read(checkpointfd,header_size));data_start=8+header_size;dense=set()
  for name,shape in selected.items():
   descriptor=manifest['tensors'][name];filename=descriptor['file'];assert '/'not in filename and filename.startswith('tensor_')and filename.endswith('.npy')
   raw=read(filename,16<<20,directory_fd=cachefd);assert digest(raw)==descriptor['file_sha256']
   value=np.load(io.BytesIO(raw),allow_pickle=False);assert value.dtype.str=='<f4'and value.shape==shape and np.isfinite(value).all()
   tensor=digest(value.tobytes(order='C'));assert tensor==descriptor['tensor_sha256']==bound['TENSOR_SHA256'][name]
   entry=header[name];start,end=entry['data_offsets'];assert entry['dtype']=='F32'and entry['shape']==list(shape)and end-start==value.size*4
   os.lseek(checkpointfd,data_start+start,os.SEEK_SET);checkpoint_tensor=digest(exact_read(checkpointfd,end-start));assert checkpoint_tensor==tensor
   source=MatrixDiagonals(value,65536,4096);plan=planner(source,strategy='chained');dense.update(plan.required_keys)
   row=dict(name=name,shape=list(shape),tensor_sha256=tensor,checkpoint_tensor_sha256=checkpoint_tensor,file=filename,file_sha256_before=digest(raw),n1=source.n1,active_diagonal_count=len(source.active),active_indices=list(source.active),required_keys=list(plan.required_keys),rotation_calls=plan.nonzero_rotation_calls)
   report['matrices'].append(row)
  for row in report['matrices']:
   row['file_sha256_after']=digest(read(row['file'],16<<20,directory_fd=cachefd));assert row['file_sha256_before']==row['file_sha256_after']
  report['checkpoint_sha256_after']=hash_fd(checkpointfd);assert report['checkpoint_sha256_after']==report['checkpoint_sha256_before']and state(os.fstat(checkpointfd))==state(checkpoint_state)and state(CHECKPOINT.stat())==state(checkpoint_state)
  assert (cache.lstat().st_dev,cache.lstat().st_ino)==(cache_state.st_dev,cache_state.st_ino)
  fixed=fixed_keys();profile=json.loads((HERE/'inputs/bsgs/p60_profile_and_refresh.json').read_bytes());refresh=set(profile['refresh_plan']['rotations'])
  graph=set(fixed['attention'])|set(fixed['norm'])|dense;required=graph|refresh;available=set(public['rotations'])
  report.update(context_id=public['context_id'],model_id=model['model_id'],model_report_sha256=MODEL_REPORT,public_key_manifest_sha256=digest(read(HERE/'inputs/actual_stage1/public_manifest.json')),fixed_groups=fixed,dense_keys=sorted(dense),graph_keys=sorted(graph),refresh_keys=sorted(refresh),required_keys=sorted(required),available_keys=sorted(available),missing_keys=sorted(required-available),unused_existing_keys=sorted(available-required),actual_key_store_covers=required<=available,model_cache_identity=[cache_state.st_dev,cache_state.st_ino],profile_matches_existing=public['parameters']['degree']==131072)
  assert not any(n in sys.modules for n in ('jax','torch','transformers'))
  report['backends_imported']=False;report['passed']=True
 finally:
  active=sys.exception();errors=[]
  for fd in (checkpointfd,cachefd):
   if fd is not None:
    try:close(fd)
    except BaseException as error:errors.append(error)
  if active is not None:
   report['error']=dict(type=type(active).__name__,message='public audit failed; raw data omitted');report['passed']=False
   for e in errors:active.add_note('public descriptor cleanup: '+type(e).__name__)
  elif errors:
   report['passed']=False;report['cleanup_errors']=[type(e).__name__for e in errors]
  report['source_hashes_after']=source_hashes();report['source_files_unchanged']=before==report['source_hashes_after']
  with tarfile.open(output/'sources.tar.gz','r:gz')as t:archived={m.name:digest(t.extractfile(m).read())for m in t.getmembers()}
  report['source_archive_verified']=archived==before;report['elapsed_s']=time.monotonic()-started;report['cleanup_completed']=not errors
  report['passed']=report['passed']and report['source_files_unchanged']and report['source_archive_verified']and not errors
  try:write(output/'report.json',report)
  except BaseException as publication:
   if active is not None:active.add_note('public report publication: '+type(publication).__name__)
   else:raise
  if active is None and errors:raise errors[0]
 return report
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('output');args=parser.parse_args();result=audit(args.output)
 print(json.dumps({k:result[k]for k in ('passed','dense_keys','missing_keys','actual_key_store_covers','elapsed_s')}))
 raise SystemExit(0 if result['passed']else 1)
