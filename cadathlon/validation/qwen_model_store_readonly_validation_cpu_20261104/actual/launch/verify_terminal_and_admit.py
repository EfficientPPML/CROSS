"""Root-owned actual terminal plus pure preparatory ModelAdmission, no body read."""
import os,stat,json,hashlib,tarfile,sys,datetime,importlib.abc
from pathlib import Path
sha=lambda b:hashlib.sha256(b).hexdigest()
canonical=lambda v:(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
HERE=Path(__file__).resolve().parent
assert sys.flags.isolated and sys.dont_write_bytecode
SOURCE=Path('/tmp/qwen-model-store-readonly-v2-20261104')
SOURCE_MAP=Path('/tmp/qwen-model-store-readonly-sources-v2-20261104.json')
SOURCE_SHA='b73e669b0cc2bc34d7930f2a8a4c620be719fe663dfa6ae7b12df7cae22c2bc0'
ARCHIVE=Path('/tmp/qwen-model-store-readonly-source-handoff-v2-20261104/sources.tar.gz')
ARCHIVE_SHA='06a7ae13b6df2adb9808af69d7906eec1e86ed811c419da9c1d064479fe95a25'
CONSUMER=Path('/tmp/qwen-keygen-authority-boundary-20261104')
CONSUMER_MAP=Path('/tmp/qwen-keygen-authority-boundary-sources-v1-20261104.json')
CONSUMER_SHA='d67a5df7aff7cd6560e88bd7731cf124ceefba18f4a2c83c443aa1ad1aa72ff8'
REPORT=Path('/tmp/qwen-model-store-readonly-actual-v1-20261104/report.json')
REPORT_SHA='cc8aaab893268b96692d9998412b121b97f7cfdd5ea47c5f3a60949fe620203c'
OUT=Path('/tmp/qwen-model-store-readonly-root-admission-20261104')
assert not OUT.exists()
def inventory(root):
 assert not root.is_symlink()and not any(p.is_symlink()for p in root.rglob('*'))
 return {p.relative_to(root).as_posix():sha(p.read_bytes())for p in root.rglob('*')if p.is_file()}
raw=SOURCE_MAP.read_bytes();assert sha(raw)==SOURCE_SHA;expected=json.loads(raw);assert canonical(expected)==raw and inventory(SOURCE)==expected and len(expected)==42
craw=CONSUMER_MAP.read_bytes();assert sha(craw)==CONSUMER_SHA;consumer_expected=json.loads(craw);assert canonical(consumer_expected)==craw and inventory(CONSUMER)==consumer_expected and len(consumer_expected)==28
assert sha(ARCHIVE.read_bytes())==ARCHIVE_SHA
with tarfile.open(ARCHIVE)as t:
 members=t.getmembers();assert len(members)==len({m.name for m in members})==42
 assert all(m.isfile()and not Path(m.name).is_absolute()and '..'not in Path(m.name).parts for m in members)
 assert{m.name:sha(t.extractfile(m).read())for m in members}==expected
launch_raw=(HERE/'execution-launch-review.json').read_bytes();assert sha(launch_raw)=='de4d26276e0be0dc6fbeb0c2c7b4831b1bf5698df4f64813cd983dcf9dcfcb8d'
grant_raw=(HERE/'grant.json').read_bytes();assert sha(grant_raw)=='eafbba1d2e463302c9d8a0305a038c204faa3ceaa531ce36b4cbd70c8f80d9f7';grant=json.loads(grant_raw)
for name in ['expected_stores','prelaunch_review']:
 row=grant[name];assert sha(Path(row['path']).read_bytes())==row['sha256']
store_raw=Path(grant['expected_stores']['path']).read_bytes();assert sha(store_raw)=='c3023d3782b3aa3b9b9607f07d291162383281a59b86be71fb9cb185fbcf8604';stores=json.loads(store_raw)
live_raw=(HERE/'live-observation.json').read_bytes();assert sha(live_raw)=='57f7f58147ca7497073ba5dfd0b70a5141d94cc21779d77f8393493413541150';live=json.loads(live_raw)['observations'][0];assert live['pid']==live['pgid']==56991
assert not Path('/proc/56991').exists()
try:os.killpg(56991,0)
except ProcessLookupError:pass
else:raise AssertionError('actual process group remains')
# Read only the public report through its held original output directory.
root_state=REPORT.parent.lstat();assert stat.S_ISDIR(root_state.st_mode)and stat.S_IMODE(root_state.st_mode)==0o700 and root_state.st_uid==os.getuid()
root_fd=os.open(REPORT.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
try:
 assert (os.fstat(root_fd).st_dev,os.fstat(root_fd).st_ino)==(root_state.st_dev,root_state.st_ino)
 assert os.listdir(root_fd)==['report.json']
 fd=os.open('report.json',os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=root_fd)
 try:
  before=os.fstat(fd);assert stat.S_ISREG(before.st_mode)and stat.S_IMODE(before.st_mode)==0o600 and before.st_uid==os.getuid()and 0<before.st_size<=2<<20
  blocks=[];size=0
  while True:
   b=os.read(fd,65536)
   if not b:break
   blocks.append(b);size+=len(b);assert size<=2<<20
  report_raw=b''.join(blocks);assert len(report_raw)==before.st_size and sha(report_raw)==REPORT_SHA
  after=os.fstat(fd);named=os.stat('report.json',dir_fd=root_fd,follow_symlinks=False)
  fields=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode,s.st_uid)
  assert fields(before)==fields(after)==fields(named)
 finally:os.close(fd)
 assert (REPORT.parent.lstat().st_dev,REPORT.parent.lstat().st_ino)==(root_state.st_dev,root_state.st_ino)
finally:os.close(root_fd)
report=json.loads(report_raw);assert canonical(report)==report_raw
assert report['passed']is report['complete']is True and report['error_type']is None and report['audit_denials']==[]
assert report['process']==dict(pid=56991,pgid=56991)
assert report['source_hashes_before']==report['source_hashes_after']==expected
assert canonical(report['stores'])==store_raw
# One post-terminal metadata pass over the exact declared names, never file bodies.
observed_roots={};observed_files=[]
for family,row in stores['roots'].items():
 p=Path(row['path']);s=p.lstat();assert stat.S_ISDIR(s.st_mode)and not p.is_symlink()
 observed_roots[family]=dict(path=str(p),device=s.st_dev,inode=s.st_ino,mode=stat.S_IMODE(s.st_mode))
for row in stores['files']:
 p=Path(stores['roots'][row['family']]['path'])/row['name'];s=p.lstat();assert stat.S_ISREG(s.st_mode)
 state=dict(device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,mode=stat.S_IMODE(s.st_mode))
 observed_files.append(dict(row,state=state))
observed=dict(roots=observed_roots,files=observed_files);assert canonical(observed)==store_raw
outer=dict(schema='qwen-model-store-validation-outer-v1',report_sha256=REPORT_SHA,aggregate_manifest_sha256=grant['aggregate_manifest_sha256'],model_id=grant['model_id'],source_manifest_sha256=SOURCE_SHA,source_archive_sha256=ARCHIVE_SHA,execution_grant_sha256=sha(grant_raw),pid=56991,pgid=56991,exit_code=0,normal_return=True,reaped=True,process_group_absent=True)
outer_raw=canonical(outer)
review=dict(schema='qwen-model-stage-root-admission-v1',passed=True,model_id=grant['model_id'],aggregate_manifest_sha256=grant['aggregate_manifest_sha256'],catalog_sha256=grant['catalog_sha256'],validation_report_sha256=REPORT_SHA,validation_outer_sha256=sha(outer_raw),source_manifest_sha256=SOURCE_SHA,source_archive_sha256=ARCHIVE_SHA,execution_grant_sha256=sha(grant_raw),current_store_binding_sha256=sha(store_raw),all_required_links_verified=True,source_archive_members_verified=True,actual_server_execution_authorized=False)
review_raw=canonical(review)
pins=dict(schema='qwen-model-expected-pins-v1',model_id=grant['model_id'],aggregate_manifest_sha256=grant['aggregate_manifest_sha256'],catalog_sha256=grant['catalog_sha256'],producer_source_manifest_sha256=SOURCE_SHA,producer_source_archive_sha256=ARCHIVE_SHA,producer_source_map=expected,execution_grant_sha256=sha(grant_raw),validation_report_sha256=REPORT_SHA,validation_outer_sha256=sha(outer_raw),root_review_sha256=sha(review_raw),consumer_source_manifest_sha256=CONSUMER_SHA)
pins_raw=canonical(pins)
# Authenticated28 is a preparation consumer only; it cannot run key generation.
forbidden={'numpy','jax','jaxlib','jaxite_word','torch','transformers','tensorflow','flax','safetensors','tokenizers','libtpu','torch_xla'}
assert not any(n.split('.')[0]in forbidden for n in sys.modules)
class DenyBackend(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]in forbidden:raise PermissionError('pure root admission forbids backend/NumPy')
sys.meta_path.insert(0,DenyBackend());sys.path.insert(0,str(CONSUMER))
from model_admission import authenticate_model,ExpectedModelPins,require_admitted
from qwen_file_contract import PublicContract
from public_source_pins import PUBLIC_PINS
public=PublicContract.from_blobs({n:(CONSUMER/'public_inputs'/n).read_bytes()for n in PUBLIC_PINS})
model=authenticate_model(public,(SOURCE/'unadmitted_aggregate_manifest.json').read_bytes(),report_raw,outer_raw,review_raw,expected_pins=ExpectedModelPins(pins_raw))
assert require_admitted(model,public,consumer_source_manifest_sha256=CONSUMER_SHA)is model
assert model.execution_authorized is False and model.model_id==grant['model_id']and model.stores.raw==store_raw
for n in ['model_admission','qwen_file_contract','qwen_public_parse','public_source_pins']:
 module=sys.modules[n];p=CONSUMER/(n+'.py');assert Path(module.__file__)==p and Path(module.__spec__.origin)==p
assert inventory(SOURCE)==expected and inventory(CONSUMER)==consumer_expected
assert not any(n.split('.')[0]in forbidden for n in sys.modules)
summary=dict(schema='qwen-model-readonly-root-terminal-admission-v1',passed=True,model_integrity_admitted=True,model_id=model.model_id,producer_source_count=42,producer_source_manifest_sha256=SOURCE_SHA,producer_source_archive_sha256=ARCHIVE_SHA,source_current_archive_exact=True,actual_report_sha256=REPORT_SHA,outer_sha256=sha(outer_raw),root_review_sha256=sha(review_raw),expected_pins_sha256=sha(pins_raw),store_binding_sha256=sha(store_raw),post_terminal_metadata_roots=3,post_terminal_metadata_files=198,post_terminal_bodies_read=False,atomic_snapshot_claimed=False,pure_admission_consumer_source_count=28,pure_admission_consumer_source_manifest_sha256=CONSUMER_SHA,consumer_scope='Reviewed preparation boundary only. Future actual keygen/client/server needs a separately reviewed exact consumer map and trusted repinning, current held-store checks and its own role grant.',model_execution_authorized=False,key_generation_authorized=False,scientific_acceptance_claimed=False,authoritative_owner='root',tool_session_id=27288,launch_tool_chunk='01593b',terminal_tool_chunk='724ea7',normal_zero_direct_reap=True,pid=56991,pgid=56991,process_group_absent=True,independent_live_observation_sha256=sha(live_raw),launch_review_sha256=sha(launch_raw),scope='Actual existing-model readonly integrity proof only; complete197physical matrices/198logical visits and113gamma vectors. Borrowedstores retained; no model forward, keys/ciphertexts/token/calibration access or re-export.',script_sha256=sha(Path(__file__).read_bytes()),utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
OUT.mkdir(mode=0o700)
for n,v in [('outer.json',outer),('root-review.json',review),('expected-pins-authority28.json',pins),('admission.json',summary)]:
 fd=os.open(OUT/n,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 try:
  data=canonical(v);offset=0
  while offset<len(data):
   written=os.write(fd,data[offset:]);assert written>0;offset+=written
  os.fsync(fd)
 finally:os.close(fd)
 assert (OUT/n).read_bytes()==data
print(json.dumps(dict(passed=True,model_integrity_admitted=True,execution_authorized=False,admission_path=str(OUT/'admission.json'),admission_sha256=sha((OUT/'admission.json').read_bytes())),sort_keys=True))
