from pathlib import Path
import hashlib,io,json,os,shutil,subprocess,tarfile
BASE=Path('/tmp/bert-toy4-checkpoint-v2-source-bound-20261104');SRC=BASE/'execution_source';MAP=BASE/'source_manifest.json';OUT=Path('/tmp/bert-toy4-spec-alias-review-20261104/negative-v1');OUT.mkdir(parents=True,exist_ok=False)
h=lambda b:hashlib.sha256(b).hexdigest();raw=MAP.read_bytes();assert h(raw)=='5161b934cbb31f55b0e38ab35822ba4a3a73ea975a5737920215392180e6b9be';expected=json.loads(raw);assert len(expected)==253
blobs={n:(SRC/n).read_bytes()for n in expected};assert {n:h(b)for n,b in blobs.items()}==expected
root=OUT/'source';root.mkdir()
for n,b in blobs.items():p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
(OUT/'source_manifest.json').write_bytes(raw)
with tarfile.open(OUT/'sources.tar.gz','w:gz')as t:
 for n,b in sorted(blobs.items()):m=tarfile.TarInfo(n);m.size=len(b);t.addfile(m,io.BytesIO(b))
rows=[]
for entry in('collect.py','component_worker.py'):
 for mode in('spec_only','file_control'):
  case=entry.removesuffix('.py')+'_'+mode;directory=OUT/case;directory.mkdir();marker=directory/'post_admission_marker';origin=str(root/'component_protocol.py')
  attr='m.__spec__=types.SimpleNamespace(origin='+repr(origin)+')'if mode=='spec_only'else'm.__file__='+repr(origin)
  script="""import runpy,sys,types,importlib.abc
from pathlib import Path
m=types.ModuleType('review_unknown_alias')
"""+attr+"""
sys.modules['review_unknown_alias']=m
class StopBeforeHelpers(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname in ('jax','jaxlib','torch','tensorflow','transformers','numpy') or fullname.startswith(('jax.','jaxlib.','jaxite_word.')):
   raise RuntimeError('review forbids backend import')
  if fullname=='component_protocol':
   Path("""+repr(str(marker))+""").write_text('source admission reached first local helper import')
   raise RuntimeError('review stopped before helper/backend execution')
sys.meta_path.insert(0,StopBeforeHelpers())
sys.argv=sys.argv[1:]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
  cmd=['/home/jianming_gatech/miniconda3/envs/jaxite/bin/python','-I','-B','-c',script,str(root/entry),'--source-manifest',str(OUT/'source_manifest.json'),'--source-manifest-sha256',h(raw),'--help']
  child=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True,env={'PATH':'/usr/bin:/bin','HOME':'/home/jianming_gatech','PYTHONDONTWRITEBYTECODE':'1'})
  stdout,stderr=child.communicate(timeout=60);(directory/'stdout.log').write_bytes(stdout);(directory/'stderr.log').write_bytes(stderr)
  try:os.killpg(child.pid,0)
  except ProcessLookupError:gone=True
  else:gone=False
  rejected=b'helper/runtime preloaded or aliased before source gate'in stderr
  assert child.returncode==1 and gone and not Path('/proc/'+str(child.pid)).exists()
  assert (marker.exists()and b'review stopped before helper/backend execution'in stderr)or(rejected and not marker.exists())
  rows.append(dict(entry=entry,mode=mode,requirement_passed=rejected,post_admission_helper_reached=marker.exists(),helper_backend_execution_stopped=True,child_exit=child.returncode,pid=child.pid,direct_wait_completed=True,pid_and_group_absent=True))
after={n:h((SRC/n).read_bytes())for n in expected};assert after==expected and {n:h((root/n).read_bytes())for n in expected}==expected
report=dict(passed=all(r['requirement_passed']for r in rows),expected_failures=sum(not r['requirement_passed']for r in rows),passing_controls=sum(r['requirement_passed']for r in rows),scope='Exact copied253 source actual collector/worker admission only; harmless preloaded alias. Independent finder stops before first local component helper or backend. No data/assets/context/key/HE execution.',source_count=253,external_manifest_sha256=h(raw),source_hashes_before=expected,source_hashes_after=after,source_archive_sha256=h((OUT/'sources.tar.gz').read_bytes()),cases=rows)
(OUT/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n');shutil.copyfile(__file__,OUT/'review.py');print(json.dumps({k:report[k]for k in('passed','expected_failures','passing_controls','source_archive_sha256')}));raise SystemExit(0 if report['passed']else 1)
