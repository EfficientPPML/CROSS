"""Source-bound current-main promotion checks, no keygen/encrypted execution."""
from pathlib import Path
import hashlib,io,json,os,subprocess,sys,tarfile,time
HERE=Path(__file__).resolve().parent;ROOT=HERE/'checkout'
def sha(data):return hashlib.sha256(data).hexdigest()
def sources():
 paths=set(HERE.glob('*.py'))|{HERE/'base_main_sha256.json',HERE/'calibration_inventory_overlap.json'}
 paths.update((HERE/'base').rglob('*.py'))
 for p in ROOT.rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts:paths.add(p)
 return {str(p.relative_to(HERE)):p.read_bytes() for p in sorted(paths)}
def main():
 output=Path(sys.argv[1]);output.mkdir(exist_ok=False,parents=True)
 raw=sources();before={name:sha(data) for name,data in raw.items()}
 with tarfile.open(output/'sources.tar.gz','w:gz') as tar:
  for name,data in raw.items():
   entry=tarfile.TarInfo(name);entry.size=len(data);tar.addfile(entry,io.BytesIO(data))
 tests=['test_chained_streaming_linear.py','test_packed_transformer.py','test_streaming_linear.py',
  'test_streaming_linear_contracts.py','test_packed_qwen.py','test_package.py','test_bert_staging.py']
 command=[sys.executable,'-m','pytest','-q',*(str(ROOT/'cadathlon/tests'/name) for name in tests),'--junitxml='+str(output/'junit.xml')]
 started=time.perf_counter()
 with (output/'run.log').open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
 after={name:sha(data) for name,data in sources().items()}
 with tarfile.open(output/'sources.tar.gz') as tar:archived={m.name:sha(tar.extractfile(m).read()) for m in tar.getmembers()}
 report=dict(scope=__doc__,passed=result.returncode==0 and before==after==archived,returncode=result.returncode,command=command,
  source_hashes_before=before,source_hashes_after=after,source_files_unchanged=before==after,
  source_archive_byte_verified=archived==before,source_files=len(before),source_archive_sha256=sha((output/'sources.tar.gz').read_bytes()),
  elapsed_s=time.perf_counter()-started,cpu_affinity=sorted(os.sched_getaffinity(0)),
  environment={k:os.environ.get(k) for k in ('JAX_PLATFORMS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
  encrypted_execution_performed=False,key_generation_performed=False)
 (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:report[k] for k in ('passed','returncode','source_files','source_files_unchanged','source_archive_byte_verified','elapsed_s')}))
 return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
