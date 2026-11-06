from pathlib import Path
import hashlib,json,os,subprocess,sys,time
HERE=Path(__file__).resolve().parent;ROOT=HERE/'checkout'
paths=[Path(__file__),HERE/'test_scoped_ntt.py',*sorted((ROOT/'jaxite_word').glob('*.py')),*sorted((ROOT/'cadathlon/common').glob('*.py')),*sorted((ROOT/'cadathlon/tests').glob('*.py'))]
hashes=lambda:{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
before=hashes();start=time.monotonic()
args=[sys.executable,'-m','pytest','-q',str(HERE/'test_scoped_ntt.py'),str(ROOT/'cadathlon/tests/test_encrypted_reference_lifecycle.py'),str(ROOT/'cadathlon/tests/test_encrypted_rotation.py')]
env=dict(os.environ,PYTHONPATH=str(ROOT)+':'+str(ROOT/'cadathlon/common'))
with (HERE/'combined_tests.log').open('w') as output:completed=subprocess.run(args,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
after=hashes();report={'passed':completed.returncode==0 and before==after,'returncode':completed.returncode,'source_files_unchanged':before==after,'source_hashes_before':before,'source_hashes_after':after,'seconds':time.monotonic()-start,'command':args,'affinity':sorted(os.sched_getaffinity(0)),'environment':{name:env.get(name) for name in ('JAX_PLATFORMS','XLA_FLAGS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','PYTHONPATH')}}
(HERE/'combined_tests.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if not k.startswith('source_hashes')}));raise SystemExit(0 if report['passed'] else 1)
