from pathlib import Path
import hashlib,json,os,subprocess,sys,time
HERE=Path(__file__).resolve().parent;ROOT=HERE/'checkout'
paths=[Path(__file__),*sorted((ROOT/'jaxite_word').rglob('*.py')),*sorted((ROOT/'cadathlon/common').rglob('*.py')),*sorted((ROOT/'cadathlon/tests').glob('*.py'))]
hashes=lambda:{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
before=hashes();started=time.monotonic();args=[sys.executable,'-m','pytest','-q','cadathlon/tests/test_dispatch_cache_policy.py','cadathlon/tests/test_encrypted_polynomial.py','cadathlon/tests/test_encrypted_rotation.py','cadathlon/tests/test_encrypted_reference_lifecycle.py']
env=dict(os.environ,PYTHONPATH=str(ROOT)+':'+str(ROOT/'cadathlon/common'))
with (HERE/'tests.log').open('w') as log:run=subprocess.run(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
after=hashes();report={'passed':run.returncode==0 and before==after,'returncode':run.returncode,'source_files_unchanged':before==after,'source_hashes_before':before,'source_hashes_after':after,'elapsed_s':time.monotonic()-started,'command':args,'affinity':sorted(os.sched_getaffinity(0)),'environment':{name:env.get(name) for name in ('JAX_PLATFORMS','XLA_FLAGS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','PYTHONPATH')}}
(HERE/'tests.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if not k.startswith('source_hashes')}));raise SystemExit(0 if report['passed'] else 1)
