from pathlib import Path
import hashlib,json,os,subprocess,sys,time
root=Path('/tmp/transformer-collector-ownedkeys-20261103')
names=['cadathlon/validation/check_transformer_suite.py','cadathlon/validation/run_bounded_qwen2_proof.py','cadathlon/tests/test_transformer_suite.py','cadathlon/tests/test_transformer_owned_keys.py','cadathlon/tests/test_bounded_qwen2_proof.py']
hashes=lambda:{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
before=hashes();started=time.monotonic();args=[sys.executable,'-m','pytest','-q',*names[2:]]
with (root/'tests_ownership_final.log').open('w') as log:run=subprocess.run(args,cwd=root,stdout=log,stderr=subprocess.STDOUT)
after=hashes();record={'passed':run.returncode==0 and before==after,'returncode':run.returncode,'source_hashes_before':before,'source_hashes_after':after,'source_files_unchanged':before==after,'elapsed_s':time.monotonic()-started,'command':args,'cwd':str(root),'affinity':sorted(os.sched_getaffinity(0)),'environment':{name:os.environ.get(name) for name in ('JAX_PLATFORMS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}}
(root/'ownership_v2_report.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if not k.startswith('source_hashes')}));raise SystemExit(0 if record['passed'] else 1)
