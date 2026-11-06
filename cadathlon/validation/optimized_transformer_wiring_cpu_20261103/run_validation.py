"""Isolated optional model-control wiring, import and lifecycle regression run."""
import hashlib,json,os,subprocess,time,sys
from pathlib import Path
ROOT=Path('/tmp/optimized-transformer-wiring-20261103/checkout');OUT=ROOT.parent
files=[p for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.py','.json','.in','.out') and not {'__pycache__','.pytest_cache'}.intersection(p.parts)]
files += [Path(__file__)]
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
start=hashes();started=time.perf_counter()
tests=['test_optimized_transformer_wiring.py','test_dispatch_cache_policy.py','test_encrypted_reference_lifecycle.py','test_bounded_reference_preflight.py','test_bert_staging.py','test_package.py','test_encrypted_refresh.py','test_packed_transformer.py','test_bounded_transformer_plan.py','test_packed_qwen.py','test_scoped_ntt.py','test_scoped_mul_ntt.py']
cmd=[sys.executable,'-m','pytest','-q',*[str(Path('cadathlon/tests')/n) for n in tests],'--junitxml='+str(OUT/'tests.xml')]
with (OUT/'tests.log').open('w') as log:r=subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT)
end=hashes();report={'scope':__doc__,'passed':r.returncode==0 and start==end,'returncode':r.returncode,'elapsed_s':time.perf_counter()-started,'command':cmd,'source_hashes_start':start,'source_hashes_end':end,'source_files_unchanged':start==end,'full_model_evaluated':False,'accelerator_evaluated':False,'ring_profile_changed':False,'bootstrap_stage_clears_changed':False}
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if not k.startswith('source_hashes')},indent=2),flush=True)
raise SystemExit(0 if report['passed'] else 1)
