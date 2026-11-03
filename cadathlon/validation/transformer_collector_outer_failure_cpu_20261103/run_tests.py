"""CPU-only source-bound collector/lifecycle validation, no HE or TPU."""
import hashlib,io,json,os,platform,subprocess,sys,tarfile,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE/'checkout';OUT=HERE/'result'
OUT.mkdir(exist_ok=False)
def sources():
    paths=[p for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.py','.in','.out')]
    return {str(p.relative_to(HERE)):p.read_bytes() for p in sorted(paths)}|{'run_tests.py':Path(__file__).read_bytes()}
raw=sources();before={n:hashlib.sha256(v).hexdigest() for n,v in raw.items()}
with tarfile.open(OUT/'sources.tar.gz','w:gz') as archive:
    for name,data in raw.items():
        info=tarfile.TarInfo(name);info.size=len(data);archive.addfile(info,io.BytesIO(data))
command=[sys.executable,'-m','pytest','-q','cadathlon/tests/test_transformer_suite.py',
    'cadathlon/tests/test_transformer_owned_keys.py','cadathlon/tests/test_bounded_qwen2_proof.py',
    '--junitxml='+str(OUT/'junit.xml')]
started=time.monotonic()
with (OUT/'pytest.log').open('w') as log:
    completed=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
after={n:hashlib.sha256(v).hexdigest() for n,v in sources().items()}
report={'scope':__doc__,'passed':completed.returncode==0 and before==after,
    'command':command,'returncode':completed.returncode,'elapsed_s':time.monotonic()-started,
    'affinity':sorted(os.sched_getaffinity(0)),'python':platform.python_version(),
    'environment':{n:os.environ.get(n) for n in ('JAX_PLATFORMS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','PYTHONDONTWRITEBYTECODE')},
    'source_hashes_start':before,'source_hashes_end':after,'source_files_unchanged':before==after,
    'source_archive_sha256':hashlib.sha256((OUT/'sources.tar.gz').read_bytes()).hexdigest(),
    'base':'/tmp/runtime-followup-integration-20261103/checkout; archived 32-file composed candidate',
    'changes':['cadathlon/validation/check_transformer_suite.py','cadathlon/tests/test_transformer_suite.py']}
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if not k.startswith('source_hashes')},indent=2))
raise SystemExit(0 if report['passed'] else 1)
