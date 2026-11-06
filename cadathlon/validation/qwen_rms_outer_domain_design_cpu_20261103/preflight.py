from pathlib import Path
import hashlib,io,json,os,subprocess,sys,tarfile,time
from plan import make_plan
P=Path(__file__).resolve().parent
files=['bounds.py','test_bounds.py','PLAN.md','plan.py','config.json','model_pins.json','preflight.py']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out=P/'preflight-v2';out.mkdir(exist_ok=False);start=time.perf_counter();before={n:sha(P/n) for n in files}
with tarfile.open(out/'sources.tar.gz','w:gz') as t:
 for n in files:
  b=(P/n).read_bytes();m=tarfile.TarInfo(n);m.size=len(b);t.addfile(m,io.BytesIO(b))
with (out/'tests.log').open('w') as f:run=subprocess.run([sys.executable,'-m','pytest','-q','test_bounds.py'],cwd=P,stdout=f,stderr=subprocess.STDOUT)
plan=make_plan();after={n:sha(P/n) for n in files}
r=dict(passed=run.returncode==0 and before==after,test_exit_code=run.returncode,source_hashes_before=before,source_hashes_after=after,source_files_unchanged=before==after,source_archive_sha256=sha(out/'sources.tar.gz'),cpu_affinity=sorted(os.sched_getaffinity(0)),elapsed_s=time.perf_counter()-start,plan=plan)
(out/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps(r,indent=2));raise SystemExit(0 if r['passed'] else 1)
