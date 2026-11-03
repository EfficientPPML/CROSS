from pathlib import Path
import hashlib,io,json,os,subprocess,sys,tarfile,time
P=Path(__file__).resolve().parent
names=['public_bound.py','test_public_bound.py','proposal.json','fresh_validation_proposal.json','model_pins.json','README.md','preflight.py']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out=P/'public-tests-v1';out.mkdir(exist_ok=False);before={n:sha(P/n) for n in names};start=time.perf_counter()
with tarfile.open(out/'sources.tar.gz','w:gz') as t:
 for n in names:
  b=(P/n).read_bytes();i=tarfile.TarInfo(n);i.size=len(b);t.addfile(i,io.BytesIO(b))
with (out/'tests.log').open('w') as f:r=subprocess.run([sys.executable,'-m','pytest','-q','test_public_bound.py'],cwd=P,stdout=f,stderr=subprocess.STDOUT)
after={n:sha(P/n) for n in names}
report={'passed':r.returncode==0 and before==after,'test_exit_code':r.returncode,'source_hashes_before':before,'source_hashes_after':after,'source_archive_sha256':sha(out/'sources.tar.gz'),'source_files_unchanged':before==after,'elapsed_s':time.perf_counter()-start,'cpu_affinity':sorted(os.sched_getaffinity(0)),'model_evaluated':False,'checkpoint_weights_read':False,'new_validation_tokenized':False,'coefficient_selection':False,'scope':'Synthetic exact-real public-bound unit checks and design/fixture freeze only.'}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));raise SystemExit(0 if report['passed'] else 1)
