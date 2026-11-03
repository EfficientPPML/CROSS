from pathlib import Path
import hashlib,io,json,os,subprocess,sys,tarfile,time
P=Path(__file__).resolve().parent;out=P/'preflight-v1';out.mkdir(exist_ok=False);start=time.perf_counter()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();names=['domain_rule.py','test_domain_rule.py','proposal.json','domain_preview.json','README.md','preflight.py'];before={n:sha(P/n) for n in names}
with tarfile.open(out/'sources.tar.gz','w:gz') as t:
 for n in names:b=(P/n).read_bytes();i=tarfile.TarInfo(n);i.size=len(b);t.addfile(i,io.BytesIO(b))
with (out/'tests.log').open('w') as f:r=subprocess.run([sys.executable,'-m','pytest','-q','test_domain_rule.py'],cwd=P,stdout=f,stderr=subprocess.STDOUT)
after={n:sha(P/n) for n in names};report={'passed':r.returncode==0 and before==after,'test_exit_code':r.returncode,'source_hashes_before':before,'source_hashes_after':after,'source_files_unchanged':before==after,'source_archive_sha256':sha(out/'sources.tar.gz'),'elapsed_s':time.perf_counter()-start,'cpu_affinity':sorted(os.sched_getaffinity(0)),'coefficient_selection':False,'model_evaluated':False,'scope':'Exact public domain/cost rule tests only; eight final certificates pending.'}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'passed':report['passed'],'sources':len(names),'source_archive_sha256':report['source_archive_sha256']}));raise SystemExit(0 if report['passed'] else 1)
