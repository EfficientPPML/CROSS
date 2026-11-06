from pathlib import Path
import hashlib,io,json,os,subprocess,sys,tarfile,time
P=Path(__file__).resolve().parent;out=P/'preflight-v1';out.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths={str(p.relative_to(P)):p for p in [*P.glob('*.py'),*P.glob('*.json'),*(P/'inputs').glob('*.json')]}
before={n:sha(p) for n,p in paths.items()};started=time.perf_counter()
with tarfile.open(out/'sources.tar.gz','w:gz') as t:
 for n,p in paths.items():b=p.read_bytes();i=tarfile.TarInfo(n);i.size=len(b);t.addfile(i,io.BytesIO(b))
with (out/'tests.log').open('w') as f:r=subprocess.run([sys.executable,'-m','pytest','-q','test_public_bound.py','test_exact_dyadic.py'],cwd=P,stdout=f,stderr=subprocess.STDOUT)
after={n:sha(p) for n,p in paths.items()}
report={'passed':r.returncode==0 and before==after,'test_exit_code':r.returncode,'source_hashes_before':before,'source_hashes_after':after,'source_archive_sha256':sha(out/'sources.tar.gz'),'elapsed_s':time.perf_counter()-started,'cpu_affinity':sorted(os.sched_getaffinity(0)),'checkpoint_weights_read':False,'model_evaluated':False,'source_files_unchanged':before==after}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'passed':report['passed'],'source_count':len(before),'source_archive_sha256':report['source_archive_sha256']}));raise SystemExit(0 if report['passed'] else 1)
