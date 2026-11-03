"""Freeze proposal bytes before tokenizer-only and public-certificate checks."""
from pathlib import Path
import hashlib,io,json,os,subprocess,sys,tarfile,time
HERE=Path(__file__).resolve().parent
NAMES=['author_corpus.py','check_inputs.py','reciprocal_schedule.py','test_protocol.py','run_preflight.py','corpus.json','protocol.json','model_pins.json','reciprocal_schedule.json','validation_inputs_only.json','README.md']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=HERE/'final-preflight';out.mkdir(exist_ok=False)
    before={n:sha(HERE/n) for n in NAMES};start=time.perf_counter()
    with tarfile.open(out/'sources.tar.gz','w:gz') as archive:
        for n in NAMES:
            data=(HERE/n).read_bytes();item=tarfile.TarInfo(n);item.size=len(data);archive.addfile(item,io.BytesIO(data))
    results=[]
    for name,args in [('tokenizer',['check_inputs.py']),('public_tests',['-m','unittest','-v','test_protocol'])]:
        with (out/(name+'.log')).open('w') as f:r=subprocess.run([sys.executable,*args],cwd=HERE,stdout=f,stderr=subprocess.STDOUT,check=False)
        results.append({'check':name,'returncode':r.returncode})
    after={n:sha(HERE/n) for n in NAMES}
    report={'passed':all(x['returncode']==0 for x in results) and before==after,'checks':results,'source_hashes_before':before,'source_hashes_after':after,'source_files_unchanged':before==after,'source_archive_sha256':sha(out/'sources.tar.gz'),'selected_source_count':len(NAMES),'no_model_evaluation':True,'no_he_or_jax':True,'elapsed_s':time.perf_counter()-start,'cpu_affinity':sorted(os.sched_getaffinity(0))}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
