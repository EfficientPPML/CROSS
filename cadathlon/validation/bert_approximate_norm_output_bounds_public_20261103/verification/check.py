"""Source-bound tests and independent exact certificate/asset verification."""
from pathlib import Path
import hashlib,io,json,subprocess,sys,tarfile,time
HERE=Path(__file__).resolve().parent.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    sys.set_int_max_str_digits(100000);out=HERE/'verification/source-bound-v1';out.mkdir(exist_ok=False);started=time.monotonic()
    public=json.loads((HERE/'public-v2/report.json').read_text())
    paths=[HERE/n for n in public['source_hashes_before']]+[HERE/'verification/verify.py',Path(__file__).resolve()]
    before={str(p.relative_to(HERE)):sha(p) for p in paths}
    assert all(before[n]==h for n,h in public['source_hashes_before'].items())
    prerequisite={n:sha(HERE/n) for n in ('public-v2/report.json','public-v2/sources.tar.gz')}
    with tarfile.open(out/'sources.tar.gz','w:gz') as t:
        for p in paths:b=p.read_bytes();m=tarfile.TarInfo(str(p.relative_to(HERE)));m.size=len(b);t.addfile(m,io.BytesIO(b))
    report=dict(passed=False,source_hashes_before=before,public_prerequisite_sha256=prerequisite)
    try:
        cases=[('tests',['-m','pytest','-q','test_exact_bounds.py']),('independent',['verification/verify.py','public-v2','--checkpoint','/home/jianming_gatech/cross_validation_assets/bert/f150c1d609d1e50dd5e2e5408661cfac8339277c/model.safetensors'])]
        for name,args in cases:
            with (out/(name+'.log')).open('w') as stream:p=subprocess.run([sys.executable,*args],cwd=HERE,stdout=stream,stderr=subprocess.STDOUT)
            report[name+'_returncode']=p.returncode
            if p.returncode:raise RuntimeError(name+' failed')
        report['passed']=True
    finally:
        after={str(p.relative_to(HERE)):sha(p) for p in paths}
        with tarfile.open(out/'sources.tar.gz') as t:archived={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers()}
        report.update(source_hashes_after=after,source_files_unchanged=before==after==archived,source_archive_sha256=sha(out/'sources.tar.gz'),elapsed_s=time.monotonic()-started)
        report['passed']=report['passed'] and report['source_files_unchanged'] and prerequisite=={n:sha(HERE/n) for n in prerequisite}
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('passed','source_files_unchanged','elapsed_s')}))
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
