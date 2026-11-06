"""Source-bound public-only planner/tests; no model or HE execution."""
from pathlib import Path
import hashlib,io,json,subprocess,sys,tarfile,time,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def sources():
    return [*sorted((P/'inputs').iterdir()),*[P/n for n in ('input_sources.json','plan.py','test_plan.py','README.md','collect_public.py','verify.py')]]
if __name__=='__main__':
    out=P/'public-v1';out.mkdir(exist_ok=False);paths=sources();before={str(p.relative_to(P)):sha(p) for p in paths};start=time.monotonic()
    with tarfile.open(out/'sources.tar.gz','w:gz') as t:
        for p in paths:
            b=p.read_bytes();i=tarfile.TarInfo(str(p.relative_to(P)));i.size=len(b);t.addfile(i,io.BytesIO(b))
    with (out/'tests.log').open('w') as f:r=subprocess.run([sys.executable,'-m','pytest','-q','test_plan.py','--rootdir=.',f'--junitxml={out}/tests.xml'],cwd=P,stdout=f,stderr=subprocess.STDOUT)
    with (out/'plan.log').open('w') as f:p=subprocess.run([sys.executable,'plan.py'],cwd=P,stdout=f,stderr=subprocess.STDOUT)
    after={str(p.relative_to(P)):sha(p) for p in paths}
    with tarfile.open(out/'sources.tar.gz') as t:
        members=t.getmembers();assert len(members)==len({m.name for m in members})==len(before)
        archived={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in members if m.isfile()}
    suites=ET.parse(out/'tests.xml').getroot().findall('testsuite');tests={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','errors','failures','skipped')}
    report=dict(passed=r.returncode==p.returncode==0 and before==after==archived,test_exit_code=r.returncode,plan_exit_code=p.returncode,tests=tests,
        source_hashes_before=before,source_hashes_after=after,source_archive_sha256=sha(out/'sources.tar.gz'),elapsed_s=time.monotonic()-start,
        model_executed=False,encrypted_execution=False,validation_outcomes_read=False,scope='Public manifest/profile/gamma report and exact-real/level/resource tests only.')
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    import shutil
    shutil.copy2(P/'report.json',out/'plan.json')
    print(json.dumps(dict(passed=report['passed'],tests=tests,sources=len(before),archive=report['source_archive_sha256'])))
    raise SystemExit(0 if report['passed'] else 1)
