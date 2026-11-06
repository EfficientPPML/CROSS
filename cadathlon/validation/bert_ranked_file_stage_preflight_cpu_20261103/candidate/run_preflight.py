"""Source-bound mocked/public validation only; deliberately no execution switch."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tarfile,time,xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent
TESTS=('test_ranked_case_gate.py','test_ranked_acceptance.py','test_ranked_handlers.py',
       'test_ranked_private_reference.py','test_ranked_private_io.py','test_ranked_stage_gate.py')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n')
def sources():
    origin=json.loads((HERE/'source_origin.json').read_text())
    paths={name:HERE/name for name in origin}
    paths.update({p.name:p for pattern in ('*.py','*.json','README.md') for p in HERE.glob(pattern)})
    return paths


def run():
    start=time.perf_counter();out=HERE/'source-bound-v1';out.mkdir(exist_ok=False)
    paths=sources();before={n:sha(p) for n,p in paths.items()}
    origin=json.loads((HERE/'source_origin.json').read_text())
    assert all(sha(Path(d['path']))==d['sha256']==before[n] for n,d in origin.items())
    with tarfile.open(out/'sources.tar.gz','w:gz') as tar:
        for n,p in sorted(paths.items()):tar.add(p,arcname=n,recursive=False)
    report=dict(passed=False,scope='Mocked ranked file-handler orchestration/public producer gates only; no actual private/key/CT/model reads or HE',
        pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)),source_hashes_before=before,
        source_archive_sha256=sha(out/'sources.tar.gz'),private_retained_bodies_read=False,
        actual_encrypted_batch=False,ranked_accuracy_or_timing_claim=False)
    write(out/'report.json',report)
    script="import json,sys,pytest; code=pytest.main(sys.argv[1:]); forbidden=sorted(n for n in ('jax','jaxite_word','torch','transformers') if n in sys.modules); print(json.dumps(dict(pytest_returncode=int(code),forbidden_backends_loaded=forbidden))); raise SystemExit(int(code) if not forbidden else 91)"
    with (out/'tests.log').open('wb') as log:
        result=subprocess.run([sys.executable,'-c',script,'-q',*TESTS,'--junitxml='+str(out/'tests.xml')],cwd=HERE,stdout=log,stderr=subprocess.STDOUT)
    report['test_returncode']=result.returncode
    xml=ET.parse(out/'tests.xml').getroot();suites=[xml] if xml.tag=='testsuite' else list(xml)
    counts={n:sum(int(s.attrib.get(n,0)) for s in suites) for n in ('tests','errors','failures','skipped')};report['test_counts']=counts
    after={n:sha(p) for n,p in sources().items()};report['source_hashes_after']=after
    report['source_files_unchanged']=before==after
    report['all_predecessor_copies_and_originals_unchanged']=all(sha(Path(d['path']))==d['sha256']==after[n] for n,d in origin.items())
    with tarfile.open(out/'sources.tar.gz') as tar:
        members=tar.getmembers();report['source_archive_verified']=(len(members)==len(before)==len({m.name for m in members})
            and all(m.isfile() for m in members) and {m.name:hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in members}==before)
    audit=json.loads((out/'tests.log').read_text().splitlines()[-1]);report['backend_audit']=audit
    report['elapsed_s']=time.perf_counter()-start
    report['passed']=(result.returncode==0 and counts==dict(tests=141,errors=0,failures=0,skipped=0)
        and before==after and report['all_predecessor_copies_and_originals_unchanged'] and report['source_archive_verified']
        and audit==dict(pytest_returncode=0,forbidden_backends_loaded=[]))
    write(out/'report.json',report)
    print(json.dumps({k:report[k] for k in ('passed','test_counts','elapsed_s','source_archive_sha256')}))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(run())
