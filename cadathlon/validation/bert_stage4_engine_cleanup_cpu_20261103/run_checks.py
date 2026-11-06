"""Source-bound mocked lifecycle checks only; no crypto or model execution."""
from pathlib import Path
import hashlib,io,json,os,signal,subprocess,sys,tarfile,time
ROOT=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def collect(output):
    output.mkdir(exist_ok=False);start=time.monotonic();child=None;primary=None
    files=[ROOT/'run_checks.py',ROOT/'source_origin.json']
    for part in ('before','candidate'):files+=sorted((ROOT/part).glob('*.py'))
    paths={str(p.relative_to(ROOT)):p for p in files}
    before={n:sha(p) for n,p in paths.items()}
    report=dict(passed=False,source_hashes_before=before,scope='Mocked engine cleanup and real process-local signals only; no model/crypto/forward.')
    write(output/'report.json',report)
    with tarfile.open(output/'sources.tar.gz','w:gz') as archive:
        for n,p in paths.items():archive.add(p,arcname=n,recursive=False)
    try:
        command=[sys.executable,'-m','pytest','-q','--tb=short','test_full_model_engine.py','test_cleanup_regressions.py','--junitxml='+str(output/'tests.xml')]
        report['command']=command;report['cpu_affinity']=sorted(os.sched_getaffinity(0))
        with (output/'tests.log').open('wb') as log:
            child=subprocess.Popen(command,cwd=ROOT/'candidate',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            report['child_pid']=child.pid;report['test_exit_code']=child.wait()
        import xml.etree.ElementTree as ET
        suites=ET.parse(output/'tests.xml').getroot().findall('testsuite')
        report['tests']={n:sum(int(s.attrib.get(n,0)) for s in suites) for n in ('tests','failures','errors','skipped')}
        report['passed']=report['test_exit_code']==0 and report['tests']==dict(tests=21,failures=0,errors=0,skipped=0)
    except BaseException as error:primary=error;report['error']=repr(error);raise
    finally:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid,signal.SIGKILL);child.wait()
            except BaseException as error:
                report['passed']=False
                if primary is not None:primary.add_note('test child cleanup: '+repr(error))
                else:raise
        report['source_hashes_after']={n:sha(p) for n,p in paths.items()}
        report['sources_unchanged']=report['source_hashes_after']==before
        report['source_archive_sha256']=sha(output/'sources.tar.gz')
        with tarfile.open(output/'sources.tar.gz') as archive:
            members=archive.getmembers()
            report['archive_verified']=(len(members)==len(before)==len({m.name for m in members}) and all(m.isfile() for m in members)
                and {m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in members}==before)
        origins=json.loads((ROOT/'source_origin.json').read_text())
        report['original_frozen_sources_unchanged']=all(sha(Path(r['source']))==r['sha256'] for r in origins.values())
        report['child_reaped']=child is None or child.returncode is not None
        report['passed']=bool(report['passed'] and report['sources_unchanged'] and report['archive_verified'] and report['original_frozen_sources_unchanged'] and report['child_reaped'])
        report['elapsed_s']=time.monotonic()-start
        try:write(output/'report.json',report)
        except BaseException as error:
            if primary is None:raise
            primary.add_note('test report publication: '+repr(error))
    print(json.dumps({k:report[k] for k in ('passed','tests','source_archive_sha256','elapsed_s')}))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(collect(Path(sys.argv[1]).resolve()))
