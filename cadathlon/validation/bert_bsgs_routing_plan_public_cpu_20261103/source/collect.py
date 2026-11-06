"""Source-bound public NumPy slot algebra; no encrypted execution."""
from pathlib import Path
import argparse,hashlib,io,json,os,signal,subprocess,sys,tarfile,time
import xml.etree.ElementTree as ET
import evidence as fixture
from cleanup import cleanup_signals,termination_signals
from terminal import terminal_scope

ROOT=Path(__file__).resolve().parent
def digest(raw):return hashlib.sha256(raw).hexdigest()
def sources():return {n:p.read_bytes() for n,p in fixture.source_paths().items()}
def hashes(raw):return {n:digest(v) for n,v in raw.items()}
def write(path,value):fixture.write_json(path,value)

def run(output):
    output=Path(output)
    if output.parent.resolve()!=ROOT:raise ValueError('fresh direct lab child required')
    output.mkdir(exist_ok=False);start=time.monotonic();primary=None;failure=None;child=None;snapshot=None;before={}
    report=dict(passed=False,scope='Public NumPy BSGS slot algebra and operation counts; no HE or measured performance.',
        encrypted_execution=False,wrapper_pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)))
    def attempt(label,fn):
        nonlocal failure
        try:return fn()
        except BaseException as error:
            report.setdefault('cleanup_errors',[]).append(dict(operation=label,type=type(error).__name__))
            if primary is not None:primary.add_note(label+': '+repr(error))
            elif failure is None:failure=error
            elif not isinstance(error,Exception) and isinstance(failure,Exception):
                error.add_note('earlier cleanup: '+repr(failure));failure=error
            else:failure.add_note(label+': '+repr(error))
    try:
        write(output/'report.json',report);raw=sources();before=hashes(raw)
        with tarfile.open(output/'sources.tar.gz','w:gz') as archive:
            for name,data in raw.items():
                item=tarfile.TarInfo(name);item.size=len(data);archive.addfile(item,io.BytesIO(data))
        fixture.verify_archive(output/'sources.tar.gz',before)
        snapshot=output/'snapshot';snapshot.mkdir()
        for name,data in raw.items():
            path=snapshot/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        report.update(source_hashes_before=before,source_archive_sha256=digest((output/'sources.tar.gz').read_bytes()),source_count=len(before))
        env=dict(os.environ,JAX_PLATFORMS='cpu',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
            PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
        command=[sys.executable,'-m','pytest','-q','test_plan.py','--junitxml='+str(output/'tests.xml')]
        with (output/'tests.log').open('wb') as stream:
            child=subprocess.Popen(command,cwd=snapshot,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            report.update(child_pid=child.pid,command=command);write(output/'report.json',report)
            code=child.wait();report.update(returncode=code,process_reaped=True)
        if code!=0:raise RuntimeError('public routing proof failed; evidence retained')
        suites=ET.parse(output/'tests.xml').getroot().iter('testsuite')
        rows=list(suites);counts={n:sum(int(s.get(n,0)) for s in rows) for n in ('tests','failures','errors','skipped')}
        if counts!=dict(tests=392,failures=0,errors=0,skipped=0):raise ValueError('all392 public routing checks required')
        import plan
        public_plan=plan.public_plan();write(output/'public_plan.json',public_plan)
        report.update(operation_complete=True,test_counts=counts,full_slot_comparisons=768,compared_slots=768*65536)
    except BaseException as error:
        primary=error;report['error']=dict(type=type(error).__name__,message=str(error))
    finally:
        with terminal_scope(report,output/'report.json',write,cleanup_signals,lambda:(primary,failure)):
            if child is not None and child.poll() is None:
                attempt('stop owned preflight child',lambda:os.killpg(child.pid,signal.SIGKILL))
                attempt('reap owned preflight child',child.wait)
            report['process_reaped']=child is None or child.poll() is not None
            after=attempt('source final readback',lambda:hashes(sources()))
            copied=attempt('snapshot final readback',lambda:{n:digest((snapshot/n).read_bytes()) for n in before}) if snapshot else None
            verified=attempt('archive final readback',lambda:fixture.verify_archive(output/'sources.tar.gz',before)) if before else False
            report.update(source_hashes_after=after,snapshot_hashes=copied,source_files_unchanged=bool(before) and before==after==copied,
                source_archive_verified=verified is True,cleanup_completed=failure is None and report['process_reaped'],elapsed_s=time.monotonic()-start)
            report['passed']=bool(report.get('operation_complete') and primary is None and failure is None
                and report['source_files_unchanged'] and report['source_archive_verified'] and report['cleanup_completed'])
            write(output/'report.json',report)
            if primary is not None:raise primary
            if failure is not None:raise failure
    print(json.dumps({n:report[n] for n in ('passed','source_count','test_counts','elapsed_s')}));return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    with termination_signals():report=run(parser.parse_args().output)
    return 0 if report['passed'] is True and report['terminal_scope_completed'] is True else 1
if __name__=='__main__':raise SystemExit(main())
