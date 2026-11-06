"""Freeze and exercise mocked five-handler integration; never execute HE."""
from pathlib import Path
from contextlib import contextmanager
import argparse
import hashlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent
TRANSPORT=Path('/tmp/bert-full-model-transport-20261103')
TESTS=('test_file_stage_contract.py','test_file_stage_io.py','test_full_model_engine.py',
       'test_request_lifecycle.py','test_file_stages.py','test_file_stage_collector.py')
LOCAL=('file_stages.py','file_stage_contract.py','file_stage_io.py','request_lifecycle.py',
       'full_model_engine.py','client_inputs.py','client_keys.py','key_contexts.py',
       'server_public_model.py','response_scale.py','collect_file_stages.py',*TESTS,
       'server_inputs/p60_profile_and_refresh.json','server_inputs/candidate_manifest.json',
       'server_inputs/twelve_layer_plan.json')


def sources():
    paths={name:HERE/name for name in LOCAL}
    paths.update({name:TRANSPORT/name for name in ('transport.py','transport_contract.py','bundle_io.py','p60_profile_and_refresh.json')})
    return {name:path.read_bytes() for name,path in paths.items()}


def hashes(raw):return {n:hashlib.sha256(v).hexdigest() for n,v in raw.items()}


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def verify_archive(path,expected):
    with tarfile.open(path) as archive:
        members=archive.getmembers()
        return (len(members)==len({m.name for m in members})==len(expected)
                and all(m.isfile() for m in members)
                and {m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in members}==expected)


@contextmanager
def cleanup_signals(restore=None):
    numbers=(signal.SIGINT,signal.SIGTERM,signal.SIGHUP)
    mask=signal.pthread_sigmask(signal.SIG_BLOCK,numbers);old={}
    try:
        for number in numbers:old[number]=signal.signal(number,signal.SIG_IGN)
        signal.pthread_sigmask(signal.SIG_SETMASK,mask)
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_BLOCK,numbers)
        for number,handler in old.items():signal.signal(number,(restore or {}).get(number,handler))
        signal.pthread_sigmask(signal.SIG_SETMASK,mask)


def group_alive(pid):
    try:os.killpg(pid,0);return True
    except ProcessLookupError:return False


def reap(child):
    if child is None:return dict(reaped=True,group_gone=True)
    leftover=child.poll() is not None and group_alive(child.pid)
    if group_alive(child.pid):
        try:os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:pass
    child.wait(timeout=30)
    until=time.monotonic()+5
    while group_alive(child.pid) and time.monotonic()<until:time.sleep(.02)
    if group_alive(child.pid):raise RuntimeError('owned test process group survives cleanup')
    if leftover:raise RuntimeError('owned descendant outlived normal leader; group cleaned')
    return dict(pid=child.pid,returncode=child.returncode,reaped=True,group_gone=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve();output.mkdir(mode=0o700,parents=False,exist_ok=False)
    start=time.perf_counter();child=work=None;primary=cancel=None;cleanup=[];before={};handlers={}
    report=dict(passed=False,scope=__doc__,actual_key_generation=False,encrypted_execution=False,
                joined_numerical_model_evaluated=False,process_id=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)))
    def interrupted(number,frame):raise KeyboardInterrupt(f'collector interrupted by signal{number}')
    try:
        write(output/'report.json',report)
        for number in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
            handlers[number]=signal.signal(number,interrupted)
        from client_keys import _OwnedDirectory
        raw=sources();before=hashes(raw);snapshot=output/'snapshot';snapshot.mkdir()
        with tarfile.open(output/'sources.tar.gz','w:gz') as archive:
            for name,data in raw.items():
                target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
                info=tarfile.TarInfo(name);info.size=len(data);archive.addfile(info,io.BytesIO(data))
        if not verify_archive(output/'sources.tar.gz',before):raise ValueError('startup archive differs')
        work=_OwnedDirectory(output/'test-work')
        env=dict(os.environ,PYTHONPATH=str(snapshot),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with (output/'tests.log').open('wb') as log:
            child=subprocess.Popen([sys.executable,'-m','pytest','-q',*TESTS,
                '--basetemp',str(work.path/'pytest'),'--junitxml',str(output/'tests.xml')],
                cwd=snapshot,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            report['test_pid']=child.pid;report['test_returncode']=child.wait();report['test_child_reaped']=True
        suites=ET.parse(output/'tests.xml').getroot()
        counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
        report['tests']=counts
        report['passed']=report['test_returncode']==0 and counts==dict(tests=112,failures=0,errors=0,skipped=0)
    except BaseException as error:primary=error;report['error']=repr(error);raise
    finally:
        with cleanup_signals(handlers):
            def attempt(label,operation):
                nonlocal cancel
                try:return operation()
                except BaseException as error:
                    cleanup.append(label+': '+repr(error))
                    if primary is not None:primary.add_note(cleanup[-1])
                    elif not isinstance(error,Exception) and cancel is None:cancel=error
            process=attempt('owned test process group reap',lambda:reap(child))
            if work is not None:
                if process and process['group_gone']:
                    attempt('owned test tree removal',work.remove)
                report['owned_test_tree_removed']=not work.path.exists()
                attempt('owned test handles close',work.close)
                report['owned_test_handles_closed']=work.fd is None and work.parent_fd is None
            after=attempt('current source readback',lambda:hashes(sources()))
            snapshot_hashes=attempt('snapshot readback',lambda:{n:hashlib.sha256((output/'snapshot'/n).read_bytes()).hexdigest() for n in before})
            archived=attempt('archive readback',lambda:verify_archive(output/'sources.tar.gz',before)) if before else False
            archive_sha=attempt('archive digest',lambda:hashlib.sha256((output/'sources.tar.gz').read_bytes()).hexdigest()) if before else None
            report.update(source_hashes_before=before,source_hashes_after=after,snapshot_hashes_after=snapshot_hashes,
                source_files_unchanged=bool(before) and before==after==snapshot_hashes,source_archive_verified=archived is True,
                cleanup_errors=cleanup,elapsed_s=time.perf_counter()-start,process_cleanup=process,source_archive_sha256=archive_sha)
            report['passed']=bool(report['passed'] and report['source_files_unchanged'] and archived
                and report.get('owned_test_tree_removed') and report.get('owned_test_handles_closed') and not cleanup and process and process['reaped'] and process['group_gone'])
            try:write(output/'report.json',report)
            except BaseException as error:
                keep=primary if primary is not None else cancel
                if keep is None:raise
                keep.add_note('terminal publication: '+repr(error))
            if primary is None and cancel is not None:raise cancel
    print(json.dumps({k:report[k] for k in ('passed','tests','elapsed_s','source_archive_sha256')}))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
