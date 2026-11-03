"""Collect one cold packed bounded Qwen2 TPU proof after a device lease.

The normal Qwen runner performs encryption, keyless evaluation, decryption and
its unchanged numerical/security gates. This collector reuses the public-suite
process and artifact checks. It does not impose a healthy-work deadline, claim
all six fixtures, evaluate the pretrained model, or calibrate scoring.
"""
from __future__ import annotations
import argparse
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import time
from contextlib import contextmanager

import check_transformer_suite as suite

ROOT = Path(__file__).resolve().parents[2]


def paths():
    problem = ROOT/'cadathlon/agent_pset/p4_qwen'
    return problem, problem/'src/qwen_bounded_reference.py', problem/'src/runner.py'


def manifest():
    problem, implementation, _ = paths()
    result = suite.source_manifest(problem, implementation)
    result[str(Path(__file__).resolve())] = suite.sha256(Path(__file__).resolve())
    return dict(sorted(result.items()))


def archive_sources(destination, sources):
    """Archive exactly the bytes hashed before child launch; detect read races."""
    members = {}
    with tarfile.open(destination, 'w:gz') as archive:
        for source, expected in sources.items():
            path = Path(source)
            payload = path.read_bytes()
            import hashlib
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError(f'source changed during snapshot: {path}')
            name = ('repository/'+str(path.relative_to(ROOT)) if path.is_relative_to(ROOT)
                    else 'external/'+expected[:16]+'-'+path.name)
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(payload), 0o644
            archive.addfile(info, io.BytesIO(payload))
            members[source] = name
    return members


def filesystem(path):
    usage = shutil.disk_usage(path)
    return dict(path=str(path), total_bytes=usage.total, used_bytes=usage.used,
                free_bytes=usage.free, observed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))


@contextmanager
def execution_environment(key_directory):
    selected = {'CROSS_PUBLIC_KEY_DIRECTORY':str(key_directory),
                'CROSS_LEVEL_PARAMS_MAX':'1', 'CROSS_BARRETT_NTT_CONTEXT_CACHE_CAPACITY':'2',
                'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1'}
    previous = {key:os.environ.get(key) for key in selected}
    try:
        os.environ.update(selected)
        yield selected
    finally:
        for key, value in previous.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value


def validate_case(record, log, implementation):
    if record.get('returncode') != 0 or record.get('error'):
        raise ValueError(record.get('error','Qwen2 child did not exit successfully'))
    result = record.get('result')
    suite.validate_result(result, model='qwen', case='test2.in', implementation=implementation,
                          warmups=0, repeats=1)
    if result.get('implementation_status') != 'experimental-unvalidated':
        raise ValueError('experimental implementation identity is missing')
    kinds = result.get('requested_device_kinds')
    if (not isinstance(kinds,list) or len(kinds)!=1 or not isinstance(kinds[0],str)
            or 'TPU' not in kinds[0]):
        raise ValueError('actual selected TPU device kind is missing')
    process_time = suite.number(record.get('process_elapsed_s'),'process_elapsed_s',positive=True)
    if process_time+1e-5 < result['total_s']:
        raise ValueError('parent elapsed time is shorter than runner total')
    progress = {'started':log.count('bounded homomorphic refresh starting'),
                'completed':log.count('bounded homomorphic refresh complete')}
    if progress != {'started':3, 'completed':3}:
        raise ValueError('Qwen2 requires exactly three completed bounded refreshes')
    return progress


def run(args):
    args.output = args.output.resolve()
    args.key_directory = args.key_directory.resolve(strict=True)
    args.output.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    record = {'scope':'One cold encrypted packed bounded public Qwen test2; not a six-case sweep or pretrained model.',
              'model':'qwen','selected_cases':['test2.in'],'implementation':str(paths()[1]),
              'passed':False,'complete':False,'all_six_passed':False,'full_model_evaluated':False,
              'scoring_calibrated':False,'warmups':0,'repeats':1,'case_timeout_s':None,
              'cleanup_grace_s':args.cleanup_grace_s,'expected_refresh_count':3,'process_reaped':False,
              'public_key_store_removed':False,'software':suite.software_versions(),
              'wrapper_pid':os.getpid(),'cpu_affinity':sorted(os.sched_getaffinity(0)),
              'source_scope':'Repository core/demos/common/problem Python, six public input/output fixtures, collector and wrapper; installed libraries identified by version.',
              'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
              'public_key_filesystem_before':filesystem(args.key_directory)}
    caught = None
    owned_keys = None
    child = None
    case_directory = args.output/'test2'
    case_directory.mkdir()
    problem, implementation, runner = paths()
    try:
        record['source_hashes_start'] = manifest()
        source_archive = args.output/'sources.tar.gz'
        record['archive_members'] = archive_sources(source_archive,record['source_hashes_start'])
        record['source_archive_sha256'] = suite.sha256(source_archive)
        owned_keys = Path(tempfile.mkdtemp(prefix='cross-bounded-qwen2-',dir=args.key_directory))
        record['owned_public_key_directory'] = str(owned_keys)
        command = suite.runner_command(runner, ['--case',str(problem/'test/test2.in'),
            '--implementation',str(implementation),'--require-tpu','--warmups','0','--repeats','1'])
        record['command'] = command
        with execution_environment(owned_keys) as selected:
            record['environment'] = {key:os.environ.get(key) for key in suite.ENVIRONMENT}
            record['child_environment_overrides'] = suite.child_environment_overrides()
            suite.write_json(args.output/'report.json',record)
            print(f'Starting bounded Qwen2; wrapper PID {os.getpid()}; raw output {case_directory}',flush=True)
            child = suite.run_case(command,case_directory,timeout=None,
                                   cleanup_grace_s=args.cleanup_grace_s)
        record['case'] = child
        record['refresh_progress'] = validate_case(child,(case_directory/'stderr.log').read_text(),implementation)
        record['case_gate_passed'] = True
    except BaseException as error:
        caught = error
        record['error'] = f'{type(error).__name__}: {error}'
    finally:
        if child is None and (case_directory/'process.json').is_file():
            try:
                child = json.loads((case_directory/'process.json').read_text())
                record['case'] = child
            except (OSError,ValueError) as error:
                record['process_read_error'] = str(error)
        pid = child.get('pid') if isinstance(child,dict) else None
        record['process_reaped'] = (type(pid) is int and pid>1 and not suite.process_group_alive(pid))
        # run_case owns/reaps its process group before returning or raising.
        # Remove only this wrapper's key directory, never a shared parent.
        if owned_keys is not None:
            try:
                if pid is not None and not record['process_reaped']:
                    raise RuntimeError('owned process group is still present; key files retained')
                shutil.rmtree(owned_keys)
                record['public_key_store_removed'] = not owned_keys.exists()
            except BaseException as error:
                record['cleanup_error'] = f'{type(error).__name__}: {error}'
                if caught is None:caught=error
        try:
            record['source_hashes_end'] = manifest()
            record['source_files_unchanged'] = record.get('source_hashes_start')==record['source_hashes_end']
        except BaseException as error:
            record['source_files_unchanged'] = False
            record['source_check_error'] = str(error)
            if caught is None:caught=error
        if not record['source_files_unchanged']:
            record.setdefault('error','source or wrapper changed during proof')
        try:
            record['public_key_filesystem_after'] = filesystem(args.key_directory)
        except BaseException as error:
            record['filesystem_check_error'] = str(error)
            if caught is None:caught=error
        record['elapsed_s'] = time.monotonic()-started
        record['passed'] = (caught is None and record.get('case_gate_passed') is True
            and record['process_reaped'] and record['public_key_store_removed']
            and record['source_files_unchanged'])
        suite.write_json(args.output/'report.json',record)
    print(json.dumps({key:record[key] for key in ('passed','elapsed_s','process_reaped','public_key_store_removed')}
                     | {'error':record.get('error'),'report':str(args.output/'report.json')}),flush=True)
    if caught is not None and not isinstance(caught,Exception):raise caught
    return 0 if record['passed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='new proof directory; existing output is rejected')
    parser.add_argument('--key-directory',type=Path,default=Path('/dev/shm'))
    parser.add_argument('--cleanup-grace-s',type=float,default=30.,help='grace after explicit cancellation; not a work deadline')
    args=parser.parse_args()
    if not math.isfinite(args.cleanup_grace_s) or args.cleanup_grace_s<=0:
        parser.error('cleanup grace must be positive and finite')
    with suite.termination_signals():
        return run(args)


if __name__=='__main__':
    raise SystemExit(main())
