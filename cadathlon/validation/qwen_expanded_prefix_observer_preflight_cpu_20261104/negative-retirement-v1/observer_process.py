"""Owned observer-only collection and source-bound synthetic preflight."""
import hashlib
import importlib
import importlib.metadata
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from freeze_io import Output, cleanup_origin, mark_cleanup, finish, parse, read
import input_gate
from cleanup_origin import protect_body, choose

SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)


def accumulate(primary, error, phase, body_primary=None):
    if body_primary is not None:
        protect_body(body_primary)
    return choose(primary,error,phase,body_primary=body_primary)


@contextmanager
def signals(state=None, shield=False):
    """Single-fire outer cancellation; deferred cleanup cancellation.

    Main-thread masking narrows installation/restoration windows, not a
    process-wide atomicity claim with native threads. All originals are saved
    before setters, and masks are queried before mutating operations.
    """
    state = {} if state is None else state
    state['restored'] = False
    old = {}; errors = []; deferred = []; previous = None
    primary = body = None; fired = False
    def handler(number, frame):
        nonlocal fired
        if shield:
            deferred.append(number)
        elif not fired:
            fired = True
            raise KeyboardInterrupt('termination signal')
    def attempt(label,action):
        try:return action()
        except BaseException as error:
            errors.append((label,mark_cleanup(error)));return None
    try:
        previous = signal.pthread_sigmask(signal.SIG_BLOCK, ())
        signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)
        for number in SIGNALS:
            old[number] = signal.getsignal(number)
        for number in SIGNALS:
            signal.signal(number, handler)
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)
        yield
    except BaseException as error:
        primary = error
        body = None if cleanup_origin(error) else protect_body(error)
        raise
    finally:
        # Capture the current mask separately before a possibly mutating setter.
        restore_mask = attempt('capture restoration mask',lambda:signal.pthread_sigmask(signal.SIG_BLOCK, ()))
        attempt('block during signal restoration',lambda:signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS))
        restored = True
        for number, handler_before in old.items():
            try:signal.signal(number,handler_before)
            except BaseException as error:
                restored=False;errors.append(('restore signal',mark_cleanup(error)))
        target = previous if previous is not None else restore_mask
        if target is not None:
            try:signal.pthread_sigmask(signal.SIG_SETMASK,target)
            except BaseException as error:
                restored=False;errors.append(('restore mask',mark_cleanup(error)))
        else:restored=False
        if deferred:errors.append(('deferred cleanup cancellation',mark_cleanup(KeyboardInterrupt('signals deferred during cleanup'))))
        state['restored'] = restored and not errors
        pending=primary
        for label,error in errors:pending=accumulate(pending,error,label,body)
        if pending is not None and pending is not primary:raise pending


def source_end(root, expected):
    names = set()
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('terminal source symlink')
        if path.is_file():
            names.add(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise ValueError('terminal nonregular source')
    if names != set(expected):
        raise ValueError('terminal source inventory differs')
    after = {name: hashlib.sha256(read(root/name, limit=8*1024*1024)).hexdigest() for name in expected}
    if after != expected:
        raise ValueError('terminal source hashes differ')
    return after


def validate_paths(args, blobs, root):
    external = input_gate.inspect(blobs)
    protected = [root.resolve(), Path(external['model_root'])]
    protected += [Path(value) for value in external['protected_paths']]
    for name in ('manifest','grant'):
        value=getattr(args,name,None)
        if value:protected.append(Path(value).resolve())
    values=[]
    for name in ('output','private_output'):
        value=getattr(args,name,None)
        if value is None:continue
        path=Path(value)
        if not path.is_absolute() or path!=path.resolve() or '..' in path.parts:
            raise ValueError('canonical absolute output path required')
        if any(path==p or p in path.parents or path in p.parents for p in protected):
            raise ValueError('output overlaps protected source/input/model/grant')
        values.append(path)
    if len(values)==2 and (values[0]==values[1] or values[0] in values[1].parents or values[1] in values[0].parents):
        raise ValueError('public and private outputs must be disjoint')


def dependencies(blobs):
    expected=parse(blobs['dependencies.json'])
    if set(expected)!={'python','numpy','torch','transformers','safetensors','tokenizers','pytest'}:
        raise ValueError('Exact observer dependency metadata required')
    actual={name:sys.version if name=='python' else importlib.metadata.version(name) for name in expected}
    if actual!=expected:raise ValueError('installed dependency versions differ')
    return actual


def install_audit(blobs, allow_inputs, synthetic_root=None):
    external=input_gate.inspect(blobs)
    calibration=external['calibration']['path']
    model_root=external['model_root'];protected=[Path(p) for p in external['protected_paths']]
    allowed={calibration}|{model_root+'/'+name for name in external['model_assets']}
    denials=[]
    def audit(event,args):
        reason=None
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.path.abspath(os.fsdecode(args[0])));resolved=path.resolve()
            protected_body=(str(path) in allowed or any(path==p or p in path.parents or resolved==p or p in resolved.parents for p in protected)
                or str(path).startswith('/home/jianming_gatech/cross_validation_assets/')
                or str(resolved).startswith('/home/jianming_gatech/cross_validation_assets/'))
            synthetic=(not allow_inputs and synthetic_root is not None and
                Path(synthetic_root) in path.parents and Path(synthetic_root) in resolved.parents)
            sensitive=protected_body or path.suffix in ('.safetensors','.parquet','.npy','.npz') and not synthetic
            flags=args[2] if len(args)>2 and type(args[2])is int else 0
            writing=bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if sensitive and (not allow_inputs or str(path) not in allowed or path!=resolved or writing):
                reason='external calibration/validation/model body'
        elif event in ('socket.connect','socket.connect_ex','socket.bind','socket.getaddrinfo'):
            reason='network'
        if reason:
            denials.append(reason);raise PermissionError('observer audit denied '+reason)
    sys.addaudithook(audit)
    return denials


def grant_check(args,blobs):
    authorization=parse(blobs['execution_authorization.json'])
    if type(authorization)is not dict or set(authorization)!={'execute'} or authorization['execute']is not False:
        raise ValueError('preparation source remains disabled; separate exact grant required')
    external=input_gate.inspect(blobs)
    if not args.grant or not args.grant_sha256:raise PermissionError('separate reviewed observer-only launch grant required')
    grant=parse(read(args.grant,args.grant_sha256,limit=1<<20))
    evidence=os.path.abspath(args.output)
    if args.mode=='observer-worker':
        if Path(evidence).name!='worker':raise ValueError('owned collector worker output required')
        evidence=str(Path(evidence).parent)
    expected=dict(schema='qwen-expanded-prefix-observer-grant-v1',execute=True,scope='initial-native-exact3200-baseline-free64-only',
        source_manifest_sha256=args.manifest_sha256,protocol_sha256=external['producer']['protocol_sha256'],
        baseline_manifest_sha256=external['baseline_manifest_sha256'],producer=external['producer'],calibration=external['calibration'],
        model_pins_sha256=hashlib.sha256(blobs['public/model_pins.json']).hexdigest(),
        private_output=os.path.abspath(args.private_output or ''),evidence_output=evidence,
        model_forward=True,tokenizer_execution=False,coefficient_selection=False,validation_read=False,
        cpu_affinity=[112,113,114,115],torch_threads=4,torch_interop_threads=1)
    fields=set(expected)|{'preflight_report_sha256','independent_review_sha256'}
    if type(grant)is not dict or set(grant)!=fields or any(not input_gate.typed_equal(grant[k],v) for k,v in expected.items()):
        raise ValueError('exact observer-only grant binding differs')
    if any(not input_gate.is_digest(grant[k]) for k in ('preflight_report_sha256','independent_review_sha256')):
        raise ValueError('review/preflight digest required')
    if not args.private_output:raise ValueError('fresh private output required')
    if sorted(os.sched_getaffinity(0))!=[112,113,114,115]:raise ValueError('observer requires CPU112-115')
    return grant


def enable_model_imports():
    found=[]
    for finder in sys.meta_path:
        if type(finder).__name__=='FrozenImports' and hasattr(finder,'allow_model'):
            found.append(finder)
    if len(found)!=1 or found[0].allow_model is not False:
        raise ValueError('one freshly authenticated import gate required')
    found[0].allow_model=True


def supervise(output, report, action, cleanup, root, expected):
    """Close body resources and restore signals before final inode-bound publication."""
    start = time.monotonic()
    primary = body_primary = None
    state = {}
    cleanup_complete = True
    cleanup_ran = False
    report['source_files_unchanged'] = False
    def verify_source():
        report['source_hashes_end'] = source_end(root, expected)
        report['source_files_unchanged'] = True
    def cleanup_all():
        nonlocal primary, cleanup_complete, cleanup_ran
        cleanup_ran = True
        shield = signals(shield=True)
        entered = False
        try:
            shield.__enter__()
            entered = True
        except BaseException as error:
            cleanup_complete = False
            primary = accumulate(primary, error, 'cleanup shielding setup failure', body_primary)
        for operation in (*cleanup, verify_source, lambda: output.close()):
            try:
                operation()
            except BaseException as error:
                cleanup_complete = False
                primary = accumulate(primary, error, 'body cleanup failure', body_primary)
        if entered:
            try:
                shield.__exit__(None,None,None)
            except BaseException as error:
                cleanup_complete = False
                primary = accumulate(primary, error, 'cleanup shielding restoration failure', body_primary)
    try:
        with signals(state):
            try:
                action()
            except BaseException as error:
                primary = error
                body_primary = None if cleanup_origin(error) else protect_body(error)
                report.update(passed=False, error_type=type(error).__name__)
            finally:
                cleanup_all()
    except BaseException as error:
        report['passed'] = False
        if primary is None and not cleanup_origin(error):
            body_primary = protect_body(error)
        primary = accumulate(primary, error, 'outer signal scope failure', body_primary)
    if not cleanup_ran:
        cleanup_all()
    report.update(elapsed_s=time.monotonic()-start, cleanup_completed=cleanup_complete,
                  signal_restoration_completed=state.get('restored',False),
                  terminal_after_body_cleanup_and_signal_restoration_attempts=True,
                  acceptance_requires_independent_exit0_reap_and_exact_report_hash=True)
    if not cleanup_complete or not state.get('restored',False) or primary is not None:
        report['passed'] = False
    if report.get('passed') is not True:
        report['initial_selection_admissible'] = False
    publication = None
    try:
        publication = Output.reopen(output.path, output.owner)
        publication.json('report.json', report)
    except BaseException as error:
        primary = accumulate(primary, error, 'final publication failure', body_primary)
    finally:
        if publication is not None:
            try:
                publication.close()
            except BaseException as error:
                primary = accumulate(primary, error, 'final publication close failure', body_primary)
    if primary is not None:
        raise primary
    return 0 if report['passed'] else 1


def worker(args, expected, blobs, origins, root):
    validate_paths(args,blobs,root);grant_check(args,blobs)
    denials=install_audit(blobs,True)
    output=Output(args.output);private=[None];reader=[None]
    report=dict(passed=False,complete=False,pid=os.getpid(),source_hashes_start=expected,
        source_manifest_sha256=args.manifest_sha256,bootstrap_origins=origins,
        private_outputs_retained=True,external_inputs_never_removed=True,
        model_forward=False,coefficient_selection=False,tokenizer_execution=False,
        validation_inputs_read=False,grant_sha256=args.grant_sha256,audit_denials=denials,
        calibration_reauthenticated=False,private_directory_handle_closed=False)
    def progress(row):
        fields={'index','kind','sequence_length','stage','semantic_pass'}
        if type(row)is not dict or not set(row)<=fields or any(type(v)not in(str,int,bool) for v in row.values()):
            raise ValueError('public progress contains unsupported/private fields')
        import json
        print(json.dumps(row,sort_keys=True),flush=True)
    def action():
        report['dependencies']=dependencies(blobs)
        private[0]=Output(args.private_output);reader[0]=input_gate.Reader(blobs)
        report['private_output']=dict(path=private[0].path,identity=list(private[0].owner),mode='0700')
        enable_model_imports()
        import model_runtime
        report['model_forward']=True
        report.update(model_runtime.execute(blobs,reader[0],private[0],progress))
        report['tokenizers_module_imported']=any(n=='tokenizers' or n.startswith('tokenizers.') for n in sys.modules)
        report['tokenizer_policy']='transitive module import permitted; no tokenizer construction in pinned runtime and tokenizer assets denied'
        if denials:raise ValueError('forbidden access attempted')
    def recheck_input():
        if reader[0] is not None:
            reader[0].recheck()
            report['calibration_reauthenticated']=reader[0].loaded
    def close_private():
        if private[0] is not None:
            private[0].close()
            report['private_directory_handle_closed']=True
    return supervise(output,report,action,[recheck_input,close_private],root,expected)


def test_worker(args, expected, blobs, origins, root):
    validate_paths(args, blobs, root)
    denials = install_audit(blobs, False, synthetic_root=Path(args.output)/'synthetic')
    output = Output(args.output)
    report = dict(passed=False, complete=False, pid=os.getpid(), source_hashes_start=expected,
                  source_manifest_sha256=args.manifest_sha256, bootstrap_origins=origins,
                  scope='Source-bound synthetic observer preflight; no actual token/model bodies or backend construction.',
                  audit_denials=denials)
    def action():
        report['dependencies'] = dependencies(blobs)
        import pytest
        result = pytest.main(['-q', '-p', 'no:cacheprovider', '--import-mode=importlib',
                              '--basetemp='+str(Path(args.output)/'synthetic'),
                              '--junitxml='+str(Path(args.output)/'tests.xml')]
                             + [str(root/name) for name in expected if name.startswith('test_') and name.endswith('.py')])
        tree = ET.parse(Path(args.output)/'tests.xml')
        cases = tree.findall('.//testcase')
        report.update(tests=len(cases), failures=len(tree.findall('.//failure')),
                      errors=len(tree.findall('.//error')), skipped=len(tree.findall('.//skipped')),
                      pytest_returncode=int(result))
        installed = []
        for name in ('torch','transformers','jax','jaxlib','jaxite_word','tensorflow','tokenizers','safetensors'):
            try:
                importlib.import_module(name)
            except ImportError:
                installed.append(name)
            else:
                raise AssertionError('forbidden backend imported')
        report['installed_import_denials'] = installed
        expectations = parse(blobs['preflight_expectations.json'])
        report.update(complete=True, passed=result==0 and len(cases)==expectations['synthetic_tests']
                      and len(installed)==expectations['installed_import_denials'] and not report['skipped'] and not denials)
    return supervise(output, report, action, [], root, expected)


def group_absent(pid):
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return True
    return False


def retire(child, primary=None):
    if child is None:
        return
    # Healthy waits have no timeout. Cleanup begins only after body completion/error.
    child.poll()
    if not group_absent(child.pid):
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic()+30
        while not group_absent(child.pid) and time.monotonic()<deadline:
            child.poll()
            time.sleep(.02)
        if not group_absent(child.pid):
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    child.wait()
    if not group_absent(child.pid):
        raise RuntimeError('owned child group remains')


def child_environment():
    env=dict(os.environ)
    for name in tuple(env):
        if name.startswith(('JAX_','XLA_','TPU_')) or name in ('PYTHONPATH','PYTHONHOME','PYTHONOPTIMIZE'):
            env.pop(name,None)
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='')
    return env


def collect(args, raw, expected, blobs, origins, root):
    validate_paths(args, blobs, root)
    actual = args.mode == 'collect-observer'
    if actual:
        grant_check(args, blobs)
    denials = install_audit(blobs, False)
    output = Output(args.output)
    report = dict(passed=False, complete=False, audit_denials=denials, scope='Initial calibration-only observer' if actual else 'Public synthetic observer preflight',
                  source_hashes_start=expected, source_manifest_sha256=args.manifest_sha256,
                  bootstrap_origins=origins, pid=os.getpid(), actual_observer_stage=actual,
                  model_forward=False, coefficient_selection=False)
    child = [None]
    logs = []
    def action():
        report['dependencies'] = dependencies(blobs)
        output.write('source_manifest.json', raw)
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
            for name, data in sorted(blobs.items()):
                item = tarfile.TarInfo(name)
                item.size = len(data)
                item.mode = 0o600
                archive.addfile(item, io.BytesIO(data))
        report['source_archive'] = output.write('sources.tar.gz', buffer.getvalue())
        execution = Path(args.output)/'execution_source'
        execution.mkdir(mode=0o700)
        for name, data in sorted(blobs.items()):
            path = execution/name
            if path.parent != execution:
                path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd = os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC, 0o600)
            error = None
            try:
                view = memoryview(data)
                while view:
                    count = os.write(fd, view)
                    if count <= 0:
                        raise OSError('source copy made no progress')
                    view = view[count:]
            except BaseException as exc:
                error = exc
                raise
            finally:
                finish(error, lambda: os.close(fd))
        source_end(execution, expected)
        command = [sys.executable, '-I', '-B', str(execution/'entry.py'),
                   '--manifest', str(Path(args.output)/'source_manifest.json'),
                   '--manifest-sha256', args.manifest_sha256,
                   '--mode', 'observer-worker' if actual else 'test-worker',
                   '--output', str(Path(args.output)/'worker')]
        if actual:
            command += ['--private-output', args.private_output, '--grant', args.grant,
                        '--grant-sha256', args.grant_sha256]
        for name in ('stdout.log','stderr.log'):
            fd = os.open(Path(args.output)/name, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC,0o600)
            primary = None
            try:
                logs.append(os.fdopen(fd,'wb',buffering=0))
            except BaseException as error:
                primary = error
                finish(primary, lambda:os.close(fd))
                raise
        child[0] = subprocess.Popen(command, stdout=logs[0], stderr=logs[1], start_new_session=True,
                                    env=child_environment())
        report.update(child_pid=child[0].pid, child_pgid=child[0].pid)
        code = child[0].wait()
        report.update(child_returncode=code, child_reaped=True, child_group_absent=group_absent(child[0].pid))
        worker_data = read(Path(args.output)/'worker/report.json')
        observed = parse(worker_data)
        report['worker_report_sha256'] = hashlib.sha256(worker_data).hexdigest()
        report['worker'] = observed
        report.update(complete=True, passed=code==0 and observed.get('passed') is True
                      and observed.get('cleanup_completed')is True and observed.get('signal_restoration_completed')is True
                      and observed.get('source_hashes_start')==expected and observed.get('source_hashes_end')==expected
                      and report['child_group_absent'] and not denials)
    def child_cleanup():
        retire(child[0])
        if child[0] is not None:
            report.update(child_returncode=child[0].returncode, child_reaped=child[0].returncode is not None,
                          child_group_absent=group_absent(child[0].pid))
    def log_cleanup():
        primary = None
        for stream in logs:
            try:
                stream.close()
            except BaseException as error:
                primary = accumulate(primary,error,'log close failure')
        if primary is not None:
            raise primary
    return supervise(output, report, action, [child_cleanup,log_cleanup], root, expected)


def dispatch(args, raw, expected, blobs, origins):
    root = Path(__file__).parent
    if args.mode.startswith('collect-'):
        return collect(args, raw, expected, blobs, origins, root)
    if args.mode == 'test-worker':
        return test_worker(args, expected, blobs, origins, root)
    return worker(args, expected, blobs, origins, root)
