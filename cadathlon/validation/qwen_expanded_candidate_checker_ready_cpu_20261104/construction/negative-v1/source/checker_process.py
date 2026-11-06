"""Owned one-candidate collection; public preparation never admits bodies."""
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
import backend_environment
import checker_admission
from proof_limits import source_limit, check_source_sizes

SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)


def checker_paths(blobs):
    value = parse(blobs['checker_protected_paths.json'])
    if (type(value) is not dict or set(value) != {'schema', 'paths'} or
            value['schema'] != 'qwen-candidate-protected-paths-v1' or
            type(value['paths']) is not list or len(value['paths']) < 5 or
            any(type(p) is not str or not Path(p).is_absolute() or '..' in Path(p).parts
                for p in value['paths'])):
        raise ValueError('Protected initial-observation trees required')
    return tuple(value['paths'])


def policy(blobs):
    value = parse(blobs['checker_runtime_policy.json'])
    expected = dict(schema='qwen-expanded-candidate-runtime-policy-v1',
        cores=[172,173,174,175], torch_threads=4, torch_interop_threads=1,
        candidate_budget=3, measured_contexts=3200, native_forwards=3232, validation_enabled=False)
    if not input_gate.typed_equal(value, expected):
        raise ValueError('Exact preparation resource/workload policy required')
    return value


def grant_check(args, blobs):
    # Public candidate authentication is deliberately before filesystem grant I/O.
    binding = checker_admission.require_candidate(blobs)
    if not input_gate.typed_equal(parse(blobs['execution_authorization.json']), {'execute': False}):
        raise ValueError('Source does not itself authorize candidate execution')
    external = input_gate.inspect(blobs); runtime = policy(blobs)
    if not args.grant or not args.grant_sha256:
        raise PermissionError('Independent one-candidate execution grant required')
    grant = parse(read(args.grant, args.grant_sha256, limit=1 << 20))
    evidence = Path(args.output)
    if args.mode == 'candidate-worker':
        if evidence.name != 'worker':
            raise ValueError('Owned candidate worker output required')
        evidence = evidence.parent
    fixed = dict(schema='qwen-expanded-candidate-checker-grant-v1', execute=True,
        scope='one-frozen-candidate-own3200-calibration-only', candidate_ordinal=binding.ordinal,
        candidate_manifest_sha256=binding.manifest_sha256, candidate_binding_sha256=binding.binding_sha256,
        initial_report_sha256=binding.initial_report_sha256,
        initial_proof_sha256=binding.initial_proof_sha256,
        selection_report_sha256=binding.selection_proof_sha256,
        selector_outer_sha256=binding.selector_outer_sha256,
        source_manifest_sha256=args.manifest_sha256, protocol_sha256=binding.protocol_sha256,
        calibration=external['calibration'], model_pins_sha256=checker_admission.sha(blobs['public/model_pins.json']),
        runtime_policy=runtime, backend_environment=dict(backend_environment.REQUIRED),
        evidence_output=str(evidence), private_output=args.private_output,
        model_forward=True, tokenizer_execution=False, coefficient_selection=False,
        validation_read=False, encrypted_execution=False, candidate_budget=3)
    if (type(grant) is not dict or set(grant) != set(fixed) | {'preflight_report_sha256', 'independent_review_sha256'} or
            any(not input_gate.typed_equal(grant[k], v) for k, v in fixed.items()) or
            any(not input_gate.is_digest(grant[k]) for k in ('preflight_report_sha256', 'independent_review_sha256'))):
        raise ValueError('Exact independent candidate-only grant binding differs')
    if not args.private_output or sorted(os.sched_getaffinity(0)) != runtime['cores']:
        raise ValueError('Fresh private output and assigned candidate CPU affinity required')
    return binding, grant


def accept_candidate_worker(observed, code, expected, args, blobs):
    binding = checker_admission.require_candidate(blobs)
    flags = ('complete', 'cleanup_completed', 'signal_restoration_completed', 'source_files_unchanged',
        'terminal_after_body_cleanup_and_signal_restoration_attempts',
        'acceptance_requires_independent_exit0_reap_and_exact_report_hash',
        'calibration_reauthenticated', 'private_directory_handle_closed', 'private_outputs_retained',
        'external_inputs_never_removed', 'model_asset_postchecks_completed', 'model_cleanup_completed',
        'native_capture_cleanup_completed', 'ledger_complete')
    if (any(observed.get(k) is not True for k in flags) or observed.get('audit_denials') != [] or
            'error_type' in observed or observed.get('source_manifest_sha256') != args.manifest_sha256 or
            observed.get('source_hashes_start') != expected or observed.get('source_hashes_end') != expected or
            observed.get('manifest_sha256') != binding.manifest_sha256 or
            observed.get('grant_sha256') != args.grant_sha256 or
            observed.get('initial_report_sha256') != binding.initial_report_sha256 or
            observed.get('selection_proof_sha256') != binding.selection_proof_sha256):
        raise ValueError('Candidate terminal provenance/cleanup is incomplete')
    for name in ('retained_entries', 'expected_entries'):
        checker_admission.require_int(observed.get(name), 3200)
    checker_admission.require_int(observed.get('candidate_ordinal'), binding.ordinal)
    for name in ('coefficient_selection_performed', 'encrypted_execution', 'validation_inputs_read',
                 'tokenizer_execution', 'contains_tokens_logits_or_failure_text'):
        if observed.get(name) is not False:
            raise ValueError('Candidate phase/private-public contract differs')
    if type(code) is not int or type(observed.get('passed')) is not bool or code != (0 if observed['passed'] else 1):
        raise ValueError('Candidate scientific outcome and direct child exit differ')
    if observed['passed']:
        for name in ('model_loaded', 'candidate_validation_admissible', 'complete_measured_context_coverage',
                     'trajectories_complete', 'all_length_invariance'):
            if observed.get(name) is not True:
                raise ValueError('Complete candidate coverage required')
        for name in ('native_forward_started', 'native_forward_completed'):
            checker_admission.require_int(observed.get(name), 3232)
        if not input_gate.typed_equal(observed.get('phase_counts'), dict(native=3200, exact=3200, candidate=3200)):
            raise ValueError('Exact own3200 phase counts required')
    elif observed.get('candidate_validation_admissible') is not False:
        raise ValueError('Failed candidate must not admit validation')
    if observed.get('failed_attempt_terminal_candidate') is True and not failed_revision_coverage(observed):
        raise ValueError('Failed revision ledger lacks exact native/context coverage')
    return observed


def failed_revision_coverage(row):
    """Candidate-only stops may leave free suffixes; comparison failures may not."""
    if (row.get('passed') is not False or row.get('revision_eligible_ledger') is not True or
            row.get('ledger_complete') is not True or row.get('all_length_invariance') is not True or
            row.get('model_loaded') is not True):
        return False
    kinds = row.get('by_kind')
    if type(kinds) is not dict or set(kinds) != {'prefill','teacher','free'}:
        return False
    unmeasured = 0
    for name, expected in (('prefill',3072),('teacher',64),('free',64)):
        value = kinds[name]
        if type(value) is not dict:
            return False
        for key in ('expected','records','top1_denominator'):
            if type(value.get(key)) is not int or value[key] != expected:
                return False
        counters = ('complete','candidate_invalid','comparison_invalid','unmeasured')
        if (any(type(value.get(k)) is not int or not 0 <= value[k] <= expected for k in counters) or
                sum(value[k] for k in counters) != expected or value['comparison_invalid'] != 0 or
                name != 'free' and value['unmeasured'] != 0):
            return False
        unmeasured += value['unmeasured']
    observed = 3200-unmeasured
    if (not input_gate.typed_equal(row.get('phase_counts'),dict(native=observed,exact=observed,candidate=observed)) or
            type(row.get('semantic_passes')) is not int or row['semantic_passes'] != observed):
        return False
    for name in ('native_forward_started','native_forward_completed'):
        if type(row.get(name)) is not int or row[name] != observed+32:
            return False
    for origin in ('native','exact'):
        value = row.get('input_ranges',{}).get(origin,{})
        if type(value.get('contexts')) is not int or value['contexts'] != observed:
            return False
    return True


def worker(args, expected, blobs, origins, root):
    validate_paths(args, blobs, root)
    binding, grant = grant_check(args, blobs)
    denials = install_audit(blobs, True)
    output = Output(args.output); private = [None]; reader = [None]
    telemetry = dict(model_loaded=False, native_forward_started=0, native_forward_completed=0)
    report = dict(passed=False, complete=False, pid=os.getpid(), source_hashes_start=expected,
        source_manifest_sha256=args.manifest_sha256, bootstrap_origins=origins,
        private_outputs_retained=True, external_inputs_never_removed=True, model_forward=False,
        **telemetry, model_execution_requested=True, coefficient_selection=False, tokenizer_execution=False,
        validation_inputs_read=False, grant_sha256=args.grant_sha256, audit_denials=denials,
        calibration_reauthenticated=False, private_directory_handle_closed=False,
        candidate_validation_admissible=False, failed_attempt_terminal_candidate=False,
        manifest_sha256=binding.manifest_sha256, candidate_ordinal=binding.ordinal,
        initial_report_sha256=binding.initial_report_sha256, selection_proof_sha256=binding.selection_proof_sha256)
    def progress(row):
        fields = {'index','kind','sequence_length','stage','semantic_pass','model_loaded',
                  'native_forward_started','native_forward_completed'}
        if type(row) is not dict or not set(row) <= fields or any(type(v) not in (str,int,bool) for v in row.values()):
            raise ValueError('Public progress contains unsupported/private fields')
        if row.get('stage') not in ('native','exact','candidate','model_loaded',
                                    'native_forward_started','native_forward_completed'):
            raise ValueError('Declared bounded public phase required')
        if 'index' in row and (type(row['index']) is not int or not 0 <= row['index'] < 3200):
            raise ValueError('Bounded public context index required')
        if 'sequence_length' in row and (type(row['sequence_length']) is not int or not 1 <= row['sequence_length'] <= 32):
            raise ValueError('Bounded public sequence length required')
        if 'kind' in row and row['kind'] not in ('prefill','teacher','free'):
            raise ValueError('Declared public context kind required')
        if 'semantic_pass' in row and type(row['semantic_pass']) is not bool:
            raise ValueError('Typed public semantic status required')
        for name in ('native_forward_started', 'native_forward_completed'):
            if name in row:
                value = row[name]
                if (type(value) is not int or value != telemetry[name] + 1 or value > 3232 or
                        telemetry['model_loaded'] is not True or row['stage'] != name):
                    raise ValueError('Monotone bounded native counters required')
                telemetry[name] = value
        if telemetry['native_forward_completed'] > telemetry['native_forward_started']:
            raise ValueError('Native completion exceeds attempts')
        if 'model_loaded' in row:
            if row['model_loaded'] is not True or row['stage'] != 'model_loaded' or telemetry['model_loaded']:
                raise ValueError('Exact actual model-loaded event required')
            telemetry['model_loaded'] = True
        report.update(telemetry, model_forward=telemetry['native_forward_started'] > 0)
        import json
        print(json.dumps(row, sort_keys=True), flush=True)
    def action():
        report['dependencies'] = dependencies(blobs)
        private[0] = Output(args.private_output); reader[0] = input_gate.Reader(blobs)
        report['private_output'] = dict(path=private[0].path, identity=list(private[0].owner), mode='0700')
        enable_model_imports()
        import checker_runtime
        result = checker_runtime.execute_candidate(blobs, reader[0], private[0], progress)
        if type(result) is not dict or any(not input_gate.typed_equal(result.get(k), v) for k, v in telemetry.items()):
            raise ValueError('Runtime counters differ from observed lifecycle events')
        if type(result.get('passed')) is not bool:
            raise ValueError('Typed scientific result required')
        if result.get('manifest_sha256') != binding.manifest_sha256 or result.get('ledger_complete') is not True:
            raise ValueError('Full ordered candidate ledger required after normal return')
        if result['passed']:
            if (telemetry != dict(model_loaded=True,native_forward_started=3232,native_forward_completed=3232) or
                    any(result.get(k) is not True for k in ('trajectories_complete','all_length_invariance',
                        'complete_measured_context_coverage','model_asset_postchecks_completed',
                        'model_cleanup_completed','native_capture_cleanup_completed')) or
                    not input_gate.typed_equal(result.get('phase_counts'),dict(native=3200,exact=3200,candidate=3200))):
                raise ValueError('Candidate pass requires actual complete native3232 and own3200 coverage')
        # The runtime owns only scientific fields. Terminal authority stays here.
        forbidden = {'source_hashes_start','source_hashes_end','source_manifest_sha256','audit_denials',
            'cleanup_completed','signal_restoration_completed','grant_sha256','pid','private_output',
            'candidate_validation_admissible','failed_attempt_terminal_candidate','error_type'}
        if set(result) & forbidden:
            raise ValueError('Runtime attempted to overwrite process authority')
        report.update(result, complete=True)
        report['candidate_validation_admissible'] = result.get('passed') is True
        report['failed_attempt_terminal_candidate'] = failed_revision_coverage(result)
        if denials:
            raise ValueError('Forbidden access attempted')
    def recheck_input():
        if reader[0] is not None:
            reader[0].recheck(); report['calibration_reauthenticated'] = reader[0].loaded
    def close_private():
        if private[0] is not None:
            private[0].close(); report['private_directory_handle_closed'] = True
    return supervise(output, report, action, [recheck_input, close_private], root, expected)


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
    primary = body = None; fired = False; entered = False
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
        entered = True
        yield
    except BaseException as error:
        if shield and not entered:
            primary = mark_cleanup(error)
            body = None
        else:
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
    check_source_sizes({name: (root/name).stat().st_size for name in expected})
    after = {}; sizes = {}
    for name in expected:
        data = read(root/name, limit=source_limit(name))
        sizes[name] = len(data); check_source_sizes(sizes)
        after[name] = hashlib.sha256(data).hexdigest()
    if after != expected:
        raise ValueError('terminal source hashes differ')
    return after


def validate_paths(args, blobs, root):
    external = input_gate.inspect(blobs)
    protected = [root.resolve(), Path(external['model_root'])]
    protected += [Path(value) for value in external['protected_paths']]
    protected += [Path(value) for value in checker_paths(blobs)]
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


def install_audit(blobs, allow_inputs, diagnostics=None):
    external=input_gate.inspect(blobs)
    calibration=external['calibration']['path']
    model_root=external['model_root'];protected=[Path(p) for p in external['protected_paths']]+[Path(p) for p in checker_paths(blobs)]
    allowed={calibration}|{model_root+'/'+name for name in external['model_assets']}
    denials=[]
    def audit(event,args):
        reason=None
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.path.abspath(os.fsdecode(args[0])));resolved=path.resolve()
            protected_body=(str(path) in allowed or any(path==p or p in path.parents or resolved==p or p in resolved.parents for p in protected)
                or str(path).startswith('/home/jianming_gatech/cross_validation_assets/')
                or str(resolved).startswith('/home/jianming_gatech/cross_validation_assets/'))
            sensitive=protected_body or path.suffix in ('.safetensors','.parquet','.npy','.npz')
            flags=args[2] if len(args)>2 and type(args[2])is int else 0
            writing=bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if sensitive and (not allow_inputs or str(path) not in allowed or path!=resolved or writing):
                reason='external calibration/validation/model body'
        elif event in ('socket.connect','socket.connect_ex','socket.bind','socket.getaddrinfo'):
            reason='network'
        if reason:
            if diagnostics is not None:
                stack=[];frame=sys._getframe(1)
                for _ in range(16):
                    if frame is None:break
                    stack.append(dict(file=frame.f_code.co_filename,function=frame.f_code.co_name,line=frame.f_lineno))
                    frame=frame.f_back
                del frame
                diagnostics.append(dict(event=event,reason=reason,stack=stack))
            denials.append(reason);raise PermissionError('observer audit denied '+reason)
    sys.addaudithook(audit)
    return denials




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
                if cleanup_origin(error):
                    cleanup_complete = False
                report.update(passed=False, error_type=type(error).__name__)
            finally:
                cleanup_all()
    except BaseException as error:
        report['passed'] = False
        if cleanup_origin(error):
            cleanup_complete = False
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
        report['candidate_validation_admissible'] = False
    if primary is not None or not cleanup_complete or not state.get('restored', False):
        report['failed_attempt_terminal_candidate'] = False
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





def import_worker(args, expected, blobs, origins, root):
    if any(getattr(args,n,None) for n in ('private_output','grant','grant_sha256')):
        raise ValueError('Import-only smoke cannot accept an actual grant or private output')
    validate_paths(args,blobs,root)
    diagnostics=[]
    denials=install_audit(blobs,False,diagnostics=diagnostics)
    output=Output(args.output)
    report=dict(passed=False,complete=False,pid=os.getpid(),source_hashes_start=expected,
        source_manifest_sha256=args.manifest_sha256,bootstrap_origins=origins,audit_denials=denials,
        scope='Backend class import only; no model constructor, forward, tokenizer or input/asset body access',
        denied_access_diagnostics=diagnostics,
        model_forward=False,model_constructed=False,tokenizer_execution=False,
        coefficient_selection=False,validation_inputs_read=False)
    def action():
        report['dependencies']=dependencies(blobs)
        report['backend_environment']=backend_environment.check()
        enable_model_imports()
        import import_smoke
        report.update(import_smoke.run(blobs))
        report.update(complete=True,passed=not denials)
    return supervise(output,report,action,[],root,expected)

def test_worker(args, expected, blobs, origins, root):
    validate_paths(args, blobs, root)
    denials = install_audit(blobs, False)
    output = Output(args.output)
    report = dict(passed=False, complete=False, pid=os.getpid(), source_hashes_start=expected,
                  source_manifest_sha256=args.manifest_sha256, bootstrap_origins=origins,
                  scope='Source-bound synthetic candidate preflight; no actual token/model bodies or backend construction.',
                  audit_denials=denials)
    def action():
        report['dependencies'] = dependencies(blobs)
        import pytest
        result = pytest.main(['-q', '-p', 'no:cacheprovider', '--import-mode=importlib',
                              '--basetemp='+str(Path(args.output)/'synthetic'),
                              '--junitxml='+str(Path(args.output)/'tests.xml')]
                             + [str(root/name) for name in expected if name.startswith('test_') and name.endswith('.py') and name != 'test_entry.py'])
        tree = ET.parse(Path(args.output)/'tests.xml')
        cases = tree.findall('.//testcase')
        report.update(tests=len(cases), failures=len(tree.findall('.//failure')),
                      errors=len(tree.findall('.//error')), skipped=len(tree.findall('.//skipped')),
                      pytest_returncode=int(result))
        installed = []
        for name in ('torch','transformers','jax','jaxlib','jaxite_word','tensorflow','tokenizers','safetensors','flax','torch_xla'):
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
    body=None if primary is None or cleanup_origin(primary) else protect_body(primary)
    chosen=primary;failed=False
    def attempt(label,action):
        nonlocal chosen,failed
        try:return action()
        except BaseException as error:
            failed=True;chosen=choose(chosen,error,label,body_primary=body);return None
    def send(number):
        try:os.killpg(child.pid,number)
        except ProcessLookupError:pass
    attempt('initial child poll',child.poll)
    empty=attempt('initial group observation',lambda:group_absent(child.pid))
    if empty is not True:
        attempt('terminate owned group',lambda:send(signal.SIGTERM))
        deadline=time.monotonic()+30
        # Faults skip the grace, but never skip KILL/direct wait/final observation.
        while not failed and time.monotonic()<deadline:
            attempt('poll after TERM',child.poll)
            empty=attempt('group after TERM',lambda:group_absent(child.pid))
            if empty is True or failed:break
            attempt('TERM grace sleep',lambda:time.sleep(.02))
        if empty is not True:attempt('kill remaining owned group',lambda:send(signal.SIGKILL))
    attempt('final direct-child wait',lambda:child.wait(timeout=30))
    empty=attempt('final group observation',lambda:group_absent(child.pid))
    if empty is not True:
        chosen=choose(chosen,RuntimeError('owned child group remains or state is unknown'),'retirement incomplete',body_primary=body)
    if chosen is not None and chosen is not primary:raise chosen


def child_environment():
    env=dict(os.environ)
    for name in tuple(env):
        if name.startswith(('JAX_','XLA_','TPU_')) or name in ('PYTHONPATH','PYTHONHOME','PYTHONOPTIMIZE'):
            env.pop(name,None)
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='')
    env.update(backend_environment.REQUIRED)
    return env


def collect(args, raw, expected, blobs, origins, root):
    validate_paths(args, blobs, root)
    actual = args.mode == 'collect-candidate'
    smoke = args.mode == 'collect-import-smoke'
    if smoke and any(getattr(args,n,None) for n in ('private_output','grant','grant_sha256')):
        raise ValueError('Import-only smoke cannot accept an actual grant or private output')
    if actual:
        grant_check(args, blobs)
    denials = install_audit(blobs, False)
    output = Output(args.output)
    report = dict(passed=False, complete=False, audit_denials=denials, scope='One frozen calibration-only candidate' if actual else 'Authenticated import-only smoke' if smoke else 'Public synthetic observer preflight',
                  source_hashes_start=expected, source_manifest_sha256=args.manifest_sha256,
                  bootstrap_origins=origins, pid=os.getpid(), actual_candidate_stage=actual,
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
                   '--mode', 'candidate-worker' if actual else 'import-smoke-worker' if smoke else 'test-worker',
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
                      and observed.get('complete') is True
                      and observed.get('cleanup_completed')is True and observed.get('signal_restoration_completed')is True
                      and observed.get('terminal_after_body_cleanup_and_signal_restoration_attempts')is True
                      and observed.get('acceptance_requires_independent_exit0_reap_and_exact_report_hash')is True
                      and observed.get('source_hashes_start')==expected and observed.get('source_hashes_end')==expected
                      and report['child_group_absent'] and not denials)
        if actual:
            accept_candidate_worker(observed, code, expected, args, blobs)
            report['candidate_validation_admissible'] = report['passed'] and observed.get('candidate_validation_admissible') is True
            report['failed_attempt_terminal_candidate'] = code == 1 and observed.get('failed_attempt_terminal_candidate') is True and report['child_group_absent']
            report['passed'] = report['candidate_validation_admissible']
    def child_cleanup():
        failure=None
        try:retire(child[0])
        except BaseException as error:failure=error
        finally:
            if child[0] is not None:
                report.update(child_returncode=child[0].returncode,child_reaped=child[0].returncode is not None)
                try:report['child_group_absent']=group_absent(child[0].pid)
                except BaseException as error:
                    report['child_group_absent']=False
                    failure=choose(failure,error,'final child-group reporting')
        if failure is not None:raise failure
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
    if args.mode == 'import-smoke-worker':
        return import_worker(args, expected, blobs, origins, root)
    if args.mode == 'test-worker':
        return test_worker(args, expected, blobs, origins, root)
    return worker(args, expected, blobs, origins, root)
