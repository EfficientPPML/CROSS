"""Separate heldout/row0 process orchestration over frozen139 lifecycle helpers."""
import hashlib,io,json,os,sys,subprocess,tarfile
from pathlib import Path
import input_gate,backend_environment
import checker_process as life
from freeze_io import Output,read,parse,finish
from cleanup_origin import choose

# Exact reviewed lifecycle functions, including shield construction fallback.
supervise=life.supervise
source_end=life.source_end
retire=life.retire
group_absent=life.group_absent
dependencies=life.dependencies
child_environment=life.child_environment
enable_model_imports=life.enable_model_imports


def mode(args):
    if args.mode in ('collect-heldout','heldout-worker'):return 'heldout'
    if args.mode in ('collect-row0','row0-worker'):return 'row0'
    raise ValueError('Exact actual heldout/row0 mode required')


def protected(blobs,root):
    external=input_gate.inspect(blobs)
    declared=parse(blobs['heldout_protected_paths.json'])
    if type(declared)is not dict or set(declared)!={'schema','paths'}or declared['schema']!='qwen-heldout-protected-predecessors-v1'or type(declared['paths'])is not list or len(declared['paths'])!=2:
        raise ValueError('Exact source-bound predecessor paths required')
    extra=[lexical_path(value)for value in declared['paths']]
    if len(set(extra))!=2:raise ValueError('Distinct protected predecessors required')
    return [*extra,root.resolve(),Path(external['model_root']),*[Path(p)for p in external['protected_paths']],
            *[Path(p)for p in life.checker_paths(blobs)]]


def lexical_path(value):
    if type(value)is not str:raise ValueError('Canonical absolute path required')
    path=Path(value)
    if not path.is_absolute()or '..'in path.parts or str(path)!=value or value.startswith('//')or '\x00'in value:
        raise ValueError('Canonical absolute path required')
    return path


def canonical_path(value):
    path=lexical_path(value)
    if path!=path.resolve():raise ValueError('Canonical absolute path required')
    return path


def overlaps(path,roots):return any(path==p or path in p.parents or p in path.parents for p in roots)


def validate_paths(args,blobs,root):
    guards=protected(blobs,root)
    for name in ('manifest','grant'):
        value=getattr(args,name,None)
        if value:
            path=lexical_path(value)
            if name=='grant'and overlaps(path,guards):raise ValueError('Public grant overlaps protected body/source')
            path=canonical_path(value)
            if name=='grant'and overlaps(path,guards):raise ValueError('Public grant overlaps protected body/source')
            guards.append(path)
    values=[]
    for name in ('output','private_output'):
        value=getattr(args,name,None)
        if value is None:continue
        path=lexical_path(value)
        if overlaps(path,guards+values):raise ValueError('Owned outputs must be disjoint from all borrowed trees')
        path=canonical_path(value)
        if overlaps(path,guards+values):raise ValueError('Owned outputs must be disjoint from all borrowed trees')
        values.append(path)


def admit(args,blobs,root):
    """Read bounded public proof only, authenticate before output/input/model."""
    import heldout_authorization as gate
    validate_paths(args,blobs,root)
    if not args.grant or not args.grant_sha256:raise PermissionError('Independent frozen-workflow grant required')
    raw=read(args.grant,args.grant_sha256,limit=1<<20)
    requests=gate.public_requests(raw)
    guards=protected(blobs,root)+[canonical_path(args.output),canonical_path(args.private_output),canonical_path(args.grant)]
    if type(requests)is not dict:raise ValueError('Exact public evidence role map required')
    evidence={}
    for name,description in requests.items():
        path=lexical_path(description['path'])
        if overlaps(path,guards):raise ValueError('Public evidence cannot read a protected input/model/owned tree')
        path=canonical_path(description['path'])
        if overlaps(path,guards):raise ValueError('Public evidence cannot read a protected input/model/owned tree')
        evidence[name]=read(path,description['sha256'],limit=description['bytes'])
        if len(evidence[name])!=description['bytes']:raise ValueError('Exact public evidence byte count required')
    output=Path(args.output)
    if args.mode.endswith('-worker'):
        if output.name!='worker':raise ValueError('Worker evidence is collector-owned worker directory')
        output=output.parent
    binding=gate.authenticate(raw,args.grant_sha256,evidence,mode=mode(args),
        source_manifest_sha256=args.manifest_sha256,evidence_output=str(output),private_output=args.private_output,blobs=blobs)
    if sorted(os.sched_getaffinity(0))!=binding.runtime_policy['cores']:raise ValueError('Granted four-core affinity required')
    def recheck():
        if read(args.grant,args.grant_sha256,limit=1<<20)!=raw:raise ValueError('Public grant changed')
        for name,description in requests.items():
            if read(description['path'],description['sha256'],limit=description['bytes'])!=evidence[name]:raise ValueError('Public evidence changed')
        gate.require_authorized(binding,blobs)
    return binding,recheck


def install_audit(blobs,root,*,binding=None,allow_processes=False):
    external=input_gate.inspect(blobs);guards=[g for g in protected(blobs,root)if g!=root.resolve()]
    allowed=set()
    if binding is not None:
        allowed={external['model_root']+'/'+name for name in external['model_assets']}
        if binding.mode=='heldout':allowed.add(binding.validation_descriptor['path'])
    denials=[]
    def audit(event,args):
        reason=None
        if event=='open'and args and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.path.abspath(os.fsdecode(args[0])))
            resolved=path if any(path==g or g in path.parents for g in guards)else path.resolve()
            sensitive=(any(path==g or g in path.parents or resolved==g or g in resolved.parents for g in guards)or path.suffix in('.safetensors','.parquet','.npy','.npz')or
                       str(path).startswith('/home/jianming_gatech/cross_validation_assets/')or str(resolved).startswith('/home/jianming_gatech/cross_validation_assets/'))
            flags=args[2]if len(args)>2 and type(args[2])is int else 0
            writing=bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if sensitive and(str(path)not in allowed or path!=resolved or writing):reason='unauthorized input/model body'
        elif event in('socket.connect','socket.connect_ex','socket.bind','socket.getaddrinfo'):reason='network'
        elif not allow_processes and event in('subprocess.Popen','os.system','os.fork','os.forkpty','os.posix_spawn','os.exec'):reason='process creation'
        if reason:
            denials.append(reason);raise PermissionError('heldout audit denied '+reason)
    sys.addaudithook(audit);return denials


def reject_denials(denials):
    if denials:raise PermissionError('Forbidden operation was attempted')


def accept_worker(row,code,expected,args,binding):
    n=1088 if binding.mode=='heldout'else 1;calls=1120 if n==1088 else 2
    flags=('complete','cleanup_completed','signal_restoration_completed','source_files_unchanged',
        'terminal_after_body_cleanup_and_signal_restoration_attempts','acceptance_requires_independent_exit0_reap_and_exact_report_hash',
        'public_proofs_reauthenticated','private_directory_handle_closed','private_outputs_retained','external_inputs_never_removed',
        'model_asset_postchecks_completed','model_cleanup_completed','native_capture_cleanup_completed','ledger_complete','selection_permanently_closed')
    if any(row.get(k)is not True for k in flags)or row.get('audit_denials')!=[]or 'error_type'in row:raise ValueError('Heldout terminal cleanup/proof incomplete')
    fixed=dict(schema='qwen-heldout-workflow-worker-v1',workload_mode=binding.mode,source_manifest_sha256=args.manifest_sha256,
        source_hashes_start=expected,source_hashes_end=expected,manifest_sha256=binding.manifest_sha256,
        freeze_receipt_sha256=binding.frozen.receipt_sha256,grant_sha256=args.grant_sha256,
        retained_entries=n,expected_entries=n,candidate_ordinal=1,revision_eligible_ledger=False,
        coefficient_selection_performed=False,encrypted_execution=False,tokenizer_execution=False,
        validation_inputs_read=binding.mode=='heldout',validation_reauthenticated=binding.mode=='heldout',
        validation_read_attempted=binding.mode=='heldout',validation_input_authenticated=binding.mode=='heldout',
        validation_read_status='confirmed'if binding.mode=='heldout'else'not_started',contains_tokens_logits_or_failure_text=False)
    if any(not input_gate.typed_equal(row.get(k),v)for k,v in fixed.items()):raise ValueError('Heldout terminal identity/phase differs')
    if type(code)is not int or type(row.get('passed'))is not bool or code!=(0 if row['passed']else 1):raise ValueError('Scientific outcome and child exit differ')
    if row.get('candidate_validation_admissible')is True or row.get('failed_attempt_terminal_candidate')is True:raise ValueError('Heldout cannot authorize feedback or revision')
    if row['passed']:
        for k in('model_loaded','complete_measured_context_coverage','trajectories_complete','all_length_invariance'):
            if row.get(k)is not True:raise ValueError('Complete frozen workload coverage required')
        for k in('native_forward_started','native_forward_completed'):
            if type(row.get(k))is not int or row[k]!=calls:raise ValueError('Actual native call count differs')
        if not input_gate.typed_equal(row.get('phase_counts'),dict(native=n,exact=n,candidate=n)):raise ValueError('Exact full workload phase counts required')
    return row


def worker(args,expected,blobs,origins,root):
    binding,recheck=admit(args,blobs,root);kind=binding.mode;n=1088 if kind=='heldout'else 1;maxcalls=1120 if n==1088 else 2
    denials=install_audit(blobs,root,binding=binding)
    output=Output(args.output);private=[None];reader=[None]
    telemetry=dict(model_loaded=False,native_forward_started=0,native_forward_completed=0)
    report=dict(schema='qwen-heldout-workflow-worker-v1',passed=False,complete=False,pid=os.getpid(),
        source_hashes_start=expected,source_manifest_sha256=args.manifest_sha256,bootstrap_origins=origins,
        private_outputs_retained=True,external_inputs_never_removed=True,model_forward=False,**telemetry,
        coefficient_selection=False,tokenizer_execution=False,validation_inputs_read=False,grant_sha256=args.grant_sha256,
        audit_denials=denials,validation_reauthenticated=False,validation_read_attempted=False,
        validation_input_authenticated=False,validation_read_status='not_started',public_proofs_reauthenticated=False,private_directory_handle_closed=False,
        manifest_sha256=binding.manifest_sha256,candidate_ordinal=1,freeze_receipt_sha256=binding.frozen.receipt_sha256,
        workload_mode=kind,selection_permanently_closed=True,revision_eligible_ledger=False)
    def progress(row):
        fields={'index','kind','sequence_length','stage','semantic_pass','model_loaded','native_forward_started','native_forward_completed'}
        if type(row)is not dict or not set(row)<=fields or any(type(v)not in(str,int,bool)for v in row.values()):raise ValueError('Public progress includes private/unsupported data')
        if row.get('stage')not in('native','exact','candidate','model_loaded','native_forward_started','native_forward_completed'):raise ValueError('Exact public phase required')
        if 'index'in row and(type(row['index'])is not int or not 0<=row['index']<n):raise ValueError('Bounded workload index required')
        if 'sequence_length'in row and(type(row['sequence_length'])is not int or not 1<=row['sequence_length']<=32):raise ValueError('Bounded sequence length required')
        if 'kind'in row and row['kind']not in(('prefill','teacher','free')if kind=='heldout'else('row0',)):raise ValueError('Exact workload context kind required')
        if 'semantic_pass'in row and type(row['semantic_pass'])is not bool:raise ValueError('Typed semantic flag required')
        for name in('native_forward_started','native_forward_completed'):
            if name in row:
                if type(row[name])is not int or row[name]!=telemetry[name]+1 or row[name]>maxcalls or not telemetry['model_loaded']or row['stage']!=name:raise ValueError('Actual monotone native counters required')
                telemetry[name]=row[name]
        if telemetry['native_forward_completed']>telemetry['native_forward_started']:raise ValueError('Completion exceeds attempts')
        if 'model_loaded'in row:
            if row['model_loaded']is not True or row['stage']!='model_loaded'or telemetry['model_loaded']:raise ValueError('Actual model-loaded event required')
            telemetry['model_loaded']=True
        report.update(telemetry,model_forward=telemetry['native_forward_started']>0)
        print(json.dumps(row,sort_keys=True),flush=True)
    def action():
        report['dependencies']=dependencies(blobs);private[0]=Output(args.private_output)
        report['private_output']=dict(path=private[0].path,identity=list(private[0].owner),mode='0700')
        if kind=='heldout':
            from heldout_input_gate import ValidationReader
            class TrackedReader(ValidationReader):
                def __call__(self,*a,**kw):
                    report['validation_read_attempted']=True
                    if report['validation_inputs_read']is not True:
                        report['validation_inputs_read']=None;report['validation_read_status']='attempted'
                    try:return super().__call__(*a,**kw)
                    finally:
                        if self.loaded is True:
                            report.update(validation_inputs_read=True,validation_input_authenticated=True,validation_read_status='confirmed')
            reader[0]=TrackedReader(blobs,binding.frozen)
        enable_model_imports()
        import heldout_runtime
        result=heldout_runtime.execute_heldout(blobs,binding,reader[0],private[0],progress)
        if type(result)is not dict or any(not input_gate.typed_equal(result.get(k),v)for k,v in telemetry.items()):raise ValueError('Runtime counters differ from actual events')
        if type(result.get('passed'))is not bool or result.get('ledger_complete')is not True:raise ValueError('Complete typed scientific ledger required')
        forbidden={'schema','source_hashes_start','source_hashes_end','source_manifest_sha256','audit_denials','cleanup_completed',
            'signal_restoration_completed','grant_sha256','pid','private_output','error_type','candidate_validation_admissible','failed_attempt_terminal_candidate',
            'validation_read_attempted','validation_input_authenticated','validation_read_status','validation_reauthenticated',
            'public_proofs_reauthenticated','private_directory_handle_closed'}
        # Numerical dispatcher schema is diagnostic; process schema stays fixed.
        result=dict(result);result.pop('schema',None)
        if set(result)&forbidden:raise ValueError('Runtime cannot overwrite process authority')
        if result.get('validation_inputs_read')is not report['validation_inputs_read']:
            raise ValueError('Runtime cannot overwrite actual validation read state')
        if result.get('manifest_sha256')!=binding.manifest_sha256 or result.get('freeze_receipt_sha256')!=binding.frozen.receipt_sha256 or result.get('workload_mode')!=kind or result.get('selection_permanently_closed')is not True or result.get('revision_eligible_ledger')is not False:raise ValueError('Frozen workload binding changed')
        if result['passed']and(telemetry!=dict(model_loaded=True,native_forward_started=maxcalls,native_forward_completed=maxcalls)or
            any(result.get(k)is not True for k in('trajectories_complete','all_length_invariance','complete_measured_context_coverage','model_asset_postchecks_completed','model_cleanup_completed','native_capture_cleanup_completed'))or
            not input_gate.typed_equal(result.get('phase_counts'),dict(native=n,exact=n,candidate=n))):raise ValueError('Complete actual workload counters required')
        report.update(result,complete=True);reject_denials(denials)
    def recheck_proof():recheck();report['public_proofs_reauthenticated']=True
    def recheck_input():
        if reader[0]is not None:
            if reader[0].loaded is True:
                report.update(validation_read_attempted=True,validation_inputs_read=True,validation_input_authenticated=True,validation_read_status='confirmed')
            reader[0].recheck();report['validation_reauthenticated']=reader[0].loaded
    def close_private():
        if private[0]is not None:private[0].close();report['private_directory_handle_closed']=True
    return supervise(output,report,action,[recheck_input,recheck_proof,close_private,lambda:reject_denials(denials)],root,expected)


def collect(args, raw, expected, blobs, origins, root):
    validate_paths(args, blobs, root)
    actual = args.mode in ('collect-heldout','collect-row0')
    smoke = args.mode == 'collect-import-smoke'
    if smoke and any(getattr(args,n,None) for n in ('private_output','grant','grant_sha256')):
        raise ValueError('Import-only smoke cannot accept an actual grant or private output')
    if actual:
        binding,recheck = admit(args,blobs,root)
    denials = install_audit(blobs,root,allow_processes=True)
    output = Output(args.output)
    report = dict(passed=False, complete=False, audit_denials=denials, schema='qwen-heldout-workflow-parent-v1', scope='Frozen heldout1088 or separate row0' if actual else 'Authenticated import-only smoke' if smoke else 'Public synthetic observer preflight',
                  source_hashes_start=expected, source_manifest_sha256=args.manifest_sha256,
                  bootstrap_origins=origins, pid=os.getpid(), actual_heldout_stage=actual,
                  model_forward=False, coefficient_selection=False)
    if actual:
        report.update(workload_mode=binding.mode,manifest_sha256=binding.manifest_sha256,freeze_receipt_sha256=binding.frozen.receipt_sha256,
            grant_sha256=args.grant_sha256,selection_permanently_closed=True,revision_eligible_ledger=False,public_proofs_reauthenticated=False)
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
        command = [sys.executable, '-I', '-B', str(execution/'heldout_entry.py'),
                   '--manifest', str(Path(args.output)/'source_manifest.json'),
                   '--manifest-sha256', args.manifest_sha256,
                   '--mode', binding.mode+'-worker' if actual else 'import-smoke-worker' if smoke else 'test-worker',
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
            accept_worker(observed,code,expected,args,binding)
            report['passed']=observed['passed'] and report['passed']
            report['selection_permanently_closed']=True
            report['revision_eligible_ledger']=False
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
                primary = life.accumulate(primary,error,'log close failure')
        if primary is not None:
            raise primary
    def proof_cleanup():
        if actual:recheck();report['public_proofs_reauthenticated']=True
    return supervise(output, report, action, [child_cleanup,log_cleanup,proof_cleanup,lambda:reject_denials(denials)], root, expected)



def test_worker(args,expected,blobs,origins,root):
    validate_paths(args,blobs,root)
    if any(getattr(args,n,None)for n in('private_output','grant','grant_sha256')):raise ValueError('Synthetic preflight accepts no actual grant')
    denials=install_audit(blobs,root,allow_processes=True);output=Output(args.output)
    report=dict(passed=False,complete=False,pid=os.getpid(),source_hashes_start=expected,source_manifest_sha256=args.manifest_sha256,
        bootstrap_origins=origins,audit_denials=denials,scope='Synthetic heldout/row0 preparation only',model_forward=False,
        validation_inputs_read=False,coefficient_selection=False)
    def action():
        import pytest,xml.etree.ElementTree as ET,importlib
        config=parse(blobs['heldout_preflight_expectations.json'])
        if type(config)is not dict or set(config)!={'tests','synthetic_tests','installed_import_denials'}or type(config['tests'])is not list:raise ValueError('Exact explicit preflight selection required')
        names=config['tests']
        if len(set(names))!=len(names)or any(type(n)is not str or not n.startswith('test_')or not n.endswith('.py')or '/'in n or n not in expected for n in names):raise ValueError('Declared local test sources required')
        report['dependencies']=dependencies(blobs)
        result=pytest.main(['-q','-p','no:cacheprovider','--import-mode=importlib','--basetemp='+str(Path(args.output)/'synthetic'),'--junitxml='+str(Path(args.output)/'tests.xml')]+[str(root/n)for n in names])
        tree=ET.parse(Path(args.output)/'tests.xml');cases=tree.findall('.//testcase')
        report.update(tests=len(cases),failures=len(tree.findall('.//failure')),errors=len(tree.findall('.//error')),skipped=len(tree.findall('.//skipped')),pytest_returncode=int(result))
        denied=[]
        for name in('torch','transformers','jax','jaxlib','jaxite_word','tensorflow','tokenizers','safetensors','flax','torch_xla'):
            try:importlib.import_module(name)
            except ImportError:denied.append(name)
            else:raise AssertionError('Forbidden backend imported')
        report['installed_import_denials']=denied
        report.update(complete=True,passed=result==0 and len(cases)==config['synthetic_tests']and len(denied)==config['installed_import_denials']and not report['failures']and not report['errors']and not report['skipped']and not denials)
    return supervise(output,report,action,[lambda:reject_denials(denials)],root,expected)


def dispatch(args,raw,expected,blobs,origins):
    root=Path(__file__).parent
    if args.mode.startswith('collect-'):return collect(args,raw,expected,blobs,origins,root)
    if args.mode=='test-worker':return test_worker(args,expected,blobs,origins,root)
    if args.mode=='import-smoke-worker':return life.import_worker(args,expected,blobs,origins,root)
    return worker(args,expected,blobs,origins,root)
