"""Disabled-by-default source-frozen reconstruction worker and owned collector.

No model backend is imported. Only a separately pinned root reconstruction
grant admits calibration and candidate journals. No outer/freeze receipt is
issued here; the root owns those independent acts after this process is reaped.
"""
from contextlib import contextmanager
import gzip
import hashlib
import io
import os
from pathlib import Path
import sys
import tarfile
import time
import xml.etree.ElementTree as ET
from cleanup import cleanup_signals,termination_signals
from cleanup_origin import cleanup_origin,protect_body,choose
from component_processes import OwnedChild
from freeze_io import Output,read,json_bytes
from producer_admission import admit,NAMES
from verify_records import parse
from verify_statistics import same,fields,integer,digest
from reconstruction_scope import run

def sha(raw):return hashlib.sha256(raw).hexdigest()

def protected_paths(blobs):
    row=parse(blobs['reconstruction_protected_paths.json']);fields(row,('schema','paths'));same(row['schema'],'qwen-candidate-reconstruction-protected-paths-v1')
    paths=row['paths']
    if type(paths)is not list or not 1<=len(paths)<=128 or any(type(p)is not str or not Path(p).is_absolute()or '..'in Path(p).parts for p in paths):raise ValueError('Fixed source-bound protected trees required')
    return tuple(Path(p)for p in paths)

def guard_path(value,protected,*,output=False):
    path=Path(value)
    if not path.is_absolute()or '..'in path.parts:raise ValueError('Canonical absolute role path required')
    for p in(path,path.resolve()):
        for raw in protected:
            for root in(raw,raw.resolve()):
                if p==root or root in p.parents or output and p in root.parents:raise PermissionError('Role path overlaps source-bound protected tree')
    return path

class Audit:
    """Ordinary Python absolute-open/import/network defense, not OS isolation.

    Relative dir_fd operations are separately checked by the held descriptor
    readers/writers. This hook does not claim to recover a missing dir_fd from
    Python's open audit event.
    """
    def __init__(self,args,root,protected=()):
        self.denials=[];self.private=[];self.exact=set();self.source=Path(root).resolve();self.output=Path(args.output).resolve()
        self.protected=protected
        self.system=[Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve(),Path('/usr'),Path('/lib'),Path('/lib64'),Path('/etc')]
        for n in('grant','request'):
            value=getattr(args,n,None)
            if value:self.exact.add(str(guard_path(value,protected).resolve()))
    def allow_public(self,path):self.exact.add(str(guard_path(path,self.protected).resolve()))
    def allow_private(self,cap):self.private=[Path(cap.private_path),Path(parse(cap.calibration_bytes)['path']).parent]
    def install(self):
        def deny(label):self.denials.append(label);raise PermissionError('Reconstruction audit denied '+label)
        def hook(event,args):
            if event in('socket.connect','socket.bind','socket.getaddrinfo'):deny(event)
            if event=='import'and args and str(args[0]).split('.')[0]in{'jax','jaxlib','jaxite_word','tensorflow','flax','torch_xla','torch','transformers','safetensors','tokenizers','numpy'}:deny('backend import')
            if event!='open'or not args or isinstance(args[0],int):return
            if not isinstance(args[0],(str,bytes,os.PathLike)):return
            p=Path(os.fsdecode(args[0]))
            if not p.is_absolute():return # Held dir_fd validation is authoritative.
            p=p.resolve();mode=args[1]if len(args)>1 else None;flags=args[2]if len(args)>2 else 0
            writing=(type(flags)is int and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))or(type(mode)is str and any(x in mode for x in'wa+'))
            owned=p==self.output or self.output in p.parents
            if writing:
                if not owned and str(p)not in('/dev/null',):deny('write outside owned evidence')
                return
            roots=[self.source,self.output,*self.system,*self.private]
            if str(p)not in self.exact and str(p)not in('/dev/null','/dev/urandom','/proc/self/stat')and not any(p==r or r in p.parents for r in roots):deny('read outside admitted scope')
        sys.addaudithook(hook)

def archive_bytes(blobs):
    buf=io.BytesIO()
    with gzip.GzipFile(fileobj=buf,mode='wb',filename='',mtime=0)as z:
        with tarfile.open(fileobj=z,mode='w')as tar:
            for name,body in sorted(blobs.items()):
                info=tarfile.TarInfo(name);info.mode=0o600;info.size=len(body);tar.addfile(info,io.BytesIO(body))
    return buf.getvalue()

def source_end(root,expected):
    names=set()
    for path in Path(root).rglob('*'):
        if path.is_symlink():raise ValueError('Reconstruction source symlink')
        if path.is_file():names.add(path.relative_to(root).as_posix())
        elif not path.is_dir():raise ValueError('Nonregular reconstruction source')
        elif not any(n.startswith(path.relative_to(root).as_posix()+'/')for n in expected):raise ValueError('Unexpected reconstruction source directory')
    if names!=set(expected):raise ValueError('Reconstruction source inventory changed')
    current={name:sha(read(Path(root)/name,limit=8<<20))for name in expected}
    same(current,expected);return current

def public_request(args,raw,expected,blobs,audit=None):
    protected=protected_paths(blobs)
    guard_path(args.output,protected,output=True)
    for key in('grant','request'):
        value=getattr(args,key,None)
        if value:guard_path(value,protected)
    same(parse(blobs['execution_authorization.json']),{'execute':False})
    if not args.grant or not args.grant_sha256 or not args.request or not args.request_sha256:raise PermissionError('Independent root reconstruction grant and public request required')
    grant_raw=read(args.grant,args.grant_sha256,limit=1<<20);grant=parse(grant_raw)
    if grant.get('execute')is not True:raise PermissionError('Root reconstruction execution is disabled')
    same(grant.get('reconstruction_source_manifest_sha256'),args.manifest_sha256)
    same(grant.get('reconstruction_source_archive_sha256'),sha(archive_bytes(blobs)))
    same(grant.get('reconstruction_script_sha256'),sha(blobs['entry.py']))
    same(grant.get('request_sha256'),args.request_sha256)
    output=Path(args.output)
    if args.mode=='reconstruct-worker':
        if output.name!='worker':raise ValueError('Exact owned worker output required')
        output=output.parent
    same(grant.get('evidence_output'),str(output))
    private=[Path(grant['private_output']['path']),Path(grant['calibration']['path']).parent]
    if any(output==r or r in output.parents or output in r.parents for r in private):raise ValueError('Evidence output overlaps borrowed private inputs')
    def public_path(path):
        p=guard_path(path,protected)
        if not p.is_absolute()or p!=p.resolve()or '..'in p.parts or any(p==r or r in p.parents for r in private):raise PermissionError('Public proof path overlaps private scope or alias')
        return p
    request_raw=read(public_path(args.request),args.request_sha256,limit=1<<20);request=parse(request_raw)
    fields(request,('schema','proofs','protected_private_roots'));same(request['schema'],'qwen-candidate-reconstruction-request-v1')
    roots=request['protected_private_roots']
    if type(roots)is not list or any(type(p)is not str or not Path(p).is_absolute()or Path(p)!=Path(p).resolve()for p in roots):raise ValueError('Exact protected private root metadata required')
    private.extend(Path(p)for p in roots)
    fields(request['proofs'],set(NAMES)-{'reconstruction_grant'})
    public={'reconstruction_grant':grant_raw};pins={'reconstruction_grant':args.grant_sha256}
    for name,d in request['proofs'].items():
        fields(d,('path','sha256','bytes'));digest(d['sha256']);integer(d['bytes'],1,96<<20 if name=='source_archive'else 64<<20)
        path=public_path(d['path'])
        if audit is not None:audit.allow_public(path)
        body=read(path,d['sha256'],limit=d['bytes'])
        if len(body)!=d['bytes']:raise ValueError('Exact public proof length differs')
        public[name]=body;pins[name]=d['sha256']
    cap=admit(public,pins)
    source_end(Path(__file__).parent,expected)
    return cap,request

@contextmanager
def shield(state):
    state['restored']=False
    with cleanup_signals():yield
    state['restored']=True

def prepare_pytest_temp(output):
    temporary=Output(Path(output.path)/'runtime-temporary')
    try:
        os.environ['TMPDIR']=temporary.path
        import tempfile
        tempfile.tempdir=None
        if Path(tempfile.gettempdir())!=Path(temporary.path):raise ValueError('Owned preflight temporary directory required')
    finally:temporary.close()

def supervise(output,report,action,finalizers):
    """Acquisition belongs to caller; existing output collision is never owned."""
    primary=body=None;state={'restored':False};cleanup_errors=[];result=None;outer_completed=False
    def attempt(label,op):
        nonlocal primary
        try:return op()
        except BaseException as error:
            cleanup_errors.append(label);primary=choose(primary,error,label,body_primary=body);return None
    def finish():
        scope=None;entered=False
        try:scope=shield(state);scope.__enter__();entered=True
        except BaseException as error:
            cleanup_errors.append('shield entry')
            nonlocal primary
            primary=choose(primary,error,'shield entry',body_primary=body)
        for label,op in finalizers:attempt(label,op)
        attempt('body output close',output.close)
        if entered:attempt('shield exit',lambda:scope.__exit__(None,None,None))
    ran=False
    try:
        with termination_signals():
            try:result=action()
            except BaseException as error:
                primary=error;body=None if cleanup_origin(error)else protect_body(error)
            finally:ran=True;finish()
        outer_completed=True
    except BaseException as error:
        if primary is None and not cleanup_origin(error):body=protect_body(error)
        primary=choose(primary,error,'outer termination scope',body_primary=body)
    if not ran:finish()
    report.update(cleanup_completed=not cleanup_errors,signal_restoration_completed=state['restored']and outer_completed,outer_termination_completed=outer_completed,
        terminal_after_body_cleanup_and_signal_restoration_attempts=True,requires_independent_outer_receipt=True)
    if primary is None and not cleanup_errors and state['restored']and outer_completed:
        report.update(complete=True,passed=True)
    else:
        report.update(complete=False,passed=False,error_type=type(primary).__name__ if primary is not None else 'CleanupIncomplete')
    publication=None
    try:
        with cleanup_signals():
            try:
                publication=Output.reopen(output.path,output.owner)
                if report['passed']and result is not None:
                    report['review']=publication.json('review.json',result)
                publication.json('report.json',report)
            except BaseException as error:primary=choose(primary,error,'terminal publication',body_primary=body)
            finally:
                if publication is not None:
                    try:publication.close(primary)
                    except BaseException as error:primary=choose(primary,error,'terminal publication close',body_primary=body)
    except BaseException as error:primary=choose(primary,error,'publication signal scope',body_primary=body)
    # Publication/close can fail after bytes exist. No file self-certifies that
    # last close; nonzero exit and the required independent outer veto admission.
    return 0 if primary is None and report['passed']else 1

def worker(args,raw,expected,blobs,origins,root):
    actual=args.mode=='reconstruct-worker'
    protected=protected_paths(blobs);guard_path(args.output,protected,output=True)
    audit=Audit(args,root,protected);audit.install()
    admitted=public_request(args,raw,expected,blobs,audit)[0]if actual else None
    if not actual and any(getattr(args,k,None)for k in('grant','grant_sha256','request','request_sha256')):raise ValueError('Synthetic worker cannot receive actual proof descriptors')
    output=Output(args.output)
    report=dict(schema='qwen-reconstruction-worker-v1',pid=os.getpid(),passed=False,complete=False,source_manifest_sha256=args.manifest_sha256,
        source_hashes_start=expected,actual_reconstruction=actual,model_forward=False,validation_read=False,private_read_attempted=False,private_read_confirmed=False,private_read_state='not_attempted',audit_denials=audit.denials)
    def sources():report['source_hashes_end']=source_end(root,expected)
    def action():
        if actual:
            # Public admission is complete, and this explicit attempted flag
            # precedes every private acquisition/read, including failed opens.
            report['private_read_attempted']=True
            report['private_read_state']='attempted_unknown';audit.allow_private(admitted)
            result=run(admitted,source_check=sources,cleanup_scope=shield)
            report.update(private_read_confirmed=True,private_read_state='confirmed',verification_passed=True,scientific_pass=result['scientific_pass'],complete_scientific_failure_eligible=result['complete_scientific_failure_eligible'])
            if audit.denials:raise ValueError('Caught audit denial vetoes reconstruction')
            return result
        prepare_pytest_temp(output)
        import pytest
        xml=Path(output.path)/'tests.xml'
        code=pytest.main(['-q','-p','no:cacheprovider','--noconftest','--rootdir',str(root),'--basetemp',str(Path(output.path)/'synthetic-temporary'),'--junitxml',str(xml),*[str(root/n)for n in sorted(expected)if n.startswith('test_')and n.endswith('.py')]])
        tree=ET.fromstring(read(xml,limit=8<<20));cases=tree.findall('.//testcase')
        report.update(tests=len(cases),failures=len(tree.findall('.//failure')),errors=len(tree.findall('.//error')),skipped=len(tree.findall('.//skipped')))
        expected_tests=parse(blobs['preflight_expectations.json'])['tests']
        if code!=0 or len(cases)!=expected_tests or report['failures']or report['errors']or report['skipped']or audit.denials:raise ValueError('All synthetic reconstruction checks and zero audit denials required')
        return None
    return supervise(output,report,action,[('worker source recheck',sources)])

def copy_execution(output,blobs):
    root=Path(output.path)/'execution_source';owners=[];primary=None
    try:
        owners.append(Output(root));by_path={root:owners[0]}
        for name,data in sorted(blobs.items()):
            target=root/name
            missing=[];parent=target.parent
            while parent not in by_path:missing.append(parent);parent=parent.parent
            for parent in reversed(missing):
                by_path[parent]=Output(parent);owners.append(by_path[parent])
            output.check();by_path[target.parent].write(target.name,data)
    except BaseException as error:primary=protect_body(error);raise
    finally:
        errors=[]
        for owner in reversed(owners):
            try:owner.close(primary)
            except BaseException as error:errors.append(error)
        for error in errors:primary=choose(primary,error,'execution copy close')
        if errors:raise primary
    return root

def collect(args,raw,expected,blobs,origins,root):
    actual=args.mode=='collect-reconstruct'
    protected=protected_paths(blobs);guard_path(args.output,protected,output=True)
    for key in('grant','request'):
        value=getattr(args,key,None)
        if value:guard_path(value,protected)
    audit=Audit(args,root,protected);audit.install()
    if actual:public_request(args,raw,expected,blobs,audit)
    elif any(getattr(args,k,None)for k in('grant','grant_sha256','request','request_sha256')):raise ValueError('Preflight cannot accept actual producer proofs')
    output=Output(args.output) # Fresh acquisition precedes all publication paths.
    report=dict(schema='qwen-reconstruction-collector-v1',pid=os.getpid(),passed=False,complete=False,source_manifest_sha256=args.manifest_sha256,
        source_hashes_start=expected,actual_reconstruction=actual,model_forward=False,validation_read=False,audit_denials=audit.denials)
    child=None;logs=[];pending_fd=None;execution=None
    def sources():
        report['source_hashes_end']=source_end(root,expected)
        if execution is not None:report['execution_source_hashes_end']=source_end(execution,expected)
    def action():
        nonlocal child,pending_fd,execution
        report['source_archive']=output.write('sources.tar.gz',archive_bytes(blobs));output.write('source_manifest.json',raw)
        execution=copy_execution(output,blobs);report['execution_source_hashes_start']=source_end(execution,expected)
        command=[sys.executable,'-I','-B',str(execution/'entry.py'),'--manifest',str(Path(output.path)/'source_manifest.json'),'--manifest-sha256',args.manifest_sha256,
            '--mode','reconstruct-worker'if actual else'test-worker','--output',str(Path(output.path)/'worker')]
        if actual:command += ['--grant',args.grant,'--grant-sha256',args.grant_sha256,'--request',args.request,'--request-sha256',args.request_sha256]
        for name in('stdout.log','stderr.log'):
            pending_fd=os.open(name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC,0o600,dir_fd=output.fd);logs.append(pending_fd);pending_fd=None
        env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTEST_ADDOPTS='',PYTEST_PLUGINS='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        child=OwnedChild(command,cwd=str(execution),stdout=logs[0],stderr=logs[1],env=env);report['child']=child.record
        code=child.wait();report['child']=dict(child.record)
        data=read(Path(output.path)/'worker/report.json',limit=8<<20);observed=parse(data);report['worker_report_sha256']=sha(data);report['worker']=observed
        if code!=0 or any(observed.get(k)is not True for k in('passed','complete','cleanup_completed','signal_restoration_completed','terminal_after_body_cleanup_and_signal_restoration_attempts','requires_independent_outer_receipt')):raise ValueError('Independently completed worker required')
        same(observed['source_hashes_start'],expected);same(observed['source_hashes_end'],expected)
        if observed.get('audit_denials')!=[]or audit.denials:raise ValueError('Audit denial vetoes collector acceptance')
        if actual:
            review=read(Path(output.path)/'worker/review.json',observed['review']['sha256'],limit=8<<20);report['worker_review_sha256']=sha(review)
            report.update(verification_passed=True,scientific_pass=observed['scientific_pass'],complete_scientific_failure_eligible=observed['complete_scientific_failure_eligible'])
        return None
    def close_child():
        if child is not None:
            try:child.close()
            finally:report['child']=dict(child.record)
    def close_logs():
        failure=None
        for fd in logs+([]if pending_fd is None else[pending_fd]):
            try:os.close(fd)
            except BaseException as error:failure=choose(failure,error,'log descriptor close')
        if failure is not None:raise failure
    return supervise(output,report,action,[('owned child retirement',close_child),('log close',close_logs),('all source rechecks',sources)])

def dispatch(args,raw,expected,blobs,origins):
    root=Path(__file__).parent
    output=Path(args.output)
    protected=protected_paths(blobs);guard_path(output,protected,output=True)
    for key in('manifest','grant','request'):
        value=getattr(args,key,None)
        if value:guard_path(value,protected)
    if not output.is_absolute()or output!=output.resolve()or output==root or root in output.parents or output in root.parents:raise ValueError('Canonical output must be disjoint from reconstruction sources')
    if args.mode in('collect-preflight','collect-reconstruct'):return collect(args,raw,expected,blobs,origins,root)
    return worker(args,raw,expected,blobs,origins,root)
