"""Direct readonly validator: external admission precedes NumPy and store IO.

No recording factory, key/context/model/HE or export path is selected. Root owns
the direct process and must independently attest its final report/exit/reap.
"""
import hashlib
import os
from pathlib import Path
import resource
import stat
import sys
import threading
from cleanup import cleanup_signals,termination_signals
from cleanup_origin import protect_body,select
from qwen_finalizer import Finalizer
from qwen_file_contract import canonical,sha,same
from qwen_public_parse import parse
from model_admission import MODEL_ID,AGGREGATE_SHA256,CATALOG_SHA256,CHECKS,SCOPE_FALSE
from model_validation_grant import authenticate,public_requests,policies,FORBIDDEN
from model_output import Output,read,canonical_path,overlap,finish

_QUARANTINE=[]


def _accessor(fn): return (getattr(fn,'__self__',None),getattr(fn,'__func__',fn))
def _same_accessor(a,b): return all(x is y for x,y in zip(_accessor(a),_accessor(b)))
def _file_state(info):
    return dict(device=info.st_dev,inode=info.st_ino,size=info.st_size,
        mtime_ns=info.st_mtime_ns,ctime_ns=info.st_ctime_ns,mode=stat.S_IMODE(info.st_mode))


def _role_path(value,blocked):
    p=canonical_path(value)
    if p.suffix.lower()in ('.npy','.npz','.safetensors','.parquet') or any(overlap(p,b)for b in blocked):
        raise ValueError('public role path overlaps source/data/output')
    return p


def admit(args,raw_map,expected,blobs):
    policy,dep=policies(blobs)
    root=canonical_path(str(Path(__file__).parent));output=canonical_path(args.output)
    static=[Path(x)for x in policy['protected_paths']+policy['bootstrap_borrowed_roots']]
    manifest=_role_path(args.manifest,[root,output,*static])
    grant_path=_role_path(args.grant,[root,output,manifest,*static])
    if any(overlap(output,p)for p in [root,manifest,grant_path,*static]):
        raise ValueError('fresh output leaf overlaps a protected role')
    raw_grant=read(grant_path,limit=policy['maximum_grant_bytes'],expected=args.grant_sha256)
    requests=public_requests(raw_grant)
    selected={};paths={}
    for role,maximum in (('expected_stores',policy['maximum_store_metadata_bytes']),('prelaunch_review',policy['maximum_review_bytes'])):
        p=_role_path(requests[role]['path'],[root,output,manifest,grant_path,*static,*paths.values()])
        paths[role]=p
        selected[role]=read(p,limit=maximum,expected=requests[role]['sha256'])
    launch=authenticate(raw_grant,args.grant_sha256,selected['expected_stores'],selected['prelaunch_review'],
        source_root=str(root),output_path=str(output),source_map_bytes=raw_map,
        source_manifest_sha256=args.manifest_sha256,blobs=blobs)
    # This inspects only authenticated metadata. Actual filesystem/root checks
    # are later, and no parameter descriptor is followed here.
    expected_store=launch.stores.check();roots=[]
    for row in expected_store['roots'].values():
        p=canonical_path(row['path'])
        if any(overlap(p,b)for b in [root,output,manifest,grant_path,*paths.values(),*roots,*map(Path,policy['protected_paths'])]):
            raise ValueError('borrowed store overlaps an unrelated protected role')
        roots.append(p)
    if any(overlap(output,p)for p in roots): raise ValueError('output overlaps borrowed store')
    # The parent can contain protected siblings. Its trust/identity checks are
    # performed by Output; only the leaf is subject to the disjointness test.
    try: output.lstat()
    except FileNotFoundError: pass
    else: raise FileExistsError('evidence output must be fresh')
    return launch,policy,dep,[manifest,grant_path,*paths.values()]


def capacity(launch,policy,dep):
    if sorted(os.sched_getaffinity(0))!=policy['cores']: raise ValueError('exact readonly CPU lane required')
    if (list(sys.version_info[:2])!=dep['python_major_minor'] or os.path.abspath(sys.executable)!=dep['interpreter']
            or Path(sys.executable).resolve()!=Path(dep['interpreter']).resolve()):
        raise ValueError('selected interpreter differs')
    for name,value in dep['environment'].items():
        if os.environ.get(name)!=value: raise ValueError('selected host thread environment differs')
    soft,_=resource.getrlimit(resource.RLIMIT_NOFILE)
    baseline=len(os.listdir('/proc/self/fd'))
    required=baseline+policy['planning_body_fds']+policy['runtime_fd_allowance']
    if soft!=resource.RLIM_INFINITY and (soft<policy['minimum_soft_nofile'] or soft<required):
        raise ValueError('insufficient descriptor baseline/headroom')
    rows=launch.stores.check()['files'];extent=sum(r['state']['size']for r in rows if r['family']!='gamma')
    # procfs reports st_size=0; this explicit kernel-metadata read is not the
    # bounded ordinary-file reader and never selects a data descriptor path.
    primary=None;fd=os.open('/proc/meminfo',os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
    try:
        blocks=[];size=0
        while size<=1<<20:
            block=os.read(fd,min(65536,(1<<20)+1-size))
            if not block: break
            blocks.append(block);size+=len(block)
        if not 0<size<=1<<20: raise ValueError('bounded kernel memory metadata required')
        raw=b''.join(blocks)
    except BaseException as error:
        primary=protect_body(error);raise
    finally:
        original=fd;fd=None
        def close_metadata():
            try: os.close(original)
            except BaseException as error:
                _QUARANTINE.append(('kernel metadata',original,error));raise
        finish(primary,close_metadata,'kernel metadata descriptor close')
    lines=[line for line in raw.decode('ascii').splitlines()if line.startswith('MemAvailable:')]
    if len(lines)!=1: raise ValueError('one available-memory observation required')
    tokens=lines[0].split()
    if len(tokens)!=3 or tokens[2]!='kB': raise ValueError('available-memory unit differs')
    available=int(tokens[1])*1024
    if available<extent+policy['memory_margin_bytes']: raise ValueError('insufficient conservative mapping headroom')
    space=os.statvfs(Path(launch.output_path).parent)
    if space.f_bavail*space.f_frsize<policy['evidence_reserve_bytes']: raise ValueError('insufficient fresh evidence reserve')
    return dict(open_fd_baseline=baseline,required_fd_capacity=required,soft_nofile=soft,
        mapped_extent=extent,available_memory=available,available_evidence_bytes=space.f_bavail*space.f_frsize)


class ReadScope:
    """Trusted-source Python audit and dirfd attribution, not an OS sandbox."""
    def __init__(self,launch,policy,dep,public_paths):
        self.denials=[];self.root=Path(launch.source_root);self.output=Path(launch.output_path)
        self.dep=dep;self.forbidden=set(FORBIDDEN);self.local=threading.local()
        self.original_open=os.open;self.wrapper=None;self.installed=False
        self.closed=False;self.failure=None;self.body_enabled=False
        stores=launch.stores.check();self.roots={Path(r['path'])for r in stores['roots'].values()}
        self.body={Path(stores['roots'][r['family']]['path'])/r['name']:r for r in stores['files']}
        self.public=set(public_paths)|{self.root/name for name,_ in launch.blobs}|set(map(Path,dep['numpy_sources']))
        self.libraries=tuple(Path(p).resolve()for p in dep['library_roots'])
        self.protected=tuple(map(Path,policy['protected_paths']))

    def deny(self,reason):
        self.denials.append(reason);raise PermissionError('readonly model audit denied '+reason)

    def resolve_open(self,value,dir_fd=None):
        if type(value)is int:
            return Path(os.readlink('/proc/self/fd/'+str(value)))
        p=Path(os.fsdecode(value))
        if not p.is_absolute():
            if dir_fd is None: p=Path(os.getcwd())/p
            else: p=Path(os.readlink('/proc/self/fd/'+str(dir_fd)))/p
        return p

    def access(self,p,flags):
        if type(flags)is not int: self.deny('untyped open flags')
        if not p.is_absolute() or p.resolve()!=p or '..'in p.parts: self.deny('open alias/path')
        write=bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if write:
            required=os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW
            if p.parent!=self.output or p.name not in('report.json','failure.json') or flags&required!=required or flags&(os.O_TRUNC|os.O_APPEND):
                self.deny('write outside exclusive evidence')
            return
        if p in self.body:
            if not self.body_enabled: self.deny('body before final source/resource gate')
            return
        if any(p==r or r in p.parents for r in self.protected): self.deny('protected body')
        if p in self.public or p in self.roots or p in(self.output,self.output.parent): return
        if p==self.root or self.root in p.parents:
            if p.is_dir(): return
            self.deny('undeclared source read')
        if p in (Path('/proc/meminfo'),Path('/proc/self/status')): return
        # Only installed code/shared libraries, never arbitrary package data.
        if any(p==r or r in p.parents for r in self.libraries) and (p.suffix in('.py','.pyc','.so','.pyd')or '.so.'in p.name): return
        self.deny('unselected file read')

    def hook(self,event,args):
        if event=='open':
            if not args: self.deny('missing open arguments')
            context=getattr(self.local,'open_context',None)
            p=context[1]if context is not None else self.resolve_open(args[0])
            flags=args[2]if len(args)>2 else None
            self.access(p,flags)
        elif event=='mmap.__new__':
            fd,length,access,offset=args
            if type(fd)is not int or fd<0: self.deny('unselected anonymous Python mmap')
            p=self.resolve_open(fd)
            if not self.body_enabled or p not in self.body or access!=1 or offset!=0 or length not in(0,self.body[p]['state']['size']):
                self.deny('unselected writable or foreign mapping')
            same(_file_state(os.fstat(fd)),self.body[p]['state'],'mapping descriptor expectation differs')
        elif event=='os.mkdir':
            p=self.resolve_open(args[0],args[2]if len(args)>2 and args[2]!=-1 else None)
            if p!=self.output or args[1]!=0o700: self.deny('unexpected directory creation')
        elif event in ('os.remove','os.rmdir','os.rename','os.link','os.symlink','os.chmod','os.chown','os.utime','os.truncate'):
            self.deny('mutation or removal')
        elif event.startswith('socket.') or event in ('os.fork','os.forkpty','os.posix_spawn','os.posix_spawnp','os.system','subprocess.Popen','os.exec'):
            self.deny('network or process creation')
        elif event=='import' and args and str(args[0]).split('.')[0]in self.forbidden:
            self.deny('forbidden backend import')

    def install(self):
        original=self.original_open
        def opened(file,flags,mode=0o777,*,dir_fd=None):
            p=self.resolve_open(file,dir_fd);self.access(p,flags)
            previous=getattr(self.local,'open_context',None)
            self.local.open_context=(file,p)
            try: return original(file,flags,mode,dir_fd=dir_fd)
            finally: self.local.open_context=previous
        self.wrapper=opened
        sys.addaudithook(self.hook);self.installed=True;os.open=opened

    def check(self):
        if self.failure is not None: raise self.failure
        if not self.installed or self.closed or os.open is not self.wrapper: raise ValueError('original readonly open attribution changed')
        if self.denials: raise PermissionError('sticky readonly audit denial')

    def close(self):
        if self.closed: return
        try:
            if os.open is not self.wrapper: raise ValueError('readonly open wrapper changed before restoration')
        finally:
            os.open=self.original_open;self.closed=True


def dependency_check(dep,*,loaded):
    for p,h in dep['numpy_sources'].items(): read(p,limit=8<<20,expected=h)
    forbidden=[n for n in sys.modules if n.split('.')[0]in FORBIDDEN]
    if forbidden: raise ValueError('unselected backend module loaded')
    if loaded:
        for name,expected in dep['numpy_origins'].items():
            module=sys.modules.get(name)
            for value in (getattr(module,'__file__',None),getattr(getattr(module,'__spec__',None),'origin',None)):
                if not isinstance(value,(str,os.PathLike)) or Path(value).resolve()!=Path(expected):
                    raise ValueError('genuine selected NumPy module origin differs')
    elif any(n.split('.')[0]=='numpy'for n in sys.modules):
        raise ValueError('NumPy imported before admitted dependency gate')


def observed_stores(owner,expected_store):
    """Reconstruct all records from original live descriptors, not grant echoes."""
    owner.check();expected=expected_store.check();roots={};files=[]
    for family,held in owner.roots.items():
        held.check();info=os.fstat(held.fd)
        roots[family]=dict(path=str(held.path),device=info.st_dev,inode=info.st_ino,mode=stat.S_IMODE(info.st_mode))
    catalog=parse(owner.catalog.matrices)
    descriptors={(r['family'],r['descriptor']['file']):r['descriptor']for r in catalog.values()}
    for (family,name),lease in owner.matrices.items():
        lease.file.check();description=descriptors[(family,name)]
        # Digests are the fixed catalog descriptors whose complete file/raw
        # readbacks succeeded; state itself is obtained freshly from the FD.
        files.append(dict(family=family,name=name,state=_file_state(os.fstat(lease.file.fd)),
            file_sha256=lease.file.expected,raw_sha256=description['sha256']))
    gamma=owner.gamma_lease;gamma.check()
    files.append(dict(family='gamma',name=gamma.name,state=_file_state(os.fstat(gamma.fd)),file_sha256=gamma.expected,raw_sha256=None))
    result=dict(roots=roots,files=sorted(files,key=lambda r:(r['family'],r['name'])))
    same(result,expected,'observed held stores differ from external grant expectation')
    return result


def run(args,raw_map,expected,blobs,finder,snapshot,origins):
    launch,policy,dep,public_paths=admit(args,raw_map,expected,blobs)
    capacity_observation=capacity(launch,policy,dep)
    audit=ReadScope(launch,policy,dep,public_paths)
    output=None;binding=None;owner=None;callbacks={};owner_catalog=None;catalog_state=None
    primary=None;cleanup_report={};cleanup_ran=False;signals_restored=False
    body_complete=False;body_entered=False;loaded=False;observed=None;source_after=None
    logical_calls=gamma_calls=0;cleanup_ok=False
    stores=launch.stores;public=stores.aggregate.public
    original=(launch,stores,public,launch.blobs,launch.grant_bytes,stores.raw)
    def guard():
        audit.check();launch.check()
        if any(a is not b for a,b in zip((launch,stores,public,launch.blobs,launch.grant_bytes,stores.raw),original)):
            raise ValueError('original readonly launch binding changed')
        if owner is not None:
            for name,fn in callbacks.items():
                if not _same_accessor(getattr(owner,name,None),fn): raise ValueError('original model owner callback changed')
            if owner_catalog is not None and (owner.catalog is not owner_catalog or owner.expected_store is not stores or
                    (owner_catalog.matrices,owner_catalog.gammas,owner_catalog.gamma_container_sha256,owner_catalog.provenance)!=catalog_state):
                raise ValueError('original model owner admission changed')
    def source_check():
        nonlocal source_after
        guard();values=snapshot(expected)
        if values!=blobs: raise ValueError('current source bytes differ from authenticated startup')
        origins(expected);source_after={n:sha(raw)for n,raw in values.items()}
        return source_after
    def cleanup_all():
        nonlocal primary,cleanup_ran,cleanup_ok
        cleanup_ran=True;final=Finalizer(cleanup_report,primary)
        captured_owner=owner;captured=dict(callbacks)
        def retire():
            final.attempt('original validation ownership',guard)
            if 'readback'in captured: final.attempt('original complete parameter readback',captured['readback'])
            final.attempt('ownership after final readback',guard)
            if 'close'in captured: final.attempt('original parameter retirement',lambda:captured['close'](primary=final.selected()))
            final.attempt('ownership after parameter retirement',guard)
            final.attempt('selected dependency source/origins',lambda:dependency_check(dep,loaded=loaded))
            final.attempt('terminal selected source check',source_check)
            if output is not None: final.attempt('original evidence body handles close',output.close)
        final.shielded(cleanup_signals,retire);primary=final.selected()
        cleanup_ok=not cleanup_report.get('cleanup_errors')
        if primary is not None:
            _QUARANTINE.append((launch,original,stores,captured_owner,captured,owner_catalog,catalog_state,output,primary))
    try:
        audit.install()
        try:
            with termination_signals():
                try:
                    dependency_check(dep,loaded=False);source_check()
                    output=Output(launch.output_path);binding=output.binding()
                    finder.allow_numpy=True
                    from qwen_public_parameters import ParameterCatalog,open_identity_bound_parameters
                    loaded=True;dependency_check(dep,loaded=True);source_check()
                    catalog=ParameterCatalog.from_public(public)
                    expected_catalog=dict(matrices=parse(catalog.matrices),gammas=parse(catalog.gammas),
                        gamma_container_sha256=catalog.gamma_container_sha256,provenance=parse(catalog.provenance))
                    same(expected_catalog,stores.aggregate.catalog,'full catalog rederivation differs before store acquisition')
                    logical=parse(catalog.matrices);gammas=parse(catalog.gammas)
                    physical={(v['family'],v['descriptor']['file'])for v in logical.values()}
                    if (len(logical),len(physical),len(gammas))!=(198,197,113): raise ValueError('complete selected catalog required')
                    source_check();audit.body_enabled=True;body_entered=True
                    owner=open_identity_bound_parameters(public,stores)
                    callbacks['close']=owner.close
                    for name in ('check','matrix','gamma','readback'): callbacks[name]=getattr(owner,name)
                    owner_catalog=owner.catalog
                    if type(owner_catalog)is not ParameterCatalog: raise TypeError('actual selected parameter catalog required')
                    catalog_state=(owner_catalog.matrices,owner_catalog.gammas,owner_catalog.gamma_container_sha256,owner_catalog.provenance)
                    for name in sorted(logical):
                        guard();callbacks['matrix'](name);logical_calls+=1;guard();callbacks['check']()
                    if type(owner.matrices)is not dict or set(owner.matrices)!=physical: raise ValueError('all197 physical leases required')
                    for name in sorted(gammas):
                        guard();callbacks['gamma'](name);gamma_calls+=1;guard();callbacks['check']()
                    if type(owner.gamma_values)is not dict or set(owner.gamma_values)!=set(gammas): raise ValueError('all113 gammas required')
                    callbacks['readback']();guard();observed=observed_stores(owner,stores);source_check()
                    body_complete=True
                except BaseException as error: primary=protect_body(error)
                finally: cleanup_all()
            signals_restored=True
        except BaseException as error:
            primary=select(primary,[('body signal scope',protect_body(error))])
        finally:
            if not cleanup_ran: cleanup_all()
        success=primary is None and cleanup_ok and signals_restored and body_complete and source_after==expected and not audit.denials
        if success:
            report=dict(schema='qwen-model-store-validation-v1',mode='reuse_existing_exports',passed=True,complete=True,error_type=None,
                model_id=MODEL_ID,aggregate_manifest_sha256=AGGREGATE_SHA256,catalog_sha256=CATALOG_SHA256,
                execution_grant_sha256=launch.grant_sha256,source_manifest_sha256=launch.source_manifest_sha256,
                source_archive_sha256=parse(launch.grant_bytes)['source_archive_sha256'],source_hashes_before=expected,
                source_hashes_after=source_after,process=dict(pid=os.getpid(),pgid=os.getpgrp()),stores=observed,
                counts=dict(logical_matrices=198,physical_matrices=197,gamma_vectors=113,
                    physical_matrix_raw_bytes=4767875072,gamma_raw_bytes=524288),checks={n:True for n in CHECKS},
                scope=dict(parameter_body_reads_performed=True,**{n:False for n in SCOPE_FALSE}),audit_denials=[])
        else:
            report=dict(schema='qwen-model-store-validation-failure-v1',passed=False,complete=False,
                model_id=MODEL_ID,execution_grant_sha256=launch.grant_sha256,source_manifest_sha256=launch.source_manifest_sha256,
                process=dict(pid=os.getpid(),pgid=os.getpgrp()),error_type=type(primary).__name__ if primary is not None else'IncompleteValidation',
                body_integrity_admitted=False,parameter_body_reads_performed=True if body_complete else None if body_entered else False,
                body_read_status='complete'if body_complete else'entered_unconfirmed'if body_entered else'not_entered',
                completed_logical_visits=logical_calls,completed_gamma_visits=gamma_calls,
                cleanup_completed=cleanup_ok,signals_restored=signals_restored,audit_denials=list(audit.denials))
        publication=None;terminal=Finalizer({},primary)
        def publish():
            nonlocal publication
            if binding is None: return
            publication=Output.reopen(binding)
            try: publication.write('report.json'if success else'failure.json',canonical(report))
            finally: terminal.attempt('final original publication close',publication.close)
        terminal.shielded(cleanup_signals,publish)
        terminal.attempt('post-publication sticky audit veto',audit.check)
        terminal.attempt('original open-attribution restoration',audit.close)
        terminal.raise_if_failed()
        if not success: raise RuntimeError('readonly model validation incomplete')
        return 0
    except BaseException as error:
        protect_body(error)
        final=Finalizer({},error)
        if audit.installed and not audit.closed: final.attempt('late original open-attribution restoration',audit.close)
        _QUARANTINE.append((launch,stores,owner,dict(callbacks),output,binding,capacity_observation,error))
        final.raise_if_failed()
