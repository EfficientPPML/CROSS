"""Prospective dedicated full28 server process; current resource gate denies.

This source does not issue grants, scientific outcomes or outer receipts.
The device/workspace admission remains unimplemented, so even a structurally
valid future packet cannot enable a backend or body through this version.
"""
from dataclasses import dataclass
import os
import sys
from pathlib import Path
from types import FunctionType,ModuleType
from cleanup import cleanup_signals,termination_signals
from cleanup_origin import protect_body
from qwen_finalizer import Finalizer
from qwen_file_contract import canonical,sha,same,hash_value
from qwen_public_parse import parse
from client_role_common import raw_object,obj
from qwen_server_authority_packet import public_from_sources,inspect_packet,authenticate_packet,PACKET_LIMIT
from qwen_server_admission import ExpectedServerPhasePins
from qwen_server_dependencies import (policies,exact_environment,hash_installed,imported_origins,
    AccessScope,object_bytes,DEPENDENCY_FIELDS)
from qwen_server_resource import check_host_capacity,require_complete_capacity
from qwen_server_public_memory import check_public_memory
from qwen_key_contract import ExpectedIds,PublicProfile,authenticate_public
from qwen_key_files_admission import bind_files
from keygen_output import Output,read,canonical_path,overlap
import qwen_admission_snapshot as admission_snapshot

_QUARANTINE=[]
_ISSUER=object()
_RUNTIME_CONTROLS={}
EXPECTED_ID_FIELDS=('context_id','public_manifest_sha256','keygen_source_sha256','model_id',
    'producer_source_manifest_sha256','consumer_source_manifest_sha256','private_manifest_sha256')
# Exactly the callback's local source closure. Entry/terminal still check the
# entire external manifest; hot checks never reread the public proof packet.
SNAPSHOT_BINDING_SOURCES=(
    'qwen_admission_snapshot.py','qwen_server_authority_packet.py','qwen_server_admission.py',
    'qwen_scientific_admission.py','qwen_file_contract.py','model_admission.py','keygen_authority.py',
    'client_role_common.py','freeze_admission.py','freeze_issuer.py','qwen_key_contract.py',
    'qwen_public_parse.py','public_source_pins.py','keygen_boundary_pins.py','scientific_source_pins.py',
    'qwen_server_process.py','qwen_server_backend.py','qwen_server_entry.py',
    'qwen_server_dependencies.py','keygen_output.py','cleanup_origin.py')
MODEL_STORAGE_SCOPES=(
    '/tmp/qwen-qkv-attention-join-20261103/public_qkv_parameters',
    '/tmp/qwen-decoder-parameters-20261103/actual-v1/public_matrices',
    '/tmp/qwen-rmsnorm-owner-20261103')


def _accessor(value):return(getattr(value,'__self__',None),getattr(value,'__func__',value))
def _same_accessor(a,b):return all(x is y for x,y in zip(_accessor(a),_accessor(b)))


def _function_binding(fn):
    if type(fn)is not FunctionType:raise TypeError('original source-fixed Python callback required')
    kw=fn.__kwdefaults__
    if kw is not None and type(kw)is not dict:raise TypeError('exact callback keyword defaults required')
    return(fn,fn.__code__,fn.__defaults__,kw,tuple(kw.items())if kw is not None else(),
        fn.__globals__,fn.__module__,fn.__name__,fn.__qualname__)


def _check_function(saved):
    fn,code,defaults,kw,items,namespace,module,name,qualname=saved
    if(fn.__code__ is not code or fn.__defaults__ is not defaults or fn.__kwdefaults__ is not kw
            or fn.__globals__ is not namespace or fn.__module__!=module or fn.__name__!=name or fn.__qualname__!=qualname):
        raise ValueError('original source callback code/defaults/origin changed')
    if kw is not None and(set(kw)!=set(k for k,v in items)or any(kw[k]is not v for k,v in items)):
        raise ValueError('original callback keyword defaults changed')


def _module_binding(module,name,path,*,entry=False):
    if type(module)is not ModuleType or sys.modules.get(name)is not module or module.__name__!=name:
        raise ValueError('original selected binding module required')
    spec=module.__spec__;file=module.__file__
    if type(file)is not str or file!=path or Path(file).resolve()!=Path(file):raise ValueError('canonical original binding file required')
    origin=None if entry else getattr(spec,'origin',None)
    if(entry and spec is not None)or(not entry and(type(origin)is not str or origin!=path)):
        raise ValueError('original independent binding spec required')
    exports=[];classes=[];functions=[]
    for key,value in tuple(module.__dict__.items()):
        if type(value)is FunctionType:
            exports.append((key,value));functions.append(_function_binding(value))
        elif type(value)is type:
            exports.append((key,value))
            if value.__module__!=name:continue
            members=tuple(value.__dict__.items());classes.append((value,members))
            for _,member in members:
                if type(member)is FunctionType:functions.append(_function_binding(member))
                elif type(member)is property:
                    for fn in(member.fget,member.fset,member.fdel):
                        if fn is not None:functions.append(_function_binding(fn))
                elif type(member)in(staticmethod,classmethod):functions.append(_function_binding(member.__func__))
    return(module,name,spec,file,origin,tuple(exports),tuple(classes),tuple(functions),entry)


def _check_module_binding(saved):
    module,name,spec,file,origin,exports,classes,functions,entry=saved
    if(sys.modules.get(name)is not module or module.__name__!=name or module.__spec__ is not spec
            or module.__file__ is not file or(not entry and spec.origin is not origin)):
        raise ValueError('original binding module/spec/file changed')
    for key,value in exports:
        if module.__dict__.get(key)is not value:raise ValueError('original binding module export changed')
    for cls,members in classes:
        now=cls.__dict__
        if set(now)!=set(k for k,v in members)or any(now[k]is not v for k,v in members):
            raise ValueError('original binding class descriptor changed')
    for fn in functions:_check_function(fn)


def _capture_snapshot(packet,root,expected,blobs,dep,gate,snapshot,origins,retained):
    """One local graph owner. The callback is nonrecursive and has no body gate."""
    retained.update(packet=packet,root=root,expected=expected,blobs=blobs,dep=dep,gate=gate,
        snapshot=snapshot,origins=origins,status='capturing',guard=None)
    expected_bytes=canonical(expected);dep_bytes=canonical(dep)
    source_rows=tuple((name,expected[name],blobs[name])for name in SNAPSHOT_BINDING_SOURCES)
    all_blobs=tuple(blobs.items());retained.update(source_rows=source_rows,all_blobs=all_blobs)
    if any(type(raw)is not bytes or sha(raw)!=digest for _,digest,raw in source_rows):
        raise ValueError('selected snapshot binding source differs')
    callbacks=tuple(_function_binding(fn)for fn in(snapshot,origins,exact_environment,read))
    # The backend module's top level is stdlib/pure declarations only. Capture
    # it before any application callback, not lazily after a replacement seam.
    modules=[]
    for relative,_,_ in source_rows:
        name='__main__'if relative=='qwen_server_entry.py'else relative[:-3]
        modules.append(_module_binding(sys.modules.get(name),name,str(Path(root)/relative),entry=name=='__main__'))
    modules=tuple(modules);retained.update(modules=modules,callbacks=callbacks)
    guard_module=admission_snapshot;guard_class=guard_module.AdmissionSnapshot
    capture=guard_module.capture_admitted;check_method=guard_class.check;retire_method=guard_class.retire
    retained.update(guard_module=guard_module,guard_class=guard_class,capture=capture,
        check_method=check_method,retire_method=retire_method)
    module_checker=_check_module_binding;module_checker_code=module_checker.__code__
    function_checker=_check_function;function_checker_code=function_checker.__code__
    callback_identity=None
    def implementation_bindings():
        if(_check_module_binding is not module_checker or module_checker.__code__ is not module_checker_code
                or _check_function is not function_checker or function_checker.__code__ is not function_checker_code):
            raise ValueError('original local binding checker changed before use')
        if(admission_snapshot is not guard_module or sys.modules.get('qwen_admission_snapshot')is not guard_module
                or guard_module.AdmissionSnapshot is not guard_class or guard_module.capture_admitted is not capture
                or guard_class.check is not check_method or guard_class.retire is not retire_method):
            raise ValueError('original snapshot module/class/accessor changed')
        for row in modules:module_checker(row)
        for row in callbacks:function_checker(row)
        if callback_identity is not None:function_checker(callback_identity)
    def check_bindings():
        implementation_bindings()
        if canonical(expected)!=expected_bytes or canonical(dep)!=dep_bytes:
            raise ValueError('original source/environment policy changed')
        if set(blobs)!=set(n for n,b in all_blobs)or any(blobs[n]is not raw for n,raw in all_blobs):
            raise ValueError('original authenticated source bytes changed')
        for name,digest,raw in source_rows:
            if read(Path(root)/name,limit=8<<20,expected=digest)!=raw:
                raise ValueError('selected snapshot binding source file changed')
        origins(expected,core_enabled=gate.core_enabled);exact_environment(dep)
        implementation_bindings()
    callback_identity=_function_binding(check_bindings)
    retained.update(binding_callback=check_bindings,binding_callback_identity=callback_identity)
    try:
        check_bindings()
        guard=capture(packet,check_bindings=check_bindings)
        retained['guard']=guard
        implementation_bindings()
        if type(guard)is not guard_class:raise TypeError('original exact snapshot required')
        retained['status']='active'
        def check():
            try:
                if retained['status']!='active' or retained['guard']is not guard:
                    raise PermissionError('original active server snapshot required')
                implementation_bindings();check_method(guard,packet)
                implementation_bindings()
                return packet
            except BaseException:
                retained['status']='failed';raise
        def retire():
            try:
                if retained['status']!='active' or retained['guard']is not guard:
                    raise PermissionError('only a clean original snapshot may retire')
                implementation_bindings();retire_method(guard,packet)
                retained['status']='retired'
            except BaseException:
                retained['status']='failed';raise
        retained.update(check=check,retire=retire)
        check();return check,retire
    except BaseException:
        retained['status']='failed';raise


def _expected_ids_bytes(value):
    if type(value)is not ExpectedIds:raise TypeError('exact derived expected IDs required')
    value.check('server')
    fields=object.__getattribute__(value,'__dict__')
    if type(fields)is not dict or set(fields)!=set(EXPECTED_ID_FIELDS):raise ValueError('exact seven expected ID fields required')
    return canonical({name:fields[name]for name in EXPECTED_ID_FIELDS})


def _runtime_action(value,action,arg):
    row=_RUNTIME_CONTROLS.get(id(value))
    if type(value)is not RuntimeAdmission or row is None or row[0]is not value:
        raise PermissionError('original process-owned server runtime required')
    return row[1](value,action,arg)


def require_runtime(value,*,_dispatch=_runtime_action):
    # Backend calls the source-fixed dispatcher, never a replaceable instance
    # .check attribute or a newly supplied dataclass _control callable.
    # The original dispatcher reaches the original registry/control and guard
    # even if the module's public dispatcher export has been replaced. Actual
    # source callers never supply the private retention keyword.
    _dispatch(value,'check',None);return value


def resource_pins(phase):
    pins=phase.expected_pins.check()
    return canonical({name:pins[name]for name in('source_manifest_sha256','source_archive_sha256','phase_grant_sha256')}|
        {name:pins['policies'][name]for name in('dependency_policy_sha256','resource_policy_sha256','resource_review_sha256')})


def admit(args,raw_map,expected,blobs):
    status=raw_object(blobs['qwen_server_entry_status.json'],1<<16,'server source readiness')
    same(status,dict(schema='qwen-server-entry-status-v1',actual_entry_ready=True,actual_grant_present=False,
        closure_preflight_passed=True,dependency_policy_instance_supplied=True,resource_policy_instance_supplied=False,
        source_only=False),'reviewed actual-ready server successor required')
    root=canonical_path(str(Path(__file__).parent))
    static=raw_object(blobs['qwen_server_static_policy.json'],1<<16,'server protected paths')
    obj(static,('schema','protected_paths'),'server protected paths')
    if static['schema']!='qwen-server-static-policy-v1'or type(static['protected_paths'])is not list:
        raise ValueError('exact source-selected server protected policy required')
    protected=tuple(canonical_path(p)for p in static['protected_paths'])
    scopes=tuple(map(canonical_path,MODEL_STORAGE_SCOPES))
    if not set(scopes)<=set(protected):raise ValueError('source-fixed model exceptions must remain generally protected')
    records={}
    def public(path,digest,limit,extra=()):
        hash_value(digest);p=canonical_path(path)
        if p.suffix.lower()in('.npy','.npz','.parquet','.safetensors')or any(overlap(p,r)for r in(root,*protected,*extra)):
            raise ValueError('public proof path overlaps source, library, body or output')
        raw=read(p,limit=limit,expected=digest)
        if str(p)in records and records[str(p)]!=(digest,limit,raw):raise ValueError('public proof path has inconsistent bindings')
        records[str(p)]=(digest,limit,raw);return raw
    packet_raw=public(args.packet,args.packet_sha256,PACKET_LIMIT)
    archive_raw=public(args.archive,args.archive_sha256,64<<20)
    inspection=inspect_packet(public_from_sources(blobs),packet_raw,expected_sha256=args.packet_sha256)
    phase_pins=ExpectedServerPhasePins(inspection.phase_pins_bytes);pins=phase_pins.check()
    same(pins['source_map'],expected,'complete selected server source differs')
    if sha(raw_map)!=args.manifest_sha256 or pins['source_manifest_sha256']!=sha(raw_map):raise ValueError('external server source map differs')
    if pins['source_archive_sha256']!=args.archive_sha256 or sha(archive_raw)!=args.archive_sha256:raise ValueError('external server archive differs')
    if inspection.source_root!=str(root)or args.role!='full28_server':raise ValueError('independently granted server source/role differs')
    if inspection.dependency_policy!=blobs['qwen_server_dependencies.json']or inspection.role_policy!=blobs['qwen_server_role_policy.json']:
        raise ValueError('dependency and role policies must be selected source bytes')
    # Inspect only independently pinned path metadata here. Complete dependency
    # semantics are revalidated after full scientific admission below.
    early_dep=object_bytes(inspection.dependency_policy,DEPENDENCY_FIELDS,'qwen-server-tpu-dependencies-v1')
    if type(early_dep['library_roots'])is not list or not early_dep['library_roots']or type(early_dep['installed_sources'])is not dict or not early_dep['installed_sources']:
        raise ValueError('explicit installed source/library path inventory required')
    libraries=tuple(canonical_path(p)for p in early_dep['library_roots'])
    installed_files=tuple(canonical_path(p)for p in early_dep['installed_sources'])
    model_roots=tuple(map(canonical_path,inspection.model_roots));key_roots=tuple(map(canonical_path,inspection.key_roots))
    request_root=canonical_path(inspection.request_root);outputs=tuple(map(canonical_path,inspection.output_roots))
    if any(not any(p==s or s in p.parents for s in scopes)for p in model_roots):raise ValueError('authenticated model root is outside selected storage scope')
    grant=raw_object(parse(packet_raw)['phase_grant'].encode('utf-8'),1<<16,'preliminary server phase grant')
    public_key_root=canonical_path(grant['public_key_store']['path'])
    private_roots=tuple(p for p in key_roots if p!=public_key_root)
    if len(key_roots)!=2 or len(private_roots)!=1 or public_key_root not in key_roots:raise ValueError('exact separate public/private generated roots required')
    always_protected=tuple(p for p in protected if p not in scopes)+private_roots
    if any(overlap(p,r)for p in model_roots for r in always_protected):raise ValueError('model exception overlaps private/token/calibration data')
    if any(overlap(p,r)for p in(public_key_root,request_root,*outputs)for r in protected):raise ValueError('server key/input/output scope overlaps protected data')
    forbidden=tuple(map(canonical_path,inspection.forbidden_proof_roots))+libraries+installed_files
    for name in records:
        if any(overlap(Path(name),p)for p in forbidden):raise ValueError('bootstrap public file overlaps authenticated data/library scope')
    for library in libraries:
        if any(overlap(library,p)for p in(root,*protected,*model_roots,*key_roots,request_root,*outputs)):
            raise ValueError('installed library overlaps server source/data/output')
    for leaf in outputs:
        try:leaf.lstat()
        except FileNotFoundError:pass
        else:raise FileExistsError('independently granted server output must be fresh')
    # Resolve and reject every descriptor before opening the first larger proof.
    for descriptor in inspection.proof_descriptors:
        p=canonical_path(descriptor.path)
        if p.suffix.lower()in('.npy','.npz','.parquet','.safetensors')or any(overlap(p,r)for r in(root,*protected,*forbidden)):
            raise ValueError('scientific proof overlaps protected body/source/library')
    metadata_memory=check_public_memory(phase_pins,parse(packet_raw)['phase_grant'].encode('utf-8'),
        inspection.resource_policy,inspection.resource_review,source_manifest_sha256=sha(raw_map),source_archive_sha256=args.archive_sha256)
    proofs={row.name:public(row.path,row.sha256,row.limit,forbidden)for row in inspection.proof_descriptors}
    packet=authenticate_packet(inspection.public,packet_raw,expected_sha256=args.packet_sha256,scientific_proofs=proofs)
    dep,res=policies(packet)
    metadata=packet.public.check();profile=PublicProfile.from_bytes(metadata['inputs/profile_audit.json'],metadata['inputs/actual_key_plan.json'])
    return packet,profile,dep,res,records,tuple(map(str,protected)),tuple(map(str,always_protected)),metadata_memory


@dataclass(frozen=True)
class RuntimeAdmission:
    phase:object
    core:object
    _issuer:object
    _control:object
    _expected_ids_bytes:bytes
    _input_digest:str
    def check(self,*,_require=require_runtime):
        return _require(self)
    def allow_public_key_reads(self,*,_dispatch=_runtime_action):_dispatch(self,'keys',None)
    def admit_server_bodies(self,handle,*,_dispatch=_runtime_action):_dispatch(self,'bodies',handle)
    def require_public_key_read_scope(self,*,_dispatch=_runtime_action):_dispatch(self,'require_keys',None)
    def require_server_body_scope(self,*,_dispatch=_runtime_action):_dispatch(self,'require_bodies',None)
    @property
    def expected_ids_bytes(self,*,_require=require_runtime):
        _require(self);return self._expected_ids_bytes
    @property
    def input_digest(self,*,_require=require_runtime):
        _require(self);return self._input_digest


# Pure backend declarations import the original dispatcher/class only after
# these definitions exist. No backend initialization or body access occurs.
import qwen_server_backend as server_backend_module


def _runtime(packet,core,audit,audit_methods,source_check,capacity_check,files,guard_check,derived,*,_require_runtime=require_runtime):
    """Current complete-capacity callback always denies before this is reached.

    The retained guard replaces only hot full-proof replay. Original source,
    live audit/core/read scopes and full transition boundaries remain separate.
    """
    capacity_check()
    guard_check()
    control_callbacks=tuple(_function_binding(fn)for fn in(guard_check,source_check,capacity_check))
    callback_checker=_check_function;callback_checker_code=callback_checker.__code__
    ids,ids_bytes,input_digest=derived
    if type(ids_bytes)is not bytes or type(input_digest)is not str or _expected_ids_bytes(ids)!=ids_bytes:
        raise ValueError('original once-derived immutable runtime metadata required')
    ids_fields=object.__getattribute__(ids,'__dict__');ids_items=tuple(ids_fields.items())
    hash_value(input_digest)
    phase=packet.phase;original={};active=True;state=dict(keys=False,bodies=False,pending=None,transition=None,handle=None,ctx=None,failure=None)
    def require_active():
        if not active or state['failure']is not None:raise PermissionError('failed or retired original server runtime')
    def control(value,action,arg):
        nonlocal active
        try:
            require_active()
            if _check_function is not callback_checker or callback_checker.__code__ is not callback_checker_code:
                raise ValueError('original runtime callback checker changed before use')
            for callback in control_callbacks:callback_checker(callback)
            guard_check()  # Checks source/class descriptors before field access.
            if value is not original['runtime']or object.__getattribute__(value,'__dict__')is not original['dict']:
                raise ValueError('original server runtime/dictionary changed')
            fields=object.__getattribute__(value,'__dict__')
            if set(fields)!=set(original['fields'])or any(fields[n]is not v for n,v in original['fields'].items()):
                raise ValueError('original server runtime field/derived identity changed')
            now_ids=object.__getattribute__(ids,'__dict__')
            if(now_ids is not ids_fields or set(now_ids)!=set(n for n,v in ids_items)
                    or any(now_ids[n]is not v for n,v in ids_items)or _expected_ids_bytes(ids)!=ids_bytes):
                raise ValueError('original derived expected IDs changed')
            core.check()
            for name,fn in audit_methods.items():
                if not _same_accessor(getattr(audit,name),fn):raise ValueError('original audit transition callback changed')
            audit_methods['check']()
            if audit.enabled is not True or audit.active is not True:
                raise ValueError('original active publication scope changed')
            flags=(audit.keys_enabled,audit.bodies_enabled)
            if any(type(flag)is not bool for flag in flags):raise ValueError('exact audit transition flags required')
            allowed={(state['keys'],state['bodies'])}
            if state['pending']=='keys':allowed={(False,False),(True,False)}
            elif state['pending']=='bodies':allowed={(True,False),(True,True)}
            if flags not in allowed:raise ValueError('original server body-read transition changed')
            if action=='check':return
            if action=='require_keys':
                if not state['keys']and state['pending']!='keys':raise PermissionError('server public-key scope not admitted')
                return
            if action=='require_bodies':
                if not state['bodies']and state['pending']!='bodies':raise PermissionError('server model/request scope not admitted')
                return
            if state['pending']is not None or state['transition']is not None:raise ValueError('nested server read transition forbidden')
            if action=='keys':
                if state['keys']or state['bodies']:raise ValueError('single public-key transition required')
                state['transition']='keys'
                try:
                    source_check();require_active();capacity_check();require_active();state['pending']='keys'
                    try:audit_methods['enable_public_keys'](value);require_active();state['keys']=True
                    finally:state['pending']=None
                    _require_runtime(value);return
                finally:state['transition']=None
            if action=='bodies':
                if not state['keys']or state['bodies']or state['handle']is not None:raise ValueError('one server body transition after loaded public keys required')
                if arg.admission is not files or arg.public is not files.keys.public or arg.ctx is None:raise ValueError('original complete public-key loader handle required')
                ctx=arg.ctx
                if ctx.secret_key is not None or ctx._param_cache.secret_key is not None or ctx.parameters.get('secret_key')is not None:raise ValueError('keyless original server context required')
                if canonical(arg.public_manifest)!=phase.readiness.public_manifest_bytes:raise ValueError('loaded original public manifest differs')
                state['transition']='bodies'
                try:
                    source_check();require_active();capacity_check();require_active()
                    state['handle']=arg;state['ctx']=ctx;state['pending']='bodies'
                    try:audit_methods['enable_server_bodies'](value);require_active();state['bodies']=True
                    finally:state['pending']=None
                    _require_runtime(value);return
                finally:state['transition']=None
            raise ValueError('unselected server runtime transition')
        except BaseException as error:
            active=False
            if state['failure']is None:state['failure']=error
            raise
    runtime=RuntimeAdmission(phase,core,_ISSUER,control,ids_bytes,input_digest)
    original.update(runtime=runtime,dict=object.__getattribute__(runtime,'__dict__'),
        fields=dict(object.__getattribute__(runtime,'__dict__')))
    _RUNTIME_CONTROLS[id(runtime)]=(runtime,control)
    def retire():
        nonlocal active
        _require_runtime(runtime)
        if state['handle']is not None and state['handle'].ctx is not None:raise ValueError('original loader must retire context before server terminal')
        active=False
        del _RUNTIME_CONTROLS[id(runtime)]
    return runtime,retire


def run(args,raw_map,expected,blobs,gate,snapshot,origins,*,_require_runtime=require_runtime):
    packet,profile,dep,res,records,protected,always_protected,metadata_memory=admit(args,raw_map,expected,blobs)
    phase=packet.phase;grant=parse(phase.grant_bytes);root=grant['source_root'];pins=resource_pins(phase)
    primary=None;audit=None;audit_methods={};audit_original_open=os.open;audit_wrapper=None;publication_enabled=False
    core=None;runtime=None;retire_runtime=None;evidence=None;methods={};response=None
    installed_before=None;platforms=None;capacity_observations=[];cleanup_report={}
    snapshot_retained={};guard_check=None;retire_snapshot=None;derived=None
    public_records=tuple(records.items());packet_check=type(packet).check
    complete_capacity=require_complete_capacity
    def audit_guard():
        if audit is None:return
        for name,fn in audit_methods.items():
            if not _same_accessor(getattr(audit,name),fn):raise ValueError('original server audit callback changed')
        if audit.original_open is not audit_original_open or audit.wrapper is not audit_wrapper:
            raise ValueError('original server audit open binding changed')
        if audit.enabled is not publication_enabled or audit.active is not True:
            raise ValueError('original server audit publication/active state changed')
    def source_check():
        if guard_check is not None:guard_check()
        if snapshot(expected)!=blobs:raise ValueError('complete server source changed')
        origins(expected,core_enabled=gate.core_enabled);packet_check(packet);exact_environment(dep)
        if guard_check is not None:guard_check()
        if audit is not None:
            audit_guard();audit_methods['check']()
        if runtime is not None:_require_runtime(runtime)
        if core is not None:core.check()
        if set(records)!=set(path for path,row in public_records)or any(records[path]is not row for path,row in public_records):
            raise ValueError('original public proof read bindings changed')
        for path,(digest,limit,raw)in public_records:
            if read(path,limit=limit,expected=digest)!=raw:raise ValueError('original public proof changed')
    def capacity_check():
        # Exact source-selected callback currently raises unconditionally.
        # Host snapshots never substitute for device/workspace admission.
        return complete_capacity(phase,packet.resource_policy,packet.resource_review,
            expected_pins=pins,host_observations=tuple(capacity_observations))
    def owner_guard(owner,callbacks):
        if owner is not None:
            for name,fn in callbacks.items():
                if not _same_accessor(getattr(owner,name),fn):raise ValueError('original server evidence callback changed')
    try:
        exact_environment(dep);source_check()
        guard_check,retire_snapshot=_capture_snapshot(packet,root,expected,blobs,dep,gate,snapshot,origins,snapshot_retained)
        guard_check()
        ids=phase.expected_ids();input_digest=phase.input_digest
        derived=(ids,_expected_ids_bytes(ids),input_digest);snapshot_retained['derived']=derived
        hash_value(input_digest);guard_check()
        audit=AccessScope(root,expected,records,phase,profile,dep,res,protected,always_protected=always_protected)
        # Retain the original close before any fallible hook installation.
        audit_methods['close']=audit.close
        for name in('check','install','enable_public_keys','enable_server_bodies'):audit_methods[name]=getattr(audit,name)
        audit_methods['install']();audit_wrapper=audit.wrapper;audit_guard()
        capacity_observations.append(check_host_capacity(phase,packet.resource_policy,packet.resource_review,expected_pins=pins,after_import=False))
        capacity_check()  # Current default denial occurs before any core/backend enablement.
        installed_before=hash_installed(dep)
        from qwen_backend_identity import install_core
        core=install_core(root,blobs,expected,gate=gate);gate.core_enabled=True
        core.load(('he_params','key_gen','ckks_ctx','polynomial'))
        import jax
        import numpy
        platforms=[str(d.platform)for d in jax.devices()]
        if not platforms or any(p!='tpu'for p in platforms)or jax.config.jax_enable_x64 is not True:raise ValueError('genuine TPU/x64 runtime required')
        imported_origins(dep,root);same(hash_installed(dep),installed_before,'installed source changed during backend import')
        capacity_observations.append(check_host_capacity(phase,packet.resource_policy,packet.resource_review,expected_pins=pins,after_import=True))
        capacity_check();source_check();audit.enabled=True;publication_enabled=True
        destination=grant['outputs']['evidence'];evidence=Output(destination['path'],expected_parent=canonical(destination['parent']))
        methods['close']=evidence.close
        for name in('check','write'):methods[name]=getattr(evidence,name)
        public_keys=authenticate_public(phase.readiness.public_manifest_bytes,profile,derived[0],role='server')
        files=bind_files(phase.public,public_keys,canonical(parse(phase.request_bytes)['public']))
        runtime,retire_runtime=_runtime(packet,core,audit,dict(audit_methods),source_check,capacity_check,files,guard_check,derived)
        runtime.allow_public_key_reads()
        from qwen_server_backend import ServerBackend
        backend=ServerBackend(runtime)
        from qwen_stage_composition import _server_for_stage
        with termination_signals():
            response=_server_for_stage(files,grant['public_key_store']['path'],parse(phase.request_bytes),
                key_dependencies=backend.dependencies,factories=backend.factories)
        _require_runtime(runtime);source_check()
        if type(response)is not bytes or not response:raise ValueError('original complete response manifest bytes required')
        origins_after=imported_origins(dep,root);same(hash_installed(dep),installed_before,'installed source changed during server stage')
        report=dict(schema='qwen-actual-server-worker-report-v1',passed=True,complete=True,error_type=None,
            source_manifest_sha256=sha(raw_map),source_archive_sha256=args.archive_sha256,public_packet_sha256=args.packet_sha256,
            phase_grant_sha256=sha(phase.grant_bytes),scientific_statement_sha256=sha(phase.scientific.statement_bytes),
            request_contract_sha256=sha(phase.request_bytes),request_manifest_sha256=sha(phase.request_manifest_bytes),
            input_digest=derived[2],response_manifest_sha256=sha(response),response_manifest=response.decode('utf-8'),
            source_hashes_before=expected,source_hashes_after=expected,policies=phase.expected_pins.check()['policies'],
            process=dict(pid=os.getpid(),pgid=os.getpgrp()),audit_denials=[],
            backend=dict(platforms=platforms,jax_enable_x64=True,installed_sources_sha256=sha(canonical(installed_before)),module_origins_sha256=sha(canonical(origins_after))),
            public_metadata_memory=metadata_memory,host_capacity=capacity_observations,
            checks=dict(original_loader_context_and_file_retirement_completed=True,source_postcheck_completed=True,installed_source_postcheck_completed=True,termination_scope_completed=True),
            scope=dict(full28_server_forward=True,public_key_reads=True,model_body_reads=True,request_ciphertext_reads=True,response_ciphertext_writes=True,
                private_key_reads=False,token_or_calibration_reads=False,key_generation=False,client_codec=False,scientific_acceptance_claimed=False,downstream_execution_authorized=False),
            outer_terminal_receipt_included=False)
        owner_guard(evidence,methods);methods['check']();methods['write']('report.json',canonical(report));methods['check']();audit_guard();audit_methods['check']()
    except BaseException as error:
        if snapshot_retained and snapshot_retained['status']!='retired':snapshot_retained['status']='failed'
        primary=protect_body(error)
    finally:
        original=evidence;original_methods=dict(methods);original_runtime=runtime;original_retire=retire_runtime
        final=Finalizer(cleanup_report,primary)
        def retire():
            final.attempt('complete server source/public proof postcheck',source_check)
            if audit is not None:
                final.attempt('original server audit callback/state identity',audit_guard)
                final.attempt('server sticky audit after publication',audit_methods['check'])
            if original_retire is not None:final.attempt('original server runtime retirement',original_retire)
            if primary is not None and'write'in original_methods:
                def incomplete():
                    owner_guard(original,original_methods);original_methods['check']()
                    original_methods['write']('failure.json',canonical(dict(schema='qwen-server-incomplete-v1',passed=False,complete=False,
                        error_type=type(primary).__name__,source_manifest_sha256=sha(raw_map),phase_grant_sha256=sha(phase.grant_bytes),
                        process=dict(pid=os.getpid(),pgid=os.getpgrp()),actual_readiness=False)))
                final.attempt('bounded class-only incomplete evidence',incomplete)
            if'close'in original_methods:
                final.attempt('original evidence callback identity',lambda:owner_guard(original,original_methods))
                final.attempt('original server evidence retirement',lambda:original_methods['close'](primary=final.selected()))
            if audit is not None:
                final.attempt('terminal sticky audit veto',audit_methods['check'])
                def close_audit():
                    try:audit_methods['close']()
                    finally:os.open=audit_original_open
                final.attempt('original server audit retirement',close_audit)
        final.shielded(cleanup_signals,retire)
        # The snapshot remains held through source, original owner, audit and
        # signal restoration. Failed cleanup never releases or resets it.
        if final.selected()is None and retire_snapshot is not None:
            final.attempt('clean original server admission snapshot retirement',retire_snapshot)
        chosen=final.selected()
        if chosen is not None:
            _QUARANTINE.append((packet,core,audit,original_runtime,original_retire,original,original_methods,response,
                snapshot_retained,guard_check,retire_snapshot,derived,chosen));raise chosen
    return 0
