"""Prospective dedicated full28 server process; current resource gate denies.

This source does not issue grants, scientific outcomes or outer receipts.
The device/workspace admission remains unimplemented, so even a structurally
valid future packet cannot enable a backend or body through this version.
"""
from dataclasses import dataclass
import os
from pathlib import Path
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
from qwen_key_contract import PublicProfile,authenticate_public
from qwen_key_files_admission import bind_files
from keygen_output import Output,read,canonical_path,overlap

_QUARANTINE=[]
_ISSUER=object()
MODEL_STORAGE_SCOPES=(
    '/tmp/qwen-qkv-attention-join-20261103/public_qkv_parameters',
    '/tmp/qwen-decoder-parameters-20261103/actual-v1/public_matrices',
    '/tmp/qwen-rmsnorm-owner-20261103')


def _accessor(value):return(getattr(value,'__self__',None),getattr(value,'__func__',value))
def _same_accessor(a,b):return all(x is y for x,y in zip(_accessor(a),_accessor(b)))


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
    def check(self):
        if type(self)is not RuntimeAdmission or self._issuer is not _ISSUER:raise PermissionError('original source-fixed server runtime required')
        self._control(self,'check',None);return self
    def allow_public_key_reads(self):self._control(self,'keys',None)
    def admit_server_bodies(self,handle):self._control(self,'bodies',handle)
    def require_public_key_read_scope(self):self._control(self,'require_keys',None)
    def require_server_body_scope(self):self._control(self,'require_bodies',None)


def _runtime(packet,core,audit,audit_methods,source_check,capacity_check,files):
    """Current complete-capacity callback always denies before this is reached.

    The separate immutable admission guard is still under source review. This
    draft retains full pure reauthentication until that guard is selected.
    """
    capacity_check()
    phase=packet.phase;original={};active=True;state=dict(keys=False,bodies=False,pending=None,handle=None,ctx=None)
    def control(value,action,arg):
        if not active or value is not original['runtime']or value.phase is not phase or value.core is not core or value._issuer is not _ISSUER or value._control is not control:
            raise ValueError('original server runtime binding changed')
        packet.check();core.check()
        for name,fn in audit_methods.items():
            if not _same_accessor(getattr(audit,name),fn):raise ValueError('original audit transition callback changed')
        audit_methods['check']()
        if audit.enabled is not True or audit.active is not True:
            raise ValueError('original active publication scope changed')
        flags=(audit.keys_enabled,audit.bodies_enabled)
        if any(type(value)is not bool for value in flags):raise ValueError('exact audit transition flags required')
        allowed={(state['keys'],state['bodies'])}
        if state['pending']=='keys':allowed={(False,False),(True,False)}
        elif state['pending']=='bodies':allowed={(True,False),(True,True)}
        if flags not in allowed:
            raise ValueError('original server body-read transition changed')
        if action=='check':return
        if action=='require_keys':
            if not state['keys']and state['pending']!='keys':raise PermissionError('server public-key scope not admitted')
            return
        if action=='require_bodies':
            if not state['bodies']and state['pending']!='bodies':raise PermissionError('server model/request scope not admitted')
            return
        if state['pending']is not None:raise ValueError('nested server read transition forbidden')
        if action=='keys':
            if state['keys']or state['bodies']:raise ValueError('single public-key transition required')
            source_check();capacity_check();state['pending']='keys'
            try:audit_methods['enable_public_keys'](value);state['keys']=True
            finally:state['pending']=None
            value.check();return
        if action=='bodies':
            if not state['keys']or state['bodies']or state['handle']is not None:raise ValueError('one server body transition after loaded public keys required')
            if arg.admission is not files or arg.public is not files.keys.public or arg.ctx is None:raise ValueError('original complete public-key loader handle required')
            ctx=arg.ctx
            if ctx.secret_key is not None or ctx._param_cache.secret_key is not None or ctx.parameters.get('secret_key')is not None:raise ValueError('keyless original server context required')
            if canonical(arg.public_manifest)!=phase.readiness.public_manifest_bytes:raise ValueError('loaded original public manifest differs')
            source_check();capacity_check();state['handle']=arg;state['ctx']=ctx;state['pending']='bodies'
            try:audit_methods['enable_server_bodies'](value);state['bodies']=True
            finally:state['pending']=None
            value.check();return
        raise ValueError('unselected server runtime transition')
    runtime=RuntimeAdmission(phase,core,_ISSUER,control);original['runtime']=runtime
    def retire():
        nonlocal active
        runtime.check()
        if state['handle']is not None and state['handle'].ctx is not None:raise ValueError('original loader must retire context before server terminal')
        active=False
    return runtime,retire


def run(args,raw_map,expected,blobs,gate,snapshot,origins):
    packet,profile,dep,res,records,protected,always_protected,metadata_memory=admit(args,raw_map,expected,blobs)
    phase=packet.phase;grant=parse(phase.grant_bytes);root=grant['source_root'];pins=resource_pins(phase)
    primary=None;audit=None;audit_methods={};audit_original_open=os.open;audit_wrapper=None;publication_enabled=False
    core=None;runtime=None;retire_runtime=None;evidence=None;methods={};response=None
    installed_before=None;platforms=None;capacity_observations=[];cleanup_report={}
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
        if snapshot(expected)!=blobs:raise ValueError('complete server source changed')
        origins(expected,core_enabled=gate.core_enabled);packet.check();exact_environment(dep)
        if audit is not None:
            audit_guard();audit_methods['check']()
        if runtime is not None:runtime.check()
        if core is not None:core.check()
        for path,(digest,limit,raw)in records.items():
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
        public_keys=authenticate_public(phase.readiness.public_manifest_bytes,profile,phase.expected_ids(),role='server')
        files=bind_files(phase.public,public_keys,canonical(parse(phase.request_bytes)['public']))
        runtime,retire_runtime=_runtime(packet,core,audit,dict(audit_methods),source_check,capacity_check,files)
        runtime.allow_public_key_reads()
        from qwen_server_backend import ServerBackend
        backend=ServerBackend(runtime)
        from qwen_stage_composition import _server_for_stage
        with termination_signals():
            response=_server_for_stage(files,grant['public_key_store']['path'],parse(phase.request_bytes),
                key_dependencies=backend.dependencies,factories=backend.factories)
        runtime.check();source_check()
        if type(response)is not bytes or not response:raise ValueError('original complete response manifest bytes required')
        origins_after=imported_origins(dep,root);same(hash_installed(dep),installed_before,'installed source changed during server stage')
        report=dict(schema='qwen-actual-server-worker-report-v1',passed=True,complete=True,error_type=None,
            source_manifest_sha256=sha(raw_map),source_archive_sha256=args.archive_sha256,public_packet_sha256=args.packet_sha256,
            phase_grant_sha256=sha(phase.grant_bytes),scientific_statement_sha256=sha(phase.scientific.statement_bytes),
            request_contract_sha256=sha(phase.request_bytes),request_manifest_sha256=sha(phase.request_manifest_bytes),
            input_digest=phase.input_digest,response_manifest_sha256=sha(response),response_manifest=response.decode('utf-8'),
            source_hashes_before=expected,source_hashes_after=expected,policies=phase.expected_pins.check()['policies'],
            process=dict(pid=os.getpid(),pgid=os.getpgrp()),audit_denials=[],
            backend=dict(platforms=platforms,jax_enable_x64=True,installed_sources_sha256=sha(canonical(installed_before)),module_origins_sha256=sha(canonical(origins_after))),
            public_metadata_memory=metadata_memory,host_capacity=capacity_observations,
            checks=dict(original_loader_context_and_file_retirement_completed=True,source_postcheck_completed=True,installed_source_postcheck_completed=True,termination_scope_completed=True),
            scope=dict(full28_server_forward=True,public_key_reads=True,model_body_reads=True,request_ciphertext_reads=True,response_ciphertext_writes=True,
                private_key_reads=False,token_or_calibration_reads=False,key_generation=False,client_codec=False,scientific_acceptance_claimed=False,downstream_execution_authorized=False),
            outer_terminal_receipt_included=False)
        owner_guard(evidence,methods);methods['check']();methods['write']('report.json',canonical(report));methods['check']();audit_guard();audit_methods['check']()
    except BaseException as error:primary=protect_body(error)
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
        chosen=final.selected()
        if chosen is not None:
            _QUARANTINE.append((packet,core,audit,original_runtime,original_retire,original,original_methods,response,chosen));raise chosen
    return 0
