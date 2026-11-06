"""Source-only three-role composition; no actual runtime issuer exists yet.

The exact callgraph below binds reviewed component owners. RuntimeAdmission is
an explicitly unavailable capability until a separately reviewed source/import,
installed/native policy, audit/device and complete capacity issuer is supplied.
Logical byte totals do not implement that authority. The public entry denies
before any source/body/backend operation, and no instance is created here.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import os,stat,weakref
from cleanup_origin import protect_body
from qwen_finalizer import Finalizer
from q5_key_contract import canonical,parse,sha,ExpectedIds,KEYGEN,authenticate_public,authenticate_private
from q5_key_producer import GenerationIntent,GeneratedManifests,CORE_MANIFEST_SHA256,bindings,same
from q5_key_admission import bind_files
from q5_role_authority import PhaseAdmission
from q5_checkpoint_contract import canonical as legacy_canonical,parse as legacy_parse,numerical_plan,checkpoint_row
_QUARANTINE=[]
_INPUTS={}
_INPUTS_CLOSED=weakref.WeakSet()

def _same(a,b):return(getattr(a,'__self__',None)is getattr(b,'__self__',None)and getattr(a,'__func__',a)is getattr(b,'__func__',b))

def _identity(path):
    p=Path(path);info=p.lstat()
    if not stat.S_ISDIR(info.st_mode)or info.st_uid!=os.geteuid()or stat.S_IMODE(info.st_mode)!=0o700:raise ValueError('original owned0700 directory required')
    return dict(path=str(p),device=info.st_dev,inode=info.st_ino,mode=stat.S_IMODE(info.st_mode),uid=info.st_uid)

@dataclass(frozen=True)
class RuntimeAdmission:
    """Data/API declaration, not a forgeable activation or resource capability."""
    phase:PhaseAdmission
    source:object
    core:object
    @property
    def role(self):return self.phase.check()['role']
    def check(self):
        if type(self)is not RuntimeAdmission or type(self.phase)is not PhaseAdmission:raise TypeError('exact runtime/phase required')
        self.phase.check()
        # Deliberate missing actual issuer: no policy instance or measurement
        # can be inferred from the recording/source-only preparation artifacts.
        raise PermissionError('complete Q5 installed/native/audit/capacity authority is not implemented or granted')
    def require_backend_scope(self):self.check()
    def require_generation_scope(self,intent=None):self.check()
    def require_generation_capacity(self,intent):self.check()
    def require_key_scope(self,admission,client_private):self.check()
    def require_server_body_scope(self):self.check()
    def admit_server_bodies(self,handle):self.check()
    def admit_client_private(self):self.check()
    def check_common_definitions(self,definitions):self.check()
    def check_loaded_handle(self,handle):self.check()
    def expected_key_root(self,*,private):self.check()
    def bind_internal_generation(self,intent,result,roots):self.check()
    def source_check(self):self.check()

class _EncryptedInput:
    """Acquire before the original encrypt call; retain originals independently."""
    def __init__(self,backend,handle,slots):
        self.backend=backend;self.handle=handle;self.ctx=handle.ctx;self.slots=slots;self.value=None;self.closed=False
        _INPUTS[id(self)]=dict(owner=self,fixed=(backend,handle,self.ctx,slots,backend.encrypt_slots),
            owned=[],value=None,state=None,entered=False,closed=False,failure=None)
    def _record(self):
        if self in _INPUTS_CLOSED:raise ValueError('encrypted input already retired')
        r=_INPUTS.get(id(self))
        if r is None or r['owner']is not self:raise ValueError('original input owner record missing')
        return r
    def check(self,*,loaded=True):
        r=self._record();b,h,ctx,slots,encrypt=r['fixed'];b.check();b.runtime.check_loaded_handle(h)
        if self.backend is not b or self.handle is not h or self.ctx is not ctx or self.slots is not slots or h.ctx is not ctx or b.encrypt_slots is not encrypt or self.value is not r['value']or self.closed is not r['closed']:
            raise ValueError('original setup encryption owner changed')
        if r['closed']:raise ValueError('input encryption consumer closed')
        if r['failure']is not None:raise r['failure']
        if loaded:
            from q5_checkpoint_arithmetic import same_state
            if r['state']is None:raise ValueError('input encryption did not complete')
            same_state(r['state'],numerical_plan()['profile'])
        return r
    def load(self):
        r=self.check(loaded=False)
        if r['entered']:raise ValueError('one original setup encryption required')
        r['entered']=True;b,h,ctx,slots,encrypt=r['fixed']
        try:
            value=encrypt(ctx,slots,scale=b.profile.parameters['scaling_factor']);r['value']=value;self.value=value;r['owned'].append(value)
            r['owned'].append(value.polynomial);value.polynomial.block_until_ready();b.secure(ctx,[value])
            from q5_checkpoint_arithmetic import state
            r['state']=state(value,numerical_plan()['profile']);self.check();return value
        except BaseException as error:r['failure']=protect_body(error);_QUARANTINE.append((self,r,error));raise
    def close(self,primary=None):
        if self in _INPUTS_CLOSED:
            if primary is not None:raise primary
            return
        r=self._record()
        if r['closed']:
            if primary is not None:raise primary
            if r['failure']is not None:raise r['failure']
            return
        owned=tuple(r['owned']);final=Finalizer({},primary)
        final.attempt('original encryption binding',lambda:self.check(loaded=False))
        for value in owned:
            wait=getattr(value,'block_until_ready',None)
            if callable(wait):final.attempt('original encrypted input barrier',wait)
        r['closed']=True;self.closed=True;error=final.selected()
        if error is not None or r['failure']is not None:
            error=error if error is not None else r['failure'];r['failure']=error;_QUARANTINE.append((self,r,owned,error));raise error
        self.backend=None;self.handle=None;self.ctx=None;self.slots=None;self.value=None;r.clear();_INPUTS.pop(id(self));_INPUTS_CLOSED.add(self)

def _input_row():
    plan=numerical_plan();r=plan['checkpoints'][0]
    result=dict(index=0,trace_index=r['clear_trace_index'],kind=r['kind'],label=r['label'],stack=r['stack'],level=49,scale=plan['profile']['scale'],nsd=1)
    checkpoint_row(result,0);return result

def _internal_keys(intent,result):
    """Same-setup consistency only; caller must retain actual original result."""
    if type(result)is not GeneratedManifests:raise TypeError('original generation result required')
    v,profile=intent.check();completion=parse(result.completion_bytes,1<<20)
    if completion['intent_sha256']!=sha(intent.raw)or completion['cleanup_completed']is not True or completion['source_postcheck_completed']is not True or completion['terminal_ready']is not False:
        raise ValueError('complete original internal writer readback required')
    expected=ExpectedIds(completion['context_id'],sha(result.public_bytes),KEYGEN,v['model_id'],v['request_producer_source_manifest_sha256'],v['response_consumer_source_manifest_sha256'],sha(result.private_bytes))
    public=authenticate_public(result.public_bytes,profile,expected,role='client');private=authenticate_private(result.private_bytes,public)
    identity={n:getattr(expected,n)for n in ('context_id','model_id','producer_source_manifest_sha256','consumer_source_manifest_sha256')}
    return bind_files(public,canonical(identity),private)

def setup_bound(runtime):
    """One preauthorized setup role; returned manifest still has no terminal authority."""
    runtime.check();runtime.require_generation_scope()
    from q5_actual_dependencies import ActualBackend
    from q5_key_producer import _generate_bound
    from q5_key_loader import _load_bound
    from q5_fixture_assets import generate_setup_assets,AssetWriter,model_id
    from q5_checkpoint_transport import InputWriter
    from q5_checkpoint_roles import SerialRole
    backend=ActualBackend(runtime);g=runtime.phase.check();outputs=g['outputs'];np=backend.np;owners=[];callbacks=[];primary=None;manifest=None;generated=None
    def acquire(owner,names):
        owners.append(owner);methods={};callbacks.append(methods);methods['close']=owner.close
        for name in names:methods[name]=getattr(owner,name)
        return methods
    def guard():
        runtime.check();backend.check()
        for owner,row in zip(owners,callbacks):
            if any(not _same(getattr(owner,n),f)for n,f in row.items()):raise ValueError('original setup owner callback changed')
    with SerialRole('setup'):
        try:
            weights,embeddings=generate_setup_assets(np,backend.profile,check=guard)
            if model_id(backend.profile)!=g['model_id']:raise ValueError('external fixed Q5 model identity differs before key RNG')
            writer=AssetWriter(np,backend.profile,outputs['assets']['path'],legacy_canonical(outputs['assets']['parent']))
            asset=acquire(writer,('write_assets','check'));assets_raw=asset['write_assets'](weights);guard();asset['check']()
            parents={k:{n:outputs[k+'_keys']['parent'][n]for n in ('path','device','inode','mode')}for k in ('public','private')}
            raw=canonical(dict(schema='qwen-toy5-generation-intent-v1',nonce=g['nonce'],model_id=g['model_id'],generation_source_manifest_sha256=g['source_manifest_sha256'],generation_source_archive_sha256=g['source_archive_sha256'],core_source_manifest_sha256=CORE_MANIFEST_SHA256,request_producer_source_manifest_sha256=g['roles']['setup'],response_consumer_source_manifest_sha256=g['roles']['client'],keygen_source_sha256=KEYGEN,public_root=outputs['public_keys']['path'],private_root=outputs['private_keys']['path'],parents=parents,bindings=bindings(),parameters=backend.profile.parameters,logical_resources=backend.profile.resources,setup_preauthorization_sha256=runtime.phase.expected.check()['grant_sha256'],actual_execution_enabled=False))
            intent=GenerationIntent(backend.profile,raw);generated=_generate_bound(intent,backend.generation_dependencies(intent));guard()
            completion=parse(generated.completion_bytes,1<<20);key_roots={k:_identity(outputs[k+'_keys']['path'])for k in ('public','private')}
            for row,label in zip(completion['roots'],('public','private')):
                if row['root']!=key_roots[label]['path']or row['device']!=key_roots[label]['device']or row['inode']!=key_roots[label]['inode']:raise ValueError('original generated root differs before same-setup reopen')
            runtime.bind_internal_generation(intent,generated,key_roots);admission=_internal_keys(intent,generated)
            with _load_bound(admission,outputs['public_keys']['path'],outputs['private_keys']['path'],dependencies=backend.dependencies)as handle:
                original_ctx=handle.ctx;runtime.check_loaded_handle(handle)
                # Original packing is selected explicitly, never dense/oracle.
                from encrypted_packed_transformer import PackedPairLayout
                layout=PackedPairLayout(2,4,32,65536);slots=layout.pack_client_embeddings(embeddings)[0]
                consumer=_EncryptedInput(backend,handle,slots);ct=acquire(consumer,('load','check'));body=None
                try:
                    value=ct['load']();ct['check']();guard()
                    context=admission.public.expected.context_id
                    input_bindings=dict(context_id=context,model_id=g['model_id'],setup_source_manifest_sha256=g['source_manifest_sha256'],setup_grant_sha256=runtime.phase.expected.check()['grant_sha256'])
                    publisher=InputWriter(np,outputs['input']['path'],legacy_canonical(outputs['input']['parent']),input_bindings);pub=acquire(publisher,('write_input','finish','check'))
                    original=pub['write_input'](value,_input_row());guard();input_raw=pub['finish'](original);guard()
                except BaseException as error:body=protect_body(error);raise
                finally:
                    final=Finalizer({},body)
                    # Each original owned consumer is retired before key scope
                    # exits; later callbacks cannot replace these bound methods.
                    for i in reversed(range(len(callbacks))):
                        close=callbacks[i]['close'];final.attempt('original setup consumer close within key scope',lambda close=close:close(body))
                        final.attempt('post-close setup binding guard',guard)
                    final.attempt('original setup context after cleanup',lambda:runtime.check_loaded_handle(handle))
                    if handle.ctx is not original_ctx:final.attempt('borrowed context identity',lambda:(_ for _ in()).throw(ValueError('borrowed setup context changed')))
                    error=final.selected()
                    if error is not None:raise error
            manifest=canonical(dict(schema='qwen-toy5-setup-manifest-v1',bindings=bindings(),context_id=admission.public.expected.context_id,model_id=g['model_id'],roles=g['roles'],public_manifest_sha256=sha(generated.public_bytes),private_manifest_sha256=sha(generated.private_bytes),public_root=key_roots['public'],private_root=key_roots['private'],asset_root=legacy_parse(assets_raw)['root'],assets_manifest_sha256=sha(assets_raw),input_manifest_sha256=sha(input_raw),original_input=original,public_arrays=203,private_arrays=1,public_weights=35,input_ciphertexts=1,terminal_authority=False))
            guard();return dict(manifest=manifest,public_manifest=generated.public_bytes,private_manifest=generated.private_bytes,assets_manifest=assets_raw,input_manifest=input_raw)
        except BaseException as error:primary=protect_body(error);_QUARANTINE.append((runtime,backend,tuple(owners),tuple(callbacks),generated,manifest,error));raise
        finally:
            # Covers failures before entering the key loader. Successful closes
            # are idempotent; uncertain/failed closes are not retried by owners.
            final=Finalizer({},primary)
            for row in reversed(callbacks):final.attempt('original partial setup owner close',lambda close=row['close']:close(primary))
            final.attempt('final setup source binding',guard)
            error=final.selected()
            if error is not None:raise error

def server_bound(runtime):
    runtime.check();g=runtime.phase.check();setup=runtime.phase.setup;setup.check()
    from q5_actual_dependencies import ActualBackend
    from q5_key_loader import _load_bound
    from q5_checkpoint_roles import server_bound as run_server
    from q5_fixture_assets import AssetBinding,PublicAssets
    from q5_checkpoint_transport import Exporter
    from q5_ciphertext_owner import InputOwner
    backend=ActualBackend(runtime);m=setup.check();admission=setup.key_files('server');output=g['outputs']['checkpoints']
    @contextmanager
    def loader():
        with _load_bound(admission,g['borrowed_roots']['public_keys']['path'],None,dependencies=backend.dependencies)as handle:
            runtime.check_loaded_handle(handle);ctx=handle.ctx
            def check():
                backend.check();runtime.check_loaded_handle(handle)
                if handle.ctx is not ctx:raise ValueError('original borrowed server context changed')
            yield SimpleNamespace(ctx=ctx,check=check,key_owner=handle)
            check()
    def assets(handle):
        runtime.require_server_body_scope()
        return PublicAssets(backend.np,AssetBinding(backend.profile,setup.assets_manifest_bytes,setup.expected.check()['assets_manifest'],legacy_canonical(m['asset_root'])))
    def original_input(handle):return InputOwner(backend.np,backend,handle.ctx,m['original_input'],check=handle.check)
    def publisher():
        pins=setup.expected.check();bound=dict(setup_report_sha256=pins['setup_report'],setup_outer_sha256=pins['setup_outer'],context_id=m['context_id'],model_id=m['model_id'],source_manifest_sha256=g['source_manifest_sha256'],grant_sha256=runtime.phase.expected.check()['grant_sha256'])
        return Exporter(backend.np,output['path'],legacy_canonical(output['parent']),m['original_input'],bound)
    def arithmetic(handle):
        runtime.check_loaded_handle(handle.key_owner);runtime.admit_server_bodies(handle.key_owner)
        return backend.arithmetic()
    return run_server(backend.np,loader,assets,original_input,arithmetic,publisher,admission_check=runtime.check)

def client_bound(runtime,*,failure_note=None):
    runtime.check();g=runtime.phase.check();setup=runtime.phase.setup;m=setup.check()
    from q5_actual_dependencies import ActualBackend
    from q5_key_loader import _load_bound
    from q5_fixture_assets import AssetBinding,PublicAssets
    from q5_checkpoint_roles import client_bound as run_client
    backend=ActualBackend(runtime);owner=None;primary=None
    @contextmanager
    def private_loader():
        # run_client admits all47 entire opaque files before this factory call,
        # including before opening even the private manifest metadata.
        runtime.admit_client_private()
        from q5_key_readers import read_metadata
        directory=backend.dependencies.directory_factory(g['borrowed_roots']['private_keys']['path'],private=True)
        close=directory.close;body=None
        try:
            private_bytes=read_metadata(directory,'private_manifest.json',1<<16);directory.check()
            admission=setup.key_files('client',private_bytes)
        except BaseException as error:body=protect_body(error);raise
        finally:
            final=Finalizer({},body);final.attempt('original private manifest directory close',close)
            error=final.selected()
            if error is not None:raise error
        with _load_bound(admission,g['borrowed_roots']['public_keys']['path'],g['borrowed_roots']['private_keys']['path'],dependencies=backend.dependencies)as handle:
            adapter=backend.client_handle(handle);yield adapter;adapter.check()
    try:
        owner=PublicAssets(backend.np,AssetBinding(backend.profile,setup.assets_manifest_bytes,setup.expected.check()['assets_manifest'],legacy_canonical(m['asset_root'])))
        close=owner.close;check=owner.check;load=owner.load;weights=load();check()
        case=numerical_plan()['case'];np=backend.np;embeddings=weights['model.embed_tokens.weight'][np.asarray(case['token_ids'],dtype=np.int64)]
        from encrypted_transformer import public_toy_approximations
        rules=public_toy_approximations('qwen');reference_root=Path(runtime.source.root)/'reference'
        result=run_client(np,runtime.phase.server,private_loader,weights,embeddings,rules,reference_root,admission_check=runtime.check,failure_note=failure_note)
        check();return result
    except BaseException as error:primary=protect_body(error);_QUARANTINE.append((runtime,backend,owner,error));raise
    finally:
        final=Finalizer({},primary)
        if owner is not None:
            final.attempt('original public client weights close',lambda:close(primary))
            final.attempt('original client weight callbacks after close',lambda:_verify_callbacks(owner,{'close':close,'check':check,'load':load}))
        final.attempt('terminal client source binding',runtime.source_check)
        error=final.selected()
        if error is not None:raise error

def _verify_callbacks(owner,callbacks):
    if any(not _same(getattr(owner,n),f)for n,f in callbacks.items()):raise ValueError('original callback changed during cleanup')

def run_actual(*args,**kwargs):
    raise PermissionError('actual Q5 setup/server/client issuer and separate resource/role grants are unavailable')


def compose_bound(runtime):
    """Internal source callgraph only; no actual RuntimeAdmission can be issued.

    A complete report precedes evidence FD retirement, so its existence alone
    is never a terminal admission. Any late exception escapes to exit70.
    """
    runtime.check();g=runtime.phase.check();output=None;close=None;write=None;check=None;primary=None;report=None;status=None;result=None
    try:
        from q5_evidence_output import EvidenceOutput
        out=g['outputs']['evidence'];output=EvidenceOutput(out['path'],legacy_canonical(out['parent']),dict(role=g['role'],source_manifest_sha256=g['source_manifest_sha256']))
        close=output.close;write=output.write;check=output.check
        def guard(*,live=True):
            runtime.check();_verify_callbacks(output,{'close':close,'write':write,'check':check})
            if live:check()
        guard()
        if g['role']=='setup':
            result=setup_bound(runtime);manifest=result['manifest'];write('setup_manifest.json',manifest);context=parse(manifest,1<<20)['context_id'];passed=True
        elif g['role']=='server':
            manifest=server_bound(runtime);context=runtime.phase.setup.check()['context_id'];passed=True
        else:
            def failure_note(index,error_type):
                guard();write('failed_checkpoint.json',canonical(dict(schema='qwen-toy5-failed-checkpoint-v1',index=index,error_type=error_type,complete=False,terminal_authority=False)));guard()
            result=client_bound(runtime,failure_note=failure_note);manifest=canonical(result);write('client_result.json',manifest)
            context=runtime.phase.setup.check()['context_id'];passed=result['original_final_gate']['passed']
            if type(passed)is not bool:raise ValueError('exact original client numeric gate result required')
        guard();runtime.source_check()
        report=dict(schema='qwen-toy5-role-report-v1',role=g['role'],complete=True,passed=passed,context_id=context,model_id=g['model_id'],source_manifest_sha256=g['source_manifest_sha256'],source_archive_sha256=g['source_archive_sha256'],grant_sha256=runtime.phase.expected.check()['grant_sha256'],manifest_sha256=sha(manifest),cleanup_completed=True,audit_denials=[])
        write('report.json',legacy_canonical(report));guard();status=0 if passed else 1
    except BaseException as error:
        primary=protect_body(error)
        if write is not None:
            final=Finalizer({},primary)
            final.attempt('owned class-only incomplete failure evidence',lambda:write('failure.json',canonical(dict(schema='qwen-toy5-role-failure-v1',role=g['role'],complete=False,error_type=type(error).__name__,terminal_authority=False))))
        _QUARANTINE.append((runtime,output,close,write,check,result,report,error));raise
    finally:
        final=Finalizer({},primary)
        if close is not None:
            final.attempt('original evidence owner close',lambda:close(primary))
            final.attempt('original evidence callbacks after close',lambda:_verify_callbacks(output,{'close':close,'write':write,'check':check}))
        final.attempt('final original source check',runtime.source_check)
        error=final.selected()
        if error is not None:
            _QUARANTINE.append((runtime,output,close,write,check,result,report,error));raise error
    if type(status)is not int or status not in(0,1):raise RuntimeError('no complete normal Q5 result')
    return status
