"""Recording-only client stages; actual factories deny before argument access.

Original embedding placement, file formats, default codec signatures and numeric
policy are inherited. Explicit dependencies are trusted synthetic seams, not
execution capabilities. No backend runtime factory or key generation is added.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
import json

from cleanup import cleanup_signals
from cleanup_origin import protect_body
from qwen_finalizer import Finalizer
from qwen_key_files_admission import FilesAdmission
from qwen_key_loader import RecordingDependencies, _load_for_recording
from qwen_file_contract import (canonical, same, hash_value, validate_request,
    validate_request_manifest, ciphertext_binding)
from qwen_file_lifecycle import prepare_client_embeddings, ResponseFiles
from qwen_backend_composition import _metadata, _snapshot, _unchanged
from qwen_publisher import Publisher

_QUARANTINE=[]


@dataclass(frozen=True)
class RecordingInputFactories:
    publisher: object
    secure: object
    shield: object = cleanup_signals
    synthetic_only: bool = True


@dataclass(frozen=True)
class RecordingOutputFactories:
    response_files: object
    materialize: object
    secure: object
    shield: object = cleanup_signals
    synthetic_only: bool = True


@dataclass(frozen=True)
class PublishedRequest:
    contract_bytes: bytes
    manifest_bytes: bytes
    input_digest: str


def create_client_input(*args, **kwargs):
    raise PermissionError('actual Qwen client input requires independent source, producer and role admission')


def create_client_output(*args, **kwargs):
    raise PermissionError('actual Qwen client output requires independent source, producer and role admission')


def _identity(function):
    return (getattr(function,'__self__',None),getattr(function,'__func__',function))


def _capture(owner,name,callbacks):
    function=getattr(owner,name)
    if not callable(function):raise TypeError('original owner callback required')
    callbacks.append((owner,name,function,_identity(function)))
    return function


def _check_callbacks(callbacks):
    for owner,name,function,identity in callbacks:
        current=_identity(getattr(owner,name,None))
        if any(a is not b for a,b in zip(current,identity)):
            raise ValueError('original client callback changed: '+name)


def _admit(admission,key_dependencies,factories,kind,names):
    # Full profile/source and independent expected identities precede any
    # supplied callback, file/key body, context or owned output operation.
    if type(admission)is not FilesAdmission:raise TypeError('complete file/key admission required first')
    admission.check()
    if _QUARANTINE:raise RuntimeError('failed client stage requires dedicated-process retirement')
    if admission.keys.public.role!='client' or admission.keys.private is None:
        raise ValueError('complete private client admission required')
    if type(key_dependencies)is not RecordingDependencies or key_dependencies.synthetic_only is not True:
        raise PermissionError('explicit recording key dependencies required')
    if type(factories)is not kind or factories.synthetic_only is not True:
        raise PermissionError('explicit recording client factories required')
    selected=tuple(getattr(factories,name)for name in names)
    if any(not callable(fn)for fn in selected):raise TypeError('explicit recording callback required')
    key_names=('security_check','runtime_factory','directory_factory','map_array')
    keys=tuple(getattr(key_dependencies,name)for name in key_names)
    if any(not callable(fn)for fn in keys):raise TypeError('explicit recording key callback required')
    def check():
        admission.check()
        if factories.synthetic_only is not True or key_dependencies.synthetic_only is not True:
            raise ValueError('recording scope changed')
        if any(getattr(factories,name)is not fn for name,fn in zip(names,selected)):
            raise ValueError('original client factory changed')
        if any(getattr(key_dependencies,name)is not fn for name,fn in zip(key_names,keys)):
            raise ValueError('original key factory changed')
    return selected,check


@contextmanager
def _consumer_scope(admission,public_keys,client_private,*,key_dependencies,
                    codec_name,check_stage,shield):
    """Own ciphertext consumers INSIDE the nested key-context lifetime.

    The run-local inventory captures original/partial arrays before validation.
    A primary or uncertain barrier retains every original owner until process
    exit and propagates into the key loader. No caller can retire the key scope
    merely by replacing a returned handle or mutable owner attribute.
    """
    with _load_for_recording(admission,public_keys,client_private,dependencies=key_dependencies)as handle:
        ctx=handle.ctx;public_manifest=handle.public_manifest
        manifest_bytes=canonical(public_manifest);refs=[];callbacks=[];primary=None
        codec=_capture(ctx,codec_name,callbacks)
        lease=None
        def check():
            check_stage()
            if (handle.ctx is not ctx or handle.admission is not admission or
                    handle.public is not admission.keys.public or
                    handle.public_manifest is not public_manifest or canonical(public_manifest)!=manifest_bytes):
                raise ValueError('original client key/context owner changed')
            _check_callbacks(callbacks)
            if lease is not None and (lease.ctx is not ctx or lease.handle is not handle or
                    lease.codec is not codec or lease.check is not check or lease.retain is not retain):
                raise ValueError('original client consumer handle changed')
            for saved in refs:
                if len(saved)==4:_unchanged(saved)
        def retain(value,binding):
            # The exact original arrays remain referenced even if metadata,
            # synchronization or a callback raises before a complete snapshot.
            original=(value,getattr(value,'polynomial',None),getattr(value,'moduli_array',None))
            refs.append(original)
            _metadata(value,binding)
            saved=_snapshot(value)
            if saved[1]is not original[1]or saved[2]is not original[2]:
                raise ValueError('client ciphertext arrays changed during admission')
            refs[-1]=saved
            for array in original[1:3]:array.block_until_ready()
            check()
            return value
        lease=SimpleNamespace(ctx=ctx,handle=handle,codec=codec,check=check,retain=retain)
        try:
            check();yield lease;check()
        except BaseException as error:primary=protect_body(error)
        finally:
            final=Finalizer({},primary)
            def retire():
                final.attempt('client ownership before retirement',check)
                seen=set()
                for saved in refs:
                    for array in saved[1:3]:
                        if array is not None and id(array)not in seen:
                            seen.add(id(array))
                            final.attempt('original client array barrier',lambda a=array:a.block_until_ready())
                final.attempt('client ownership after retirement',check)
            final.shielded(shield,retire)
            selected=final.selected()
            if selected is not None:
                _QUARANTINE.append((admission,handle,ctx,public_manifest,manifest_bytes,
                    tuple(refs),tuple(callbacks),lease,check_stage,shield,selected))
            else:
                refs.clear()
            # A selected error must cross the nested loader's entered boundary
            # so its complete key/context/transfers backing is quarantined too.
            final.raise_if_failed()


def publish_input_for_recording(admission,public_keys,client_private,*,request_id,
                                embeddings,positions,key_dependencies,factories):
    selected,base_check=_admit(admission,key_dependencies,factories,RecordingInputFactories,
                               ('publisher','secure','shield'))
    public=admission.public_contract
    prepared=prepare_client_embeddings(public,admission.keys.check(),request_id,embeddings,positions)
    contract=json.loads(prepared.contract_bytes);validate_request(public,contract)
    same(admission.keys.check(),contract['public'],'client input public identity differs')
    tiles=prepared.tiles;positions_original=prepared.original_positions
    raw=None;publisher=None;callbacks=[];primary=None
    close=finish=write=None
    def check_stage():
        base_check();_check_callbacks(callbacks)
        if (prepared.tiles is not tiles or prepared.original_positions is not positions_original or
                prepared.contract_bytes!=canonical(contract)):
            raise ValueError('original prepared embedding ownership changed')
    with _consumer_scope(admission,public_keys,client_private,key_dependencies=key_dependencies,
            codec_name='encrypt_slots',check_stage=check_stage,shield=selected[2])as lease:
        try:
            lease.check()
            publisher=selected[0](lease.handle,json.loads(prepared.contract_bytes))
            close=_capture(publisher,'close',callbacks)
            finish=_capture(publisher,'finish',callbacks)
            write=_capture(publisher,'write_ciphertext',callbacks)
            lease.check()
            if (type(publisher)is not Publisher or publisher.public is not public or
                    publisher.request!=prepared.contract_bytes or publisher.role!='request' or
                    publisher.input_digest is not None):
                raise ValueError('exact original request publisher required')
            for index,tile in enumerate(tiles):
                binding=ciphertext_binding(public,contract,'request',index)
                lease.check()
                value=lease.codec(tile,scale=binding['scale'])
                lease.retain(value,binding)
                selected[1](lease.ctx,value);lease.check()
                artifact=write(index,value,json.loads(canonical(binding)))
                lease.check()
                same(artifact['binding'],binding,'published fresh input metadata differs')
            raw=finish();lease.check()
            if type(raw)is not bytes:raise TypeError('canonical upload bytes required')
            manifest=json.loads(raw);validate_request_manifest(public,contract,manifest)
            if raw!=canonical(manifest):raise ValueError('canonical complete upload required')
        except BaseException as error:primary=protect_body(error)
        finally:
            final=Finalizer({},primary)
            def retire_publisher():
                final.attempt('original client ownership before publisher close',lease.check)
                if close is not None:final.attempt('original request publisher close',lambda:close(primary=final.selected()))
                final.attempt('original client ownership after publisher close',lease.check)
            final.shielded(selected[2],retire_publisher)
            error=final.selected()
            if error is not None:
                _QUARANTINE.append((admission,prepared,tiles,lease,selected,publisher,tuple(callbacks),error))
            final.raise_if_failed()
    # Key readbacks, context retirement and guards have also completed here.
    return PublishedRequest(prepared.contract_bytes,raw,json.loads(raw)['input_digest'])


def consume_output_for_recording(admission,public_keys,client_private,request,input_digest,
                                 *,key_dependencies,factories):
    selected,base_check=_admit(admission,key_dependencies,factories,RecordingOutputFactories,
                               ('response_files','materialize','secure','shield'))
    public=admission.public_contract;contract=validate_request(public,request);hash_value(input_digest)
    contract_bytes=canonical(contract)
    same(admission.keys.check(),contract['public'],'client output public identity differs')
    response=None;close=check_files=run_client=None;callbacks=[];primary=None;result=None
    def check_stage():
        base_check();_check_callbacks(callbacks)
        if canonical(contract)!=contract_bytes:raise ValueError('original client request changed')
        if response is not None:
            if (type(response)is not ResponseFiles or response.public is not public or
                    response.contract!=canonical(contract)or response.input_digest!=input_digest or
                    len(response.records)!=38):
                raise ValueError('complete response belongs to another client request')
    try:
        # The real held ResponseFiles constructor authenticates all38 records
        # before this function can invoke even the private loader callback.
        response=selected[0](public,json.loads(contract_bytes),input_digest)
        close=_capture(response,'close',callbacks)
        check_files=_capture(response,'check',callbacks)
        run_client=_capture(response,'run_client',callbacks)
        check_stage();check_files(readback=True);check_stage()
        def private_loader():
            check_stage();check_files(readback=True)
            return _consumer_scope(admission,public_keys,client_private,key_dependencies=key_dependencies,
                codec_name='decrypt_slots',check_stage=check_stage,shield=selected[3])
        def decode(lease,payload,binding):
            lease.check()
            binding_bytes=canonical(binding);expected=json.loads(binding_bytes)
            value=selected[1](lease.ctx,payload,json.loads(binding_bytes))
            lease.retain(value,expected)
            selected[2](lease.ctx,value);lease.check()
            # Same defaultbc978 facade; no optional decoder or suppression of
            # its ordinary approximation check. Full-slot snapshots and finite
            # checks happen in unchanged ResponseFiles.run_client.
            output=lease.codec(value,scale=expected['scale'],slots_to_decode=65536,
                               validate_approximation=True)
            lease.check();return output
        result=run_client(private_loader,decode)
        check_stage();check_files(readback=True);check_stage()
    except BaseException as error:primary=protect_body(error)
    finally:
        final=Finalizer({},primary)
        def retire_response():
            final.attempt('original client ownership before response close',check_stage)
            if close is not None:final.attempt('original response close',lambda:close(primary=final.selected()))
            final.attempt('original client ownership after response close',check_stage)
        final.shielded(selected[3],retire_response)
        error=final.selected()
        if error is not None:_QUARANTINE.append((admission,response,selected,tuple(callbacks),result,error))
        final.raise_if_failed()
    return result
