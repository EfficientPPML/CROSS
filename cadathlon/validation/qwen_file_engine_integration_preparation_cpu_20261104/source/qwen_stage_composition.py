"""Explicit recording bridge; every actual role factory remains disabled."""
from dataclasses import dataclass
import json,sys
from cleanup import cleanup_signals
from cleanup_origin import protect_body
from qwen_finalizer import Finalizer
from qwen_file_contract import canonical,same,validate_request
from qwen_key_files_admission import FilesAdmission
from qwen_key_loader import RecordingDependencies,_load_for_recording
from qwen_backend_composition import Dependencies,BorrowedServerEngine
from qwen_publisher import Publisher

_QUARANTINE=[]

@dataclass(frozen=True)
class RecordingStageFactories:
    request_files:object
    parameter_files:object
    publisher:object
    engine_dependencies:object
    shield:object=cleanup_signals
    synthetic_only:bool=True


def create_reference_session(*args,**kwargs):
    raise PermissionError('full Qwen session actual role awaits independent source/grant/terminal admission')


def server_for_recording(admission,public_keys,request,*,key_dependencies,factories):
    """Connect the admitted recording key scope, real engine glue and publisher.

    Factory callbacks are trusted synthetic seams, never execution capabilities.
    The actual factories still deny. This function creates no native dependency
    factory, reads no model/token input itself, and never generates keys.
    """
    if type(admission)is not FilesAdmission:raise TypeError('complete file/key admission required first')
    admission.check();public=admission.public_contract;c=validate_request(public,request)
    same(admission.keys.check(),c['public'],'loader/file request identity differs')
    if admission.keys.public.role!='server':raise ValueError('keyless server role required')
    if type(key_dependencies)is not RecordingDependencies or key_dependencies.synthetic_only is not True:
        raise PermissionError('explicit recording key dependencies required')
    if type(factories)is not RecordingStageFactories or factories.synthetic_only is not True:
        raise PermissionError('explicit recording file/engine factories required')
    for name in ('request_files','parameter_files','publisher','engine_dependencies','shield'):
        if not callable(getattr(factories,name)):raise TypeError('explicit trusted recording factory required')
    factory_snapshot=(factories.request_files,factories.parameter_files,factories.publisher,factories.engine_dependencies,factories.shield)
    raw=None
    with _load_for_recording(admission,public_keys,None,dependencies=key_dependencies)as handle:
        ctx=handle.ctx;upload=parameters=publisher=engine=None;primary=None
        upload_close=parameter_close=publisher_close=publisher_finish=publisher_write=None
        public_manifest=handle.public_manifest;public_manifest_bytes=canonical(public_manifest)
        owner_callbacks=[]
        def callback_identity(fn):
            return (getattr(fn,'__self__',None),getattr(fn,'__func__',fn))
        def capture(owner,name):
            fn=getattr(owner,name)
            if not callable(fn):raise TypeError('original owner callback required')
            owner_callbacks.append((owner,name,fn,callback_identity(fn)))
            return fn
        def guard_handle():
            if handle.ctx is not ctx or handle.admission is not admission or handle.public is not admission.keys.public:
                raise ValueError('borrowed loader handle changed')
            if handle.public_manifest is not public_manifest or canonical(public_manifest)!=public_manifest_bytes:
                raise ValueError('borrowed loader public manifest changed')
            admission.check()
            for owner,name,fn,identity in owner_callbacks:
                current=callback_identity(getattr(owner,name,None))
                if any(a is not b for a,b in zip(current,identity)):
                    raise ValueError('original owner callback changed: '+name)
            if any(getattr(factories,name)is not original for name,original in zip(('request_files','parameter_files','publisher','engine_dependencies','shield'),factory_snapshot)):
                raise ValueError('recording stage factory changed')
        try:
            guard_handle()
            upload=factory_snapshot[0](handle)
            upload_close=capture(upload,'close')
            guard_handle()
            parameters=factory_snapshot[1](handle)
            parameter_close=capture(parameters,'close')
            guard_handle()
            publisher=factory_snapshot[2](handle,upload.input_digest)
            publisher_close=capture(publisher,'close')
            publisher_finish=capture(publisher,'finish')
            publisher_write=capture(publisher,'write_ciphertext')
            guard_handle()
            if (type(publisher)is not Publisher or publisher.public is not public or publisher.request!=canonical(c)
                    or publisher.role!='response' or publisher.input_digest!=upload.input_digest):
                raise ValueError('exact response publisher belongs to a different request')
            deps=factory_snapshot[3](ctx)
            if type(deps)is not Dependencies:raise TypeError('explicit ordinary engine dependencies required')
            guard_handle()
            engine=BorrowedServerEngine(public,c,handle.admission.keys,ctx=ctx,key_owner=handle,
                request_files=upload,parameter_files=parameters,dependencies=deps,cleanup_shield=factory_snapshot[4])
            response=engine.run(publisher_write)
            guard_handle();raw=publisher_finish();guard_handle()
            same(json.loads(raw),response,'published response differs from complete engine artifacts')
        except BaseException as error:primary=protect_body(error)
        finally:
            final=Finalizer({},primary)
            def close_owned():
                # Output publication and borrowed file consumers retire while
                # the real loader scope still owns context and key backing.
                final.attempt('original ownership before cleanup',guard_handle)
                if publisher_close is not None:final.attempt('publisher close',lambda:publisher_close(primary=final.selected()))
                final.attempt('original ownership after publisher cleanup',guard_handle)
                if parameter_close is not None:final.attempt('parameter backing close',lambda:parameter_close(primary=final.selected()))
                final.attempt('original ownership after parameter cleanup',guard_handle)
                if upload_close is not None:final.attempt('request backing close',lambda:upload_close(primary=final.selected()))
                final.attempt('original ownership after all cleanup',guard_handle)
            final.shielded(factory_snapshot[4],close_owned)
            selected=final.selected()
            if selected is not None:
                _QUARANTINE.append((handle,ctx,admission,factory_snapshot,upload,parameters,publisher,engine,tuple(owner_callbacks),public_manifest,public_manifest_bytes,selected))
            final.raise_if_failed()
    # Only return after loader postchecks, readiness, cleanup and guard restore.
    # This still does not constitute a producer process/outer terminal receipt.
    return raw
