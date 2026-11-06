"""New joined boundaries only; no inherited suites are selected by this file."""
from pathlib import Path
from types import SimpleNamespace as NS
from contextlib import nullcontext
import json,os,sys
import numpy as np
import pytest
HERE=Path(__file__).resolve().parent
# Explicit test-only fixtures contain tests/metadata, no competing runtime modules.
sys.path.extend((str(HERE/'recording_sources'),str(HERE/'recording_sources/reader43')))
import engine_fixture
from test_key_loader_bridge import setup_bridge
from test_qwen_key_contract import profile
import qwen_stage_composition as stage
import qwen_key_loader as loader
import qwen_backend_composition as backend
import qwen_publisher as publisher
from qwen_file_contract import canonical,request
from cleanup_origin import mark,is_cleanup
from qwen_finalizer import Finalizer


def environment(public,profile,tmp_path,monkeypatch):
    env=setup_bridge(tmp_path,monkeypatch,profile,public)
    state=NS(case=None,publisher=None,handle=None,parameter_close_error=None,publish_error=None)
    physical=np.zeros((1,2,256,512,6),dtype=np.uint32)
    monkeypatch.setattr(engine_fixture.Array,'__array__',lambda self,dtype=None,copy=None:physical,raising=False)
    c=request(public,json.loads(env.admission.keys.identity_bytes),'recording_case',[0])
    def upload(handle):
        state.handle=handle;state.case=engine_fixture.setup(public,length=1,ctx_override=handle.ctx)
        assert state.case.request==c
        case=state.case
        def close(primary=None):
            env.rec.log.append(('request-close',))
            assert handle.ctx is case.ctx and not any(r[0]=='mapping-close'for r in env.rec.log)
            if primary is not None:raise primary
            assert case.upload.retired
        case.upload.close=close
        return case.upload
    def parameters(handle):
        case=state.case
        def close(primary=None):
            env.rec.log.append(('parameter-close',))
            assert handle.ctx is case.ctx and not any(r[0]=='mapping-close'for r in env.rec.log)
            if state.parameter_close_error:raise state.parameter_close_error
            if primary is not None:raise primary
            assert case.params.retired
        case.params.close=close
        return case.params
    def output(handle,input_digest):
        state.publisher=publisher.Publisher(public,c,'response',tmp_path/'response',input_digest=input_digest)
        original=state.publisher.write_ciphertext
        def write(*args):
            if state.publish_error:raise state.publish_error
            return original(*args)
        state.publisher.write_ciphertext=write
        return state.publisher
    def dependencies(ctx):
        assert ctx is state.case.ctx
        return state.case.args['dependencies']
    factories=stage.RecordingStageFactories(upload,parameters,output,dependencies,nullcontext)
    def run():return stage.server_for_recording(env.admission,env.public,c,key_dependencies=env.dependencies,factories=factories)
    return env,state,c,factories,run


def test_joined335_loader_engine38_writer_closes_consumers_before_key_retirement(public,profile,tmp_path,monkeypatch):
    env,s,c,_,run=environment(public,profile,tmp_path,monkeypatch);raw=run()
    assert len(json.loads(raw)['artifacts'])==38 and s.publisher.published and s.publisher.closed
    assert s.handle.ctx is None and not loader._FAILURE_QUARANTINE and not backend._QUARANTINE
    assert len(env.rec.maps)==335 and len([r for r in env.rec.log if r[0]=='mapping-close'])==335
    closed=env.rec.log.index(('parameter-close',));request_closed=env.rec.log.index(('request-close',))
    retired=next(i for i,r in enumerate(env.rec.log)if r[0]=='codec-cache-evict')
    assert closed<request_closed<retired
    assert s.case.upload.retired and s.case.params.retired
    env.restored()


@pytest.mark.parametrize('cancel_type',[KeyboardInterrupt,SystemExit])
def test_late_publisher_close_error_after_manifest_then_consumer_cancellation(public,profile,tmp_path,monkeypatch,cancel_type):
    env,s,c,_,run=environment(public,profile,tmp_path,monkeypatch)
    ordinary=OSError('publisher close only');cancel=cancel_type('later parameter cleanup')
    real=os.close;triggered=[]
    def close(fd):
        target=s.publisher.files[-1].fd if s.publisher is not None and s.publisher.published else None
        # Descriptor.close detaches before os.close, so match the retained inode
        # via all original file state only until our one intended close fires.
        info=None
        if not triggered and s.publisher is not None and s.publisher.published:
            try:info=os.fstat(fd)
            except OSError:pass
            expected=s.publisher.files[-1].inode
            if info is not None and (info.st_dev,info.st_ino)==expected:
                triggered.append(fd);real(fd);raise ordinary
        return real(fd)
    monkeypatch.setattr(publisher.os,'close',close);s.parameter_close_error=cancel
    with pytest.raises(cancel_type)as caught:run()
    assert caught.value is cancel and is_cleanup(cancel)and triggered
    assert (tmp_path/'response/response.json').is_file()and s.publisher.closed and s.publisher.failed is ordinary
    assert s.case.params.retired and s.case.upload.retired
    held=loader._FAILURE_QUARANTINE[-1];assert len(held['public_mappings'])==335 and held['context']is s.case.ctx
    assert not any(r[0]=='mapping-close'for r in env.rec.log)
    assert stage._QUARANTINE[-1][-1]is cancel
    env.restored()


def test_genuine_publication_body_preserves_identity_through_cleanup_and_loader(public,profile,tmp_path,monkeypatch):
    env,s,_,_,run=environment(public,profile,tmp_path,monkeypatch)
    body=ValueError('publisher body');s.publish_error=body;s.parameter_close_error=KeyboardInterrupt('cleanup')
    with pytest.raises(ValueError)as caught:run()
    assert caught.value is body and not is_cleanup(body)
    assert not s.publisher.published and not (tmp_path/'response/response.json').exists()
    assert backend._QUARANTINE and len(loader._FAILURE_QUARANTINE[-1]['public_mappings'])==335
    assert not any(r[0]=='mapping-close'for r in env.rec.log)
    env.restored()


@pytest.mark.parametrize('cancel_type',[KeyboardInterrupt,SystemExit])
def test_engine_uses_same_selected_exception_for_quarantine_and_propagation(public,cancel_type):
    c=engine_fixture.setup(public,length=1);engine=backend.BorrowedServerEngine(**c.args)
    ordinary=mark(OSError('marked publisher cleanup'));cancel=cancel_type('later barrier')
    def publish(*args):
        c.state.inputs[0].polynomial.error=cancel
        raise ordinary
    with pytest.raises(cancel_type)as caught:engine.run(publish)
    assert caught.value is cancel and engine.failed is cancel and backend._QUARANTINE[-1][-1]is cancel
    assert is_cleanup(cancel)


def test_full_identity_failure_precedes_key_directory_and_factories(public,profile,tmp_path,monkeypatch):
    env,s,c,fac,run=environment(public,profile,tmp_path,monkeypatch);c['public']['model_id']='f'*64
    with pytest.raises(ValueError,match='identity'):run()
    assert s.case is None and not any(r[0]in('recording-runtime','context-construct')for r in env.rec.log)
    assert not (tmp_path/'response').exists()


def test_actual_role_factories_still_deny_without_touching_arguments():
    class Trap:
        def __getattribute__(self,name):raise AssertionError(name)
    for fn in (stage.create_reference_session,backend.create_reference_session,publisher.create_actual_publisher,loader.open_client,loader.open_server):
        with pytest.raises((PermissionError,TypeError)):
            if fn is loader.open_client:fn(Trap(),Trap(),Trap())
            elif fn is loader.open_server:fn(Trap(),Trap())
            else:fn(Trap())


def test_all_selected_finalizers_and_cleanup_modules_are_explicitly_one_identity():
    import qwen_file_lifecycle,qwen_public_parameters,qwen_finalizer,cleanup_origin,cleanup,key_descriptor_io
    assert backend.Finalizer is stage.Finalizer is qwen_file_lifecycle.Finalizer is qwen_public_parameters.Finalizer is qwen_finalizer.Finalizer
    assert loader.select is qwen_finalizer.select is publisher.select is cleanup.select is cleanup_origin.select
    assert key_descriptor_io.select is cleanup_origin.select


def test_close_accessor_replacement_is_rejected_and_original_cleanup_runs(public,profile,tmp_path,monkeypatch):
    from dataclasses import replace
    env,s,c,fac,_=environment(public,profile,tmp_path,monkeypatch)
    old=fac.publisher
    def output(handle,digest):
        out=old(handle,digest);write=out.write_ciphertext
        def replaced(*args):
            row=write(*args);s.case.params.close=lambda primary=None:None;return row
        out.write_ciphertext=replaced;return out
    fac=replace(fac,publisher=output)
    with pytest.raises(ValueError):stage.server_for_recording(env.admission,env.public,c,key_dependencies=env.dependencies,factories=fac)
    assert ('parameter-close',)in env.rec.log and loader._FAILURE_QUARANTINE

def test_handle_mutation_after_finish_is_rejected_before_key_retirement(public,profile,tmp_path,monkeypatch):
    from dataclasses import replace
    env,s,c,fac,_=environment(public,profile,tmp_path,monkeypatch);old=fac.publisher
    def output(handle,digest):
        out=old(handle,digest);finish=out.finish
        def replaced():raw=finish();handle.public=object();return raw
        out.finish=replaced;return out
    fac=replace(fac,publisher=output)
    with pytest.raises(ValueError):stage.server_for_recording(env.admission,env.public,c,key_dependencies=env.dependencies,factories=fac)
    assert loader._FAILURE_QUARANTINE and not any(r[0]=='mapping-close'for r in env.rec.log)

def test_factory_mutation_during_cleanup_is_rejected_before_loader_return(public,profile,tmp_path,monkeypatch):
    from dataclasses import replace
    env,s,c,fac,_=environment(public,profile,tmp_path,monkeypatch);old=fac.parameter_files;holder={}
    def parameters(handle):
        owner=old(handle);close=owner.close
        def changed(primary=None):
            close(primary=primary);object.__setattr__(holder['fac'],'engine_dependencies',lambda ctx:None)
        owner.close=changed;return owner
    fac=replace(fac,parameter_files=parameters);holder['fac']=fac
    with pytest.raises(ValueError):stage.server_for_recording(env.admission,env.public,c,key_dependencies=env.dependencies,factories=fac)
    assert loader._FAILURE_QUARANTINE and not any(r[0]=='mapping-close'for r in env.rec.log)
