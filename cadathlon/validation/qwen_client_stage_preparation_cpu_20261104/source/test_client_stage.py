"""Only new client boundaries: real synthetic files, recording codecs/keys.

No inherited test suite is selected. Context construction/security/key arrays
and main-thread signals remain explicit recording seams. Payload files use the
real Qwen physical dimensions and owned Publisher/ResponseFiles implementations.
"""
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import json,sys
import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
sys.path.extend((str(HERE/'recording_sources'),str(HERE/'recording_sources/reader43')))
from test_key_loader_bridge import setup_bridge
from test_qwen_key_contract import profile
import client_stage as stage
import qwen_key_loader as loader
import qwen_publisher as publication
import qwen_file_lifecycle as lifecycle
from qwen_file_contract import canonical,sha,digest,request,ciphertext_binding,DELTA
from cleanup_origin import mark,is_cleanup


@pytest.fixture(autouse=True)
def retire_synthetic_quarantines():
    stage._QUARANTINE.clear();publication._QUARANTINE.clear();lifecycle._QUARANTINE.clear()
    yield
    stage._QUARANTINE.clear();publication._QUARANTINE.clear();lifecycle._QUARANTINE.clear()


class Array:
    def __init__(self,value,log,label):self.value=value;self.log=log;self.label=label;self.error=None
    def block_until_ready(self):
        self.log.append(('ct-barrier',self.label))
        if self.error is not None:raise self.error
        return self
    def __array__(self,dtype=None,copy=None):
        return self.value if dtype is None else self.value.astype(dtype)


class Ciphertext:
    def __init__(self,payload,binding,log,index):
        self.polynomial=Array(payload,log,('payload',index))
        self.moduli_array=Array(np.asarray(binding['moduli'],dtype=np.int64),log,('moduli',index))
        self.shape=payload.shape;self.degree_layout=(256,512);self.precision=32
        self._ckks_scale=binding['scale'];self._ckks_nsd=1;self.q=list(binding['moduli']);self.index=index
    def validate(self):pass
    def get_moduli(self):return self.q


def environment(public,profile,tmp_path,monkeypatch,positions=(7,),*,output=False):
    env=setup_bridge(tmp_path,monkeypatch,profile,public,'client')
    state=NS(ctx=None,handle=None,publisher=None,response=None,inputs=[],cts=[],decoded=[],scratch=np.empty(65536),
             encrypt_hook=None,decode_hook=None,materialize_hook=None,publisher_hook=None,secure_hook=None)
    contract=request(public,json.loads(env.admission.keys.identity_bytes),'client_recording',list(positions))
    original=env.rec.runtime.Context
    class Context(original):
        def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);state.ctx=self
        def encrypt_slots(self,slots,scale=None):
            index=len(state.inputs);env.rec.log.append(('encrypt',index,scale))
            state.inputs.append(np.array(slots,copy=True))
            binding=ciphertext_binding(public,contract,'request',index)
            assert scale==binding['scale'] and type(scale)is float
            value=Ciphertext(np.zeros(binding['shape'],dtype=np.uint32),binding,env.rec.log,index)
            state.cts.append(value)
            if state.encrypt_hook:state.encrypt_hook(self,value,index)
            return value
        def decrypt_slots(self,value=None,*,scale=None,slots_to_decode=None,validate_approximation=True):
            if value is None:return 'record-only codec' # original recording constructor check
            call=(value.index,scale,slots_to_decode,validate_approximation)
            state.decoded.append(call);env.rec.log.append(('decode',value.index))
            assert type(scale)is float and scale==value._ckks_scale
            assert slots_to_decode==65536 and validate_approximation is True
            state.scratch.fill(value.index+0.25)
            if state.decode_hook:state.decode_hook(self,value)
            return state.scratch
    env.rec.runtime.Context=Context
    def secure(ctx,value):
        assert ctx is state.ctx;env.rec.log.append(('secure',value.index))
        if state.secure_hook:state.secure_hook(ctx,value)
    def publisher(handle,c):
        state.handle=handle
        p=publication.Publisher(public,c,'request',tmp_path/'upload',shield=nullcontext)
        state.publisher=p
        if state.publisher_hook:state.publisher_hook(p,handle)
        return p
    factories=stage.RecordingInputFactories(publisher,secure,nullcontext)
    def input_run(embeddings=None):
        if embeddings is None:embeddings=np.arange(len(positions)*1024,dtype=np.float64).reshape(1,len(positions),1024)/1024
        return stage.publish_input_for_recording(env.admission,env.public,env.private,request_id='client_recording',
            embeddings=embeddings,positions=positions,key_dependencies=env.dependencies,factories=factories)
    return env,state,contract,factories,input_run


def response_environment(public,profile,tmp_path,monkeypatch):
    env,state,c,_,_=environment(public,profile,tmp_path,monkeypatch,positions=(3,11,20))
    input_digest='5'*64;root=tmp_path/'response';scales=[]
    binding=ciphertext_binding(public,c,'response',0,actual_scale=DELTA,input_digest=input_digest)
    payload=np.zeros(binding['shape'],dtype=np.uint32)
    with publication.Publisher(public,c,'response',root,input_digest=input_digest,shield=nullcontext)as p:
        for index in range(38):
            payload.flat[0]=index;scale=DELTA*(1+index/1000);scales.append(scale)
            p.write_payload(index,payload,moduli=binding['moduli'],degree_layout=[256,512],precision=32,actual_scale=scale,nsd=1)
        raw=p.finish()
    report=canonical(dict(schema='qwen-organizer-server-report-v1',producer_source_manifest_sha256=c['public']['producer_source_manifest_sha256'],consumer_source_manifest_sha256=c['public']['consumer_source_manifest_sha256'],context_id=c['public']['context_id'],model_id=c['public']['model_id'],request_contract_sha256=digest(c),input_digest=input_digest,response_manifest_sha256=sha(raw),output_count=38,body_completed=True,source_unchanged=True,parameters_rechecked=True,cleanup_completed=True,termination_scope_completed=True,passed=True))
    outer=canonical(dict(schema='qwen-organizer-server-outer-v1',report_sha256=sha(report),producer_source_manifest_sha256=c['public']['producer_source_manifest_sha256'],request_contract_sha256=digest(c),server_pid=111,server_pgid=112,exit_code=0,reaped=True,process_group_absent=True))
    terminal=lifecycle.ProducerTerminal.authenticate(public,c,input_digest,raw,report_bytes=report,report_sha256=sha(report),outer_bytes=outer,outer_sha256=sha(outer))
    def files(p,contract,h):
        env.rec.log.append(('response-auth-start',))
        state.response=lifecycle.ResponseFiles(p,contract,h,raw,root,terminal=terminal,producer_report_sha256=sha(report),producer_outer_sha256=sha(outer))
        env.rec.log.append(('response-auth-complete',len(state.response.records)))
        return state.response
    def materialize(ctx,array,b):
        assert ctx is state.ctx and not array.flags.writeable
        index=int(array.flat[0])
        if state.materialize_hook:state.materialize_hook(ctx,array,b,index)
        value=Ciphertext(array,b,env.rec.log,index);state.cts.append(value);return value
    def secure(ctx,value):assert ctx is state.ctx;env.rec.log.append(('secure',value.index))
    factories=stage.RecordingOutputFactories(files,materialize,secure,nullcontext)
    def run():return stage.consume_output_for_recording(env.admission,env.public,env.private,c,input_digest,key_dependencies=env.dependencies,factories=factories)
    return env,state,c,factories,run,root,scales


def assert_failed_key_retention(env,state):
    assert loader._FAILURE_QUARANTINE[-1]['context']is state.ctx
    assert [r[1]for r in env.rec.log if r[0]=='mapping-close']==[m.name for m in env.rec.maps[:335]if m.name!='public_key.npy']
    assert not any(r[0]=='codec-cache-evict'for r in env.rec.log)
    assert stage._QUARANTINE
    env.restored()


@pytest.mark.parametrize('positions',[(7,),tuple(2+3*i for i in range(17))])
def test_original_rows_positions_one_or_two_fresh_batch_one_inputs(public,profile,tmp_path,monkeypatch,positions):
    env,s,c,fac,run=environment(public,profile,tmp_path,monkeypatch,positions)
    embeddings=np.arange(len(positions)*1024,dtype=np.float64).reshape(1,len(positions),1024)/1024
    result=run(embeddings);assert json.loads(result.contract_bytes)==c
    manifest=json.loads(result.manifest_bytes);assert result.input_digest==manifest['input_digest']
    assert len(s.inputs)==len(manifest['artifacts'])==(len(positions)+15)//16
    for token in range(len(positions)):
        for feature in (0,7,1023):assert s.inputs[token//16][(token%16)*4096+feature]==embeddings[0,token,feature]
    for tile_index,tile in enumerate(s.inputs):
        active=min(16,len(positions)-tile_index*16)
        grid=tile.reshape(16,4096);assert np.all(grid[:active,1024:]==0)and np.all(grid[active:]==0)
        row=manifest['artifacts'][tile_index]['binding'];assert row['level']==49 and row['shape']==[1,2,256,512,100]
    assert s.publisher.closed and s.publisher.published and s.handle.ctx is None
    cache=next(i for i,r in enumerate(env.rec.log)if r[0]=='codec-cache-evict')
    assert max(i for i,r in enumerate(env.rec.log)if r[0]=='ct-barrier')<cache
    assert not loader._FAILURE_QUARANTINE;env.restored()


def test_all38_authenticated_before_private_default_decode_and_full_scratch_snapshots(public,profile,tmp_path,monkeypatch):
    env,s,c,fac,run,root,scales=response_environment(public,profile,tmp_path,monkeypatch)
    result=run();assert result.shape==(1,151936)and result.dtype==np.float64 and not result.flags.writeable
    for column in (0,4095,4096,98304,151551,151552,151935):assert result[0,column]==column//4096+0.25
    assert s.decoded==[(i,scales[i],65536,True)for i in range(38)]
    assert env.rec.log.index(('response-auth-complete',38))<next(i for i,r in enumerate(env.rec.log)if r[0]=='map')
    retirement=next(i for i,r in enumerate(env.rec.log)if r[0]=='codec-cache-evict')
    assert max(i for i,r in enumerate(env.rec.log)if r[0]=='ct-barrier')<retirement
    assert s.response.closed and not loader._FAILURE_QUARANTINE;env.restored()


def test_last_response_hash_failure_blocks_all_private_loader_work(public,profile,tmp_path,monkeypatch):
    env,s,c,fac,run,root,_=response_environment(public,profile,tmp_path,monkeypatch)
    (root/'block37.npz').write_bytes(b'corrupt synthetic response')
    with pytest.raises(ValueError):run()
    assert not any(r[0]in('map','context-construct','recording-runtime')for r in env.rec.log)
    assert s.ctx is None


def test_nonfinite_original_embedding_rejects_before_keys_or_output(public,profile,tmp_path,monkeypatch):
    env,s,c,fac,run=environment(public,profile,tmp_path,monkeypatch)
    value=np.zeros((1,1,1024));value[0,0,3]=np.nan
    with pytest.raises(ValueError):run(value)
    assert s.ctx is s.publisher is None and not env.rec.maps and not (tmp_path/'upload').exists()


def test_full_admission_failure_precedes_callbacks_and_key_readers(public,profile,tmp_path,monkeypatch):
    env,s,c,fac,run=environment(public,profile,tmp_path,monkeypatch)
    object.__setattr__(env.admission.keys,'identity_bytes',canonical(dict(schema='wrong')))
    with pytest.raises((ValueError,TypeError)):run()
    assert s.ctx is s.publisher is None and not env.rec.maps


def test_partial_ciphertext_metadata_failure_still_barriers_original_arrays(public,profile,tmp_path,monkeypatch):
    env,s,c,fac,run=environment(public,profile,tmp_path,monkeypatch)
    s.encrypt_hook=lambda ctx,value,index:setattr(value,'shape',(1,))
    with pytest.raises(ValueError):run()
    assert ('ct-barrier',('payload',0))in env.rec.log and ('ct-barrier',('moduli',0))in env.rec.log
    assert_failed_key_retention(env,s)


@pytest.mark.parametrize('kind',[ValueError,KeyboardInterrupt,SystemExit])
def test_decode_body_failure_identity_and_consumer_retirement_cross_key_scope(public,profile,tmp_path,monkeypatch,kind):
    env,s,c,fac,run,root,_=response_environment(public,profile,tmp_path,monkeypatch);body=kind('synthetic decode body')
    def fail(ctx,value):raise body
    s.decode_hook=fail
    with pytest.raises(kind)as caught:run()
    assert caught.value is body and not is_cleanup(body)
    assert env.rec.log.count(('ct-barrier',('payload',0)))>=2
    assert_failed_key_retention(env,s)


def test_late_publisher_close_cleanup_error_then_consumer_cancel(public,profile,tmp_path,monkeypatch):
    env,s,c,fac,run=environment(public,profile,tmp_path,monkeypatch);ordinary=OSError('publisher cleanup');cancel=KeyboardInterrupt('consumer barrier')
    def hook(p,handle):
        close=p.close
        def fail(primary=None):
            close(primary=primary);s.cts[0].polynomial.error=cancel;raise ordinary
        p.close=fail
    s.publisher_hook=hook
    with pytest.raises(KeyboardInterrupt)as caught:run()
    assert caught.value is cancel and is_cleanup(ordinary)and is_cleanup(cancel)
    assert s.publisher.closed and (tmp_path/'upload/request.json').is_file()
    assert_failed_key_retention(env,s)


def test_replaced_encrypt_codec_rejected_with_original_arrays_retained(public,profile,tmp_path,monkeypatch):
    env,s,c,fac,run=environment(public,profile,tmp_path,monkeypatch)
    s.encrypt_hook=lambda ctx,value,index:setattr(ctx,'encrypt_slots',lambda *a,**k:None)
    with pytest.raises(ValueError,match='callback'):run()
    assert_failed_key_retention(env,s)


def test_replaced_publisher_close_rejected_but_original_close_runs(public,profile,tmp_path,monkeypatch):
    env,s,c,fac,run=environment(public,profile,tmp_path,monkeypatch);closed=[]
    def hook(p,handle):
        close=p.close;write=p.write_ciphertext
        def original_close(primary=None):closed.append(True);return close(primary=primary)
        def changed(*args):row=write(*args);p.close=lambda primary=None:None;return row
        p.close=original_close;p.write_ciphertext=changed
    s.publisher_hook=hook
    with pytest.raises(ValueError,match='callback'):run()
    assert closed and s.publisher.closed;assert_failed_key_retention(env,s)


def test_handle_mutation_after_input_finish_rejects_before_key_retirement(public,profile,tmp_path,monkeypatch):
    env,s,c,fac,run=environment(public,profile,tmp_path,monkeypatch)
    def hook(p,handle):
        finish=p.finish
        def changed():raw=finish();handle.public=object();return raw
        p.finish=changed
    s.publisher_hook=hook
    with pytest.raises(ValueError,match='owner'):run()
    assert s.publisher.closed and s.publisher.published;assert_failed_key_retention(env,s)


def test_factory_mutation_during_publisher_close_is_rejected(public,profile,tmp_path,monkeypatch):
    env,s,c,fac,run=environment(public,profile,tmp_path,monkeypatch)
    def hook(p,handle):
        close=p.close
        def changed(primary=None):close(primary=primary);object.__setattr__(fac,'secure',lambda *args:None)
        p.close=changed
    s.publisher_hook=hook
    with pytest.raises(ValueError,match='factory'):run()
    assert_failed_key_retention(env,s)


def test_output_padding_nonfinite_is_not_hidden_by_vocabulary_slice(public,profile,tmp_path,monkeypatch):
    env,s,c,fac,run,root,_=response_environment(public,profile,tmp_path,monkeypatch)
    s.decode_hook=lambda ctx,value:s.scratch.__setitem__(65535,np.nan)
    with pytest.raises(ValueError,match='finite'):run()
    assert_failed_key_retention(env,s)


def test_last_decode_then_retirement_barrier_failure_retains_all38(public,profile,tmp_path,monkeypatch):
    env,s,c,fac,run,root,_=response_environment(public,profile,tmp_path,monkeypatch);error=OSError('retirement barrier')
    def last(ctx,value):
        if value.index==37:s.cts[0].polynomial.error=error
    s.decode_hook=last
    with pytest.raises(OSError)as caught:run()
    assert caught.value is error and len(s.decoded)==len(s.cts)==38
    assert_failed_key_retention(env,s)


def test_materializer_cannot_change_the_preserved_response_binding(public,profile,tmp_path,monkeypatch):
    env,s,c,fac,run,root,_=response_environment(public,profile,tmp_path,monkeypatch)
    s.materialize_hook=lambda ctx,array,b,index:b.__setitem__('scale',b['scale']*2)
    with pytest.raises(ValueError,match='scale'):run()
    assert_failed_key_retention(env,s)


def test_actual_client_factories_deny_before_inspection():
    class Trap:
        def __getattribute__(self,name):raise AssertionError(name)
    for fn in (stage.create_client_input,stage.create_client_output):
        with pytest.raises(PermissionError):fn(Trap())
