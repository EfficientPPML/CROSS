import contextlib,io,json,os
import numpy as np
import pytest
import qwen_file_lifecycle as l
from qwen_file_contract import *

def payload_bytes(binding,*,dtype=np.uint32):
    stream=io.BytesIO()
    np.savez(stream,payload=np.zeros(tuple(binding['shape']),dtype=dtype),moduli=np.asarray(binding['moduli'],dtype=np.int64),
        degree_layout=np.asarray(binding['degree_layout'],dtype=np.int64),precision=np.asarray(32,dtype=np.int64),
        scale=np.asarray(binding['scale'],dtype=np.float64),nsd=np.asarray(1,dtype=np.int64))
    return stream.getvalue()

def terminal(public,c,input_hash,response_bytes,*,change_report=None,change_outer=None):
    report=dict(schema='qwen-organizer-server-report-v1',producer_source_manifest_sha256=c['public']['producer_source_manifest_sha256'],
        consumer_source_manifest_sha256=c['public']['consumer_source_manifest_sha256'],context_id=c['public']['context_id'],model_id=c['public']['model_id'],
        request_contract_sha256=digest(c),input_digest=input_hash,response_manifest_sha256=sha(response_bytes),output_count=38,
        body_completed=True,source_unchanged=True,parameters_rechecked=True,cleanup_completed=True,termination_scope_completed=True,passed=True)
    if change_report:change_report(report)
    raw=canonical(report)
    outer=dict(schema='qwen-organizer-server-outer-v1',report_sha256=sha(raw),producer_source_manifest_sha256=c['public']['producer_source_manifest_sha256'],
        request_contract_sha256=digest(c),server_pid=1234,server_pgid=1233,exit_code=0,reaped=True,process_group_absent=True)
    if change_outer:change_outer(outer)
    outer_raw=canonical(outer)
    return l.ProducerTerminal(raw,outer_raw),sha(raw),sha(outer_raw)

def files(tmp_path,public,c,*,bad_index=None,bad_header=False):
    root=tmp_path/'responses';root.mkdir();input_hash='a'*64
    binding=ciphertext_binding(public,c,'response',0,actual_scale=DELTA,input_digest=input_hash)
    raw=payload_bytes(binding)
    base=root/'block00.npz';base.write_bytes(raw)
    for i in range(1,38):os.link(base,root/f'block{i:02}.npz')
    artifacts=[artifact(public,c,'response',i,payload_sha256=sha(raw),actual_scale=DELTA,input_digest=input_hash)for i in range(38)]
    if bad_index is not None:
        path=root/f'block{bad_index:02}.npz';path.unlink();path.write_bytes(b'changed')
    if bad_header:
        path=root/'block00.npz';path.unlink();wrong=payload_bytes(binding,dtype=np.uint64);path.write_bytes(wrong)
        artifacts[0]['payload_sha256']=sha(wrong)
    response_bytes=canonical(response(public,c,input_hash,artifacts));(root/'response.json').write_bytes(response_bytes)
    proof,rpin,opin=terminal(public,c,input_hash,response_bytes)
    return dict(public=public,contract=c,input_digest=input_hash,response_bytes=response_bytes,root=root,terminal=proof,
        producer_report_sha256=rpin,producer_outer_sha256=opin)

def test_all38_real_synthetic_rns_before_private_and_full_vocabulary(tmp_path,public,request_case):
    args=files(tmp_path,public,request_case);owner=l.ResponseFiles(**args);events=[]
    assert len(owner.records)==38 and len(owner.files)==39
    @contextlib.contextmanager
    def private():
        events.append('private-open');assert len(owner.records)==38
        try:yield object()
        finally:events.append('private-close')
    def decode(ctx,payload,binding):
        assert payload.shape==(1,2,256,512,6)and not payload.flags.writeable
        index=binding['index'];events.append(index)
        slots=np.full(65536,1e30,dtype=np.float64);start=binding['logical']['start'];width=binding['logical']['stop']-start
        slots[:width]=np.arange(start,start+width,dtype=np.float64);return slots
    output=owner.run_client(private,decode)
    assert np.array_equal(output,np.arange(151936,dtype=np.float64).reshape(1,-1))
    assert events==['private-open',*range(38),'private-close']
    with pytest.raises(ValueError):owner.run_client(private,decode)
    owner.close();assert all(f.fd is None for f in owner.files)

@pytest.mark.parametrize('index',[0,19,37])
def test_corrupt_actual_payload_prevents_admission_before_private(tmp_path,public,request_case,index):
    args=files(tmp_path,public,request_case,bad_index=index)
    with pytest.raises(ValueError,match='hash differs'):l.ResponseFiles(**args)

def test_actual_rns_header_rejected_before_backend_or_private(tmp_path,public,request_case):
    args=files(tmp_path,public,request_case,bad_header=True)
    with pytest.raises(ValueError):l.ResponseFiles(**args)

@pytest.mark.parametrize('where,field,value',[
    ('report','passed',False),('report','cleanup_completed',False),('report','output_count',38.0),
    ('report','consumer_source_manifest_sha256','3'*64),('report','parameters_rechecked',False),
    ('outer','exit_code',1),('outer','reaped',False),('outer','process_group_absent',False),('outer','server_pid',True)])
def test_public_producer_gate_precedes_any_response_open(tmp_path,public,request_case,monkeypatch,where,field,value):
    args=files(tmp_path,public,request_case)
    proof,rpin,opin=terminal(public,request_case,args['input_digest'],args['response_bytes'],
        change_report=(lambda r:r.update({field:value}))if where=='report'else None,
        change_outer=(lambda r:r.update({field:value}))if where=='outer'else None)
    args.update(terminal=proof,producer_report_sha256=rpin,producer_outer_sha256=opin)
    def denied(*a,**k):raise AssertionError('file opening preceded public proof rejection')
    monkeypatch.setattr(l,'HeldDirectory',denied)
    with pytest.raises(ValueError):l.ResponseFiles(**args)

def test_independently_pinned_outer_hash_required_before_files(tmp_path,public,request_case,monkeypatch):
    args=files(tmp_path,public,request_case);args['producer_outer_sha256']='f'*64
    monkeypatch.setattr(l,'HeldDirectory',lambda *a:(_ for _ in ()).throw(AssertionError('unexpected file open')))
    with pytest.raises(ValueError,match='externally pinned'):l.ResponseFiles(**args)

def test_retained_inode_rechecked_before_private_loader(tmp_path,public,request_case):
    args=files(tmp_path,public,request_case);owner=l.ResponseFiles(**args);opened=[]
    path=args['root']/'block37.npz';path.unlink();path.write_bytes(b'new bytes')
    with pytest.raises(ValueError):owner.run_client(lambda:opened.append(True),lambda *a:None)
    assert opened==[]
    with pytest.raises(ValueError):owner.close()

@pytest.mark.parametrize('body_kind',[RuntimeError,KeyboardInterrupt,SystemExit])
def test_decode_primary_survives_cleanup_cancellation_and_retains_inputs(tmp_path,public,request_case,body_kind):
    args=files(tmp_path,public,request_case);owner=l.ResponseFiles(**args);body=body_kind('genuine decode failure');events=[]
    class Private:
        def __enter__(self):events.append('enter');return self
        def __exit__(self,kind,error,tb):
            events.append(error);raise KeyboardInterrupt('cleanup-only cancellation')
    def decode(*a):raise body
    with pytest.raises(body_kind)as caught:owner.run_client(Private,decode)
    assert caught.value is body and events==['enter',body]
    assert owner.failed is body and len(owner.records)==38 and all(f.fd is not None for f in owner.files)
    with pytest.raises(body_kind):owner.close()
    # No actual loader/HE state exists: synthetic emulation of process retirement.
    owner.failed=None;owner.close()

def test_nonfinite_all_slot_decode_cannot_be_hidden_by_row_slice(tmp_path,public,request_case):
    args=files(tmp_path,public,request_case);owner=l.ResponseFiles(**args)
    @contextlib.contextmanager
    def private():yield None
    def decode(*args):
        slots=np.zeros(65536,dtype=np.float64);slots[-1]=np.nan;return slots
    with pytest.raises(ValueError,match='all-slot'):owner.run_client(private,decode)
    owner.failed=None;owner.close()

def test_actual_reference_entry_disabled_before_argument_access():
    class Hostile:
        def __getattribute__(self,name):raise AssertionError('argument touched')
    with pytest.raises(PermissionError,match='disabled'):l.create_reference_session(Hostile())
