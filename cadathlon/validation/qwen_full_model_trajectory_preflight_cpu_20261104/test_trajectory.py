"""Synthetic boundary/metric/instrumentation tests only; no learned trajectory."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,os
import numpy as np
import pytest
import trajectory_assets as assets
from trajectory_observer import Observer,metric,packed_expected,TrajectoryFailure
P=Path(__file__).resolve().parent

def test_fixed_runtime_is_exact87():
    req=assets.requirement();assert len(assets.runtime_gate(req))==87
    assert req['fixtures']==[{'name':'checkpoint_embedding_row0_S1','sequence':1,'selector':'embedding_row0','positions':[0]},
        {'name':'first_original_calibration_S8','sequence':8,'selector':'first_file_order_sequence_length8','positions':list(range(8))}]

def test_gate_before_token_read(tmp_path,monkeypatch):
    root=tmp_path/'calibration';root.mkdir();p=root/'report.json';p.write_text('{}')
    req={'calibration':{'root':str(root),'report':p.name,'report_sha256':'0'*64,'tokens':'never_open.bin'}}
    with pytest.raises(ValueError,match='report changed'):assets.calibration_gate(req)
    assert not (root/'never_open.bin').exists()

@pytest.mark.parametrize('bad',[True,1.0,-1,151936,'3'])
def test_exact_typed_original_token_ids(tmp_path,bad):
    p=tmp_path/'records.json';p.write_text(json.dumps([dict(sequence_length=8,token_ids=[bad]+[2]*7)]))
    req=dict(calibration=dict(root=str(tmp_path),tokens=p.name,tokens_sha256=assets.sha(p)))
    with pytest.raises(ValueError,match='token schema'):assets.selected_ids(req)

def test_first_file_order_not_domain_or_output_selection(tmp_path):
    p=tmp_path/'records.json';p.write_text(json.dumps([dict(sequence_length=16,token_ids=list(range(16))),dict(id='first',sequence_length=8,token_ids=list(range(8))),dict(id='second',sequence_length=8,token_ids=[10]*8)]))
    req=dict(calibration=dict(root=str(tmp_path),tokens=p.name,tokens_sha256=assets.sha(p)))
    ids,record=assets.selected_ids(req);assert ids==((0,),tuple(range(8))) and 'token_ids'not in record

@pytest.mark.parametrize('bad',[np.full(3,np.nan),np.full(3,np.inf),np.full(3,1e308),np.ones(4),np.array(['x','y','z'])])
def test_metric_json_failure(bad):
    r=metric(bad,np.full(3,-1e308));assert not r['passed'];json.dumps(r,allow_nan=False)

def test_finite_extreme_overflow_has_no_infinite_metric():
    r=metric(np.full(3,1e308),np.full(3,-1e308));assert r['max_abs_error']is None

def test_head_unpacking_and_padding():
    a=np.arange(1*16*8*128,dtype=np.float64).reshape(1,16,8,128)
    p=packed_expected(a,'heads',8).reshape(16,4096)
    np.testing.assert_array_equal(p[:8,:2048],a[0].transpose(1,0,2).reshape(8,2048))
    assert not p[8:].any() and not p[:,2048:].any()

def observer(sequence=1):
    m=json.loads((P/'inputs/candidate_manifest.json').read_bytes())
    ops={};blocks=[]
    for i in range(28):
        for kind,width in (('input.inverse_sqrt',1024),('q_linear',2048),('k_linear',1024),('v_linear',1024),('gqa',2048),('output_linear',1024),('post.inverse_sqrt',1024),('gate_linear',3072),('up_linear',3072),('silu',3072),('gated_mlp',1024)):
            ops[f'{i}.{kind}']=np.zeros((1,sequence,width))
        for kind,heads in (('q.inverse_sqrt',16),('k.inverse_sqrt',8),('q_rope',16),('k_rope',8)):ops[f'{i}.{kind}']=np.zeros((1,heads,sequence,128))
        blocks.append(np.zeros((1,sequence,1024)))
    ops['final.inverse_sqrt']=np.zeros((1,1024))
    ref=SimpleNamespace(operators=ops,blocks=blocks,logits=np.zeros((1,151936)))
    return Observer(ref,m,tuple(range(sequence)))

def note(o,label,data=None):o(label,SimpleNamespace(polynomial=np.zeros(65536) if data is None else data),{})

def test_all421_public_operator_labels_and28_blocks38_vocab():
    o=observer()
    for i in range(28):
        p=f'layer{i}.'
        for kind in ('input','q','k'):note(o,p+f'attention.tile0.{kind}_norm.output')
        for kind in ('q','k','v'):note(o,p+f'attention.tile0.{kind}.linear')
        for kind in ('q','k'):note(o,p+f'attention.tile0.{kind}.rope.output')
        note(o,p+'attention.output0')
        for kind in ('o','up','down'):note(o,p+f'tail.tile0.{kind}.linear')
        note(o,p+'tail.tile0.gate.linear')
        note(o,p+'tail.tile0.post_norm.output');note(o,p+'tail.tile0.silu.output');note(o,p+'block.output0')
    note(o,'final.norm.output');note(o,'final.norm.output')
    for i in range(38):note(o,f'logits.{i}')
    assert len(o.operators)==421 and len(o.blocks)==28 and len(o.logits)==38 and o.duplicates=={'final.inverse_sqrt':1}
    with pytest.raises(TrajectoryFailure,match='all113'):o.complete()

@pytest.mark.parametrize('label',['layer0.attention.tile0.q.linear','layer0.block.output0','logits.37'])
def test_inactive_physical_slot_is_gated(label):
    o=observer();x=np.zeros(65536);x[-1]=1e-6
    with pytest.raises(TrajectoryFailure,match='all-slot'):note(o,label,x)

def test_duplicate_checkpoint_cannot_hide_failure():
    o=observer();note(o,'final.norm.output');x=np.zeros(65536);x[0]=1e-6
    with pytest.raises(TrajectoryFailure,match='inconsistent'):note(o,'final.norm.output',x)

@pytest.mark.parametrize('kind,width',[('input',1024),('q',128),('k',128),('post',1024)])
def test_actual_rms_domain_inactive_dummy_and_active_anchors(kind,width):
    o=observer();rule=o.manifest['nonlinear']['0.'+kind+'.inverse_sqrt'];lo,hi=rule['domain'];x=np.zeros((16,4096));x[:,::width]=(lo+hi)/2
    phase='tail' if kind=='post'else'attention';label=f'layer0.{phase}.tile0.{kind}_norm.variance';note(o,label,x.ravel())
    assert o.domains[label]['count']==16*(4096//width)
    x[15,4096-width]=hi*2
    o=observer()
    with pytest.raises(TrajectoryFailure,match='domain'):note(o,label,x.ravel())

def test_softmax_domain_uses_actual_active_branch_not_other_rows():
    o=observer(8);x=np.zeros((16,4096));x[7]=1000
    label='layer3.attention.tile0.softmax.K3.normalized_t_branch_mask'
    x[2,:512]=np.tile(np.r_[1.,-.5,-.5,np.zeros(29)],16)
    note(o,label,x.ravel());assert o.domains[label+'.exp_argument']['maximum']==.5
    x[2,0]=1.01
    with pytest.raises(TrajectoryFailure):note(observer(8),label,x.ravel())

def test_softmax_dummy_denominator_rejected():
    o=observer(8);x=np.zeros((16,4096));x[:,::32]=1.;x[-1,-32]=3.
    with pytest.raises(TrajectoryFailure):note(o,'layer0.attention.tile0.softmax.K3.round0.denominator_anchor_dummy1',x.ravel())

def test_rational_silu_denominator_all_physical_slots():
    o=observer();layer=next(int(k.split('.')[0]) for k,r in o.manifest['nonlinear'].items() if k.endswith('.silu') and r['method']=='rational_tanh')
    x=np.ones(65536);x[-1]=2.2
    with pytest.raises(TrajectoryFailure):note(o,f'layer{layer}.tail.tile0.silu.double1.denominator',x)

def test_observer_current_nonfinite_fails_before_label_filter():
    x=np.zeros(65536);x[321]=np.nan
    with pytest.raises(TrajectoryFailure,match='nonfinite'):note(observer(),'unmeasured.internal',x)

@pytest.fixture
def mapped(tmp_path,monkeypatch):
    value=np.arange(12,dtype='<f8').reshape(3,4);p=tmp_path/'public.bin'
    with p.open('wb') as f:np.save(f,value)
    row=dict(file=p.name,file_sha256=assets.sha(p),sha256=hashlib.sha256(value.tobytes()).hexdigest(),shape=list(value.shape))
    obj=object.__new__(assets.PublicParameters);obj.req=dict(public_parameters=dict(qkv_directory=str(tmp_path),decoder_directory=str(tmp_path)))
    obj.maps={};obj.handles={};obj.records={};obj.gamma_values={};obj.closed=False;obj.qkv={'q':row};obj.decoder={};obj.gamma_rows={}
    yield obj,value,p
    if not obj.closed:obj.close()

def test_public_mapping_digest_shape_and_close(mapped):
    p,v,_=mapped;actual=p.matrix('q');np.testing.assert_array_equal(actual,v);assert not actual.flags.writeable
    assert len(p.maps)==1;p.readback();fds=[h[1] for h in p.handles.values()];p.close()
    for fd in fds:
        with pytest.raises(OSError):os.fstat(fd)

def test_public_map_parent_replacement_rejected(mapped):
    p,v,path=mapped;p.matrix('q');path.rename(path.with_suffix('.old'));path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='identity'):p.readback()

def test_fdopen_failure_closes_untransferred_fd(mapped,monkeypatch):
    p,_,_=mapped;fds=[];original=assets.os.open
    def opened(*a,**k):fd=original(*a,**k);fds.append(fd);return fd
    monkeypatch.setattr(assets.os,'open',opened)
    monkeypatch.setattr(assets.os,'fdopen',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('fdopen injected')))
    with pytest.raises(RuntimeError,match='fdopen'):p.matrix('q')
    assert len(fds)==1
    with pytest.raises(OSError):os.fstat(fds[0])

@pytest.mark.parametrize('field,value',[('shape',[4,3]),('sha256','0'*64),('file_sha256','0'*64),('file','../escape')])
def test_public_descriptor_rejects_before_return(mapped,field,value):
    p,_,_=mapped;p.qkv['q'][field]=value
    with pytest.raises(ValueError):p.matrix('q')

@pytest.mark.parametrize('kind',[RuntimeError,KeyboardInterrupt,SystemExit])
def test_partial_release_attempts_all_and_keeps_primary(kind):
    seen=[];primary=kind('primary')
    def bad():seen.append('bad');raise OSError('close')
    assets.release([bad,lambda:seen.append('last')],primary)
    assert seen==['bad','last'] and primary.__notes__

def test_release_only_cancellation_has_priority():
    seen=[]
    def a():seen.append('ordinary');raise OSError('close')
    def b():seen.append('cancel');raise KeyboardInterrupt('cancel')
    with pytest.raises(KeyboardInterrupt):assets.release([a,b,lambda:seen.append('last')])
    assert seen==['ordinary','cancel','last']
