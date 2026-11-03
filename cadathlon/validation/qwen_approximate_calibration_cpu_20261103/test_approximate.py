from pathlib import Path
from types import SimpleNamespace
import json,signal
import numpy as np
import pytest
import check_approximate as c

def complete_record(native,approx):
    return {'native_top1':int(native.argmax()),'approximate_top1':None if approx is None else int(approx.argmax()),'approximate_complete':approx is not None,'numeric_pass':approx is not None,'domain_pass':approx is not None,'exact_semantic_pass':True}

def test_teacher_and_free_native_comparisons_receive_the_actual_distinct_prefixes():
    calls=[]
    def run(ids,kind,step):
        calls.append((list(ids),kind,step));native=np.array([0.,2.,1.]);approx=np.array([3.,0.,1.]);return native,approx,complete_record(native,approx)
    initial=[4,5];r=c.decode_trace(initial,run)
    assert initial==[4,5] and len(calls)==8
    assert [x[0] for x in calls[:4]]==[[4,5]+[1]*i for i in range(4)]
    assert [x[0] for x in calls[4:]]==[[4,5]+[0]*i for i in range(4)]
    assert not r['complete_sequence_agreement']

def test_stopped_free_sequence_never_substitutes_native_tokens_or_suffix_success():
    calls=[]
    def run(ids,kind,step):
        calls.append((list(ids),kind,step));native=np.array([0.,2.,1.]);approx=None if kind=='free' and step==1 else np.array([3.,0.,1.])
        return native,approx,complete_record(native,approx)
    r=c.decode_trace([4,5],run)
    assert len(calls)==6 and r['approximate_free_tokens']==[0]
    assert r['free_suffix_stopped'] and not r['complete_sequence_agreement']
    assert [x['status'] for x in r['free_records'][2:]]==['unmeasured_suffix_after_approximate_failure']*2
    summary=c.summarize(r['free_records'],4);assert summary['complete_approximate_contexts']==1 and summary['top1_denominator']==4 and not summary['numeric_and_domain_pass']

def test_prefill_gate_uses_aggregate_counts_and_invalid_examples_stay_in_denominator():
    rows=[{'approximate_complete':True,'approximate_top1':1,'native_top1':1,'numeric_pass':True,'domain_pass':True,'exact_semantic_pass':True} for _ in range(260)]
    rows += [{'approximate_complete':False,'approximate_top1':None,'native_top1':1,'numeric_pass':False,'domain_pass':False,'exact_semantic_pass':True} for _ in range(28)]
    r=c.summarize(rows,288);assert r['top1_matches']==260 and r['top1_denominator']==288 and not r['numeric_and_domain_pass'] and not r['complete']
    assert c.math.ceil(.9*288)==260 and c.math.ceil(.9*64)==58 and c.math.ceil(.75*16)==12

def test_published_operator_and_logit_limits_are_distinct():
    x=np.ones((1,151936));y=x*1.075
    assert not c.metric(y,x,.05)['passed'] and c.metric(y,x,.10)['passed']
    assert not c.metric(np.array([np.nan]),np.zeros(1),.10)['passed']
    assert not c.metric(np.zeros(2),np.zeros(1),.10)['passed']

def test_signal_handlers_restore_after_controlled_cancellation():
    previous={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGHUP)}
    with pytest.raises(SystemExit) as caught:
        with c.termination_signals():signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
    assert caught.value.code==128+signal.SIGTERM
    assert {s:signal.getsignal(s) for s in previous}==previous

def test_initial_manifest_failure_publishes_concrete_failed_attempt(tmp_path,monkeypatch):
    import torch
    monkeypatch.setattr(torch,'set_num_threads',lambda *a:None);monkeypatch.setattr(torch,'set_num_interop_threads',lambda *a:None)
    def fail(path):raise ValueError('manifest read failed')
    monkeypatch.setattr(c,'sha',fail);out=tmp_path/'failure'
    with pytest.raises(ValueError,match='manifest read failed'):c.collect(SimpleNamespace(output=out,smoke=True))
    r=json.loads((out/'report.json').read_text());assert not r['passed'] and r['errors'][0]['message']=='manifest read failed' and not r['source_files_unchanged']

def test_final_publication_failure_preserves_primary_cancellation(tmp_path,monkeypatch):
    import torch
    monkeypatch.setattr(torch,'set_num_threads',lambda *a:None);monkeypatch.setattr(torch,'set_num_interop_threads',lambda *a:None)
    def cancel(path):raise KeyboardInterrupt('primary cancellation')
    monkeypatch.setattr(c,'sha',cancel);original=c.write;calls=[]
    def publish(path,value):
        calls.append(path)
        if len(calls)>1:raise OSError('publication failed')
        return original(path,value)
    monkeypatch.setattr(c,'write',publish)
    with pytest.raises(KeyboardInterrupt) as caught:c.collect(SimpleNamespace(output=tmp_path/'cancel',smoke=True))
    assert any('publication failed' in note for note in caught.value.__notes__)

def test_manifest_binding_and_no_validation_source_access():
    assert c.sha(c.P/'candidate-v1/manifest.json')==c.EXPECTED_MANIFEST
    text=(c.P/'check_approximate.py').read_text()
    assert 'qwen_native_workload_audit' not in text and 'validation_inputs_only.json' not in text

@pytest.mark.parametrize('name',list(json.loads((c.P/'candidate-v1/manifest.json').read_text())['source_input_sha256']))
def test_each_frozen_selection_input_rejects_substitution_before_assets(tmp_path,name):
    manifest=json.loads((c.P/'candidate-v1/manifest.json').read_text())
    for n in manifest['source_input_sha256']:(tmp_path/n).write_bytes((c.P/n).read_bytes())
    assert c.validate_manifest_inputs(tmp_path,manifest)==manifest['source_input_sha256']
    changed=tmp_path/name;changed.write_bytes(changed.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='Frozen candidate input mismatch'):c.validate_manifest_inputs(tmp_path,manifest)

def test_mutually_changed_corpus_tokenization_and_pins_cannot_rebind_frozen_manifest(tmp_path):
    manifest=json.loads((c.P/'candidate-v1/manifest.json').read_text())
    for n in manifest['source_input_sha256']:(tmp_path/n).write_bytes((c.P/n).read_bytes())
    for n in ('corpus.json','tokenized_calibration_inputs.json','model_pins.json'):
        path=tmp_path/n;value=json.loads(path.read_text())
        if isinstance(value,dict):value['self_declared_rehash']='replacement'
        else:value[0]['token_ids'][0]+=1
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='Frozen candidate input mismatch'):c.validate_manifest_inputs(tmp_path,manifest)
