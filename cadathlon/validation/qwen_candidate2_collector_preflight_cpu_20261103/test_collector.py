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


@pytest.mark.parametrize('actual,expected',[(np.array([1e308]),np.array([-1e308])),(np.array([np.longdouble('1e400')]),np.zeros(1)),(np.array([1e308]),np.zeros(1))])
def test_metric_overflow_fails_finitely_and_serializes(actual,expected):
    result=c.metric(actual,expected,.1)
    assert not result['passed'];json.dumps(result,allow_nan=False)


def test_metric_large_matching_finite_vectors_do_not_overflow_mean_square():
    result=c.metric(np.array([1e300,-1e300]),np.array([1e300,-1e300]),.1)
    assert result['passed'] and result['nrmse']==0.


@pytest.mark.parametrize('primary,cleanup',[(ValueError('prerequisite failed'),KeyboardInterrupt('cleanup interrupted')),(KeyboardInterrupt('cancelled'),RuntimeError('cleanup failed'))])
def test_cleanup_preserves_primary_and_still_publishes(tmp_path,monkeypatch,primary,cleanup):
    def fail(*args):raise primary
    def clean():raise cleanup
    monkeypatch.setattr(c.candidate_gate,'verify',fail);monkeypatch.setattr(c.gc,'collect',clean)
    out=tmp_path/'attempt'
    with pytest.raises(type(primary)) as caught:c.collect(SimpleNamespace(output=out,smoke=True))
    assert caught.value is primary
    report=json.loads((out/'report.json').read_text());assert not report['passed']
    assert any(e.get('phase')=='model_cleanup' for e in report['errors'])
    assert any('Model cleanup failed' in note for note in primary.__notes__)


def test_initial_prerequisite_failure_precedes_all_asset_reads_and_publishes(tmp_path,monkeypatch):
    def fail(*args):raise ValueError('input mismatch')
    monkeypatch.setattr(c.candidate_gate,'verify',fail)
    def forbidden(*args):raise AssertionError('assets must not be read')
    monkeypatch.setattr(c,'sha',forbidden)
    out=tmp_path/'attempt'
    with pytest.raises(ValueError,match='input mismatch'):c.collect(SimpleNamespace(output=out,smoke=True))
    report=json.loads((out/'report.json').read_text())
    assert not report['passed'] and not report['source_files_unchanged'] and not report['validation_data_read']


def test_publication_failure_does_not_replace_primary_cancellation(tmp_path,monkeypatch):
    primary=KeyboardInterrupt('cancelled')
    def fail(*args):raise primary
    monkeypatch.setattr(c.candidate_gate,'verify',fail);original=c.write;calls=[]
    def publish(path,report):
        calls.append(path)
        if len(calls)>1:raise OSError('publication failed')
        original(path,report)
    monkeypatch.setattr(c,'write',publish)
    with pytest.raises(KeyboardInterrupt) as caught:c.collect(SimpleNamespace(output=tmp_path/'attempt',smoke=True))
    assert caught.value is primary and any('publication failed' in n for n in primary.__notes__)


def test_complete_unchanged_calibration_contract_and_gate_counts():
    p=json.loads((c.P/'calibration/protocol.json').read_text())
    assert p['calibration']['prefill_contexts']==288 and p['calibration']['decode_seeds']==16 and p['calibration']['decode_steps']==4
    assert p['gates']['approximate_operator_block_nrmse']==.05 and p['gates']['approximate_full_logit_nrmse']==.10
    assert p['gates']['nrmse_floor']==.01 and c.math.ceil(.9*288)==260 and c.math.ceil(.9*64)==58
    text=(c.P/'check_approximate.py').read_text()
    assert text.index('prerequisite=candidate_gate.verify(P)')<text.index('assets={n:sha(SNAPSHOT/n)')<text.index('model,loading=')
    assert 'candidate-v1/manifest.json' not in text
