from pathlib import Path
from types import SimpleNamespace
import hashlib,io,json,os,tarfile
import numpy as np
import pytest
import validate_heldout as v
import calibration_gate as gate
import candidate_gate
import validation_gate
P=Path(__file__).resolve().parent
REQ=json.loads((P/'validation_requirement.json').read_text())
REPORT=P/'calibration_prerequisite/report.json';ARCHIVE=P/'calibration_prerequisite/runtime_sources.tar.gz'

def test_full_calibration_and_selection_prerequisites_pass_with_current_exact_runtime():
    req=validation_gate.verify(P)
    checked=gate.verify(REPORT,ARCHIVE,P,req)
    assert checked['verified'] and checked['runtime_sources']==53 and checked['calibration_contexts']==416
    assert candidate_gate.verify(P)['passed']

@pytest.mark.parametrize('field,value',[('ready',False),('full_report_sha256','0'*64),('full_source_archive_sha256','0'*64)])
def test_incomplete_or_changed_calibration_prerequisite_fails_closed(field,value):
    req=dict(REQ);req[field]=value
    with pytest.raises(ValueError):gate.verify(REPORT,ARCHIVE,P,req)

@pytest.fixture
def copied(tmp_path):
    for n in REQ['expected_runtime_source_hashes']:
        if n.startswith('software/'):continue
        p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);os.link(P/n,p)
    for n in ['validation_requirement.json','validation_protocol.json','validate_heldout.py','calibration_gate.py','validation_gate.py','test_validation_gate.py','preflight_validation.py']:
        p=tmp_path/n
        if not p.exists():os.link(P/n,p)
    return tmp_path

def replace(path,content):path.unlink();path.write_bytes(content)

@pytest.mark.parametrize('name',['arithmetic.py','native_capture.py','check_approximate.py','calibration/protocol.json','candidate-v2/manifest.json','inputs/public_gate_bounds.json'])
def test_current_runtime_substitution_fails_against_successful_calibration(copied,name):
    p=copied/name;replace(p,p.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='current runtime source'):gate.verify(REPORT,ARCHIVE,copied,REQ)

@pytest.mark.parametrize('name',['validation_requirement.json','validation_protocol.json','validate_heldout.py','calibration_gate.py'])
def test_protocol_or_wrapper_substitution_rejected_before_inputs(copied,name):
    p=copied/name;replace(p,p.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='Frozen validation'):validation_gate.verify(copied)


def test_exact_input_manifest_inventory_and_original_numeric_aggregate_gates():
    rows=v.load_validation_inputs(REQ['validation_input_sha256'])
    assert len(rows)==56 and sum(r['kind']=='prefill' for r in rows)==48 and sum(r['kind']=='decode' for r in rows)==8
    protocol=json.loads((P/'validation_protocol.json').read_text());g=protocol['gates']
    assert protocol['context_count']==112 and g['aggregate_prefill_top1_matches']==44 and g['teacher_top1_matches']==29 and g['complete_free_sequences']==6
    assert g['operator_block_nrmse']==.05 and g['full_logit_nrmse']==.10 and g['nrmse_floor']==.01


def test_changed_validation_input_rejected_before_tokenizer(tmp_path,monkeypatch):
    p=tmp_path/'inputs.json';p.write_bytes(v.VALIDATION_INPUTS.read_bytes()+b'\n');monkeypatch.setattr(v,'VALIDATION_INPUTS',p)
    with pytest.raises(ValueError,match='input manifest hash'):v.load_validation_inputs(REQ['validation_input_sha256'])


def test_failed_calibration_prevents_any_selection_or_validation_or_asset_read(tmp_path,monkeypatch):
    def rejected(*args):raise ValueError('failed calibration')
    def forbidden(*args):raise AssertionError('must not read later inputs')
    monkeypatch.setattr(v.calibration_gate,'verify',rejected);monkeypatch.setattr(v.candidate_gate,'verify',forbidden);monkeypatch.setattr(v,'load_validation_inputs',forbidden)
    out=tmp_path/'attempt'
    with pytest.raises(ValueError,match='failed calibration'):v.collect(SimpleNamespace(output=out,calibration_report=REPORT,calibration_source_archive=ARCHIVE))
    r=json.loads((out/'report.json').read_text());assert not r['passed'] and not r['validation_data_read']

@pytest.mark.parametrize('primary,cleanup',[(ValueError('failed prerequisite'),KeyboardInterrupt('cleanup cancelled')),(KeyboardInterrupt('primary cancel'),RuntimeError('cleanup failed'))])
def test_cleanup_failure_preserves_primary_and_final_report(tmp_path,monkeypatch,primary,cleanup):
    def fail(*args):raise primary
    def clean():raise cleanup
    monkeypatch.setattr(v.validation_gate,'verify',fail);monkeypatch.setattr(v.gc,'collect',clean)
    out=tmp_path/'attempt'
    with pytest.raises(type(primary)) as caught:v.collect(SimpleNamespace(output=out,calibration_report=REPORT,calibration_source_archive=ARCHIVE))
    assert caught.value is primary
    r=json.loads((out/'report.json').read_text());assert not r['passed'] and any(x.get('phase')=='model_cleanup' for x in r['errors'])
    assert any('Model cleanup failed' in note for note in primary.__notes__)


def test_final_publication_preserves_primary_cancellation(tmp_path,monkeypatch):
    primary=KeyboardInterrupt('primary')
    def fail(*args):raise primary
    monkeypatch.setattr(v.validation_gate,'verify',fail);original=v.write;calls=[]
    def write(path,report):
        calls.append(path)
        if len(calls)>1:raise OSError('publication failed')
        original(path,report)
    monkeypatch.setattr(v,'write',write)
    with pytest.raises(KeyboardInterrupt) as caught:v.collect(SimpleNamespace(output=tmp_path/'attempt',calibration_report=REPORT,calibration_source_archive=ARCHIVE))
    assert caught.value is primary and any('publication failed' in x for x in primary.__notes__)

@pytest.mark.parametrize('invalid',[np.array([1.+1.j]),np.array([1.],dtype=object),np.array(['1']),np.array([True])])
@pytest.mark.parametrize('side',['actual','expected'])
def test_semantic_and_approximate_metric_reject_nonnumeric_or_complex_dtypes(invalid,side):
    x,y=(invalid,np.ones(1)) if side=='actual' else (np.ones(1),invalid)
    for limit in (1e-5,.05,.1):assert v.metric(x,y,limit)['passed'] is False


def test_publication_and_requirement_order_is_fail_closed():
    s=(P/'validate_heldout.py').read_text()
    assert s.index('calibration_gate.verify(args.')<s.index('candidate_gate.verify(P)')<s.index('validation=load_validation_inputs(')<s.index('assets={n:sha(SNAPSHOT/n)')<s.index('model,loading=')
    assert v.metric is v.base.metric and v.decode_trace is v.base.decode_trace
