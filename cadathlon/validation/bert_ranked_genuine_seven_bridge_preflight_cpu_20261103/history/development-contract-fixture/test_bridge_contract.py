from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace as NS
import copy,json,sys
import pytest
import bridge_common as common,bridge_protocol as p,bridge_integration as i,bridge_terminal as terminal


def envelope(phase='client_preprocess_input'):
    r={n:None for n in p.parent.REQUEST_FIELDS};r.update(stage=p.STAGES[phase],size=1,seed=42,staged_root='/tmp/fresh-bridge/attempt',attempt_sha256='a'*64,predecessors=[])
    return dict(schema='cross-ranked-genuine-seven-request-v1',request=r,phase=phase,partial=None if phase not in p.REQUIRES else dict(phase=p.REQUIRES[phase],evidence='/tmp/partial',report_sha256='b'*64),authorization=dict(size0_evidence='/tmp/size0',size0_report_sha256='c'*64,storage_path='/tmp/grant',storage_sha256='d'*64))

@pytest.mark.parametrize('phase',common.NAMES)
def test_literal_seven_contract(phase):assert p.validate_envelope(envelope(phase))['request']['stage']==p.STAGES[phase]

@pytest.mark.parametrize('value',[None,False,{},[],1,'x'])
def test_malformed_terminal_returns_preserve_value_error_and_publish(tmp_path,value):
    @contextmanager
    def shield():yield
    path=tmp_path/'report.json'
    with pytest.raises(ValueError,match='terminal report object'):
        terminal.finish(path,lambda:value,common.write,termination=shield,shield=shield)
    r=common.read(path);assert not r['passed'] and not r['termination_scope_completed'] and r['error']['type']=='ValueError'

@pytest.mark.parametrize('phase',list(p.REQUIRES))
def test_split_requires_exact_previous_phase(phase):
    e=envelope(phase);e['partial']['phase']=phase
    with pytest.raises(ValueError):p.validate_envelope(e)

@pytest.mark.parametrize('bad',[3.0,True,False,'3'])
def test_typed_parent_stage(bad):
    e=envelope();e['request']['stage']=bad
    with pytest.raises(ValueError):p.validate_envelope(e)

def test_predecessor_inventory_and_complete_stage_order():
    cfg={'base_request':envelope()['request'],'authorization':envelope()['authorization']}
    proofs=[]
    for n,phase in enumerate(common.NAMES):
        e=i.envelope_for(cfg,{'next_phase':n,'proofs':proofs},phase)
        assert len(e['request']['predecessors'])==max(0,p.STAGES[phase]-3)
        if phase in p.REQUIRES:assert e['partial']['report_sha256']==proofs[-1]['report_sha256']
        if n>0:assert e['request']['key_report_sha256']==proofs[0]['report_sha256']
        if n>1:assert e['request']['model_report_sha256']==proofs[1]['report_sha256']
        proofs.append(dict(phase=phase,evidence=f'/proof/{n}',report_sha256=f'{n+1:064x}'))
    with pytest.raises(ValueError):i.envelope_for(cfg,{'next_phase':7,'proofs':proofs},common.NAMES[-1])

def test_reject_synthetic_runtime_before_authorization(monkeypatch):
    monkeypatch.setenv('CROSS_RANKED_SYNTHETIC_ONLY','1')
    with pytest.raises(ValueError,match='synthetic'):common.reject_test_environment()

def test_parent_all272_still_exact():
    pins=common.read(common.HERE/'bridge_parent_manifest.json')
    assert len(pins)==272
    assert {n:common.digest((common.HERE/n).read_bytes()) for n in pins}==pins

def test_original_all11_harness_sources_preserved():
    pins=common.read(common.HERE/'upstream/manifest.json');assert len(pins)==11
    assert {n:common.digest((common.HERE/'upstream'/n).read_bytes()) for n in pins}==pins

def test_partial_proof_not_accepted_when_preparation_only(tmp_path):
    e=envelope('client_encode_encrypt_input');raw=common.canonical({'passed':True,'execute':False});(tmp_path/'report.json').write_bytes(raw)
    e['partial'].update(evidence=str(tmp_path),report_sha256=common.digest(raw))
    with pytest.raises(ValueError,match='complete source-bound'):p.authenticate_partial(e,{'a.py':'a'*64},{})

def test_partial_proof_sha_checked_before_archive(tmp_path):
    e=envelope('client_postprocess');(tmp_path/'report.json').write_bytes(b'{}');e['partial']['evidence']=str(tmp_path)
    with pytest.raises(ValueError,match='externally pinned'):p.authenticate_partial(e,{'a.py':'a'*64},{})

def test_encode_freshness_accepts_preprocessed_but_not_old_ledger(tmp_path):
    paths={n:tmp_path/n for n in ('preprocessed_input','upload','download','decrypted_results','predictions','server_report','client_private')}
    paths['client_private'].mkdir();paths['preprocessed_input'].write_text('synthetic')
    p.require_fresh(paths,'client_encode_encrypt_input')
    (paths['client_private']/'requests.json').write_text('{}')
    with pytest.raises(FileExistsError):p.require_fresh(paths,'client_encode_encrypt_input')

def test_post_freshness_retains_existing_decoded(tmp_path):
    paths={'predictions':tmp_path/'pred','decrypted_results':tmp_path/'decoded'};paths['decrypted_results'].write_text('synthetic')
    p.require_fresh(paths,'client_postprocess');assert paths['decrypted_results'].read_text()=='synthetic'
    paths['predictions'].write_text('0')
    with pytest.raises(FileExistsError):p.require_fresh(paths,'client_postprocess')
