"""No payloads: fixed source identities, typed readiness and original54 gates."""
from pathlib import Path
import copy,json
import pytest
import validator_common as c


def test_real_consumer_and_producer_closures_are_distinct():
    expected=c.protocol.hashes(c.protocol.sources())
    profile,plan,checked=c.inspect_sources(expected)
    assert len(checked['producer_sources'])==163 and len(expected)>163
    assert {n for n,h in checked['producer_sources'].items()if expected[n]!=h}==set(c.RUNTIME_OVERLAYS)|c.METADATA_OVERLAYS
    assert profile and plan


@pytest.mark.parametrize('name',['component_protocol.py','component_worker.py','component_transport.py','component_contract.py','gain4_fixtures.py','common/encrypted_bert_normalization_gain4.py'])
def test_arithmetic_gate_or_format_drift_rejected(name,monkeypatch):
    expected=c.protocol.hashes(c.protocol.sources());expected[name]='00'*32
    monkeypatch.setattr(c.protocol,'hashes',lambda raw:expected)
    monkeypatch.setattr(c.protocol,'check_sources',lambda *a:pytest.fail('drift must fail before runtime checks'))
    with pytest.raises(ValueError,match='overlays'):c.inspect_sources(expected)


@pytest.mark.parametrize('name',list(c.RUNTIME_OVERLAYS))
def test_wrong_io_overlay_rejected(name,monkeypatch):
    expected=c.protocol.hashes(c.protocol.sources());expected[name]='00'*32
    monkeypatch.setattr(c.protocol,'hashes',lambda raw:expected)
    with pytest.raises(ValueError,match='IO overlay'):c.inspect_sources(expected)


@pytest.mark.parametrize('field,value',[('ready',1),('ready',1.0),('ready','true'),('stage','fixture'),('stage','server'),('execution_authorized',1),('execution_authorized',False)])
def test_typed_validator_authorization(tmp_path,monkeypatch,field,value):
    row=dict(schema='gain4-validator-only-authorization-v1',ready=True,stage='validate',execution_authorized=True);row[field]=value
    (tmp_path/'validator_authorization.json').write_text(json.dumps(row));monkeypatch.setattr(c,'HERE',tmp_path)
    with pytest.raises(ValueError,match='disabled'):c.require_ready()


def test_true_readiness_still_requires_bound_server(tmp_path,monkeypatch):
    (tmp_path/'validator_authorization.json').write_text(json.dumps(dict(schema='gain4-validator-only-authorization-v1',ready=True,stage='validate',execution_authorized=True)))
    monkeypatch.setattr(c,'HERE',tmp_path)
    with pytest.raises(ValueError,match='pending'):c.require_ready()


def numeric():
    proof=c.contract.numerical_plan();rows=[];derived={}
    for i,row in enumerate(proof['checkpoints']):rows.append(dict(index=i,case=row['case'],label=row['label'],slots=65536,level=row['level'],scale=row['scale'],absolute_gate=1e-6,max_abs_error=0.,passed=True))
    for case in proof['cases']:
        names={f'{p}{i:02}'for p in ('invariant','true_residual_vs_original')for i in range(1,case['updates']+1)}|{'final_true_inverse','final_true_relative_residual','final_true_normalized_quotient'}
        derived[case['name']]={n:dict(max_abs_error=0.,passed=True)for n in names}
    return dict(numeric_validation_complete=True,all54_authenticated_before_private_read=True,numeric_checkpoints=rows,derived_checks=derived,
        numeric_summary=dict(checkpoints=54,physical_slots=54*65536,maximum_absolute_error=0.,absolute_gate=1e-6,passed=True))


def test_all54_and_both_original_derived_gate_sets():assert c.validate_numeric_report(numeric())


@pytest.mark.parametrize('edit',['missing','reordered','bool_index','float_level','scale','gate','large','nan','false','derived_missing','derived_false','summary','private_order'])
def test_numeric_or_typed_metadata_cannot_be_weakened(edit):
    r=numeric();row=r['numeric_checkpoints'][0]
    if edit=='missing':r['numeric_checkpoints'].pop()
    elif edit=='reordered':r['numeric_checkpoints'].reverse()
    elif edit=='bool_index':row['index']=False
    elif edit=='float_level':row['level']=float(row['level'])
    elif edit=='scale':row['scale']*=2
    elif edit=='gate':row['absolute_gate']=1e-5
    elif edit=='large':row['max_abs_error']=1.01e-6
    elif edit=='nan':row['max_abs_error']=float('nan')
    elif edit=='false':row['passed']=False
    elif edit=='derived_missing':next(iter(r['derived_checks'].values())).popitem()
    elif edit=='derived_false':next(iter(next(iter(r['derived_checks'].values())).values()))['passed']=False
    elif edit=='summary':r['numeric_summary']['checkpoints']=54.
    else:r['all54_authenticated_before_private_read']=False
    with pytest.raises(ValueError):c.validate_numeric_report(r)
