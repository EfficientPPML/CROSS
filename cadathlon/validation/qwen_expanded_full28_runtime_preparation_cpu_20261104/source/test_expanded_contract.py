"""Public contracts and synthetic owner metadata only; no learned arrays."""
import ast,hashlib,json,math
from fractions import Fraction as F
from pathlib import Path
import pytest
from plan_full import P,contract
from full_contract import state,verify
from encrypted_qwen_rmsnorm import QwenRMSContract,rms_scale_plan,DELTA
import qwen_candidate_bridge as bridge

def raw(name):return (P/name).read_bytes()
def document(name):return json.loads(raw(name))
def test_exact_historical_artifacts_and_bridge():
    for name in ('inputs/certificates.json','inputs/config.json','inputs/decoder_parameters.json','inputs/actual_key_plan.json','qkv-export-v1/parameter_manifest.json'):
        assert raw(name)==raw('before87/'+name)
    result=bridge.derive_bridge(raw('before87/inputs/candidate_manifest.json'),raw('inputs/candidate_manifest.json'),raw('inputs/config.json'),raw('inputs/certificates.json'),raw('qkv-export-v1/parameter_manifest.json'),raw('inputs/decoder_parameters.json'))
    assert result==document('inputs/candidate_bridge.json')
    assert result['protected_certificate_applicable'] is True
    assert result['normalization_certificate_transferred'] is False
    assert result['calibration_pass_claimed'] is False and result['execution_authorized'] is False

@pytest.mark.parametrize('name',['nonlinear','softmax','exp','initial_reciprocal','reciprocal_by_keys'])
def test_protected_subtrees_match_exactly(name):
    old=bridge.protected(document('before87/inputs/candidate_manifest.json'));new=bridge.protected(document('inputs/candidate_manifest.json'))
    assert bridge.canonical(old[name])==bridge.canonical(new[name])

@pytest.mark.parametrize('which',['candidate','certificate','qkv','decoder'])
def test_bridge_rejects_rewritten_or_unpinned_bytes(which):
    manifest=raw('inputs/candidate_manifest.json');cert=raw('inputs/certificates.json')
    if which=='candidate':
        with pytest.raises(ValueError):bridge.verify_protected_certificate(raw('before87/inputs/candidate_manifest.json'),cert)
    elif which=='certificate':
        data=json.loads(cert);data['manifest_sha256']=bridge.MANIFEST
        with pytest.raises(ValueError):bridge.verify_protected_certificate(manifest,bridge.canonical(data))
    else:
        data=document('qkv-export-v1/parameter_manifest.json' if which=='qkv' else 'inputs/decoder_parameters.json');data['candidate_sha256']=bridge.MANIFEST
        with pytest.raises(ValueError):bridge.verify_parameters(manifest,bridge.canonical(data),which)

def test_new_113_rules_have_fresh_certificates_and_events():
    report=document('inputs/normalization_certificates.json');manifest=document('inputs/candidate_manifest.json')
    meta=document('inputs/site_metadata.json');oldmeta=document('before87/inputs/site_metadata.json')
    assert report['candidate_manifest_sha256']==bridge.MANIFEST and report['site_count']==113
    assert report['changed_rule_count']==90 and report['inverse_refresh_count']==260
    assert report['scientific_calibration_verified'] is False and report['execution_authorized'] is False
    assert set(report['certificates'])==set(meta)
    for name,row in report['certificates'].items():
        rule=manifest['nonlinear'][name];assert row['rule_sha256']==bridge.digest(bridge.canonical(rule))
        proof=row['certificate']
        if rule['method']=='polynomial':
            assert proof['certified'] is True
            relative=F(int(proof['relative']['numerator_hex'],16),int(proof['relative']['denominator_hex'],16))
            assert 0<=relative<=F(1,10**6)
        else:assert proof['verified'] is True
        assert meta[name]['refresh_events']==oldmeta[name]['refresh_events']
        assert meta[name]['gamma_f64le_sha256']==oldmeta[name]['gamma_f64le_sha256']
        assert meta[name]['output_level']==(3 if name=='final.inverse_sqrt' else oldmeta[name]['output_level'])

def test_only_final_raw_bound_changes_and_old_bound_has_in_domain_counterexample():
    old={r['site']:r for r in document('before87/inputs/join_plan.json')['normalizations']}
    new={r['site']:r for r in document('inputs/join_plan.json')['normalizations']}
    assert [n for n in new if new[n]['raw_input_bound']!=old[n]['raw_input_bound']]==['final.inverse_sqrt']
    rule=document('inputs/candidate_manifest.json')['nonlinear']['final.inverse_sqrt'];lo,hi=map(F,rule['domain'])
    variance=F(4096**2,1024)+F(1e-6)
    assert lo<variance<hi and 4096>old['final.inverse_sqrt']['raw_input_bound']
    # Independent exact power-of-two enclosure with the inherited9/8 guard.
    guarded_square=F(1024)*hi*F(9,8)**2
    assert F(4096)**2<guarded_square<=F(8192)**2
    assert new['final.inverse_sqrt']['raw_input_bound']==8192 and new['final.inverse_sqrt']['output_bound']==1024

def test_degree256_plan_and_all38_descriptors():
    c=contract((5,11,23));final=QwenRMSContract(c.manifest_bytes,c.site_bytes,'final.inverse_sqrt')
    assert (final.degree,final.output_level,final.events)==(256,3,())
    profile=next(r for r in document('inputs/profile_audit.json')['profiles'] if r['name']=='depth49_p60_d8')
    for scale in (math.nextafter(DELTA,-math.inf),DELTA,math.nextafter(DELTA,math.inf)):
        plan=rms_scale_plan(profile['q'],final,input_scale=scale)
        assert plan['output_level']==3 and plan['stages'][-1]['level']==3
        assert next(r for r in plan['stages'] if r['name']=='inverse')['level']==5
        assert not any(r['kind']=='refresh' for r in plan['stages'])
    assert c.final_norm_level==3 and c.final_norm_degree==256
    assert len(c.vocabulary_blocks)==38 and sum(d.stop-d.start for d in c.vocabulary_blocks)==151936
    assert all((d.input_level,d.output_level)==(3,2) for d in c.vocabulary_blocks)

@pytest.mark.parametrize('field,value',[('final_norm_level',3.0),('final_norm_level',7),('final_norm_degree',True),('final_norm_degree',64)])
def test_new_contract_fields_typed_and_immutable(field,value):
    c=contract((0,));object.__setattr__(c,field,value)
    with pytest.raises(ValueError):verify(c)

@pytest.mark.parametrize('field,value',[('input_level',3.0),('input_level',7),('output_level',True),('output_level',6)])
def test_vocabulary_level_fields_typed_and_immutable(field,value):
    c=contract((0,));object.__setattr__(c.vocabulary_blocks[0],field,value)
    with pytest.raises(ValueError):verify(c)

def test_generic_inverse_refresh_profile_and_reference_unchanged():
    common='checkout/cadathlon/common/'
    for name in ('encrypted_qwen_rms_refresh.py','encrypted_scale_planning.py','encrypted_polynomial.py','encrypted_qwen_silu.py'):
        assert raw(common+name)==raw('before87/'+common+name)
    before=ast.parse(raw('before87/'+common+'encrypted_qwen_rmsnorm.py'));after=ast.parse(raw(common+'encrypted_qwen_rmsnorm.py'))
    for tree in (before,after):
        tree.body=[n for n in tree.body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SITE_METADATA' for t in n.targets))]
    assert ast.dump(before)==ast.dump(after)
    for name in ('inputs/profile_audit.json','inputs/frozen_arithmetic.py','full_reference.py'):
        assert raw(name)==raw('before87/'+name)

def test_no_scientific_or_execution_authorization():
    value=document('execution_authorization.json')
    assert value['first_pass_freeze_sha256'] is None and value['actual_execution_grant_sha256'] is None
    assert all(value[k]is False for k in ('scientific_calibration_pass_verified','model_forward_enabled','heldout_enabled','row0_enabled','he_enabled'))
