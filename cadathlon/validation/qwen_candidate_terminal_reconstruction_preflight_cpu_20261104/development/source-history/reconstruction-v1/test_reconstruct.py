"""Independent synthetic journal checks; no saved runtime/data outcomes."""
import copy
import hashlib
import pytest
import verify_records as r
import verify_statistics as s
from synthetic_records import Fixture,manifest,inputs,domain_observations,metric

@pytest.fixture(scope='module')
def passing():return Fixture()

def run(f,reader=None,**kwargs):
    return r.reconstruct(f.calibration,f.raw,kwargs.get('ledger',f.ledger),reader or f.read,kwargs.get('worker',f.worker),kwargs.get('inventory',f.inventory))

def test_full3200_independent_reconstruction(passing):
    out=run(passing)
    assert out['scientific_pass']and not out['complete_scientific_failure_eligible']
    assert out['private_files_checked']==9601 and out['coverage']['native_forward_completed']==3232
    assert len(passing.reads)==9600 and out['coverage']['candidate_complete']==3200
    assert out['gates']['whole_free_sequences']==16 and not out['fresh_array_recomputation']

def test_complete_scientific_failed_attempt_keeps_teacher_and_own_free():
    # One numeric-failed ownfree still appends its actual differing argmax.
    # A candidate-only teacher stop continues from native; a separate free stop
    # retains all three causal suffix entries without fabricated journal files.
    f=Fixture(numeric_failed=(3136,),different_choice=(3136,),stopped=(3072,3140))
    out=run(f)
    assert not out['scientific_pass']and out['complete_scientific_failure_eligible']
    c=out['coverage'];assert(c['measured_contexts'],c['candidate_invalid'],c['unmeasured_free_suffixes'],c['native_forward_completed'])==(3197,2,3,3229)
    assert f.records[3137]['prefix'][-1]==8 and f.records[3073]['prefix'][-1]==7
    assert not any(x.startswith(('3141-','3142-','3143-'))for x in f.reads)
    assert out['gates']['whole_free_sequences']==14

@pytest.mark.parametrize('bad',['unknown','comparison_invalid','unmeasured_suffix'])
def test_forged_ledger_status_fails_before_journal(passing,bad):
    ledger=copy.deepcopy(passing.ledger);ledger['records'][0]['status']=bad
    with pytest.raises(ValueError):run(passing,lambda n:pytest.fail('body read'),ledger=ledger)

def test_exact_private_inventory_before_any_phase(passing):
    for names in(passing.inventory-{'0000-native.json'},passing.inventory|{'started.json'}):
        with pytest.raises(ValueError):run(passing,lambda n:pytest.fail('body read'),inventory=names)

@pytest.mark.parametrize('field,value',[('sequence_length',True),('original_prefill_member',1),('token_ids',[True]),('id','')])
def test_calibration_scalar_aliases_rejected_before_body(passing,field,value):
    data=copy.deepcopy(passing.calibration);data['prefills'][0][field]=value
    with pytest.raises(ValueError):r.reconstruct(data,passing.raw,passing.ledger,lambda n:pytest.fail('body read'),passing.worker,passing.inventory)

@pytest.mark.parametrize('phase,key,value',[('native','native_top1',True),('native','operator_count',420),('native','context_sha256','a'*64),
    ('exact','exact_top1',8),('candidate','numeric_pass',1),('candidate','all_logits',151935)])
def test_journal_header_choices_counts_and_typed_gates(passing,phase,key,value):
    def reader(name):
        row=copy.deepcopy(passing.read(name))
        if name==f'0000-{phase}.json':row[key]=value
        return row
    with pytest.raises(ValueError):run(passing,reader)

@pytest.mark.parametrize('mutation',['missing_norm','wrong_head_count','weighted_namespace','wrong_flat','missing_k','k1_nonzero','bool_bypass','bad_hash'])
def test_complete_input_statistics_and_perk_inventory(passing,mutation):
    row=copy.deepcopy(passing.read('0000-native.json'));obs=row['observations']
    if mutation=='missing_norm':del obs['ranges']['0.q.inverse_sqrt']
    elif mutation=='wrong_head_count':obs['ranges']['0.q.inverse_sqrt']['count']=8
    elif mutation=='weighted_namespace':obs['ranges']['0.q.inverse_sqrt.exact_output']=obs['ranges'].pop('0.q.inverse_sqrt')
    elif mutation=='wrong_flat':obs['ranges']['0.softmax.centered']['nonpositive']=0
    elif mutation=='missing_k':del obs['score_rows']['0']['1']
    elif mutation=='k1_nonzero':obs['score_rows']['0']['1']['centered']['minimum']=.1
    elif mutation=='bool_bypass':obs['score_rows']['0']['1']['k1_bypass']=1
    if mutation!='bad_hash':row['observations_sha256']=hashlib.sha256(s.canonical(obs)).hexdigest()
    else:row['observations_sha256']='a'*64
    with pytest.raises(ValueError):r.native_journal(row,r.entries()[0],tuple(row['token_ids']),passing.pin,s.InputAggregate())

@pytest.mark.parametrize('mutation',['missing_operator','extra_operator','wrong_block','wrong_limit','wrong_domain','missing_inverse','dummy_normalization','early_failed_domain','claimed_complete_failure'])
def test_candidate_all_metric_and_domain_records(passing,mutation):
    row=copy.deepcopy(passing.read('0001-candidate.json'));obs=row['observations']
    if mutation=='missing_operator':del row['operator_metrics']['0.q_linear']
    elif mutation=='extra_operator':row['operator_metrics']['unknown']=row['operator_metrics']['0.q_linear']
    elif mutation=='wrong_block':row['block_metrics'][0]['count']-=1
    elif mutation=='wrong_limit':row['full_logit_metric']['limit']=.11
    elif mutation=='wrong_domain':obs['domains']['0.input.inverse_sqrt']['domain'][1]+=1
    elif mutation=='missing_inverse':del obs['domains']['0.attention.query1.initial_inverse']
    elif mutation=='dummy_normalization':obs['normalization_variance']={}
    elif mutation=='early_failed_domain':obs['domains']['0.input.inverse_sqrt']['above']=1
    else:row['failure']=dict(type='InvalidCircuit',label='x',record={'nonfinite':1})
    row['observations_sha256']=hashlib.sha256(s.canonical(obs)).hexdigest()
    with pytest.raises(ValueError):r.candidate_journal(row,r.entries()[1],tuple(row['token_ids']),passing.pin,passing.plans[2],{})

def test_json_sorted_mapping_is_accepted(passing):
    row=r.parse(s.canonical(passing.read('0001-candidate.json')))
    assert list(row['operator_metrics'])!=list(r.operator_counts(2))
    c=r.candidate_journal(row,r.entries()[1],tuple(row['token_ids']),passing.pin,passing.plans[2],{})
    assert c['complete']

@pytest.mark.parametrize('bad',[b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":1e999}'])
def test_strict_json(bad):
    with pytest.raises(ValueError):r.parse(bad)

def test_token_hash_is_signed_little_endian():
    assert s.context_hash((1,256))==hashlib.sha256(b'\x01'+b'\x00'*7+b'\x00\x01'+b'\x00'*6).hexdigest()

def test_partial_domain_stop_is_only_last_observed_event(passing):
    plan=passing.plans[1];obs=domain_observations(plan,True);name=plan[0][0]
    failure=dict(type='InvalidCircuit',label=name,record=obs['domains'][name])
    assert s.verify_domains(obs,plan,False,failure)
    failure['label']='unrelated'
    with pytest.raises(ValueError):s.verify_domains(obs,plan,False,failure)

def test_nonfinite_internal_stop_needs_positive_count(passing):
    plan=passing.plans[1];obs=domain_observations(plan)
    for kind,record in [('InvalidCircuit',{'nonfinite':1}),('NonfiniteOutput',None)]:
        failure=dict(type=kind,label='internal'if kind=='InvalidCircuit'else None,record=record)
        s.verify_domains(obs,plan,False,failure)
    with pytest.raises(ValueError):s.verify_domains(obs,plan,False,dict(type='InvalidCircuit',label='internal',record={'nonfinite':0}))
