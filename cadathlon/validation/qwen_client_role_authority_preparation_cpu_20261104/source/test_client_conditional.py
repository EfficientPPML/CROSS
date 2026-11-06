"""Pre-keygen conditional metadata is necessary but never immediate body authority."""
import json
import pytest
from qwen_file_contract import canonical,sha
from client_role_admission import ExpectedRolePins,authenticate_phase,ExpectedPhasePins
from client_role_source_pins import FIXED_CLIENT_SOURCES,ACTUAL_CLIENT_MODULES
from synthetic_keygen_chain import clone,fake_hash
from synthetic_client_chain import phase_bundle

@pytest.mark.parametrize('same_maps',[False,True])
@pytest.mark.parametrize('which',['request_producer','response_consumer'])
def test_complete_conditional_and_phase_metadata_stay_unexecuting(client_chain,same_maps,which):
    c=client_chain(same_maps=same_maps);conditional=c.admit(which)
    result=phase_bundle(c,which).admit()
    assert conditional.execution_authorized is result.execution_authorized is False
    assert result.expected_ids().private_manifest_sha256==result.readiness.expected_pins.check()['private_manifest_sha256']
    assert (c.maps['request_producer']==c.maps['response_consumer'])is same_maps
    assert conditional.model.expected_pins.check()['consumer_source_manifest_sha256']==sha(canonical(c.maps[which]))
    generated=result.readiness.launch.expected_pins.check()['model_evidence']
    client=conditional.expected_pins.check()['model_evidence']
    assert generated['expected_pins_sha256']!=client['expected_pins_sha256']
    assert {k:v for k,v in generated.items()if k!='expected_pins_sha256'}=={k:v for k,v in client.items()if k!='expected_pins_sha256'}

def test_exact93_existing_prerequisites_all_individually_reject_repinned_substitution(client_chain):
    c=client_chain();base=c.conditional['request_producer'].pins
    assert len(FIXED_CLIENT_SOURCES)==93
    for name in FIXED_CLIENT_SOURCES:
        row=clone(base);row['source_map'][name]=fake_hash('changed '+name);row['source_manifest_sha256']=sha(canonical(row['source_map']))
        with pytest.raises(ValueError,match='fixed client dependency'):ExpectedRolePins(canonical(row)).check()

def test_all14_actual_module_names_are_required_without_claiming_they_exist(client_chain):
    base=client_chain().conditional['request_producer'].pins
    assert len(ACTUAL_CLIENT_MODULES)==14
    for name in ACTUAL_CLIENT_MODULES:
        row=clone(base);row['source_map'].pop(name);row['source_manifest_sha256']=sha(canonical(row['source_map']))
        with pytest.raises(ValueError):ExpectedRolePins(canonical(row)).check()

@pytest.mark.parametrize('case',['extra_field','unknown_role','numeric_hash','bad_relative_name','wrong_map_digest','policy_unbound'])
def test_closed_typed_role_pins_reject_semantically_repinned_values(client_chain,case):
    row=clone(client_chain().conditional['request_producer'].pins)
    if case=='extra_field':row['execute']=True
    elif case=='unknown_role':row['role']='server'
    elif case=='numeric_hash':row['readiness_review_sha256']=True
    elif case=='bad_relative_name':row['source_map']['../escape.py']=fake_hash('escape');row['source_manifest_sha256']=sha(canonical(row['source_map']))
    elif case=='wrong_map_digest':row['source_manifest_sha256']=fake_hash('different map')
    elif case=='policy_unbound':row['policies']['role_policy_sha256']=fake_hash('policy')
    with pytest.raises((ValueError,TypeError)):ExpectedRolePins(canonical(row)).check()

@pytest.mark.parametrize('case',['phase_not_required','readiness_not_required','body_entry_true','scope_extra','wrong_entry','wrong_generation_intent','wrong_generation_map','wrong_model','relative_source'])
def test_repinned_conditional_cannot_remove_future_gates_or_change_links(client_chain,case):
    def mutate(which,row):
        if case=='phase_not_required':row['conditions']['phase_instance_grant_required']=False
        elif case=='readiness_not_required':row['conditions']['generated_key_readiness_required']=False
        elif case=='body_entry_true':row['scope']['body_entry_by_this_document_alone']=True
        elif case=='scope_extra':row['scope']['execute']=True
        elif case=='wrong_entry':row['entry']='qwen_client_entry.py:wrong'
        elif case=='wrong_generation_intent':row['generation_intent_sha256']=fake_hash('other intent')
        elif case=='wrong_generation_map':row['generation_source_manifest_sha256']=fake_hash('other generation')
        elif case=='wrong_model':row['model_id']=fake_hash('other model')
        elif case=='relative_source':row['source_root']='relative/source'
    with pytest.raises((ValueError,TypeError)):client_chain(change_conditional=mutate).admit('request_producer')

def test_client_model_admission_must_be_rebound_to_exact_actual_consumer(client_chain):
    with pytest.raises(ValueError):client_chain(client_model_consumer=fake_hash('unrelated consumer28')).admit('request_producer')

def test_phase_requires_same_original_model_proof_even_when_both_chains_independently_valid(client_chain):
    # Extent/timestamp metadata changes alter an otherwise coherent model report,
    # outer and root chain. Fresh hashes alone cannot replace the original proof.
    def mutate(which,row):row['stores']['files'][0]['state']['mtime_ns']+=1
    c=client_chain(change_client_model=mutate)
    c.admit('request_producer')
    with pytest.raises(ValueError,match='model body proof'):phase_bundle(c,'request_producer').admit()

@pytest.mark.parametrize('case',['archive','grant','review'])
def test_generated_readiness_role_evidence_must_bind_conditional_document(client_chain,case):
    fields={'archive':'source_archive_sha256','grant':'execution_grant_sha256','review':'readiness_review_sha256'}
    def mutate(rows):rows['request_producer'][fields[case]]=fake_hash('different role '+case)
    c=client_chain(change_generation_roles=mutate)
    with pytest.raises(ValueError,match='actual client role'):phase_bundle(c,'request_producer').admit()

def test_conditional_object_cannot_replace_independent_later_phase_grant(client_chain):
    c=client_chain();conditional=c.admit('request_producer');ready=c.key.ready();request=c.request(ready)
    with pytest.raises(TypeError):authenticate_phase(conditional,ready,request,conditional.grant_bytes,expected_pins=conditional.expected_pins)

@pytest.mark.parametrize('case',['compact','duplicate_key'])
def test_role_pin_bytes_must_be_canonical_and_duplicate_free(client_chain,case):
    row=client_chain().conditional['request_producer'].pins
    raw=json.dumps(row,sort_keys=True,separators=(',',':')).encode()
    if case=='duplicate_key':raw=b'{"schema":"qwen-client-role-expected-pins-v1",'+raw[1:]
    with pytest.raises((ValueError,TypeError)):ExpectedRolePins(raw).check()
