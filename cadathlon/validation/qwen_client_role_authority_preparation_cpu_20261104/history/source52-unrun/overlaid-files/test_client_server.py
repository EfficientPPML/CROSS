"""Independent server envelope uses its own source map and normal-zero outer."""
import json
import pytest
from qwen_file_contract import sha,canonical
from synthetic_client_chain import server_bundle,phase_bundle
from synthetic_keygen_chain import fake_hash


def test_distinct_server_source_preserves_inner_request_provenance_and_all38_blocks(client_chain):
    c=client_chain();ready=c.key.ready();request=c.request(ready);bundle=server_bundle(c,ready,request)
    admitted=bundle.admit();phase=phase_bundle(c,'response_consumer',ready=ready,request_raw=request,server=admitted).admit()
    request_source=json.loads(request)['public']['producer_source_manifest_sha256']
    assert bundle.inner['producer_source_manifest_sha256']==request_source
    assert bundle.envelope['server_source_manifest_sha256']!=request_source
    rows=json.loads(admitted.response_bytes)['artifacts'];assert len(rows)==38
    assert [row['binding']['index']for row in rows]==list(range(38))
    assert all(row['binding']['level']==2 and row['binding']['logical']['physical_row']==0 for row in rows)
    assert rows[-1]['binding']['logical']==dict(index=37,start=151552,stop=151936,physical_row=0,level=2)
    assert phase.execution_authorized is admitted.execution_authorized is False

@pytest.mark.parametrize('case',['missing_last','duplicate_index','wrong_level','wrong_last_stop','boolean_row','wrong_context','wrong_input_digest','nonfinite_scale'])
def test_repinned_all38_response_metadata_rejects_shape_order_identity_errors(client_chain,case):
    c=client_chain();ready=c.key.ready();request=c.request(ready)
    def change(row):
        if case=='missing_last':row['artifacts'].pop()
        elif case=='duplicate_index':row['artifacts'][1]=row['artifacts'][0]
        elif case=='wrong_level':row['artifacts'][37]['binding']['level']=3
        elif case=='wrong_last_stop':row['artifacts'][37]['binding']['logical']['stop']=155648
        elif case=='boolean_row':row['artifacts'][0]['binding']['logical']['physical_row']=False
        elif case=='wrong_context':row['artifacts'][0]['binding']['context_id']=fake_hash('other context')
        elif case=='wrong_input_digest':row['input_digest']=fake_hash('other input')
        elif case=='nonfinite_scale':row['artifacts'][0]['binding']['scale']='nan'
    with pytest.raises((ValueError,TypeError)):server_bundle(c,ready,request,change_response=change).admit()

@pytest.mark.parametrize('case',['inner_source_as_server','inner_count','inner_exit','inner_boolean_pid','inner_unreaped'])
def test_repinned_inner_file_proof_cannot_be_reinterpreted_as_actual_server_source(client_chain,case):
    c=client_chain();ready=c.key.ready();request=c.request(ready);changes={}
    if case=='inner_source_as_server':changes['change_inner']=lambda r:r.__setitem__('producer_source_manifest_sha256',fake_hash('actual server map'))
    elif case=='inner_count':changes['change_inner']=lambda r:r.__setitem__('output_count',37)
    elif case=='inner_exit':changes['change_inner_outer']=lambda r:r.__setitem__('exit_code',1)
    elif case=='inner_boolean_pid':changes['change_inner_outer']=lambda r:r.__setitem__('server_pid',True)
    elif case=='inner_unreaped':changes['change_inner_outer']=lambda r:r.__setitem__('reaped',False)
    with pytest.raises((ValueError,TypeError)):server_bundle(c,ready,request,**changes).admit()

@pytest.mark.parametrize('case',['incomplete','audit_denial','changed_source','cleanup_false','private_read','scientific_claim','extra','bool_count'])
def test_repinned_server_envelope_cannot_hide_source_cleanup_or_scope_failures(client_chain,case):
    c=client_chain();ready=c.key.ready();request=c.request(ready)
    def change(row):
        if case=='incomplete':row['complete']=False
        elif case=='audit_denial':row['audit_denials']=['SYNTHETIC forbidden operation']
        elif case=='changed_source':row['source_hashes_after']['synthetic_actual_server.py']=fake_hash('changed source')
        elif case=='cleanup_false':row['checks']['original_owners_retired']=False
        elif case=='private_read':row['scope']['private_key_read']=True
        elif case=='scientific_claim':row['scope']['scientific_acceptance_claimed']=True
        elif case=='extra':row['allgood']=True
        elif case=='bool_count':row['process']['pid']=True
    with pytest.raises((ValueError,TypeError)):server_bundle(c,ready,request,change_envelope=change).admit()

@pytest.mark.parametrize('case',['exit1','exit70','not_normal','not_reaped','group_live','pid_changed'])
def test_repinned_server_outer_requires_same_process_normal_zero_and_retirement(client_chain,case):
    c=client_chain();ready=c.key.ready();request=c.request(ready)
    def change(row):
        if case=='exit1':row['exit_code']=1
        elif case=='exit70':row['exit_code']=70
        elif case=='not_normal':row['normal_return']=False
        elif case=='not_reaped':row['reaped']=False
        elif case=='group_live':row['process_group_absent']=False
        elif case=='pid_changed':row['pid']+=1
    with pytest.raises((ValueError,TypeError)):server_bundle(c,ready,request,change_outer=change).admit()

@pytest.mark.parametrize('case',['archive','current_response','scientific_authorization','enables_client','other_model_proof'])
def test_repinned_root_admission_cannot_weaken_independent_server_obligations(client_chain,case):
    c=client_chain();ready=c.key.ready();request=c.request(ready)
    def change(row):
        if case=='archive':row['source_archive_members_verified']=False
        elif case=='current_response':row['current_response_store_verified']=False
        elif case=='scientific_authorization':row['scientific_authorization_verified']=False
        elif case=='enables_client':row['client_execution_authorized']=True
        elif case=='other_model_proof':row['model_evidence']['report_sha256']=fake_hash('other report')
    with pytest.raises((ValueError,TypeError)):server_bundle(c,ready,request,change_root=change).admit()
