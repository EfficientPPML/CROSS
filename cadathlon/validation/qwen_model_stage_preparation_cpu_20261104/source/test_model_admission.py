"""Synthetic externally pinned bytes only: never a real integrity admission."""
from dataclasses import replace
import json
import pytest
from qwen_file_contract import canonical,sha
from model_admission import Aggregate,ExpectedStoreIdentity,ExpectedModelPins,authenticate_model,require_admitted,create_actual_model_admission
from qwen_public_parameters import open_actual_admitted_parameters
from model_store_validation import create_actual_model_validator


def test_synthetic_positive_chain_reauthenticates_without_execution_authority(public,aggregate,chain):
    b=chain();bound=b.call()
    assert bound.check().report_bytes==b.report_raw
    assert bound.model_id==sha(canonical(aggregate.check()['content']))
    assert bound.stores.raw==canonical(b.report['stores'])
    assert bound.execution_authorized is False
    assert require_admitted(bound,public,consumer_source_manifest_sha256='c'*64)is bound


def test_canonical_compact_source_map_is_required(chain):
    def change(p):p['producer_source_manifest_sha256']=sha((json.dumps(p['producer_source_map'],sort_keys=True,indent=2)+'\n').encode())
    b=chain(change_expected=change)
    with pytest.raises(ValueError,match='canonical expected producer map'):b.call()


def test_content_model_id_does_not_depend_on_producer_or_consumer_sources(chain):
    a=chain().call();b=chain(source_map={'new_entry.py':'e'*64},consumer='f'*64).call()
    assert a.model_id==b.model_id
    assert a.expected_pins.check()['producer_source_manifest_sha256']!=b.expected_pins.check()['producer_source_manifest_sha256']
    assert a.expected_pins.check()['consumer_source_manifest_sha256']!=b.expected_pins.check()['consumer_source_manifest_sha256']


def test_forged_retained_dataclass_and_wrong_consumer_are_rejected(public,chain):
    b=chain();bound=b.call()
    with pytest.raises(ValueError,match='consumer source'):require_admitted(bound,public,consumer_source_manifest_sha256='f'*64)
    forged=replace(bound,report_bytes=canonical(dict(b.report,complete=False)))
    with pytest.raises(ValueError):require_admitted(forged,public,consumer_source_manifest_sha256='c'*64)


@pytest.mark.parametrize('case',['failed','incomplete','error','cleanup','backend','counts','audit','source_end'],ids=['failed','incomplete','error','cleanup','backend','counts','audit','source-end'])
def test_repinned_incomplete_or_infrastructure_report_rejects(chain,case):
    def change(r):
        if case=='failed':r['passed']=False
        elif case=='incomplete':r['complete']=False
        elif case=='error':r['error_type']='OSError'
        elif case=='cleanup':r['checks']['cleanup_completed']=False
        elif case=='backend':r['scope']['backend_context_constructed']=True
        elif case=='counts':r['counts']['physical_matrices']=198
        elif case=='audit':r['audit_denials']=['forbidden']
        else:r['source_hashes_after']['entry.py']='f'*64
    with pytest.raises(ValueError):chain(change_report=change).call()


@pytest.mark.parametrize('case',['exit','reaped','group','report_hash'],ids=['exit','reaped','group','report-hash'])
def test_repinned_outer_must_bind_normal_success_and_retirement(chain,case):
    def change(o):
        if case=='exit':o['exit_code']=70
        elif case=='reaped':o['reaped']=False
        elif case=='group':o['process_group_absent']=False
        else:o['report_sha256']='f'*64
    with pytest.raises(ValueError):chain(change_outer=change).call()


@pytest.mark.parametrize('case',['archive_members','store_hash','execution'],ids=['archive-members','store-hash','execution'])
def test_repinned_root_review_cannot_omit_archive_store_or_scope_links(chain,case):
    def change(r):
        if case=='archive_members':r['source_archive_members_verified']=False
        elif case=='store_hash':r['current_store_binding_sha256']='f'*64
        else:r['actual_server_execution_authorized']=True
    with pytest.raises(ValueError):chain(change_review=change).call()


@pytest.mark.parametrize('case',['model','report_digest'],ids=['model','report-digest'])
def test_external_expected_pins_are_not_replaced_by_report_claims(chain,case):
    def change(p):p['model_id'if case=='model'else'validation_report_sha256']='f'*64
    with pytest.raises(ValueError):chain(change_expected=change).call()


@pytest.mark.parametrize('case',['duplicate','missing','boolean_mode','raw_hash'],ids=['duplicate','missing','boolean-mode','raw-hash'])
def test_external_store_inventory_and_exact_types_reject(aggregate,stores,case):
    value=json.loads(stores.raw)
    if case=='duplicate':value['files'][-1]=value['files'][0]
    elif case=='missing':value['files'].pop()
    elif case=='boolean_mode':value['roots']['qkv']['mode']=True
    else:next(r for r in value['files']if r['family']!='gamma')['raw_sha256']='f'*64
    with pytest.raises(ValueError):ExpectedStoreIdentity(aggregate,canonical(value)).check()


@pytest.mark.parametrize('entry',[create_actual_model_admission,open_actual_admitted_parameters,create_actual_model_validator],ids=['admission','parameters','validator'])
def test_actual_entries_reject_before_argument_access(entry):
    class Bomb:
        def __getattribute__(self,name):raise AssertionError('actual argument touched')
    with pytest.raises(PermissionError):entry(Bomb(),intent=Bomb())
