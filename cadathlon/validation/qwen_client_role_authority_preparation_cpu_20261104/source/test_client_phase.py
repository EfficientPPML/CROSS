"""Pure post-keygen metadata checks; no claim of real private/body ordering."""
import builtins
import json
import os
from pathlib import Path
import pytest
from qwen_file_contract import canonical
from client_role_admission import authenticate_phase,create_actual_client_input,create_actual_client_output
from client_server_terminal import create_actual_server_consumer
from synthetic_client_chain import phase_bundle
from synthetic_keygen_chain import fake_hash

@pytest.mark.parametrize('case',['execute_int','wrong_role','wrong_entry','wrong_conditional','wrong_request','extra','wrong_key_store','dtype','shape_bool','too_short','too_large','scope_model_read'])
def test_repinned_phase_requires_exact_typed_links_and_original_embedding_metadata(client_chain,case):
    def change(row):
        if case=='execute_int':row['execute']=1
        elif case=='wrong_role':row['role']='response_consumer'
        elif case=='wrong_entry':row['entry']='qwen_client_entry.py:consume_response'
        elif case=='wrong_conditional':row['conditional_grant_sha256']=fake_hash('other conditional')
        elif case=='wrong_request':row['request_contract_sha256']=fake_hash('other request')
        elif case=='extra':row['ready']=True
        elif case=='wrong_key_store':row['key_stores']['private']['inode']+=1
        elif case=='dtype':row['input']['dtype']='float32'
        elif case=='shape_bool':row['input']['shape'][0]=True
        elif case=='too_short':row['input']['state']['size']=17*1024*8
        elif case=='too_large':row['input']['state']['size']=17*1024*8+16385
        elif case=='scope_model_read':row['scope']['model_body_reads']=True
    with pytest.raises((ValueError,TypeError)):phase_bundle(client_chain(),'request_producer',change_grant=change).admit()

@pytest.mark.parametrize('case',['same_outputs','nested_output','ancestor_input','key_private','model_root','source_root','input_in_keys','wrong_parent','boolean_inode'])
def test_repinned_phase_rejects_overlapping_paths_and_untyped_parent_metadata(client_chain,case):
    c=client_chain()
    def change(row):
        if case=='same_outputs':row['outputs']['result']=row['outputs']['evidence']
        elif case=='nested_output':row['outputs']['result']['path']='/synthetic-unopened-phase/evidence/result';row['outputs']['result']['parent']['path']='/synthetic-unopened-phase/evidence'
        elif case=='ancestor_input':row['input']['path']='/synthetic-unopened-input/nested/original.npy';row['outputs']['result']['path']='/synthetic-unopened-input/nested';row['outputs']['result']['parent']['path']='/synthetic-unopened-input'
        elif case=='key_private':row['outputs']['result']['path']=row['key_stores']['private']['path']+'/result';row['outputs']['result']['parent']['path']=row['key_stores']['private']['path']
        elif case=='model_root':row['outputs']['result']['path']='/synthetic-unopened-model/qkv/result';row['outputs']['result']['parent']['path']='/synthetic-unopened-model/qkv'
        elif case=='source_root':row['outputs']['result']['path']=c.conditional['request_producer'].grant['source_root']+'/result';row['outputs']['result']['parent']['path']=c.conditional['request_producer'].grant['source_root']
        elif case=='input_in_keys':row['input']['path']=row['key_stores']['public']['path']+'/original.npy'
        elif case=='wrong_parent':row['outputs']['result']['parent']['path']='/other'
        elif case=='boolean_inode':row['outputs']['result']['parent']['inode']=True
    with pytest.raises((ValueError,TypeError),match='fresh output overlaps' if case=='ancestor_input' else None):phase_bundle(c,'request_producer',change_grant=change).admit()

def test_request_phase_rejects_attached_server_admission_and_response_requires_one(client_chain):
    c=client_chain();p=phase_bundle(c,'response_consumer')
    with pytest.raises(TypeError):authenticate_phase(p.conditional,p.ready,p.request_raw,p.raw,expected_pins=p.expected,server=None)
    inp=phase_bundle(c,'request_producer',ready=p.ready,request_raw=p.request_raw)
    with pytest.raises(ValueError):authenticate_phase(inp.conditional,inp.ready,inp.request_raw,inp.raw,expected_pins=inp.expected,server=p.server)

@pytest.mark.parametrize('case',['conditional_raw','conditional_pin','readiness_report','phase_raw','phase_pin','server_envelope'])
def test_retained_object_reauth_rejects_mutation_before_expected_ids(client_chain,case):
    p=phase_bundle(client_chain(),'response_consumer');accepted=p.admit()
    if case=='conditional_raw':object.__setattr__(accepted.conditional,'grant_bytes',b'{}')
    elif case=='conditional_pin':object.__setattr__(accepted.conditional.expected_pins,'raw',b'{}')
    elif case=='readiness_report':object.__setattr__(accepted.readiness,'report_bytes',b'{}')
    elif case=='phase_raw':object.__setattr__(accepted,'grant_bytes',b'{}')
    elif case=='phase_pin':object.__setattr__(accepted.expected_pins,'raw',b'{}')
    elif case=='server_envelope':object.__setattr__(accepted.server,'envelope_bytes',b'{}')
    with pytest.raises((ValueError,TypeError)):accepted.expected_ids()

@pytest.mark.parametrize('which',['request_producer','response_consumer'])
def test_complete_public_chain_rechecks_without_io_rng_or_backend(client_chain,monkeypatch,which):
    p=phase_bundle(client_chain(),which);calls=[]
    def deny(*args,**kwargs):calls.append(True);raise AssertionError('pure client gate attempted I/O/RNG')
    with monkeypatch.context()as patch:
        patch.setattr(builtins,'open',deny)
        for name in ('open','stat','lstat','listdir','scandir','urandom'):patch.setattr(os,name,deny)
        if hasattr(os,'getrandom'):patch.setattr(os,'getrandom',deny)
        patch.setattr(Path,'open',deny)
        admitted=p.admit();admitted.check();ids=admitted.expected_ids()
        assert ids.private_manifest_sha256 is not None
    assert calls==[]

@pytest.mark.parametrize('entry',['input','output','server'])
def test_actual_entry_denies_before_argument_inspection_or_io(monkeypatch,entry):
    class Bomb:
        def __getattribute__(self,name):raise AssertionError('actual argument touched')
    def deny(*a,**kw):raise AssertionError('actual entry attempted I/O/RNG')
    function={'input':create_actual_client_input,'output':create_actual_client_output,'server':create_actual_server_consumer}[entry]
    with monkeypatch.context()as patch:
        patch.setattr(builtins,'open',deny);patch.setattr(os,'open',deny);patch.setattr(os,'urandom',deny)
        with pytest.raises(PermissionError):function(Bomb(),input=Bomb())
