"""Readback ordering and rejection tests with bounded fake ciphertext payloads."""
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib,json,sys
import numpy as np
import pytest
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE/'runtime/stage'),str(HERE/'runtime')]
import file_stages,transport,transport_contract as tc,file_stage_contract as batches
import request_readback


@pytest.fixture
def case(tmp_path,monkeypatch):
    context='a'*64;nonce='f'*32
    paths={n:tmp_path/n for n in ('public_keys','client_private','server_data','preprocessed_input','upload')}
    paths.update(size=0,batch_size=1,model_id=file_stages.MODEL_ID,model_revision=file_stages.REVISION)
    for n in ('public_keys','client_private','server_data','upload'):paths[n].mkdir()
    (paths['server_data']/context).mkdir()
    record=dict(target_idx=1234,input_ids=[101]+[2]*127,token_type_ids=[0]*128,attention_mask=[1]*15+[0]*113)
    raw=tc.canonical(record);paths['preprocessed_input'].write_bytes(raw);input_sha=hashlib.sha256(raw).hexdigest()
    keys=dict(context_id=context,parameters=tc.parameters())
    model=dict(context_id=context,model_id='b'*64,checkpoint_sha256=tc.CHECKPOINT_SHA256,
        config_sha256=tc.CONFIG_SHA256,candidate_sha256=tc.CANDIDATE_SHA256,profile_sha256=tc.PROFILE_SHA256)
    (paths['public_keys']/'public_manifest.json').write_bytes(tc.canonical(keys))
    (paths['server_data']/context/'model_manifest.json').write_bytes(tc.canonical(model))
    prepared=file_stages.client_inputs.pack_embeddings(file_stages.client_inputs.validate_record(record),np.zeros((128,768)))
    public=file_stages._public(keys,model);contract=tc.make_contract(public,nonce,prepared.public_metadata());sample=paths['upload']/nonce;sample.mkdir()
    manifests=[]
    for i in range(8):
        (sample/f'tile{i}').mkdir()
        binding=tc.ciphertext_contract(contract,'request',i)
        manifests.append(transport._manifest_fields(binding)|dict(files={'bundle.json':'c'*64,'ct_0.npz':str(i)*64}))
    batch=batches.public_batch(public,[dict(contract=contract,input_digest=transport.request_digest(manifests,contract))],count=1)
    (paths['upload']/'batch.json').write_bytes(tc.canonical(batch))
    private=paths['client_private']/'requests.json';private.write_bytes(tc.canonical(batches.private_requests(batch,[1234],input_sha)));private.chmod(0o600)
    calls=[]
    def validate(path,*,contract,role,tile_index):
        calls.append(tile_index);binding=tc.ciphertext_contract(contract,role,tile_index)
        return np.zeros((2,3),dtype=np.uint32),binding['scale'],binding,manifests[tile_index]
    monkeypatch.setattr(transport,'validate_ciphertext',validate)
    return NS(paths=paths,context=context,sha=input_sha,calls=calls,private=private,sample=sample,manifests=manifests)


def test_complete_ordered_readback(case):
    result=request_readback.verify(case.paths,context_id=case.context,expected_input_sha256=case.sha)
    assert case.calls==list(range(8)) and result['tile_count']==8 and result['sample_count']==1
    assert result['private_binding_verified'] and result['ciphertext_payload_bytes']==8*24
    assert 'targets' not in result and 'input_ids' not in result


@pytest.mark.parametrize('bad',('input','context','private_target','batch_count','extra_sample','missing_tile'))
def test_invalid_binding_fails_before_ciphertext_reads(case,bad):
    c=case
    if bad=='input':c.paths['preprocessed_input'].write_bytes(c.paths['preprocessed_input'].read_bytes()+b' ')
    elif bad=='context':c.context='e'*64
    elif bad=='private_target':
        d=json.loads(c.private.read_text());d['targets']=[7];c.private.write_bytes(tc.canonical(d))
    elif bad=='batch_count':
        p=c.paths['upload']/'batch.json';d=json.loads(p.read_text());d['sample_count']=2;p.write_bytes(tc.canonical(d))
    elif bad=='extra_sample':(c.paths['upload']/'unexpected').mkdir()
    else:(c.sample/'tile7').rmdir()
    with pytest.raises(ValueError):request_readback.verify(c.paths,context_id=c.context,expected_input_sha256=c.sha)
    assert c.calls==[]


def test_changed_ordered_payload_digest_rejected(case):
    case.manifests[4]['files']['ct_0.npz']='9'*64
    with pytest.raises(ValueError,match='ordered ciphertext'):
        request_readback.verify(case.paths,context_id=case.context,expected_input_sha256=case.sha)


def test_private_permissions_rejected(case):
    case.private.chmod(0o644)
    with pytest.raises(ValueError,match='0600'):
        request_readback.verify(case.paths,context_id=case.context,expected_input_sha256=case.sha)
