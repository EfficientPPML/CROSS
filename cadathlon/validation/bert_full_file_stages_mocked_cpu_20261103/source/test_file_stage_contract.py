"""Batch protocol checks only; these tests do not encrypt or run a model."""
import copy
import math
import pytest
import file_stage_contract as protocol
import transport_contract as tc
from client_inputs import ValidatedInput, pack_embeddings
import numpy as np


def fixture(count=2):
    public = dict(context_id='a'*64, model_id='b'*64,
        checkpoint_sha256=tc.CHECKPOINT_SHA256, config_sha256=tc.CONFIG_SHA256,
        candidate_sha256=tc.CANDIDATE_SHA256, profile_sha256=tc.PROFILE_SHA256,
        parameters=tc.parameters())
    record = ValidatedInput(9382, (101,)+(2,)*127, (0,)*128, (1,)*15+(0,)*113)
    metadata = pack_embeddings(record, np.zeros((128,768))).public_metadata()
    requests = [dict(contract=tc.make_contract(public, 'request_'+str(i), metadata),
                     input_digest=f'{i:064x}') for i in range(count)]
    return public, protocol.public_batch(public, requests, count=count)


@pytest.mark.parametrize('count', [1,100,200,408])
def test_full_benchmark_batch_sizes_without_private_leakage(count):
    public, batch = fixture(count)
    assert protocol.validate_public_batch(batch, public=public, count=count) == batch
    targets = [9013+17*i for i in range(count)]
    private = protocol.private_requests(batch, targets, 'c'*64)
    assert protocol.validate_private_requests(private,batch,targets,'c'*64) == private
    text = tc.canonical(batch).decode()
    assert all(token not in text for token in ('target_idx','input_ids','token_type_ids','targets'))
    assert private['targets'] == targets
    assert private['public_batch_sha256'] == tc.digest(batch)


@pytest.mark.parametrize('count', [True,False,0,409,-1,1.,'1',None])
def test_invalid_batch_count(count):
    public,batch = fixture()
    with pytest.raises(ValueError): protocol.public_batch(public,batch['requests'],count=count)


@pytest.mark.parametrize('kind', ['extra','private_id','private_tokens','duplicate','context','model',
    'candidate','position','mask','bad_digest','missing','reorder_then_private','float_index','bad_nonce'])
def test_public_mutations_rejected(kind):
    public,batch = fixture(); original=copy.deepcopy(batch)
    private = protocol.private_requests(original,[101,89],'c'*64)
    row=batch['requests'][0]
    if kind=='extra':batch['extra']=0
    elif kind=='private_id':row['target_idx']=101
    elif kind=='private_tokens':row['input_ids']=[101]*128
    elif kind=='duplicate':batch['requests'][1]=copy.deepcopy(row)
    elif kind in ('context','model'):row['contract']['public'][kind+'_id']='d'*64
    elif kind=='candidate':row['contract']['public']['candidate_sha256']='d'*64
    elif kind=='position':row['contract']['input']['original_positions'][0]=1
    elif kind=='mask':row['contract']['input']['original_attention_mask'][0]=0
    elif kind=='bad_digest':row['input_digest']='D'*64
    elif kind=='missing':batch['requests'].pop()
    elif kind=='float_index':row['contract']['input']['original_positions'][0]=0.
    elif kind=='bad_nonce':row['contract']['request_id']='../elsewhere'
    else:batch['requests'].reverse()
    if kind=='reorder_then_private':
        protocol.validate_public_batch(batch,public=public,count=2)
        with pytest.raises(ValueError):protocol.validate_private_requests(private,batch,[101,89],'c'*64)
    else:
        with pytest.raises(ValueError):protocol.validate_public_batch(batch,public=public,count=2)


@pytest.mark.parametrize('targets', [[1,1],[True,2],[1.,2],[-1,2],[1],[1,2,3],(1,2)])
def test_private_target_validation(targets):
    _,batch=fixture()
    with pytest.raises(ValueError):protocol.private_requests(batch,targets,'c'*64)


@pytest.mark.parametrize('kind', ['targets','input','request','context','model','public_digest','extra'])
def test_private_changed_after_encryption(kind):
    _,batch=fixture();private=protocol.private_requests(batch,[101,89],'c'*64)
    if kind=='targets':private['targets'].reverse()
    elif kind=='input':private['input_file_sha256']='d'*64
    elif kind=='request':private['requests'][0]['input_digest']='d'*64
    elif kind in ('context','model'):private[kind+'_id']='d'*64
    elif kind=='public_digest':private['public_batch_sha256']='d'*64
    else:private['extra']=1
    # Keep the original separately, as the client does on disk. No shared
    #mutable Python object should make this a false-negative mutation test.
    _,original=fixture()
    with pytest.raises(ValueError):protocol.validate_private_requests(private,original,[101,89],'c'*64)


def responses(batch):
    return [dict(request_id=r['contract']['request_id'],input_digest=r['input_digest'],
                 response_manifest_sha256=f'{100+i:064x}',scale=17.)
            for i,r in enumerate(batch['requests'])]


def test_responses_require_original_order_and_independent_scale_check():
    _,batch=fixture();value=protocol.response_batch(batch,responses(batch));seen=[]
    def check(scale):seen.append(scale);return scale==17.
    assert protocol.validate_response_batch(value,batch,check_scale=check)==value
    assert seen==[17.,17.]
    with pytest.raises(ValueError):protocol.validate_response_batch(value,batch,check_scale=lambda x:False)
    with pytest.raises(ValueError):protocol.validate_response_batch(value,batch,check_scale=lambda x:1)


@pytest.mark.parametrize('kind', ['swap','duplicate','input','nonce','digest','extra','missing',
    'batch_digest','zero','negative','nan','infinite','bool','integer','alter_scale'])
def test_response_mutations_rejected(kind):
    _,batch=fixture();value=protocol.response_batch(batch,responses(batch));rows=value['responses']
    if kind=='swap':rows.reverse()
    elif kind=='duplicate':rows[1]=copy.deepcopy(rows[0])
    elif kind=='input':rows[0]['input_digest']='e'*64
    elif kind=='nonce':rows[0]['request_id']='request_other'
    elif kind=='digest':rows[0]['response_manifest_sha256']='z'*64
    elif kind=='extra':rows[0]['he_logits']=[0,1]
    elif kind=='missing':rows.pop()
    elif kind=='batch_digest':value['public_batch_sha256']='e'*64
    else:rows[0]['scale']={'zero':0.,'negative':-1.,'nan':math.nan,'infinite':math.inf,
                          'bool':True,'integer':17,'alter_scale':18.}[kind]
    with pytest.raises(ValueError):protocol.validate_response_batch(value,batch,check_scale=lambda x:x==17.)


def test_exact_raw_input_bytes_bound():
    assert protocol.file_digest(b'line\n') != protocol.file_digest(b'line\r\n')
    with pytest.raises(ValueError):protocol.file_digest('line\n')
