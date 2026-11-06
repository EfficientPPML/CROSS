"""Mocked five-handler integration. No real keygen, ciphertext or model claim."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS
import json
import sys
import numpy as np
import pytest
import file_stages as stages
import transport
import transport_contract as tc
import full_model_engine


@pytest.fixture
def flow(tmp_path,monkeypatch):
    paths=dict(size=0,batch_size=1,model_id=stages.MODEL_ID,model_revision=stages.REVISION)
    for name in ('public_keys','client_private','server_data','preprocessed_input','upload','download','decrypted_results','server_report'):
        paths[name]=tmp_path/name
    paths['preprocessed_input'].write_text(json.dumps(dict(target_idx=9017,input_ids=[101]+[2]*127,
        token_type_ids=[0]*128,attention_mask=[1]*15+[0]*113))+'\n')
    keys=dict(context_id='a'*64,parameters=tc.parameters())
    model=dict(context_id='a'*64,model_id='b'*64,checkpoint_sha256=tc.CHECKPOINT_SHA256,
        config_sha256=tc.CONFIG_SHA256,candidate_sha256=tc.CANDIDATE_SHA256,profile_sha256=tc.PROFILE_SHA256)
    state=NS(events=[],values={},failure=None,keys=keys,model=model)
    def keygen(public,private,**kw):
        state.events.append('keygen');Path(public).mkdir();Path(private).mkdir()
        (Path(private)/'secret.npy').write_bytes(b'fake private sentinel')
        return dict(public_manifest=keys,preflight=dict(mocked=True))
    monkeypatch.setattr(stages.client_keys,'generate',keygen)
    monkeypatch.setattr(stages.client_keys,'authenticate_public',lambda *a,**kw:keys)
    def preprocess(snapshot,path,**kw):
        state.events.append('model');Path(path).mkdir();return model
    monkeypatch.setattr(stages.server_public_model,'preprocess',preprocess)
    monkeypatch.setattr(stages.server_public_model,'load',lambda *a,**kw:({'mocked_weights':True},model))
    monkeypatch.setattr(stages,'_snapshot',lambda:tmp_path/'snapshot')
    def value(data):return NS(data=np.asarray(data),_ckks_nsd=1,_ckks_scale=keys['parameters']['scaling_factor'],
                             polynomial=NS(block_until_ready=lambda:state.events.append('sync')))
    class Client:
        def encrypt_slots(self,data):state.events.append('encrypt');return value(data)
        def decrypt_slots(self,ct):state.events.append('decrypt');return ct.data
    @contextmanager
    def client(*args,**kwargs):
        state.events.append('client_open')
        try:yield NS(ctx=Client(),public_manifest=keys,private_manifest={})
        finally:state.events.append('client_close')
    @contextmanager
    def server(*args,**kwargs):
        assert all('client_private' not in str(arg) for arg in args)
        state.events.append('server_open')
        try:yield NS(ctx=object(),public_manifest=keys)
        finally:state.events.append('server_close')
    monkeypatch.setitem(sys.modules,'key_contexts',NS(load_client=client,load_server=server))
    monkeypatch.setitem(sys.modules,'support',NS(require_secure_execution=lambda ctx,ct:state.events.append('security')))
    monkeypatch.setitem(sys.modules,'response_scale',NS(accepts_scale=lambda scale,**kw:type(scale)is float and scale==17.))
    class Embeddings:
        def __init__(self,snapshot):state.events.append('embeddings_only')
        def prepare(self,record):
            return stages.client_inputs.pack_embeddings(record,np.ones((128,768)))
    monkeypatch.setattr(stages.client_inputs,'ClientEmbeddings',Embeddings)
    def save(ct,path,*,contract,role,tile_index,expected_scale=None,input_digest=None):
        if state.failure=='save_request' and role=='request':raise KeyboardInterrupt('upload interrupted')
        binding=tc.ciphertext_contract(contract,role,tile_index,expected_scale,input_digest)
        m=transport._manifest_fields(binding)|dict(files={'bundle.json':'c'*64,'ct_0.npz':'d'*64})
        path=Path(path);path.mkdir()
        for name in ('bundle.json','ct_0.npz'):(path/name).write_bytes(b'test placeholder')
        (path/'transport.json').write_bytes(tc.canonical(m));state.values[str(path)]=(ct,m)
        return m
    def load(path,**kwargs):
        ct,m=state.values[str(path)]
        binding=tc.ciphertext_contract(kwargs['contract'],kwargs['role'],kwargs['tile_index'],kwargs.get('expected_scale'),kwargs.get('input_digest'))
        transport._validate_manifest(m,binding)
        return ct,m
    monkeypatch.setattr(stages,'save_ciphertext',save);monkeypatch.setattr(stages,'load_ciphertext',load)
    class Graph:
        def evaluate(self,tiles,weights,**kwargs):
            state.events.append('full_owner_evaluate')
            if state.failure=='evaluate':raise KeyboardInterrupt('server interrupted')
            assert len(tiles)==8 and kwargs['original_positions']==list(range(15))
            assert all(tile.data.shape==(65536,) for tile in tiles)
            data=np.zeros(65536);data[:2]=[.25,.75];result=value(data);result._ckks_scale=17.
            return NS(output=result)
    @contextmanager
    def graph(*a,**kw):
        state.events.append('graph_open')
        try:yield Graph()
        finally:state.events.append('graph_close')
    monkeypatch.setattr(full_model_engine,'server_engine',graph)
    return paths,state


def server_paths(paths):
    return {k:v for k,v in paths.items() if k not in ('client_private','preprocessed_input','decrypted_results')}


def initialize(paths):
    stages.client_key_generation(paths);stages.server_preprocess_model(server_paths(paths))


def complete(paths):
    stages.client_encode_encrypt_input(paths)
    stages.server_encrypted_compute(server_paths(paths))
    stages.client_decrypt_decode(paths)


def test_all_five_handlers_and_upstream_repeated_run(flow):
    paths,state=flow;initialize(paths);complete(paths)
    first=json.loads((paths['upload']/'batch.json').read_text())
    assert json.loads(paths['decrypted_results'].read_text())==dict(target_idx=9017,he_logits=[.25,.75])
    assert '9017' not in (paths['upload']/'batch.json').read_text()
    # The upstream harness keeps all output paths for the next iteration.
    complete(paths)
    second=json.loads((paths['upload']/'batch.json').read_text())
    assert first['requests'][0]['contract']['request_id']!=second['requests'][0]['contract']['request_id']
    assert state.events.count('keygen')==1 and state.events.count('full_owner_evaluate')==2
    assert state.events.count('encrypt')==16 and state.events.count('decrypt')==2
    assert state.events.count('client_open')==state.events.count('client_close')==4
    assert state.events.count('server_open')==state.events.count('server_close')==2
    assert (paths['client_private']/'secret.npy').read_bytes()==b'fake private sentinel'


@pytest.mark.parametrize('kind',['size','count','model','revision','server_private'])
def test_harness_contract_rejected_before_side_effects(flow,kind):
    paths,state=flow
    if kind=='size':paths['size']=True
    elif kind=='count':paths['batch_size']=2
    elif kind=='model':paths['model_id']='other'
    elif kind=='revision':paths['model_revision']='other'
    with pytest.raises(ValueError):
        if kind=='server_private':stages.server_preprocess_model(paths)
        else:stages.client_key_generation(paths)
    assert not state.events


@pytest.mark.parametrize('failure',['save_request','evaluate'])
def test_cancellation_closes_context_and_removes_only_incomplete_output(flow,failure):
    paths,state=flow;initialize(paths);state.failure=failure
    with pytest.raises(KeyboardInterrupt):
        stages.client_encode_encrypt_input(paths)
        stages.server_encrypted_compute(server_paths(paths))
    assert not paths['download'].exists() and not (paths['client_private']/'completed.json').exists()
    if failure=='save_request':
        assert not paths['upload'].exists() and not (paths['client_private']/'requests.json').exists()
        assert state.events[-1]=='client_close'
    else:
        assert paths['upload'].exists() and state.events[-2:]==['graph_close','server_close']


@pytest.mark.parametrize('kind',['input','private_digest','response_scale','response_nonce','response_digest'])
def test_client_rejects_mixed_or_changed_requests_before_decrypt(flow,kind):
    paths,state=flow;initialize(paths);stages.client_encode_encrypt_input(paths)
    stages.server_encrypted_compute(server_paths(paths))
    if kind=='input':
        row=json.loads(paths['preprocessed_input'].read_text());row['target_idx']=1
        paths['preprocessed_input'].write_text(json.dumps(row)+'\n')
    elif kind=='private_digest':
        p=paths['client_private']/'requests.json';row=json.loads(p.read_text());row['public_batch_sha256']='e'*64;p.write_bytes(tc.canonical(row))
    else:
        p=paths['download']/'batch.json';row=json.loads(p.read_text());item=row['responses'][0]
        if kind=='response_scale':item['scale']=18.
        elif kind=='response_nonce':item['request_id']='e'*32
        else:item['response_manifest_sha256']='e'*64
        p.write_bytes(tc.canonical(row))
    with pytest.raises(ValueError):stages.client_decrypt_decode(paths)
    assert 'decrypt' not in state.events and not paths['decrypted_results'].exists()
