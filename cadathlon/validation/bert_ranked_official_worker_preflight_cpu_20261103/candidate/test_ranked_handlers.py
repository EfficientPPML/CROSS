"""Real unchanged file-handler loops with tiny synthetic transport/crypto doubles.

No crypto, model or retained payload is opened. This is orchestration/lifecycle
coverage only: Graph is deliberately a double, never an HE accuracy claim.
"""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib,json,math,os,sys
import numpy as np
import pytest
import file_stages as stages
import full_model_engine,transport,transport_contract as tc,request_lifecycle as lifecycle
import ranked_readback as readback


def flow(tmp_path,monkeypatch,*,size):
    count=(1,100,200,408)[size]
    paths=dict(size=size,batch_size=count,model_id=stages.MODEL_ID,model_revision=stages.REVISION)
    for name in ('public_keys','client_private','server_data','preprocessed_input','upload','download','decrypted_results','server_report'):
        paths[name]=tmp_path/name
    records=[];positions=[];references=[]
    for i in range(count):
        K=(1,15,16,17,64,127,128)[i%7]
        pos=[0]+list(range(129-K,128)) if K>1 else [0]
        records.append(dict(target_idx=10000+3*i,input_ids=[101]+[2]*127,
            token_type_ids=[0]*128,attention_mask=[int(j in pos) for j in range(128)]))
        positions.append(pos);references.append(dict(target_idx=10000+3*i,label=i%2,native_fp32_logits=[1.,0.] if i%2==0 else [0.,1.]))
    paths['preprocessed_input'].write_text(''.join(json.dumps(r)+'\n' for r in records));paths['preprocessed_input'].chmod(0o600)
    keys=dict(context_id='a'*64,parameters=tc.parameters())
    model=dict(context_id='a'*64,model_id='b'*64,checkpoint_sha256=tc.CHECKPOINT_SHA256,
        config_sha256=tc.CONFIG_SHA256,candidate_sha256=tc.CANDIDATE_SHA256,profile_sha256=tc.PROFILE_SHA256)
    state=NS(events=[],values={},failure=None,keys=keys,model=model,encrypts=0,computes=0,decrypts=0)
    # Only tiny placeholder files are written. Preserve the actual capacity
    # calculation while providing a synthetic filesystem budget for this test.
    monkeypatch.setattr(stages.shutil,'disk_usage',lambda path:NS(total=2<<40,used=0,free=2<<40))
    def keygen(public,private,**kwargs):
        state.events.append('keygen');Path(public).mkdir(mode=0o700);Path(private).mkdir(mode=0o700)
        (Path(private)/'secret.npy').write_bytes(b'synthetic key sentinel; not a key')
        (Path(private)/'secret.npy').chmod(0o600)
        return dict(public_manifest=keys,preflight=dict(mocked=True))
    monkeypatch.setattr(stages.client_keys,'generate',keygen)
    monkeypatch.setattr(stages.client_keys,'authenticate_public',lambda *a,**kw:keys)
    def preprocess(snapshot,path,**kwargs):Path(path).mkdir(mode=0o700);return model
    monkeypatch.setattr(stages.server_public_model,'preprocess',preprocess)
    monkeypatch.setattr(stages.server_public_model,'load',lambda *a,**kw:({'synthetic_weights':True},model))
    monkeypatch.setattr(stages,'_snapshot',lambda:tmp_path/'never_read_model')
    def value(ordinal,tile,output=False):
        return NS(ordinal=ordinal,tile=tile,output=output,_ckks_nsd=1,_ckks_scale=keys['parameters']['scaling_factor'],
            polynomial=NS(block_until_ready=lambda:state.events.append('sync')))
    class Client:
        def encrypt_slots(self,slots):
            assert slots.shape==(65536,);i=state.encrypts;state.encrypts+=1
            return value(i//8%count,i%8)
        def decrypt_slots(self,ct):
            assert ct.output;state.decrypts+=1;slots=np.zeros(65536);slots[:2]=references[ct.ordinal]['native_fp32_logits'];return slots
    @contextmanager
    def client(*args,**kwargs):
        state.events.append('client_open')
        try:yield NS(ctx=Client(),public_manifest=keys)
        finally:state.events.append('client_close')
    @contextmanager
    def server(*args,**kwargs):
        state.events.append('server_open')
        try:yield NS(ctx=object(),public_manifest=keys)
        finally:state.events.append('server_close')
    monkeypatch.setitem(sys.modules,'key_contexts',NS(load_client=client,load_server=server))
    monkeypatch.setitem(sys.modules,'support',NS(require_secure_execution=lambda *args:None))
    monkeypatch.setitem(sys.modules,'response_scale',NS(accepts_scale=lambda scale,**kwargs:type(scale)is float and scale==keys['parameters']['scaling_factor']))
    class Embeddings:
        def __init__(self,*args):pass
        def prepare(self,record):
            # Actual unchanged client compaction and eight-tile packer.
            data=np.broadcast_to(np.arange(128,dtype=np.float64)[:,None],(128,768))
            return stages.client_inputs.pack_embeddings(record,data)
    monkeypatch.setattr(stages.client_inputs,'ClientEmbeddings',Embeddings)
    def save(ct,path,*,contract,role,tile_index,expected_scale=None,input_digest=None):
        if state.failure=='last_upload' and role=='request' and ct.ordinal==count-1 and tile_index==7:
            raise KeyboardInterrupt('synthetic late upload cancellation')
        binding=tc.ciphertext_contract(contract,role,tile_index,expected_scale,input_digest)
        path=Path(path);path.mkdir(mode=0o700)
        files={}
        for name in ('bundle.json','ct_0.npz'):
            raw=b'synthetic transport placeholder';(path/name).write_bytes(raw);files[name]=hashlib.sha256(raw).hexdigest()
        manifest=transport._manifest_fields(binding)|dict(files=files)
        (path/'transport.json').write_bytes(tc.canonical(manifest));state.values[str(path)]=(ct,manifest)
        return manifest
    def load(path,**kwargs):
        ct,manifest=state.values[str(path)]
        binding=tc.ciphertext_contract(kwargs['contract'],kwargs['role'],kwargs['tile_index'],kwargs.get('expected_scale'),kwargs.get('input_digest'))
        transport._validate_manifest(manifest,binding);return ct,manifest
    def validated(path,**kwargs):
        ct,manifest=load(path,**kwargs)
        binding=tc.ciphertext_contract(kwargs['contract'],kwargs['role'],kwargs['tile_index'],kwargs.get('expected_scale'),kwargs.get('input_digest'))
        shape=binding['shape'];payload=NS(shape=shape,nbytes=math.prod(shape)*4,dtype=np.dtype('uint32'))
        return payload,binding['scale'],binding,manifest
    monkeypatch.setattr(stages,'save_ciphertext',save);monkeypatch.setattr(stages,'load_ciphertext',load)
    monkeypatch.setattr(transport,'validate_ciphertext',validated)
    class Graph:
        def evaluate(self,tiles,weights,**kwargs):
            ordinal=state.computes%count
            assert len(tiles)==8 and all((v.ordinal,v.tile)==(ordinal,i) for i,v in enumerate(tiles))
            assert kwargs['original_positions']==positions[ordinal]
            if state.failure=='last_compute' and ordinal==count-1:raise KeyboardInterrupt('synthetic late server cancellation')
            state.computes+=1;return NS(output=value(ordinal,0,True))
    @contextmanager
    def graph(*args,**kwargs):
        state.events.append('graph_open')
        try:yield Graph()
        finally:state.events.append('graph_close')
    monkeypatch.setattr(full_model_engine,'server_engine',graph)
    return paths,state,references


def server_paths(paths):return {k:v for k,v in paths.items() if k not in ('client_private','preprocessed_input','decrypted_results')}
def initialize(paths):stages.client_key_generation(paths);stages.server_preprocess_model(server_paths(paths))


@pytest.mark.parametrize('size',[1,2,3])
def test_all_ranked_handler_counts_order_and_complete_receipt(tmp_path,monkeypatch,size):
    paths,state,refs=flow(tmp_path,monkeypatch,size=size);initialize(paths)
    encoded=stages.client_encode_encrypt_input(paths);batch=json.loads((paths['upload']/'batch.json').read_text())
    req,validated_batch=readback.requests(paths,public=batch['public'],count=paths['batch_size'],expected_batch_sha256=encoded['public_batch_sha256'])
    assert req['tile_count']==8*len(refs)
    stages.server_encrypted_compute(server_paths(paths));resp=readback.responses(paths,batch=validated_batch)
    assert resp['response_count']==len(refs) and len(resp['responses'])==len(refs)
    stages.client_decrypt_decode(paths)
    checked=readback.client_results(paths,batch=batch,references=refs,count=len(refs),predictions=[r['label'] for r in refs])
    assert checked['metrics']['final_ranked_acceptance'] and checked['private_receipt_validated']
    assert state.encrypts==8*len(refs) and state.computes==state.decrypts==len(refs)
    assert state.events.count('graph_open')==state.events.count('graph_close')==1
    assert state.events.count('client_open')==state.events.count('client_close')==2
    receipt=paths['client_private']/'completed.json';assert receipt.stat().st_size<16<<20
    assert len(json.loads(receipt.read_text())['nonces'])==len(refs)
    assert lifecycle.retire_completed(paths,context_id='a'*64)
    assert not paths['upload'].exists() and not paths['download'].exists()
    assert (paths['client_private']/'secret.npy').read_bytes()==b'synthetic key sentinel; not a key'


@pytest.mark.parametrize('failure',['last_upload','last_compute'])
def test_last_ranked_record_cancellation_preserves_keys_and_primary(tmp_path,monkeypatch,failure):
    paths,state,_=flow(tmp_path,monkeypatch,size=1);initialize(paths);state.failure=failure
    with pytest.raises(KeyboardInterrupt):
        stages.client_encode_encrypt_input(paths);stages.server_encrypted_compute(server_paths(paths))
    assert not paths['download'].exists() and not (paths['client_private']/'completed.json').exists()
    assert paths['upload'].exists()==(failure=='last_compute')
    assert (paths['client_private']/'secret.npy').exists()


def test_last_response_scale_rejected_before_first_decrypt(tmp_path,monkeypatch):
    paths,state,_=flow(tmp_path,monkeypatch,size=1);initialize(paths)
    stages.client_encode_encrypt_input(paths);stages.server_encrypted_compute(server_paths(paths))
    p=paths['download']/'batch.json';response=json.loads(p.read_text());response['responses'][-1]['scale']=1.;p.write_bytes(tc.canonical(response))
    with pytest.raises(ValueError):stages.client_decrypt_decode(paths)
    assert state.decrypts==0 and not paths['decrypted_results'].exists()


def test_public_readback_catches_last_payload_digest_before_client(tmp_path,monkeypatch):
    paths,state,_=flow(tmp_path,monkeypatch,size=1);initialize(paths)
    stages.client_encode_encrypt_input(paths);stages.server_encrypted_compute(server_paths(paths))
    batch=json.loads((paths['upload']/'batch.json').read_text())
    p=paths['download']/'batch.json';response=json.loads(p.read_text())
    response['responses'][-1]['response_manifest_sha256']='f'*64;p.write_bytes(tc.canonical(response))
    with pytest.raises(ValueError):readback.responses(paths,batch=batch)
    assert state.decrypts==0 and not paths['decrypted_results'].exists()


def test_ranked_partial_retirement_retry_preserves_key_sentinel(tmp_path,monkeypatch):
    paths,state,_=flow(tmp_path,monkeypatch,size=1);initialize(paths)
    stages.client_encode_encrypt_input(paths);stages.server_encrypted_compute(server_paths(paths));stages.client_decrypt_decode(paths)
    real=lifecycle.shutil.rmtree
    def remove(name,**kwargs):
        if name=='download':raise OSError('synthetic interrupted retirement')
        return real(name,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(lifecycle.shutil,'rmtree',remove)
        with pytest.raises(OSError):lifecycle.retire_completed(paths,context_id='a'*64)
    assert not paths['upload'].exists() and paths['download'].exists()
    assert lifecycle.retire_completed(paths,context_id='a'*64)
    assert (paths['client_private']/'secret.npy').read_bytes()==b'synthetic key sentinel; not a key'


def test_incomplete_ranked_batch_cannot_reuse_same_ledger(tmp_path,monkeypatch):
    paths,state,_=flow(tmp_path,monkeypatch,size=1);initialize(paths)
    stages.client_encode_encrypt_input(paths)
    original=(paths['upload']/'batch.json').read_bytes();requests=(paths['client_private']/'requests.json').read_bytes()
    with pytest.raises(FileExistsError):stages.client_encode_encrypt_input(paths)
    assert state.encrypts==800 and (paths['upload']/'batch.json').read_bytes()==original
    assert (paths['client_private']/'requests.json').read_bytes()==requests


@pytest.mark.parametrize('mutation',['last_nested_file','same_inode','new_child'])
def test_full408_receipt_replacement_prevents_any_retirement(tmp_path,monkeypatch,mutation):
    paths,state,refs=flow(tmp_path,monkeypatch,size=3);initialize(paths)
    stages.client_encode_encrypt_input(paths);stages.server_encrypted_compute(server_paths(paths));stages.client_decrypt_decode(paths)
    batch=json.loads((paths['upload']/'batch.json').read_text());p=paths['upload']/batch['requests'][-1]['contract']['request_id']/'tile7'/'ct_0.npz'
    if mutation=='last_nested_file':p.rename(p.with_suffix('.old'));p.write_bytes(b'replacement');p.with_suffix('.old').unlink()
    elif mutation=='same_inode':p.write_bytes(b'changed same inode')
    else:(p.parent/'sentinel').write_bytes(b'keep')
    with pytest.raises(ValueError):lifecycle.retire_completed(paths,context_id='a'*64)
    assert paths['upload'].exists() and paths['download'].exists() and (paths['client_private']/'completed.json').exists()
