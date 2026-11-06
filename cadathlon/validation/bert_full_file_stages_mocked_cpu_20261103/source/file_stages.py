"""Five file-stage handlers for the pinned full BERT candidate.

This isolated implementation is under integration validation. It always calls
the real CKKS adapters and joined twelve-layer owner; no native encoder or
plaintext model fallback is available here.
"""
from pathlib import Path
import hashlib
import os
import secrets
import shutil
import time
import numpy as np
import client_inputs
import client_keys
import server_public_model
import file_stage_contract as batches
from file_stage_io import new_directory, new_file, read_manifest, write_manifest
from transport_contract import canonical, digest, make_contract, public_identity
from transport import _read_regular, save_ciphertext, load_ciphertext, request_digest, manifest_digest

ROOT=Path(__file__).resolve().parent
MODEL_ID='google-bert/bert-base-cased-finetuned-mrpc'
REVISION='f150c1d609d1e50dd5e2e5408661cfac8339277c'
BATCH_SIZES=(1,100,200,408)


def _paths(paths, *, server=False):
    if type(paths) is not dict or type(paths.get('size')) is not int or not 0<=paths['size']<4:
        raise ValueError('one fixed benchmark size required')
    if (type(paths.get('batch_size')) is not int or paths['batch_size']!=BATCH_SIZES[paths['size']]
            or paths.get('model_id')!=MODEL_ID or paths.get('model_revision')!=REVISION):
        raise ValueError('fixed benchmark/model identity differs')
    if server and set(paths)&{'client_private','preprocessed_input','test_input','decrypted_results','predictions'}:
        raise ValueError('server handler must receive only public paths')
    return paths['batch_size']


def _inputs():
    profile=(ROOT/'server_inputs/p60_profile_and_refresh.json').read_bytes()
    plan=(ROOT/'server_inputs/twelve_layer_plan.json').read_bytes()
    candidate=(ROOT/'server_inputs/candidate_manifest.json').read_bytes()
    client_keys.public_contract(profile,plan)
    if hashlib.sha256(candidate).hexdigest()!=server_public_model.CANDIDATE_SHA256:
        raise ValueError('fixed candidate bytes changed')
    return dict(profile_bytes=profile,plan_bytes=plan),candidate


def _snapshot():
    explicit=os.environ.get('CROSS_BERT_SNAPSHOT')
    if explicit:return Path(explicit)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=MODEL_ID,revision=REVISION,local_files_only=True))


def _model_dir(paths,context_id):
    return Path(paths['server_data'])/context_id


def _public(keys,model):
    if model['context_id']!=keys['context_id']:
        raise ValueError('public model cache belongs to another key context')
    return public_identity(dict(context_id=keys['context_id'],model_id=model['model_id'],
        parameters=keys['parameters'],**{k:model[k] for k in
        ('checkpoint_sha256','config_sha256','candidate_sha256','profile_sha256')}))


def _records(paths,count):
    raw=_read_regular(Path(paths['preprocessed_input']),16<<20)
    records=[client_inputs.validate_record(client_inputs._json(line)) for line in raw.splitlines() if line.strip()]
    if len(records)!=count or len({r.target_idx for r in records})!=count:
        raise ValueError('exact input count with unique private target IDs required')
    return records,hashlib.sha256(raw).hexdigest()


def _capacity(path,count,per_sample):
    needed=count*per_sample+(1<<30)
    if shutil.disk_usage(Path(path).parent).free<needed:
        raise OSError('insufficient filesystem capacity for complete ciphertext batch')


def _scale_checker(inputs,candidate):
    from response_scale import accepts_scale
    return lambda scale:accepts_scale(scale,profile_bytes=inputs['profile_bytes'],manifest_bytes=candidate)


def client_key_generation(paths):
    _paths(paths);inputs,_=_inputs()
    result=client_keys.generate(paths['public_keys'],paths['client_private'],**inputs)
    return dict(context_id=result['public_manifest']['context_id'],preflight=result['preflight'])


def server_preprocess_model(paths):
    _paths(paths,server=True);inputs,candidate=_inputs()
    keys=client_keys.authenticate_public(paths['public_keys'],**inputs)
    parent=Path(paths['server_data']);parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    fd=os.open(parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        output=_model_dir(paths,keys['context_id'])
        if output.exists() or output.is_symlink():
            _,manifest=server_public_model.load(output,context_id=keys['context_id'])
        else:
            manifest=server_public_model.preprocess(_snapshot(),output,context_id=keys['context_id'],
                candidate_bytes=candidate,profile_bytes=inputs['profile_bytes'])
    finally:os.close(fd)
    return dict(model_id=manifest['model_id'],context_id=keys['context_id'])


def client_encode_encrypt_input(paths):
    count=_paths(paths);inputs,_=_inputs();records,input_sha=_records(paths,count)
    from key_contexts import load_client
    from request_lifecycle import retire_completed
    from support import require_secure_execution
    with load_client(paths['public_keys'],paths['client_private'],**inputs) as client:
        retire_completed(paths,context_id=client.public_manifest['context_id'])
        _capacity(paths['upload'],count,8*((100<<20)+65536))
        weights,model=server_public_model.load(_model_dir(paths,client.public_manifest['context_id']),
            context_id=client.public_manifest['context_id']);del weights
        public=_public(client.public_manifest,model);embeddings=client_inputs.ClientEmbeddings(_snapshot())
        requests=[]
        with new_directory(paths['upload']) as upload:
            for record in records:
                prepared=embeddings.prepare(record);nonce=secrets.token_hex(16)
                contract=make_contract(public,nonce,prepared.public_metadata());manifests=[]
                with new_directory(upload.path/nonce) as sample:
                    for i,slots in enumerate(prepared.tiles):
                        value=client.ctx.encrypt_slots(slots)
                        # Fresh encryption is NSD1 in the core codec. Retain an
                        #existing explicit value; never replace bad metadata.
                        if not hasattr(value,'_ckks_nsd'):value._ckks_nsd=1
                        require_secure_execution(client.ctx,value)
                        value.polynomial.block_until_ready()
                        manifests.append(save_ciphertext(value,sample.path/f'tile{i}',
                            contract=contract,role='request',tile_index=i))
                        del value
                requests.append(dict(contract=contract,input_digest=request_digest(manifests,contract)))
            batch=batches.public_batch(public,requests,count=count)
            private=batches.private_requests(batch,[r.target_idx for r in records],input_sha)
            with new_file(Path(paths['client_private'])/'requests.json',canonical(private)):
                write_manifest(upload,'batch.json',batch);upload.check()
    return dict(sample_count=count,public_batch_sha256=digest(batch))


def server_encrypted_compute(paths):
    count=_paths(paths,server=True);inputs,candidate=_inputs()
    from key_contexts import load_server
    from full_model_engine import server_engine
    check_scale=_scale_checker(inputs,candidate);_capacity(paths['download'],count,(6<<20)+65536)
    started=time.perf_counter();responses=[]
    with load_server(paths['public_keys'],**inputs) as server:
        weights,model=server_public_model.load(_model_dir(paths,server.public_manifest['context_id']),
            context_id=server.public_manifest['context_id']);public=_public(server.public_manifest,model)
        batch=batches.validate_public_batch(read_manifest(Path(paths['upload'])/'batch.json'),public=public,count=count)
        expected={r['contract']['request_id'] for r in batch['requests']}|{'batch.json'}
        if {p.name for p in Path(paths['upload']).iterdir()}!=expected:
            raise ValueError('unexpected upload sample inventory')
        with new_directory(paths['download']) as download:
            with server_engine(server.ctx,manifest_bytes=candidate) as graph:
                for request in batch['requests']:
                    contract=request['contract'];nonce=contract['request_id'];sample=Path(paths['upload'])/nonce
                    if {p.name for p in sample.iterdir()}!={f'tile{i}' for i in range(8)}:
                        raise ValueError('exact eight input tiles required')
                    tiles=[];manifests=[]
                    for i in range(8):
                        value,manifest=load_ciphertext(sample/f'tile{i}',contract=contract,role='request',tile_index=i)
                        tiles.append(value);manifests.append(manifest)
                    if request_digest(manifests,contract)!=request['input_digest']:
                        raise ValueError('ordered encrypted request bytes differ')
                    result=graph.evaluate(tuple(tiles),weights,original_positions=contract['input']['original_positions'],
                        context_id=public['context_id'],request_id=nonce)
                    scale=result.output._ckks_scale
                    if check_scale(scale) is not True:raise ValueError('actual output scale differs from public replay')
                    manifest=save_ciphertext(result.output,download.path/nonce,contract=contract,
                        role='response',tile_index=0,expected_scale=scale,input_digest=request['input_digest'])
                    responses.append(dict(request_id=nonce,input_digest=request['input_digest'],scale=scale,
                        response_manifest_sha256=manifest_digest(manifest)))
                    del tiles,value,result
            response=batches.response_batch(batch,responses)
            write_manifest(download,'batch.json',response);download.check()
            report=dict(full_model_compute_and_transport_seconds=time.perf_counter()-started)
            with new_file(paths['server_report'],canonical(report)):download.check()
    return dict(sample_count=count,public_batch_sha256=digest(batch))


def client_decrypt_decode(paths):
    count=_paths(paths);inputs,candidate=_inputs();records,input_sha=_records(paths,count)
    from key_contexts import load_client
    from request_lifecycle import record_completed
    from support import require_secure_execution
    private=read_manifest(Path(paths['client_private'])/'requests.json')
    raw_batch=read_manifest(Path(paths['upload'])/'batch.json')
    # The independently retained private record authenticates the complete
    #original public request before any server-controlled scale or output.
    batches.validate_private_requests(private,raw_batch,[r.target_idx for r in records],input_sha)
    batch=batches.validate_public_batch(raw_batch,public=raw_batch['public'],count=count)
    response=batches.validate_response_batch(read_manifest(Path(paths['download'])/'batch.json'),batch,
        check_scale=_scale_checker(inputs,candidate))
    if {p.name for p in Path(paths['download']).iterdir()}!={r['request_id'] for r in response['responses']}|{'batch.json'}:
        raise ValueError('unexpected response inventory')
    results=[]
    with load_client(paths['public_keys'],paths['client_private'],**inputs) as client:
        if client.public_manifest['context_id']!=batch['public']['context_id']:
            raise ValueError('response belongs to another client key context')
        for target,request,row in zip(private['targets'],batch['requests'],response['responses']):
            value,manifest=load_ciphertext(Path(paths['download'])/row['request_id'],contract=request['contract'],
                role='response',tile_index=0,expected_scale=row['scale'],input_digest=request['input_digest'])
            if manifest_digest(manifest)!=row['response_manifest_sha256']:
                raise ValueError('returned ciphertext bytes differ from bound response')
            require_secure_execution(client.ctx,value)
            decoded=np.asarray(client.ctx.decrypt_slots(value))
            if decoded.shape!=(65536,) or decoded.dtype.kind!='f' or not np.isfinite(decoded).all():
                raise ValueError('finite real full-slot decode required')
            results.append(dict(target_idx=target,he_logits=[float(decoded[0]),float(decoded[1])]))
            del value,decoded
    with new_file(paths['decrypted_results'],b''.join(canonical(r) for r in results)):
        record_completed(paths,context_id=batch['public']['context_id'],batch=batch)
    return dict(sample_count=count,public_batch_sha256=digest(batch))
