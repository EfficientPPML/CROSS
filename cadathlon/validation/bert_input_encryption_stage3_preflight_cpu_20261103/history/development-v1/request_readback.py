"""Read back the complete size0 upload without constructing or decoding CKKS."""
from pathlib import Path
import stat


def verify(paths,*,context_id,expected_input_sha256):
    import file_stages
    import file_stage_contract as batches
    from file_stage_io import read_manifest
    from transport import validate_ciphertext,request_digest,manifest_digest
    from transport_contract import digest
    records,input_sha=file_stages._records(paths,1)
    if input_sha!=expected_input_sha256:raise ValueError('private input changed before upload readback')
    public_keys=read_manifest(Path(paths['public_keys'])/'public_manifest.json')
    if public_keys['context_id']!=context_id:raise ValueError('client context differs')
    model=read_manifest(Path(paths['server_data'])/context_id/'model_manifest.json')
    public=file_stages._public(public_keys,model)
    batch=batches.validate_public_batch(read_manifest(Path(paths['upload'])/'batch.json'),public=public,count=1)
    private=read_manifest(Path(paths['client_private'])/'requests.json')
    batches.validate_private_requests(private,batch,[r.target_idx for r in records],input_sha)
    nonce=batch['requests'][0]['contract']['request_id'];sample=Path(paths['upload'])/nonce
    if {p.name for p in Path(paths['upload']).iterdir()}!={nonce,'batch.json'}:
        raise ValueError('exact size0 upload inventory required')
    if {p.name for p in sample.iterdir()}!={f'tile{i}' for i in range(8)}:
        raise ValueError('exact eight request tiles required')
    manifests=[];summaries=[];request=batch['requests'][0]
    for i in range(8):
        payload,scale,binding,manifest=validate_ciphertext(sample/f'tile{i}',contract=request['contract'],role='request',tile_index=i)
        # validate_ciphertext has authenticated bounded container/header/residue
        #bytes. No Polynomial, JAX transfer, key read or decode occurs here.
        summaries.append(dict(tile=i,payload_bytes=payload.nbytes,shape=list(payload.shape),
                              scale=scale,manifest_sha256=manifest_digest(manifest)))
        manifests.append(manifest);del payload
    if request_digest(manifests,request['contract'])!=request['input_digest']:
        raise ValueError('ordered ciphertext bytes differ from private-bound request')
    info=(Path(paths['client_private'])/'requests.json').lstat()
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode)!=0o600:
        raise ValueError('private request binding must be a regular0600 file')
    return dict(sample_count=1,tile_count=8,public_batch_sha256=digest(batch),
                input_digest=request['input_digest'],private_input_sha256=input_sha,
                private_binding_verified=True,private_request_permissions_verified=True,
                ciphertext_payload_bytes=sum(x['payload_bytes'] for x in summaries),tiles=summaries)
