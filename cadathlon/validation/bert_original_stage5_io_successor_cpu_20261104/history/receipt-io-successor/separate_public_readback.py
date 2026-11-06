"""Verify public response bytes before private input/reference/key loading.

This API is selected only by a future authorized client worker. Preparation uses
synthetic transport doubles; no retained ciphertext is read during preflight.
"""
from pathlib import Path

def verify(paths,*,server,requests):
    from file_stage_io import read_manifest
    import file_stages,file_stage_contract as batches
    from transport_contract import digest
    from transport import validate_ciphertext,manifest_digest
    batch=read_manifest(Path(paths['upload'])/'batch.json')
    if digest(batch)!=requests['public_batch_sha256']:raise ValueError('authenticated request batch changed')
    batch=batches.validate_public_batch(batch,public=batch['public'],count=1)
    inputs,candidate=file_stages._inputs()
    response=batches.validate_response_batch(read_manifest(Path(paths['download'])/'batch.json'),batch,
        check_scale=file_stages._scale_checker(inputs,candidate))
    if digest(response)!=server['response_batch_sha256']:raise ValueError('completed response batch changed')
    row=response['responses'][0];request=batch['requests'][0]
    value,scale,binding,manifest=validate_ciphertext(Path(paths['download'])/row['request_id'],
        contract=request['contract'],role='response',tile_index=0,expected_scale=row['scale'],input_digest=requests['input_digest'])
    if manifest_digest(manifest)!=row['response_manifest_sha256']or manifest_digest(manifest)!=server['response_manifest_sha256']:
        raise ValueError('completed response payload/manifest binding changed')
    if binding['level']!=2:raise ValueError('fixed client response level2 required')
    result=dict(passed=True,public_batch_sha256=requests['public_batch_sha256'],response_batch_sha256=server['response_batch_sha256'],
        response_manifest_sha256=server['response_manifest_sha256'],input_digest=requests['input_digest'],
        payload_bytes=value.nbytes,shape=list(value.shape),scale=scale,level=2,private_data_read=False)
    del value
    return result
