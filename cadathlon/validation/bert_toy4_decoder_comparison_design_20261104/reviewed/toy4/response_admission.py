"""Public-only admission of all96 current responses before private loading.

The external review digest is an explicit trusted CLI input; a server cannot
supply its own expected digest or manufacture approval through its report.
"""
from pathlib import Path
import os,stat
import component_protocol as p
import component_contract as c
import component_transport as transport
from scale_replay import client_replay

REVIEW_SCHEMA='cross-toy4-server-root-review-v1'

def review_fields(config,expected_sources,prior):
    """Public schema helper; constructing these fields does not grant approval."""
    setup,server=prior['setup'],prior['server'];request=c.validate_contract(setup['request_contract'])
    return dict(schema=REVIEW_SCHEMA,approved=True,stage='server',
        setup_receipt_sha256=config['setup_sha256'],server_receipt_sha256=config['server_sha256'],
        consumer_source_map_sha256=c.digest(expected_sources),server_source_archive_sha256=server['source_archive_sha256'],
        workspace=config['workspace'],workspace_identity=config['workspace_identity'],
        request_contract_sha256=c.digest(request),input_digest=setup['input_digest'],
        ordered_artifacts_sha256=c.digest(server['artifacts']),artifact_count=96,
        original_source_archive_sha256=c.ORIGINAL_SHA,profile_sha256=c.PROFILE_SHA,numerical_plan_sha256=c.PLAN_SHA)

def authenticate_review(config,expected_sources,prior):
    path,pin=config.get('server_root_review'),config.get('server_root_review_sha256')
    if type(path)is not str or not Path(path).is_absolute()or type(pin)is not str or not c.SHA.fullmatch(pin):
        raise ValueError('externally pinned completed server root review required before responses/private')
    raw=p.read(path,1<<20)
    if p.sha(raw)!=pin:raise ValueError('external server root review digest differs')
    review=p.parse(raw);wanted=review_fields(config,expected_sources,prior)
    if c.canonical(review)!=c.canonical(wanted):raise ValueError('exact root review request/source/terminal/artifact binding differs')
    return dict(path=path,sha256=pin,review=review)

def authenticate_all(config,expected_sources,prior):
    # The completed parent/worker/group/source gates run in worker.priors first.
    # Requiring the independently pinned review is separate from those gates.
    review=authenticate_review(config,expected_sources,prior)
    setup,server=prior['setup'],prior['server'];request=c.validate_contract(setup['request_contract']);artifacts=server['artifacts']
    if server.get('all96_server_outputs_verified')is not True or type(artifacts)is not list or len(artifacts)!=96:
        raise ValueError('all96 server artifacts required before private read')
    replay=client_replay(tuple(server['arithmetic']['refresh_scales']))
    if len(replay['trace'])!=96:raise ValueError('exact96 replay required')
    for i,(row,expected)in enumerate(zip(artifacts,replay['trace'],strict=True)):
        if type(row)is not dict or set(row)!={'index','metadata','manifest_sha256'}or type(row['index'])is not int or row['index']!=i:
            raise ValueError('ordered exact artifact metadata required')
        if type(row['manifest_sha256'])is not str or not c.SHA.fullmatch(row['manifest_sha256']):raise ValueError('manifest digest required')
        for key in ('index','kind','label','stack','level','scale','nsd'):
            if c.canonical(row['metadata'][key])!=c.canonical(expected[key]):raise ValueError('independent actual metadata replay differs')
    folder=Path(config['workspace'])/'outputs';state=transport._CloseState()
    fd=os.open(folder,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        info=os.fstat(fd);identity=[info.st_dev,info.st_ino]
        def check():
            p.check_identity(config['workspace'],config['workspace_identity']);p.check_identity(folder,identity)
            if set(os.listdir(fd))!={f'tile{i}'for i in range(96)}:raise ValueError('exact96 response directory inventory required')
        check();digests=[]
        for i,row in enumerate(artifacts):
            check()
            payload,_,_,manifest=transport.validate_ciphertext(folder/f'tile{i}',contract=request,role='response',tile_index=i,
                expected_scale=replay['trace'][i]['scale'],input_digest=setup['input_digest'])
            del payload
            digest=transport.manifest_digest(manifest)
            if digest!=row['manifest_sha256']:raise ValueError('current response differs from terminal server binding')
            digests.append(digest);check()
        if authenticate_review(config,expected_sources,prior)!=review:raise ValueError('root review changed during public admission')
        return dict(replay=replay,review=review,output_identity=identity,manifest_sha256=digests,
            all96_current_responses_authenticated_before_private=True)
    finally:state.close(fd)
