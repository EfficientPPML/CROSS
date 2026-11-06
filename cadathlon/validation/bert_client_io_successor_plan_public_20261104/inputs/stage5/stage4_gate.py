"""Externally pinned completed server proof; public metadata only, no payloads."""
from pathlib import Path
import math,stat
import key_gate
HERE=Path(__file__).resolve().parent

def _same(a,b,message):
    if key_gate.canonical(a)!=key_gate.canonical(b):raise ValueError(message)
def _identity(path,expected,kind):
    s=Path(path).lstat()
    if type(expected)is not dict or any(type(expected.get(n))is not int for n in ('device','inode')):
        raise ValueError('exact retained response identity required')
    if [s.st_dev,s.st_ino]!=[expected['device'],expected['inode']] or not (stat.S_ISDIR(s.st_mode) if kind=='directory' else stat.S_ISREG(s.st_mode)):
        raise ValueError('retained server output identity changed')

def authenticate(evidence,report_sha256,io_workspace,*,keys,model,requests):
    evidence=Path(evidence);root=Path(io_workspace)
    raw=key_gate.read(evidence/'report.json')
    if key_gate.digest(raw)!=key_gate.sha_field(report_sha256):raise ValueError('external completed stage4 report differs')
    report=key_gate.parse(raw)
    flags=('passed','execute','actual_server_compute','actual_execution_attempted','all_processes_reaped',
        'external_workspaces_never_removed','operation_complete','cleanup_completed','terminal_scope_completed',
        'source_files_unchanged','source_archive_verified','io_workspace_retained')
    if type(report)is not dict or any(report.get(n)is not True for n in flags) or report.get('error') or report.get('cleanup_errors'):
        raise ValueError('completed actual server proof required before client reads')
    inventory_raw=key_gate.read(HERE/'stage4_source_manifest.json',1<<20)
    dependencies=key_gate.parse(key_gate.read(HERE/'dependencies.json'))
    if key_gate.digest(inventory_raw)!=dependencies['stage4_source_inventory_sha256']:raise ValueError('reviewed stage4 source pin changed')
    expected=key_gate.parse(inventory_raw)
    if type(expected)is not dict or len(expected)!=217:raise ValueError('exact217 reviewed server sources required')
    if any(report.get(n)!=expected for n in ('source_hashes_start','source_hashes_end','execution_source_hashes_end')):
        raise ValueError('completed server source inventory differs')
    source=key_gate.read(evidence/'sources.tar.gz',16<<20)
    if key_gate.digest(source)!=key_gate.sha_field(report.get('source_archive_sha256')):raise ValueError('completed server source archive differs')
    key_gate.verify_source_archive(source,expected)
    process=report.get('worker_process')
    if (type(process)is not dict or type(process.get('returncode'))is not int or process['returncode']!=0 or process.get('error')
            or any(process.get(n)is not True for n in ('process_reaped','public_key_store_removed','owned_directory_handle_closed'))):
        raise ValueError('server worker terminal reaping required')
    result=process.get('result')
    if type(result)is not dict or result.get('passed')is not True:raise ValueError('successful server worker result required')
    worker_raw=key_gate.read(evidence/'worker/worker_report.json')
    if key_gate.digest(worker_raw)!=key_gate.sha_field(result.get('worker_report_sha256')):raise ValueError('server worker report linkage differs')
    worker=key_gate.parse(worker_raw)
    worker_flags=('passed','actual_server_compute_started','actual_server_compute_completed','request_readback_complete',
        'response_readback_complete','source_files_unchanged','keygen_codecs_native_private_forbidden','terminal_scope_completed')
    if (type(worker)is not dict or any(worker.get(n)is not True for n in worker_flags) or worker.get('error') or worker.get('forbidden_attempts')
            or any(worker.get(n)!=expected for n in ('source_hashes_start','source_hashes_end'))):raise ValueError('complete server-only worker proof required')
    prerequisite={n:keys[n] for n in ('context_id','key_report_sha256','key_source_archive_sha256')}|{
        n:model[n] for n in ('model_id','model_report_sha256','model_manifest_sha256')}|{
        n:requests[n] for n in ('stage3_report_sha256','public_batch_sha256','input_digest')}
    for proof in (report,worker):_same(proof.get('prerequisite'),prerequisite,'server predecessor chain differs')
    for proof in (report,worker,result):
        if proof.get('context_id')!=keys['context_id'] or proof.get('model_id')!=model['model_id']:raise ValueError('server context/model differs')
    if result.get('public_batch_sha256')!=requests['public_batch_sha256']:raise ValueError('server stdout request binding differs')
    if report.get('io_workspace')!=str(root):raise ValueError('completed server I/O path differs')
    ident=report.get('io_workspace_identity')
    if type(ident)is not list or len(ident)!=2 or any(type(x)is not int for x in ident):raise ValueError('exact I/O owner identity required')
    _identity(root,dict(device=ident[0],inode=ident[1]),'directory')
    for name,kind in (('upload','directory'),('download','directory'),('server_report','regular')):
        a=report.get('remaining_artifacts',{}).get(name);b=worker.get('remaining_artifacts',{}).get(name)
        _same(a,b,'server retained output identities differ')
        if type(a)is not dict or a.get('present')is not True or a.get('kind')!=kind:raise ValueError('complete retained server artifact required')
        _identity(root/name,a,kind)
    request_expected=dict(sample_count=1,tile_count=8,public_batch_sha256=requests['public_batch_sha256'],input_digest=requests['input_digest'],
        ciphertext_payload_bytes=sum(row['payload_bytes'] for row in requests['tiles']),tiles=requests['tiles'])
    for proof in (worker,report):_same(proof.get('request_readback'),request_expected,'server original request readback differs')
    from file_stage_io import read_manifest
    import file_stage_contract as batches,file_stages
    from transport_contract import digest,ciphertext_contract
    from transport import _validate_manifest,manifest_digest
    batch=read_manifest(root/'upload/batch.json')
    if digest(batch)!=requests['public_batch_sha256']:raise ValueError('retained public request batch changed')
    inputs,candidate=file_stages._inputs()
    response=batches.validate_response_batch(read_manifest(root/'download/batch.json'),batch,check_scale=file_stages._scale_checker(inputs,candidate))
    row=response['responses'][0];original=batch['requests'][0];nonce=row['request_id'];directory=root/'download'/nonce
    if ({p.name for p in (root/'download').iterdir()}!={'batch.json',nonce} or not stat.S_ISDIR(directory.lstat().st_mode)
            or {p.name for p in directory.iterdir()}!={'bundle.json','transport.json','ct_0.npz'}):raise ValueError('exact retained response inventory required')
    binding=ciphertext_contract(original['contract'],'response',0,row['scale'],requests['input_digest'])
    manifest=_validate_manifest(read_manifest(directory/'transport.json'),binding)
    if manifest_digest(manifest)!=row['response_manifest_sha256']:raise ValueError('retained response manifest changed')
    timing=read_manifest(root/'server_report');seconds=timing.get('full_model_compute_and_transport_seconds')
    if (set(timing)!={'full_model_compute_and_transport_seconds'} or type(seconds)not in (float,int)
            or not math.isfinite(seconds) or seconds<0):raise ValueError('finite server timing required')
    p=keys['public_manifest']['parameters'];shape=[1,2,*p['degree_layout'],6]
    summary=dict(sample_count=1,response_count=1,input_digest=requests['input_digest'],public_batch_sha256=requests['public_batch_sha256'],
        response_batch_sha256=digest(response),response_manifest_sha256=manifest_digest(manifest),payload_bytes=math.prod(shape)*4,
        shape=shape,scale=row['scale'],level=2,full_model_compute_and_transport_seconds=seconds)
    for proof in (report,worker):_same(proof.get('response_readback'),summary,'completed server response readback differs')
    return dict(stage4_report_sha256=report_sha256,source_archive_sha256=report['source_archive_sha256'],
        context_id=keys['context_id'],model_id=model['model_id'],public_batch_sha256=requests['public_batch_sha256'],
        input_digest=requests['input_digest'],response_batch_sha256=summary['response_batch_sha256'],
        response_manifest_sha256=summary['response_manifest_sha256'])
