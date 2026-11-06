"""Fresh official setup handlers and public proof authentication.

Only execute_setup may touch actual key/model payloads, under an explicitly
launched worker. Public authenticate functions read reports and manifests only.
"""
from pathlib import Path
from contextlib import contextmanager,redirect_stdout
import importlib,json,os,stat,sys
import key_gate,ranked_execution as execution
from ranked_stage_gate import _archive,case_binding


def phases(stage):
    return dict(encryption=False,decryption=False,server_compute=False,model_forward=False,key_generation=stage==1,public_model_preprocessing=stage==2)

def binding(request,case):
    return dict(case_binding=case_binding(case),attempt_sha256=request['attempt_sha256'],staged_root=str(Path(request['staged_root']).absolute()),
        key_report_sha256=None if request['stage']==1 else request['key_report_sha256'])

def _directory(path,identity):
    info=Path(path).lstat()
    if type(identity)is not list or len(identity)!=2 or any(type(x)is not int for x in identity) or not stat.S_ISDIR(info.st_mode) or identity!=[info.st_dev,info.st_ino]:raise ValueError('retained fresh setup directory changed')

def _completed(evidence,report_sha,*,stage,request,case,expected_sources):
    if type(expected_sources)is not dict or not expected_sources:raise ValueError('reviewed nonempty fresh setup source inventory required')
    for name,h in expected_sources.items():
        if type(name)is not str or Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('relative setup source member required')
        key_gate.sha_field(h)
    evidence=Path(evidence);raw=key_gate.read(evidence/'report.json',16<<20)
    if key_gate.digest(raw)!=key_gate.sha_field(report_sha):raise ValueError('externally pinned fresh setup report differs')
    report=key_gate.parse(raw)
    required=('passed','execute','actual_execution_attempted','operation_complete','all_processes_reaped','source_files_unchanged','source_archive_verified','cleanup_completed','terminal_scope_completed','termination_scope_completed','owned_artifacts_retained')
    if (type(report)is not dict or report.get('schema')!='cross-bert-ranked-setup-proof-v1' or type(report.get('stage'))is not int or report['stage']!=stage
            or any(report.get(n)is not True for n in required) or report.get('error') or report.get('cleanup_errors')):raise ValueError('completed fresh setup proof required')
    if any(report.get(n)!=expected_sources for n in ('source_hashes_start','source_hashes_end','execution_source_hashes_end')):raise ValueError('reviewed fresh setup source closure differs')
    archived=key_gate.read(evidence/'sources.tar.gz',16<<20)
    if key_gate.digest(archived)!=key_gate.sha_field(report.get('source_archive_sha256')):raise ValueError('fresh setup source archive differs')
    _archive(archived,expected_sources)
    proc=report.get('worker_process')
    if (type(proc)is not dict or type(proc.get('returncode'))is not int or proc['returncode']!=0 or proc.get('error')
            or any(proc.get(n)is not True for n in ('process_reaped','public_key_store_removed','owned_directory_handle_closed'))):raise ValueError('fresh setup process terminal cleanup incomplete')
    result=proc.get('result');worker_raw=key_gate.read(evidence/'worker/worker_report.json',16<<20)
    if type(result)is not dict or result.get('passed')is not True or result.get('worker_report_sha256')!=key_gate.digest(worker_raw):raise ValueError('fresh setup worker linkage differs')
    proof=key_gate.parse(worker_raw);wanted=binding(request|{'stage':stage},case)
    flags=('passed','actual_stage_started','actual_stage_completed','readback_complete','source_files_unchanged','cleanup_completed','terminal_scope_completed','termination_scope_completed')
    if (any(proof.get(n)is not True for n in flags) or proof.get('error') or proof.get('cleanup_errors') or proof.get('forbidden_attempts')
            or type(proof.get('stage'))is not int or proof['stage']!=stage
            or key_gate.canonical(proof.get('phases'))!=key_gate.canonical(phases(stage))
            or any(proof.get(n)!=expected_sources for n in ('source_hashes_start','source_hashes_end'))
            or any(value.get('bindings')!=wanted for value in (proof,report)) or proof.get('readback')!=report.get('readback')):
        raise ValueError('fresh setup worker phase/source/attempt proof differs')
    return report,proof['readback']


def authenticate_keys(evidence,report_sha,*,request,case,paths,expected_sources,profile_bytes,plan_bytes):
    report,readback=_completed(evidence,report_sha,stage=1,request=request,case=case,expected_sources=expected_sources)
    if any(readback.get(n)is not True for n in ('private_binding_verified','private_permissions_verified','actual_security_preflight_passed')) or type(readback.get('private_file_count_at_generation'))is not int or readback['private_file_count_at_generation']!=2:
        raise ValueError('actual fresh key generation validation incomplete')
    _directory(paths['public_keys'],readback.get('public_directory_identity'))
    # Do not read or stat the client-private store from a server gate. Its
    #generation-time validation is producer attestation, not a perpetual count.
    raw=key_gate.read(paths['public_keys']/'public_manifest.json',1<<20);public=key_gate.parse(raw)
    from client_keys import public_contract,inventory,_body,digest
    p,rotations=public_contract(profile_bytes,plan_bytes);specs=inventory(p,rotations)
    if type(public)is not dict or type(public.get('arrays'))is not dict or set(public['arrays'])!=set(specs):raise ValueError('complete fresh public key inventory differs')
    source=key_gate.sha_field(public.get('keygen_source_sha256'))
    if source!=expected_sources['runtime/jaxite_word/key_gen.py']:raise ValueError('fresh key generator source differs')
    body=_body(p,rotations,public['arrays'],source)
    if raw!=key_gate.canonical(body|{'context_id':digest(body)}) or public!=readback.get('public_manifest'):raise ValueError('fresh public manifest binding differs')
    return dict(context_id=public['context_id'],public_keys=paths['public_keys'],public_manifest=public,key_report_sha256=report_sha,key_source_archive_sha256=report['source_archive_sha256'])


def authenticate_model(evidence,report_sha,*,request,case,paths,keys,expected_sources):
    report,readback=_completed(evidence,report_sha,stage=2,request=request,case=case,expected_sources=expected_sources)
    cache=paths['server_data']/keys['context_id'];_directory(cache,readback.get('model_directory_identity'))
    raw=key_gate.read(cache/'model_manifest.json',1<<20);manifest=key_gate.parse(raw)
    import server_public_model as model
    base=model._base(keys['context_id']);records=manifest.get('tensors')
    if (type(manifest)is not dict or set(manifest)!=set(base)|{'tensors','model_id'} or any(key_gate.canonical(manifest.get(k))!=key_gate.canonical(v) for k,v in base.items())
            or manifest['model_id']!=model.digest({k:v for k,v in manifest.items() if k!='model_id'})
            or raw!=key_gate.canonical(manifest) or manifest!=readback.get('public_model_manifest')):raise ValueError('fresh model context/checkpoint/circuit identity differs')
    if type(records)is not dict or set(records)!=set(model.tensor_shapes()):raise ValueError('all196 fresh model descriptors required')
    for i,(name,shape) in enumerate(model.tensor_shapes().items()):
        d=records[name]
        if (type(d)is not dict or set(d)!={'file','shape','dtype','tensor_sha256','file_sha256'} or d['file']!=f'tensor_{i:03d}.npy'
                or key_gate.canonical(d['shape'])!=key_gate.canonical(list(shape)) or d['dtype']!='<f4' or d['tensor_sha256']!=model.TENSOR_SHA256[name]):raise ValueError('fresh public tensor descriptor differs')
        key_gate.sha_field(d['file_sha256'])
    if (type(readback.get('tensor_count'))is not int or readback['tensor_count']!=196 or type(readback.get('matrix_count'))is not int or readback['matrix_count']!=74
            or readback.get('all_readback_fp64_immutable')is not True):raise ValueError('full fresh model readback incomplete')
    return dict(context_id=keys['context_id'],model_id=manifest['model_id'],server_data=paths['server_data'],model_report_sha256=report_sha,model_manifest_sha256=key_gate.digest(raw))


@contextmanager
def setup_only(stage,calls,paths):
    """Regression guards, not an operating-system sandbox."""
    from cleanup import cleanup_signals
    if any(n.split('.')[0] in ('torch','transformers') for n in sys.modules):raise ValueError('fresh setup cannot preload native models')
    from jaxite_word import CKKSContext
    import client_keys,client_inputs,key_contexts,file_stages
    patches=[];primary=None;failure=None;active=[True]
    def deny(label):
        def stop(*a,**kw):calls.append(label);raise RuntimeError('forbidden ranked setup action')
        return stop
    def patch(owner,name):
        original=getattr(owner,name);patches.append((owner,name,original));setattr(owner,name,deny(owner.__name__+'.'+name))
    def audit(event,args):
        if not active[0]:return
        if event=='import' and args and str(args[0]).split('.')[0] in ('torch','transformers'):deny('native import')()
        if stage==2 and event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).absolute().resolve();private=paths['client_private'].resolve()
            if p==private or private in p.parents:deny('private setup read')()
    try:
        for name in ('__init__','encrypt','decrypt','encrypt_slots','decrypt_slots','encrypt_slots_batch','decrypt_slots_batch','decode'):patch(CKKSContext,name)
        for owner,name in ((client_inputs,'ClientEmbeddings'),(key_contexts,'load_client'),(key_contexts,'load_server'),
            (file_stages,'client_encode_encrypt_input'),(file_stages,'server_encrypted_compute'),(file_stages,'client_decrypt_decode')):patch(owner,name)
        patch(file_stages,'server_preprocess_model' if stage==1 else 'client_key_generation')
        if stage==2:
            patch(client_keys,'generate')
            for module in (importlib.import_module('key_gen'),importlib.import_module('jaxite_word.key_gen')):
                for name in ('gen_pke_pair','gen_evaluation_key','gen_rotation_key','key_switch_gen'):patch(module,name)
        sys.addaudithook(audit);yield
    except BaseException as error:primary=error;raise
    finally:
        active[0]=False
        with cleanup_signals():
            for owner,name,original in reversed(patches):
                try:setattr(owner,name,original)
                except BaseException as error:
                    if primary is not None:primary.add_note('setup guard restoration: '+type(error).__name__)
                    elif failure is None or not isinstance(error,Exception) and isinstance(failure,Exception):failure=error
            if primary is None and failure is not None:raise failure


def execute_setup(request,case,paths,keys,report,calls):
    import file_stages,client_keys,server_public_model as model
    stage=request['stage'];inputs,_=file_stages._inputs()
    with setup_only(stage,calls,paths),redirect_stdout(sys.stderr):
        report['actual_stage_started']=True
        if stage==1:
            paths['public_keys'].parent.mkdir(mode=0o700,parents=True,exist_ok=True)
            _,_,preflight=client_keys.preflight(paths['public_keys'],paths['client_private'],**inputs)
            result=file_stages.client_key_generation({k:paths[k] for k in ('size','batch_size','model_id','model_revision','public_keys','client_private')})
            public=key_gate.parse(key_gate.read(paths['public_keys']/'public_manifest.json',1<<20))
            private=key_gate.parse(key_gate.read(paths['client_private']/'private_manifest.json',1<<20))
            root=paths['client_private'];names={'secret.npy','private_manifest.json'}
            if result.get('context_id')!=public.get('context_id') or private.get('context_id')!=public.get('context_id') or private.get('public_manifest_sha256')!=key_gate.digest(key_gate.canonical(public)):
                raise ValueError('fresh private/public setup binding differs')
            if (set(p.name for p in root.iterdir())!=names or stat.S_IMODE(root.lstat().st_mode)!=0o700
                    or any(not stat.S_ISREG((root/n).lstat().st_mode) or stat.S_IMODE((root/n).lstat().st_mode)!=0o600 for n in names)):raise ValueError('fresh private key permissions/inventory differ')
            info=paths['public_keys'].lstat()
            summary=dict(public_manifest=public,public_directory_identity=[info.st_dev,info.st_ino],private_binding_verified=True,
                private_permissions_verified=True,private_file_count_at_generation=2,actual_security_preflight_passed=True,preflight=preflight)
        else:
            result=file_stages.server_preprocess_model({k:v for k,v in paths.items() if k in execution.SERVER_FIELDS})
            cache=paths['server_data']/keys['context_id'];arrays,manifest=model.load(cache,context_id=keys['context_id'])
            shapes=model.tensor_shapes()
            if (result.get('context_id')!=keys['context_id'] or result.get('model_id')!=manifest.get('model_id') or set(arrays)!=set(shapes)
                    or len(arrays)!=196 or any(a.dtype.str!='<f8' or a.flags.writeable or a.shape!=shapes[n] for n,a in arrays.items())):raise ValueError('fresh public model readback differs')
            matrices=sum(a.ndim==2 for a in arrays.values());del arrays;info=cache.lstat()
            summary=dict(public_model_manifest=manifest,model_directory_identity=[info.st_dev,info.st_ino],tensor_count=196,matrix_count=matrices,all_readback_fp64_immutable=True)
    if calls:raise ValueError('forbidden setup action attempted')
    report.update(actual_stage_completed=True,readback_complete=True,readback=summary)
    return summary
