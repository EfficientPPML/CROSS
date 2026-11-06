"""Fresh-process installed guard probe, never constructs keys/context/model."""
from pathlib import Path
import argparse,hashlib,importlib,json,sys,tempfile
import ranked_worker as worker


def run(stage):
    from jaxite_word import CKKSContext
    import file_stages,client_keys,client_inputs,key_contexts,request_lifecycle
    modules=[importlib.import_module(n) for n in ('key_gen','jaxite_word.key_gen')]
    key_names=('gen_pke_pair','gen_evaluation_key','gen_rotation_key','key_switch_gen')
    codecs=('encrypt','encrypt_slots','encrypt_slots_batch','decrypt','decrypt_slots','decrypt_slots_batch','decode')
    pairs=[]
    if stage in (1,2):
        import ranked_setup
        pairs=[(CKKSContext,n) for n in ('__init__',*codecs)]
        pairs += [(client_inputs,'ClientEmbeddings'),(key_contexts,'load_client'),(key_contexts,'load_server'),
            (file_stages,'client_encode_encrypt_input'),(file_stages,'server_encrypted_compute'),(file_stages,'client_decrypt_decode'),
            (file_stages,'server_preprocess_model' if stage==1 else 'client_key_generation')]
        if stage==2:pairs += [(client_keys,'generate')]+[(m,n) for m in modules for n in key_names]
        scope=lambda calls,paths:ranked_setup.setup_only(stage,calls,paths)
        allowed=(file_stages,'client_key_generation' if stage==1 else 'server_preprocess_model')
    elif stage==3:
        from transformers.models.bert import modeling_bert as bert
        pairs=[(m,n) for m in modules for n in key_names]+[(CKKSContext,n) for n in ('decrypt','decrypt_slots','decrypt_slots_batch','decode')]
        pairs += [(getattr(bert,n),'__init__') for n in ('BertModel','BertEncoder','BertLayer','BertPooler','BertForSequenceClassification','BertSelfAttention','BertAttention','BertIntermediate','BertOutput')]
        scope=lambda calls,paths:worker.guard(stage,calls,paths);allowed=(file_stages,'client_encode_encrypt_input')
    elif stage==4:
        pairs=[(m,n) for m in modules for n in key_names]+[(CKKSContext,n) for n in codecs]
        pairs += [(client_keys,'generate'),(client_inputs,'ClientEmbeddings'),(key_contexts,'load_client'),(key_contexts,'_read_private'),
            (file_stages,'client_key_generation'),(file_stages,'client_encode_encrypt_input'),(file_stages,'client_decrypt_decode'),
            (file_stages,'server_preprocess_model'),(file_stages,'_records'),(file_stages,'_snapshot')]
        scope=lambda calls,paths:worker.guard(stage,calls,paths);allowed=(file_stages,'server_encrypted_compute')
    else:
        pairs=[(m,n) for m in modules for n in key_names]+[(CKKSContext,n) for n in ('encrypt','encrypt_slots','encrypt_slots_batch','program_initialization')]
        pairs += [(client_keys,'generate'),(client_inputs,'ClientEmbeddings'),(key_contexts,'load_server'),
            (file_stages,'client_key_generation'),(file_stages,'server_preprocess_model'),(file_stages,'client_encode_encrypt_input'),
            (file_stages,'server_encrypted_compute'),(file_stages,'_snapshot'),(request_lifecycle,'retire_completed')]
        scope=lambda calls,paths:worker.guard(stage,calls,paths);allowed=(file_stages,'client_decrypt_decode')
    originals=[getattr(o,n) for o,n in pairs];good=getattr(*allowed);calls=[]
    paths={n:Path('/nonexistent-ranked-guard-probe')/n for n in ('client_private','test_input','preprocessed_input')}
    with scope(calls,paths):
        if getattr(*allowed)is not good:raise AssertionError('allowed handler changed')
        for owner,name in pairs:
            try:getattr(owner,name)(None)
            except (RuntimeError,AssertionError):pass
            else:raise AssertionError('installed forbidden action was callable')
    if len(calls)!=len(pairs) or any(getattr(o,n)is not original for (o,n),original in zip(pairs,originals)):raise AssertionError('guard count/restoration differs')
    local={}
    for name,m in tuple(sys.modules.items()):
        file=getattr(m,'__file__',None)
        if file and Path(file).is_absolute() and Path(file).resolve().is_relative_to(worker.HERE) and Path(file).suffix=='.py':
            p=Path(file).resolve();local[name]={'path':str(p.relative_to(worker.HERE)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    return dict(passed=True,stage=stage,denied_count=len(pairs),all_restored=True,allowed_handler_unchanged=True,
        context_constructed=False,key_generation=False,payload_or_model_reads=False,loaded_local_modules=local)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',type=int,choices=range(1,6),required=True);a=p.parse_args()
    print(json.dumps(run(a.stage),sort_keys=True))
