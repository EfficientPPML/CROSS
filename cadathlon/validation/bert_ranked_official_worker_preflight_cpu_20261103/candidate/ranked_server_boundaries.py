"""Public-server entrypoint and private-path regression guards, not a sandbox."""
from contextlib import contextmanager
from pathlib import Path
import importlib,os,sys
from cleanup import cleanup_signals

KEY_METHODS=('gen_pke_pair','gen_evaluation_key','gen_rotation_key','key_switch_gen')
CODECS=('encrypt','encrypt_slots','encrypt_slots_batch','decrypt','decrypt_slots','decrypt_slots_batch','decode')

@contextmanager
def server_only(calls, *, private_paths=()):
    if any(name.split('.')[0] in ('torch','transformers') for name in sys.modules):
        raise ValueError('fresh server process must not preload native model modules')
    from jaxite_word import CKKSContext
    modules=[importlib.import_module(n) for n in ('key_gen','jaxite_word.key_gen')]
    import client_keys,client_inputs,key_contexts,file_stages
    protected=tuple(Path(p).absolute().resolve() for p in private_paths)
    patches=[];active=[True];primary=None;failure=None
    def denied(label):
        def stop(*args,**kwargs):
            calls.append(label);raise RuntimeError('forbidden stage4 server action: '+label)
        return stop
    def patch(owner,name,label):
        original=getattr(owner,name);patches.append((owner,name,original));setattr(owner,name,denied(label))
    def audit(event,args):
        if not active[0]:return
        if event=='import' and args and str(args[0]).split('.')[0] in ('torch','transformers'):
            denied('native model import')()
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0])).absolute().resolve()
            if any(path==p or p in path.parents for p in protected):denied('private file access')()
    try:
        # Install before any key loader/context construction or handler work.
        for module in modules:
            for name in KEY_METHODS:patch(module,name,module.__name__+'.'+name)
        for name in CODECS:patch(CKKSContext,name,'CKKSContext.'+name)
        for owner,name in ((client_keys,'generate'),(client_inputs,'ClientEmbeddings'),
                           (key_contexts,'load_client'),(key_contexts,'_read_private'),
                           (file_stages,'client_key_generation'),(file_stages,'client_encode_encrypt_input'),
                           (file_stages,'client_decrypt_decode'),(file_stages,'server_preprocess_model'),
                           (file_stages,'_records'),(file_stages,'_snapshot')):
            patch(owner,name,owner.__name__+'.'+name)
        sys.addaudithook(audit)
        yield
    except BaseException as error:primary=error;raise
    finally:
        active[0]=False
        with cleanup_signals():
            for owner,name,original in reversed(patches):
                try:setattr(owner,name,original)
                except BaseException as error:
                    if primary is not None:primary.add_note('server guard restore: '+repr(error))
                    elif failure is None:failure=error
                    elif not isinstance(error,Exception) and isinstance(failure,Exception):
                        error.add_note('earlier guard restore: '+repr(failure));failure=error
                    else:failure.add_note('additional guard restore: '+repr(error))
            if primary is None and failure is not None:raise failure
