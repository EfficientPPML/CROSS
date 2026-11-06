"""Client-decode entrypoint guards, not an OS sandbox."""
from contextlib import contextmanager
import importlib,sys
from cleanup import cleanup_signals

@contextmanager
def client_only(calls):
    if any(n.split('.')[0] in ('torch','transformers') for n in sys.modules):raise ValueError('fresh decode process cannot preload native model modules')
    from jaxite_word import CKKSContext
    modules=[importlib.import_module(n) for n in ('key_gen','jaxite_word.key_gen')]
    if {m.__name__ for m in modules}!={'key_gen','jaxite_word.key_gen'}:raise ValueError('both keygen aliases required')
    import file_stages,client_keys,client_inputs,key_contexts,request_lifecycle
    patches=[];active=[True];primary=None;failure=None
    def deny(label):
        def stop(*a,**k):calls.append(label);raise RuntimeError('forbidden stage5 action: '+label)
        return stop
    def patch(owner,name):
        old=getattr(owner,name);patches.append((owner,name,old));setattr(owner,name,deny(getattr(owner,'__name__','owner')+'.'+name))
    def audit(event,args):
        if active[0] and event=='import' and args and str(args[0]).split('.')[0] in ('torch','transformers'):deny('native model import')()
    try:
        for module in modules:
            for n in ('gen_pke_pair','gen_evaluation_key','gen_rotation_key','key_switch_gen'):patch(module,n)
        for n in ('encrypt','encrypt_slots','encrypt_slots_batch','program_initialization'):patch(CKKSContext,n)
        for owner,n in [(client_keys,'generate'),(client_inputs,'ClientEmbeddings'),(key_contexts,'load_server'),
            (file_stages,'client_key_generation'),(file_stages,'server_preprocess_model'),(file_stages,'client_encode_encrypt_input'),
            (file_stages,'server_encrypted_compute'),(file_stages,'_snapshot'),(request_lifecycle,'retire_completed')]:patch(owner,n)
        sys.addaudithook(audit);yield
    except BaseException as error:primary=error;raise
    finally:
        active[0]=False
        with cleanup_signals():
            for owner,n,old in reversed(patches):
                try:setattr(owner,n,old)
                except BaseException as error:
                    if primary is not None:primary.add_note('client guard restoration: '+repr(error))
                    elif failure is None:failure=error
                    elif not isinstance(error,Exception) and isinstance(failure,Exception):error.add_note('earlier guard restoration: '+repr(failure));failure=error
                    else:failure.add_note('additional restoration: '+repr(error))
            if primary is None and failure is not None:raise failure
