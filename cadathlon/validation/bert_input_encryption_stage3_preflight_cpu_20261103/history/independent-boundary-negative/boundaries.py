"""Guard the client encryption process: embeddings and encryption only.

These Python entrypoint checks accompany the exact source inventory. They are
not an operating-system sandbox and do not claim numerical ciphertext accuracy.
"""
from contextlib import contextmanager
import importlib
from cleanup import cleanup_signals

KEY_METHODS=('gen_pke_pair','gen_evaluation_key','gen_rotation_key','key_switch_gen')
DECODE_METHODS=('decrypt','decrypt_slots','decrypt_slots_batch','decode')
MODEL_CLASSES=('BertModel','BertEncoder','BertLayer','BertPooler','BertForSequenceClassification',
               'BertSelfAttention','BertAttention','BertIntermediate','BertOutput')


@contextmanager
def encryption_only(calls):
    from jaxite_word import CKKSContext
    model=importlib.import_module('transformers.models.bert.modeling_bert')
    modules=[importlib.import_module(n) for n in ('key_gen','jaxite_word.key_gen')]
    if {m.__name__ for m in modules}!={'key_gen','jaxite_word.key_gen'}:
        raise ValueError('both actual key-generator aliases required')
    patches=[];primary=None;failure=None
    def patch(owner,name,label):
        original=getattr(owner,name)
        def denied(*a,**kw):
            calls.append(label)
            raise RuntimeError('forbidden client-encryption action: '+label)
        patches.append((owner,name,original));setattr(owner,name,denied)
    try:
        for module in modules:
            for name in KEY_METHODS:patch(module,name,module.__name__+'.'+name)
        for name in DECODE_METHODS:patch(CKKSContext,name,'CKKSContext.'+name)
        for name in MODEL_CLASSES:patch(getattr(model,name),'__init__',name+'.__init__')
        yield
    except BaseException as error:primary=error;raise
    finally:
        with cleanup_signals():
            for owner,name,original in reversed(patches):
                try:setattr(owner,name,original)
                except BaseException as error:
                    if primary is not None:primary.add_note('client encryption guard restore: '+repr(error))
                    elif failure is None:failure=error
                    else:failure.add_note('additional guard restore: '+repr(error))
            if primary is None and failure is not None:raise failure
