"""Copied phase/metadata/cleanup utilities; no context construction at import."""

from pathlib import Path

import gc,hashlib,math,sys

import numpy as np

SLOTS=65536;GATE=1e-6

CODEC_METHODS=('encrypt','encrypt_slots','encrypt_slots_batch','decrypt','decrypt_slots','decrypt_slots_batch','decode')

KEYGEN_METHODS=('gen_pke_pair','gen_evaluation_key','gen_rotation_key','key_switch_gen')

def metrics(actual, expected):
    actual,expected=np.asarray(actual),np.asarray(expected)
    if actual.shape!=(SLOTS,) or expected.shape!=(SLOTS,) or actual.dtype.kind!='f' or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return {'passed':False,'finite_and_shape':False,'slots_checked':SLOTS,'absolute_error_gate':GATE}
    with np.errstate(over='ignore',invalid='ignore'):
        actual=actual.astype(np.float64);expected=expected.astype(np.float64)
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return {'passed':False,'finite_and_shape':False,'finite_after_binary64_conversion':False,
            'slots_checked':SLOTS,'absolute_error_gate':GATE}
    with np.errstate(over='ignore',invalid='ignore'):
        diff=np.abs(actual-expected)
    error=float(np.max(diff))
    if not math.isfinite(error):return {'passed':False,'finite_and_shape':True,'finite_errors':False,'slots_checked':SLOTS,'absolute_error_gate':GATE}
    return {'passed':error<GATE,'finite_and_shape':True,'slots_checked':SLOTS,
        'absolute_error_gate':GATE,'max_abs_error':error,
        'rms_error':float(np.max(diff)*np.sqrt(np.mean((diff/np.max(diff))**2))) if np.max(diff)>0 else 0.,
        'max_relative_error_with_1e_12_floor':(lambda x:x if math.isfinite(x) else None)(float(np.max(diff/np.maximum(np.abs(expected),1e-12)))),
        'worst_slot_index':int(np.argmax(diff)),'per_4096_slot_block_max_abs_error':diff.reshape(16,4096).max(axis=1).tolist(),
        'boundary_interior_probe_errors':diff[:13].tolist()}

def validate_key_directory(value):
    candidate=Path(value)
    if candidate.is_symlink():raise ValueError('owned key directory cannot be a symlink')
    path=candidate.resolve(strict=True)
    if not path.is_dir() or path==Path('/dev/shm') or not path.is_relative_to('/dev/shm'):
        raise ValueError('an owned child directory under /dev/shm is required')
    return path

def install_generator_guards(registry, restorations):
    names=('key_gen','jaxite_word.key_gen')
    if any(name not in registry for name in names):
        raise AssertionError('both historical and packaged key_gen aliases must be loaded')
    modules={registry[name] for name in names}
    def forbidden(*args,**kwargs):raise AssertionError('runtime key generation forbidden')
    for module in modules:
        for name in KEYGEN_METHODS:
            original=getattr(module,name)
            restorations.append(lambda m=module,n=name,f=original:setattr(m,n,f))
            setattr(module,name,forbidden)

def install_codec_guards(context_class, server_getter, phase_getter, restorations):
    for name in CODEC_METHODS:
        original=getattr(context_class,name)
        def guarded(self,*args,_original=original,_name=name,**kwargs):
            if phase_getter()=='server' or self is server_getter():
                raise AssertionError('codec forbidden in keyless evaluation: '+_name)
            return _original(self,*args,**kwargs)
        restorations.append(lambda n=name,f=original:setattr(context_class,n,f))
        setattr(context_class,name,guarded)

def ciphertext_fingerprint(value):
    array=np.asarray(value.polynomial)
    return dict(payload_sha256=hashlib.sha256(array.tobytes()).hexdigest(),payload_shape=list(array.shape),
        payload_dtype=str(array.dtype),degree=value.degree,batch=value.batch,num_elements=value.num_elements,
        num_moduli=value.num_moduli,moduli=list(value.moduli),degree_layout=list(value.degree_layout),
        precision=value.precision,scale=value._ckks_scale,nsd=value._ckks_nsd,
        ntt_backend=type(value.ntt_ctx).__module__+'.'+type(value.ntt_ctx).__name__,
        field_backend=type(value.ntt_ctx.ff_ctx).__module__+'.'+type(value.ntt_ctx.ff_ctx).__name__)

def release_all(contexts, owner, key_store, active_error, callbacks=None):
    """Attempt all releases, preserving a numerical/cancellation failure."""
    import jax
    from jaxite_word import ckks_ctx
    errors=[]
    callbacks=[] if callbacks is None else callbacks
    while callbacks:
        release=callbacks.pop(0)
        try:release()
        except BaseException as error:errors.append(error)
        release=None
    if owner is not None:
        try:owner.close()
        except BaseException as error:errors.append(error)
    while contexts:
        context=contexts.pop()
        if context is not None:
            try:ckks_ctx.evict_context_caches(context)
            except BaseException as error:errors.append(error)
        context=None
    for release in (jax.clear_caches,gc.collect):
        try:release()
        except BaseException as error:errors.append(error)
    if key_store is not None:
        try:key_store.close()
        except BaseException as error:errors.append(error)
    if errors:
        if active_error is not None:
            for error in errors:active_error.add_note('Cleanup also failed: '+repr(error))
        else:
            primary=next((e for e in errors if not isinstance(e,Exception)),errors[0])
            for error in errors:
                if error is not primary:primary.add_note('Other cleanup failure: '+repr(error))
            raise primary
    return not errors
