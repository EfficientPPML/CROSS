"""Exact representation checks; no client codec or numerical oracle."""
import hashlib,math,time
import numpy as np
from parity_protocol import INDICES,LEVELS,MAX_BYTES,MAX_ENTRIES

def metadata(value):
    scale=getattr(value,'_ckks_scale',None);nsd=getattr(value,'_ckks_nsd',None)
    if type(scale)is not float or not math.isfinite(scale) or scale<=0 or type(nsd)is not int or nsd!=1:
        raise ValueError('typed finite positive scale and explicit NSD1 required')
    payload=np.asarray(value.polynomial)
    if payload.dtype!=np.uint32:raise ValueError('exact uint32 ciphertext payload required')
    return dict(shape=list(payload.shape),dtype=str(payload.dtype),moduli=[int(q) for q in value.moduli],
                degree_layout=list(value.degree_layout),precision=value.precision,scale=scale,scale_hex=scale.hex(),nsd=nsd)

def fingerprint(value):
    record=metadata(value);payload=np.ascontiguousarray(np.asarray(value.polynomial))
    record['payload_sha256']=hashlib.sha256(memoryview(payload).cast('B')).hexdigest()
    return record

def check_level(value,ctx,level):
    record=metadata(value);q=list(ctx._param_cache.q_moduli_at_level(level))
    if (record['shape']!=[1,2,*ctx.degree_layout,len(q)] or record['moduli']!=q
            or record['degree_layout']!=list(ctx.degree_layout) or type(record['precision'])is not int or record['precision']!=32):
        raise ValueError('actual ordered ring/shape/layout/precision differs')
    return record

def compare(left,right,source,ctx,level):
    a=check_level(left,ctx,level);b=check_level(right,ctx,level);original=check_level(source,ctx,level)
    x=np.asarray(left.polynomial);y=np.asarray(right.polynomial)
    mismatch=int(np.count_nonzero(x!=y))
    result=dict(passed=a==b==original and mismatch==0,metadata_equal=a==b==original,
                mismatched_words=mismatch,compared_words=int(x.size),left=fingerprint(left),right=fingerprint(right))
    return result

def verify_reduction(original,lower,ctx):
    before=check_level(original,ctx,49);after=check_level(lower,ctx,25)
    count=len(after['moduli'])
    same=bool(np.array_equal(np.asarray(original.polynomial)[...,:count],np.asarray(lower.polynomial)))
    result=dict(passed=same and after['moduli']==before['moduli'][:count]
        and after['scale_hex']==before['scale_hex'] and after['nsd']==before['nsd'],
        payload_prefix_equal=same,ordered_q_prefix=after['moduli']==before['moduli'][:count],
        input=before,output=after,method='actual ctx.he_level_reduce[49,25].level_reduce; no rescale or scale relabel')
    if not result['passed']:raise ValueError('legitimate level-reduction prefix/metadata differs')
    return result

def exercise(ctx,value,baseline,prepared,*,progress,security):
    initial=fingerprint(value);security(baseline,value);security(prepared,value)
    result=dict(passed=False,comparisons=[],levels=[],input=initial)
    current=value
    for level in LEVELS:
        if level==25:
            current=ctx.he_level_reduce[49,25].level_reduce(value)
            result['level_reduction']=verify_reduction(value,current,ctx)
        check_level(current,ctx,level);security(prepared,current)
        source=fingerprint(current);stats_before=dict(prepared.rotation.stats)
        for sequence,index in enumerate(INDICES):
            started=time.monotonic()
            left=baseline.rotate(current,index);right=prepared.rotate(current,index)
            security(baseline,left);security(prepared,right)
            row=compare(left,right,current,ctx,level)
            row.update(level=level,index=index,sequence=sequence,elapsed_pair_s=time.monotonic()-started,
                prepared_entries=prepared.rotation.retained_entries,prepared_bytes=prepared.rotation.retained_bytes)
            result['comparisons'].append(row)
            if (row['prepared_entries']>MAX_ENTRIES or row['prepared_bytes']>MAX_BYTES):raise ValueError('prepared budget exceeded')
            progress(row)
            if not row['passed']:raise ValueError('genuine rotation payload/metadata parity failed')
            if fingerprint(current)!=source or fingerprint(value)!=initial:raise ValueError('original ciphertext changed')
            del left,right
        stats=dict(prepared.rotation.stats)
        delta={k:stats.get(k,0)-stats_before.get(k,0) for k in set(stats)|set(stats_before)}
        level_record=dict(level=level,stats_delta=delta,retained_entries=prepared.rotation.retained_entries,
                          retained_bytes=prepared.rotation.retained_bytes,input=fingerprint(current))
        result['levels'].append(level_record)
        if delta.get('hits',0)!=4 or delta.get('published',0)!=2 or level_record['retained_entries']!=2:
            raise ValueError('fixed prepared-cache hit/publication schedule differs')
    result.update(passed=True,actual_rotation_calls=2*len(result['comparisons']),input_unchanged=fingerprint(value)==initial)
    return result

class OwnerCleanup:
    """Capture the original owners immediately; never lose retirement on swaps."""
    def __init__(self):self.records=[];self.completed=False
    def register(self,owner):
        rotation=owner.rotation
        self.records.append((owner,rotation,owner._clear_working_caches,rotation.clear))
    def close(self,primary=None):
        failure=None
        def attempt(fn):
            nonlocal failure
            try:fn()
            except BaseException as error:
                if primary is not None:primary.add_note('arithmetic retirement: '+repr(error))
                elif failure is None:failure=error
                elif not isinstance(error,Exception) and isinstance(failure,Exception):
                    error.add_note('earlier cleanup: '+repr(failure));failure=error
                else:failure.add_note('additional cleanup: '+repr(error))
        def identity(method):return getattr(method,'__self__',None),getattr(method,'__func__',method)
        for owner,rotation,clear,rotation_clear in self.records:
            valid=[False]
            def check():
                if owner.rotation is not rotation or identity(owner._clear_working_caches)!=identity(clear) or identity(rotation.clear)!=identity(rotation_clear):
                    raise ValueError('original arithmetic cleanup owner was replaced')
                valid[0]=True
            attempt(check)
            # Always retire the original rotation. If public bindings changed,
            #avoid a method that could dispatch through substituted attributes;
            #raise into loader body so every original backing is quarantined.
            if valid[0]:attempt(clear)
            attempt(lambda:rotation_clear(executables=True))
        self.completed=primary is None and failure is None
        if primary is None and failure is not None:raise failure

def close_owners(baseline,prepared,primary=None):
    """Test convenience; production registers each owner immediately at birth."""
    cleanup=OwnerCleanup()
    for owner in (baseline,prepared):
        if owner is not None:cleanup.register(owner)
    cleanup.close(primary)
