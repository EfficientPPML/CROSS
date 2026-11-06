"""Acquire-before-load ciphertext consumers; context/key ownership is borrowed.

Private records retain originals independently of public owner mirrors. Returned
partial arrays are retained immediately. This component is not launch authority;
its guard/backend are source-selected by the separately denied actual factory.
"""
import weakref
from cleanup_origin import protect_body
from qwen_finalizer import Finalizer
from q5_checkpoint_contract import canonical,parse,numerical_plan,checkpoint_row
from q5_checkpoint_arithmetic import state,same_state,metadata

_QUARANTINE=[]
_OWNERS={}
_CLOSED=weakref.WeakSet()

def _record(owner):
    if owner in _CLOSED:raise ValueError('consumer already retired')
    record=_OWNERS.get(id(owner))
    if record is None or record['owner']is not owner:raise ValueError('original ciphertext owner record missing')
    return record

def _method(value):return(getattr(value,'__self__',None),getattr(value,'__func__',value))
def _same(a,b):return all(x is y for x,y in zip(_method(a),_method(b)))
def _callbacks(owner,names):return tuple((name,getattr(owner,name))for name in names)
def _unchanged(owner,callbacks):
    if any(not _same(getattr(owner,n),f)for n,f in callbacks):raise ValueError('original consumer callback changed')

class CiphertextOwner:
    def __init__(self,np,runtime,ctx,payload,row,*,check):
        checkpoint_row(row,row['index'])
        if not callable(check):raise TypeError('explicit original consumer guard required')
        self.np=np;self.runtime=runtime;self.ctx=ctx;self.payload=payload;self.row_bytes=canonical(row);self.guard=check
        self.value=None;self.closed=False
        _OWNERS[id(self)]=dict(owner=self,fixed=(np,runtime,ctx,payload,self.row_bytes,check),
            arrays=[],value=None,state=None,entered=False,closed=False,failure=None,cache=None,cache_getter=None,ntt_owner=None,ntt_close=None,ntt_check=None)
    def _binding(self):
        r=_record(self)
        if any(a is not b for a,b in zip((self.np,self.runtime,self.ctx,self.payload,self.row_bytes,self.guard),r['fixed'])):
            raise ValueError('original ciphertext consumer binding changed')
        if self.closed is not r['closed']or self.value is not r['value']:raise ValueError('ciphertext consumer mirror changed')
        if r['closed']:raise ValueError('ciphertext consumer closed')
        if r['failure']is not None:raise r['failure']
        r['fixed'][5]()
        if r['cache']is not None:
            if getattr(r['fixed'][2],'_param_cache',None)is not r['cache']or not _same(r['cache'].get_sliced_ntt_q,r['cache_getter']):raise ValueError('original borrowed NTT cache changed')
        if r['ntt_owner']is not None:
            _unchanged(r['ntt_owner'],(('close',r['ntt_close']),('check',r['ntt_check'])))
            r['ntt_check']()
        return r
    def load(self):
        r=self._binding()
        if r['entered']:raise ValueError('ciphertext consumer load is one-use')
        r['entered']=True;np,backend,ctx,payload,raw,guard=r['fixed'];row=parse(raw)
        try:
            profile=numerical_plan()['profile'];q=tuple(profile['q_towers'][:2*(row['level']+1)])
            if type(payload)is not np.ndarray or payload.shape!=(1,2,256,512,len(q))or payload.dtype!=np.dtype('uint32'):
                raise ValueError('exact admitted opaque ciphertext payload required')
            for limb,modulus in enumerate(q):
                if np.any(payload[...,limb]>=modulus):raise ValueError('opaque residue out of range')
            guard();cache=getattr(ctx,'_param_cache',None)
            if cache is not None:
                # Borrow the original server cache, retaining it before its lazy getter.
                r['cache']=cache;r['cache_getter']=cache.get_sliced_ntt_q
                ntt_context=r['cache_getter'](row['level']);guard()
                if getattr(ctx,'_param_cache',None)is not cache or not _same(cache.get_sliced_ntt_q,r['cache_getter']):raise ValueError('original cache changed during NTT acquisition')
            else:
                # Codec-only client contexts have no HEParameterCache. Acquire
                # an explicit public NTT owner BEFORE either original __init__.
                owner=PublicNTTOwner(backend,q,check=guard);r['ntt_owner']=owner
                r['ntt_close']=owner.close;r['ntt_check']=owner.check
                ntt_context=owner.load()
            if tuple(ntt_context.ff_ctx.moduli)!=q:raise ValueError('original NTT context Q prefix differs')
            array=backend.jax.numpy.asarray(payload,dtype=backend.jax.numpy.uint32);r['arrays'].append(array)
            shapes=dict(batch=1,num_elements=2,num_moduli=len(q),degree=131072,precision=32,degree_layout=(256,512))
            parameters=dict(moduli=list(q),ntt_ctx=ntt_context)
            value=backend.polynomial.Polynomial.from_array(array,shapes,parameters);r['value']=value;self.value=value
            # Own both returned arrays before any metadata/security callback.
            r['arrays'].append(value.polynomial);r['arrays'].append(value.moduli_array)
            if value.ntt_ctx is not ntt_context:raise ValueError('injected original NTT context was not retained')
            # Reconstruct authenticated wire metadata; no computed result is relabeled.
            value._ckks_scale=row['scale'];value._ckks_nsd=row['nsd'];value.validate()
            value.polynomial.block_until_ready();guard()
            backend.secure(ctx,[value]);guard()
            if metadata(value,profile)!={k:row[k]for k in('level','scale','nsd')}:raise ValueError('materialized actual metadata differs')
            r['state']=state(value,profile);self.check();return value
        except BaseException as error:
            r['failure']=protect_body(error);_QUARANTINE.append((self,r,error));raise
    def check(self):
        r=self._binding()
        if r['state']is None or r['value']is not r['state'][0]:raise ValueError('original complete ciphertext consumer required')
        same_state(r['state'],numerical_plan()['profile']);return r['value']
    def close(self,primary=None):
        primary=protect_body(primary)
        if self in _CLOSED:
            if primary is not None:raise primary
            return
        r=_record(self)
        if r['closed']:
            if primary is not None:raise primary
            if r['failure']is not None:raise r['failure']
            return
        originals=tuple(r['arrays']);final=Finalizer({},primary)
        final.attempt('original ciphertext binding before retirement',self._binding)
        seen=set()
        for array in originals:
            if id(array)in seen:continue
            seen.add(id(array));final.attempt('original ciphertext array barrier',lambda array=array:array.block_until_ready())
        if r['ntt_close']is not None:
            final.attempt('original owned client NTT retirement',lambda:r['ntt_close'](primary))
        r['closed']=True;self.closed=True;chosen=final.selected()
        if chosen is not None or r['failure']is not None:
            error=chosen if chosen is not None else r['failure'];r['failure']=error;_QUARANTINE.append((self,r,originals,error));raise error
        # Borrowed key/cache disposal belongs exclusively to the outer loader.
        self.value=None;self.payload=None;self.ctx=None;self.runtime=None;self.guard=None;self.np=None;self.row_bytes=None
        r.clear();_OWNERS.pop(id(self));_CLOSED.add(self)

class InputOwner:
    def __init__(self,np,backend,ctx,original_input,*,check):
        if not callable(check):raise TypeError('explicit input guard required')
        self.np=np;self.backend=backend;self.ctx=ctx;self.raw=canonical(original_input);self.guard=check
        self.reader=None;self.consumer=None;self.value=None;self.closed=False
        _OWNERS[id(self)]=dict(owner=self,fixed=(np,backend,ctx,self.raw,check),reader=None,consumer=None,
            callbacks=[],value=None,entered=False,closed=False,failure=None)
    def _binding(self,*,readers=True):
        r=_record(self)
        if any(a is not b for a,b in zip((self.np,self.backend,self.ctx,self.raw,self.guard),r['fixed'])):
            raise ValueError('original input owner binding changed')
        if self.closed is not r['closed']or self.reader is not r['reader']or self.consumer is not r['consumer']or self.value is not r['value']:
            raise ValueError('original input consumer mirror changed')
        if r['closed']:raise ValueError('input owner closed')
        if r['failure']is not None:raise r['failure']
        for owner,callbacks in r['callbacks']:_unchanged(owner,callbacks)
        r['fixed'][4]()
        if readers:
            for owner,callbacks in r['callbacks']:
                for name,fn in callbacks:
                    if name=='check':fn()
        return r
    def load(self):
        from q5_checkpoint_transport import ArtifactReader
        r=self._binding(readers=False)
        if r['entered']:raise ValueError('input owner is one-use')
        r['entered']=True;np,backend,ctx,raw,guard=r['fixed'];original=parse(raw)
        try:
            reader=ArtifactReader(np,canonical(original['root']));r['reader']=reader;self.reader=reader
            callbacks=_callbacks(reader,('close','check','read'));r['callbacks'].append((reader,callbacks))
            payload=dict(callbacks)['read'](original['artifact'],retain_payload=True)
            consumer=CiphertextOwner(np,backend,ctx,payload,original['artifact']['metadata'],check=guard)
            r['consumer']=consumer;self.consumer=consumer
            callbacks=_callbacks(consumer,('close','check','load'));r['callbacks'].append((consumer,callbacks))
            value=dict(callbacks)['load']();r['value']=value;self.value=value;self.check();return value
        except BaseException as error:r['failure']=protect_body(error);_QUARANTINE.append((self,r,error));raise
    def check(self):
        r=self._binding()
        if r['consumer']is None or r['value']is not r['consumer'].value:raise ValueError('original complete input consumer required')
        return r['value']
    def close(self,primary=None):
        if self in _CLOSED:
            if primary is not None:raise primary
            return
        r=_record(self)
        if r['closed']:
            if primary is not None:raise primary
            if r['failure']is not None:raise r['failure']
            return
        callbacks=tuple(r['callbacks']);final=Finalizer({},primary)
        final.attempt('original input owner binding',lambda:self._binding(readers=False))
        for owner,methods in reversed(callbacks):
            close=dict(methods)['close'];final.attempt('original input consumer close',lambda close=close:close(primary))
            final.attempt('post-close original consumer callbacks',lambda owner=owner,methods=methods:_unchanged(owner,methods))
            final.attempt('post-close input bindings',lambda:self._binding(readers=False))
        r['closed']=True;self.closed=True;error=final.selected()
        if error is not None or r['failure']is not None:
            error=error if error is not None else r['failure'];r['failure']=error;_QUARANTINE.append((self,r,callbacks,error));raise error
        self.value=None;self.reader=None;self.consumer=None;self.np=None;self.backend=None;self.ctx=None;self.raw=None;self.guard=None
        r.clear();_OWNERS.pop(id(self));_CLOSED.add(self)


class PublicNTTOwner:
    """Original Barrett public tables, acquired before initialization; no keys.

    No shared-cache fallback is invoked. Original objects are allocated first
    and their original __init__ functions are then called with the same public
    parameters as Polynomial's default Barrett path. Partial reachable tables
    are retained on error. No claim is made about invisible constructor locals.
    """
    def __init__(self,backend,moduli,*,check):
        self.backend=backend;self.moduli=tuple(moduli);self.guard=check;self.value=None;self.closed=False
        _OWNERS[id(self)]=dict(owner=self,fixed=(backend,self.moduli,check),objects=[],arrays=[],ntt=None,
            classes=(backend.finite_field.BarrettContext,backend.ntt_mm.NTTCiphertextBarrettContext),
            initializers=(backend.finite_field.BarrettContext.__init__,backend.ntt_mm.NTTCiphertextBarrettContext.__init__),
            entered=False,closed=False,failure=None,control=None)
    def _binding(self):
        r=_record(self);backend,q,guard=r['fixed']
        if self.backend is not backend or self.moduli is not q or self.guard is not guard or self.value is not r['ntt']or self.closed is not r['closed']:raise ValueError('original NTT owner binding changed')
        if r['closed']:raise ValueError('NTT owner retired')
        if r['failure']is not None:raise r['failure']
        if backend.finite_field.BarrettContext is not r['classes'][0]or backend.ntt_mm.NTTCiphertextBarrettContext is not r['classes'][1]:raise ValueError('source-selected public NTT classes changed')
        if any(cls.__init__ is not original for cls,original in zip(r['classes'],r['initializers'])):raise ValueError('original NTT initializer changed')
        guard();return r
    def _retain_arrays(self,r):
        seen=set()
        def visit(value):
            if id(value)in seen:return
            seen.add(id(value))
            wait=getattr(value,'block_until_ready',None)
            if callable(wait):
                if all(value is not old for old in r['arrays']):r['arrays'].append(value)
            elif type(value)is dict:
                for child in value.values():visit(child)
            elif type(value)in(tuple,list):
                for child in value:visit(child)
        for owner in r['objects']:visit(vars(owner))
    def load(self):
        r=self._binding()
        if r['entered']:raise ValueError('public NTT owner one-use')
        r['entered']=True;backend,q,guard=r['fixed']
        try:
            ff=object.__new__(r['classes'][0]);r['objects'].append(ff)
            r['initializers'][0](ff,moduli=list(q));self._retain_arrays(r);guard()
            if backend.ntt_mm.ntt_ciphertext_context_for(ff)is not r['classes'][1]:raise ValueError('original Barrett NTT dispatch differs')
            ntt=object.__new__(r['classes'][1]);r['objects'].append(ntt);r['ntt']=ntt;self.value=ntt
            r['initializers'][1](ntt,moduli=list(q),parameters=dict(r=256,c=512,finite_field_context=ff))
            self._retain_arrays(r)
            for array in tuple(r['arrays']):array.block_until_ready()
            from q5_checkpoint_arithmetic import typed
            r['control']=tuple(typed(vars(owner))for owner in r['objects']);self.check();return ntt
        except BaseException as error:
            r['failure']=protect_body(error)
            try:self._retain_arrays(r)
            except BaseException as secondary:error.add_note('partial NTT array retention: '+type(secondary).__name__)
            _QUARANTINE.append((self,r,error));raise
    def check(self):
        r=self._binding()
        from q5_checkpoint_arithmetic import typed
        if r['control']is None or tuple(typed(vars(owner))for owner in r['objects'])!=r['control']:raise ValueError('original public NTT controls changed')
        if r['ntt'].ff_ctx is not r['objects'][0]or tuple(r['ntt'].ff_ctx.moduli)!=self.moduli:raise ValueError('original public NTT field owner differs')
        return r['ntt']
    def close(self,primary=None):
        if self in _CLOSED:
            if primary is not None:raise primary
            return
        r=_record(self)
        if r['closed']:
            if primary is not None:raise primary
            if r['failure']is not None:raise r['failure']
            return
        final=Finalizer({},primary);final.attempt('original public NTT owner binding',self._binding)
        final.attempt('retain partial public NTT arrays',lambda:self._retain_arrays(r))
        arrays=tuple(r['arrays'])
        for array in arrays:final.attempt('original public NTT table barrier',lambda array=array:array.block_until_ready())
        r['closed']=True;self.closed=True;error=final.selected()
        if error is not None or r['failure']is not None:
            error=error if error is not None else r['failure'];r['failure']=error;_QUARANTINE.append((self,r,arrays,error));raise error
        self.value=None;self.backend=None;self.moduli=None;self.guard=None;r.clear();_OWNERS.pop(id(self));_CLOSED.add(self)
