"""Acquire-before-load ciphertext consumers; context/key ownership is borrowed.

Private records retain originals independently of public owner mirrors. Returned
partial arrays are retained immediately. This component is not launch authority;
its guard/backend are source-selected by the separately denied actual factory.
"""
from cleanup_origin import protect_body
from qwen_finalizer import Finalizer
from q5_checkpoint_contract import canonical,parse,numerical_plan,checkpoint_row
from q5_checkpoint_arithmetic import state,same_state,metadata

_QUARANTINE=[]
_OWNERS={}

def _record(owner):
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
            arrays=[],value=None,state=None,entered=False,closed=False,failure=None)
    def _binding(self):
        r=_record(self)
        if any(a is not b for a,b in zip((self.np,self.runtime,self.ctx,self.payload,self.row_bytes,self.guard),r['fixed'])):
            raise ValueError('original ciphertext consumer binding changed')
        if self.closed is not r['closed']or self.value is not r['value']:raise ValueError('ciphertext consumer mirror changed')
        if r['closed']:raise ValueError('ciphertext consumer closed')
        if r['failure']is not None:raise r['failure']
        r['fixed'][5]()
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
            guard();array=backend.jax.numpy.asarray(payload,dtype=backend.jax.numpy.uint32);r['arrays'].append(array)
            params=dict(batch=1,num_elements=2,num_moduli=len(q),degree=131072,precision=32,moduli=list(q),degree_layout=(256,512))
            cache=getattr(ctx,'_param_cache',None)
            if cache is not None:params['ntt_ctx']=cache.get_sliced_ntt_q(row['level'])
            value=backend.polynomial.Polynomial.from_array(array,params);r['value']=value;self.value=value
            # Reconstruct authenticated wire metadata; no computed result is relabeled.
            value._ckks_scale=row['scale'];value._ckks_nsd=row['nsd'];value.validate()
            r['arrays'].append(value.polynomial);value.polynomial.block_until_ready();guard()
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
        r=_record(self);primary=protect_body(primary)
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
        r['closed']=True;self.closed=True;chosen=final.selected()
        if chosen is not None or r['failure']is not None:
            error=chosen if chosen is not None else r['failure'];r['failure']=error;_QUARANTINE.append((self,r,originals,error));raise error
        # Borrowed key/cache disposal belongs exclusively to the outer loader.
        self.value=None;self.payload=None;r['value']=None;r['state']=None;r['arrays'].clear();r['fixed']=None

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
        self.value=None;r['value']=None;r['reader']=None;r['consumer']=None;r['callbacks'].clear();r['fixed']=None
