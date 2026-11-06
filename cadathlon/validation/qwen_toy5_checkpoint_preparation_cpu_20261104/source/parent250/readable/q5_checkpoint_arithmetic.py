"""Instrument the unchanged Q5 encrypted graph; no client/clear-reference path.

Only bookkeeping subclasses are added. Every arithmetic/refresh body delegates
to the frozen199 implementation. All returned arrays are synchronized before
the save callback. A body failure quarantines local owners until process exit
and must propagate through the outer key loader, which retains public backings.
"""
from contextlib import contextmanager
from cleanup_origin import mark,select,protect_body
from collections.abc import Mapping
import hashlib,math,weakref
from q5_checkpoint_contract import numerical_plan,canonical,checkpoint_row
from q5_actual_metadata_replay import weighted_scale,product_scale,check_refresh

_FAILURE_QUARANTINE=[]

def typed(x):
    if type(x).__module__.startswith('numpy') and hasattr(x,'tobytes') and hasattr(x,'dtype'):return(type(x),id(x),x.shape,x.dtype.str,hashlib.sha256(x.tobytes()).hexdigest())
    if type(x)in(list,tuple):return(type(x),tuple(typed(v)for v in x))
    if type(x)is dict:return(dict,tuple(sorted((k,typed(v))for k,v in x.items())))
    if type(x)in(str,int,float,bool,type(None)):return(type(x),x)
    return(type(x),id(x))

def accessor(x,name):
    f=getattr(x,name);return(getattr(f,'__self__',None),getattr(f,'__func__',f))

def metadata(value,profile):
    for name,want in(('batch',1),('num_elements',2),('degree',131072),('precision',32),('r',256),('c',512),('_ckks_nsd',1)):
        if type(getattr(value,name,None))is not int or getattr(value,name)!=want:raise ValueError('exact ciphertext '+name+' required')
    n=value.num_moduli
    if type(n)is not int or n<2 or n>100 or n%2:raise ValueError('exact composite modulus count required')
    if typed(value.degree_layout)!=typed((256,512)):raise ValueError('exact ciphertext degree layout required')
    moduli=tuple(value.moduli)
    if any(type(q)is not int for q in moduli)or moduli!=tuple(profile['q_towers'][:n]):raise ValueError('ordered original Q prefix required')
    payload=value.polynomial
    if tuple(payload.shape)!=(1,2,256,512,n)or str(payload.dtype)!='uint32':raise ValueError('exact ciphertext payload layout required')
    scale=getattr(value,'_ckks_scale',None)
    if type(scale)is not float or not math.isfinite(scale)or scale<=0:raise ValueError('finite positive actual scale required')
    return dict(level=n//2-1,scale=scale,nsd=1)

def state(value,profile):
    meta=metadata(value,profile)
    # Original array objects stay strongly held during callbacks, independently
    #of mutable wrappers. Public-bound annotations are intentionally excluded:
    #the original bounded graph legitimately updates those between stages.
    return(value,value.polynomial,getattr(value,'moduli_array',None),getattr(value,'ntt_ctx',None),
        typed(meta),typed(value.moduli),typed(value.degree_layout),
        typed({k:getattr(value,k)for k in('batch','num_elements','num_moduli','degree','precision','r','c')}))

def same_state(old,profile):
    new=state(old[0],profile)
    if any(new[i]is not old[i]for i in(0,1,2,3))or new[4:]!=old[4:]:raise ValueError('retained ciphertext changed during callback')

def working_snapshot(roots):
    """Hold exact live executable/control/constant refs across the save callback.

    These caches legitimately change during graph work, so they are compared
    only immediately around callbacks. Payloads are never copied or decoded.
    """
    seen={};held=[]
    def visit(x):
        if x is None or type(x)in(str,int,float,bool):return typed(x)
        key=id(x)
        if key in seen:return('alias',seen[key])
        seen[key]=len(seen);held.append(x)
        if isinstance(x,Mapping):return(type(x),key,tuple((visit(k),visit(v))for k,v in x.items()))
        if type(x)in(tuple,list):return(type(x),key,tuple(visit(v)for v in x))
        if hasattr(x,'shape')and hasattr(x,'dtype'):return(type(x),key,tuple(x.shape),str(x.dtype))
        fields=getattr(x,'__dict__',None)
        if isinstance(fields,dict):
            return(type(x),key,tuple((k,visit(v))for k,v in sorted(fields.items())if k not in('ctx','_param_cache')))
        return(type(x),key)
    frozen=tuple(visit(x)for x in roots)
    return tuple(held),frozen

class Ledger:
    def __init__(self,checkpoint):
        if not callable(checkpoint):raise TypeError('checkpoint callback required')
        self.plan=numerical_plan();self.profile=self.plan['profile'];self.callback=checkpoint
        self.trace=[];self.full_count=0;self.refresh_observations=[];self.stack=[];self.counter={};self.live={};self.counts=dict(add=0,weighted=0,multiply=0)
        self.frames=[];self.control=None;self.control_state=None;self.refresh_scales=[];self.callback_held=();self.working=None;self.callback_controls=()
    def track(self,value):
        metadata(value,self.profile);key=id(value)
        if key not in self.live:
            self.live[key]=weakref.ref(value,lambda ref,key=key:self.live.pop(key,None))
        return value
    def snapshots(self):return tuple(state(v,self.profile)for ref in tuple(self.live.values())if(v:=ref())is not None)
    def guard(self):
        if self.control is not None and self.control()!=self.control_state:raise ValueError('borrowed graph/context/helper binding changed')
    def note(self,kind,label,value,**extra):
        self.guard();self.track(value);trace_index=self.full_count
        if trace_index>=216:raise ValueError('unexpected additional original trace event')
        expected=self.plan['full_trace'][trace_index];meta=metadata(value,self.profile)
        for key,actual in dict(index=trace_index,kind=kind,label=label,stack=list(self.stack),level=meta['level'],nsd=1,**extra).items():
            if typed(expected.get(key))!=typed(actual):raise ValueError('original trace order/contract differs: '+key)
        self.full_count+=1
        selected=[r for r in self.plan['checkpoints']if r['clear_trace_index']==trace_index]
        if not selected:return value
        index=len(self.trace)
        if len(selected)!=1 or selected[0]['index']!=index:raise ValueError('selected47 order differs')
        row=dict(index=index,trace_index=trace_index,kind=kind,label=label,stack=list(self.stack),**meta,**extra)
        checkpoint_row(row,index)
        value.polynomial.block_until_ready()
        snapshots=self.snapshots();annotations=tuple((old[0],typed({k:v for k,v in vars(old[0]).items()if k.startswith('_public_')}))for old in snapshots)
        row_bytes=canonical(row);self.callback_held=snapshots
        before_controls=working_snapshot(self.working())if self.working is not None else((),())
        self.callback_controls=before_controls[0]
        self.callback(index,value,row)
        if canonical(row)!=row_bytes:raise ValueError('checkpoint callback row changed')
        self.guard()
        if self.working is not None and working_snapshot(self.working())[1]!=before_controls[1]:raise ValueError('live compiled control/cache changed during callback')
        for old in snapshots:same_state(old,self.profile)
        for v,before in annotations:
            if typed({k:x for k,x in vars(v).items()if k.startswith('_public_')})!=before:raise ValueError('public bound annotation changed during callback')
        self.trace.append(row);self.callback_held=();self.callback_controls=();return value
    @contextmanager
    def frame(self,kind,values):
        self.guard()
        for value in values:self.track(value)
        frame=dict(kind=kind,prepared=None);self.frames.append(frame)
        try:yield frame
        finally:self.frames.pop()
    def validate_output(self,result,level,scale):
        self.track(result);actual=metadata(result,self.profile)
        if actual!=dict(level=level,scale=scale,nsd=1):raise ValueError('primitive actual metadata recurrence differs')
        self.guard();return result
    def next_label(self,kind):
        i=self.counter.get(kind,0);self.counter[kind]=i+1;return str(i)

class ArithmeticInstrumentation:
    def _ensure_capacity(self,values,required):
        values=tuple(values)
        for value in values:self.ledger.track(value)
        prepared=super()._ensure_capacity(values,required)
        for value in prepared:self.ledger.track(value)
        if self.ledger.frames:self.ledger.frames[-1]['prepared']=tuple(metadata(v,self.ledger.profile)for v in prepared)
        return prepared
    def _lower(self,value,level):
        self.ledger.track(value);scale=metadata(value,self.ledger.profile)['scale']
        result=super()._lower(value,level)
        return self.ledger.validate_output(result,level,scale)
    def add(self,left,right,*,subtract=False):
        with self.ledger.frame('add',(left,right)):
            a,b=(metadata(v,self.ledger.profile)for v in(left,right));result=super().add(left,right,subtract=subtract)
            self.ledger.counts['add']+=1
            return self.ledger.validate_output(result,min(a['level'],b['level']),a['scale'])
    def rotate(self,value,shift):
        with self.ledger.frame('rotate',(value,)):
            before=metadata(value,self.ledger.profile);result=super().rotate(value,shift)
            return self.ledger.validate_output(result,before['level'],before['scale'])
    def weighted_sum(self,values,weights,*,bias=0.):
        values=tuple(values)
        with self.ledger.frame('weighted',values)as frame:
            result=super().weighted_sum(values,weights,bias=bias);prepared=frame['prepared']
            if prepared is None:raise ValueError('production weighted preparation was not observed')
            level=min(v['level']for v in prepared);scale=weighted_scale(prepared[0]['scale'],level)
            self.ledger.counts['weighted']+=1
            return self.ledger.validate_output(result,level-1,scale)
    def multiply(self,left,right):
        with self.ledger.frame('multiply',(left,right))as frame:
            result=super().multiply(left,right);prepared=frame['prepared']
            if prepared is None or len(prepared)!=2:raise ValueError('production product preparation was not observed')
            level=min(v['level']for v in prepared);scale=product_scale(prepared[0]['scale'],prepared[1]['scale'],level)
            self.ledger.counts['multiply']+=1
            return self.ledger.validate_output(result,level-1,scale)
    def checkpoint(self,value,label):
        result=super().checkpoint(value,label)
        return self.ledger.note('phase',label,result)
    def evaluate(self,value,approximation,*,output_weights=1.):
        label=self.ledger.next_label('polynomial')+':'+approximation.name
        self.ledger.note('polynomial_before',label,value,domain=list(approximation.domain));self.ledger.stack.append('polynomial:'+label)
        try:result=super().evaluate(value,approximation,output_weights=output_weights)
        finally:self.ledger.stack.pop()
        return self.ledger.note('polynomial_after',label,result)

class RefreshInstrumentation:
    def install_alignment_observer(self):
        # Install once before control capture. Every original weighted body is
        #called exactly once with the original values/weights/bias. No return
        #metadata is assigned and no primitive call is inserted.
        original=self.evaluator.weighted_sum
        self._weighted_original=original;self._refresh_frame=None
        def observed(values,weights,*,bias=0.):
            self.ledger.guard();values=tuple(values);weights=tuple(weights)
            frame=self._refresh_frame
            if frame is None or len(values)!=1 or len(weights)!=1 or type(weights[0])is not float or type(bias)is not float:raise ValueError('original scalar refresh weighted call required')
            before=metadata(values[0],self.ledger.profile)
            self.ledger.track(values[0]);result=original(values,weights,bias=bias);self.ledger.track(result)
            after=metadata(result,self.ledger.profile)
            if after!=dict(level=before['level']-1,scale=weighted_scale(before['scale'],before['level']),nsd=1):raise ValueError('actual refresh weighted recurrence differs')
            frame.append(dict(before=before,after=after,weight=weights[0],bias=bias))
            self.ledger.guard();return result
        self.evaluator.weighted_sum=observed
        self._weighted_observed=observed
    def refresh(self,value,*,public_bound=None):
        self.ledger.guard()
        if self._refresh_frame is not None:raise ValueError('nested refresh frame')
        label=self.ledger.next_label('refresh');before=metadata(value,self.ledger.profile)
        self.ledger.note('refresh_before',label,value,public_bound=public_bound)
        calls=[];self._refresh_frame=calls
        try:result=super().refresh(value,public_bound=public_bound)
        finally:self._refresh_frame=None
        observation=dict(index=int(label),public_bound=public_bound,before=before,weighted_calls=calls,after=metadata(result,self.ledger.profile))
        check_refresh(observation,int(label));self.ledger.refresh_observations.append(observation)
        return self.ledger.note('refresh_after',label,result,public_bound=public_bound)

class GraphInstrumentation:
    def _event(self,kind,function,*args,**kwargs):
        label=self.arithmetic.ledger.next_label(kind);ledger=self.arithmetic.ledger
        ledger.note(kind+'_before',label,args[0]);ledger.stack.append(kind+':'+label)
        try:result=function(*args,**kwargs)
        finally:ledger.stack.pop()
        return ledger.note(kind+'_after',label,result)
    def linear(self,*args,**kwargs):return self._event('linear',super().linear,*args,**kwargs)
    def normalize(self,*args,**kwargs):return self._event('normalize',super().normalize,*args,**kwargs)
    def attention(self,*args,**kwargs):return self._event('attention',super().attention,*args,**kwargs)
    def normalize_heads(self,*args,**kwargs):return self._event('normalize_heads',super().normalize_heads,*args,**kwargs)
    def rope(self,*args,**kwargs):return self._event('rope',super().rope,*args,**kwargs)

def control_state(ctx,graph,refresher,provider):
    a=graph.arithmetic;cache=ctx._param_cache;profile=numerical_plan()['profile']
    if(ctx.secret_key is not None or ctx.public_key is not None or cache.secret_key is not None
            or ctx.parameters.get('secret_key')is not None or ctx.parameters.get('public_key')is not None):raise ValueError('keyless server context required')
    if cache.per_level_rotation_keys is not False:raise ValueError('reusable pregenerated keys required')
    if typed((ctx.degree,ctx.num_slots,ctx.degree_layout,ctx.composite_degree,ctx.max_level,cache.dnum))!=typed((131072,65536,(256,512),2,49,4)):raise ValueError('exact original context dimensions required')
    if typed(list(ctx.q_towers))!=typed(profile['q_towers'])or typed(list(ctx.p_towers))!=typed(profile['p_towers']):raise ValueError('original Q/P profile required')
    if typed(list(cache.q_towers))!=typed(profile['q_towers'])or typed(list(cache.p_towers))!=typed(profile['p_towers']):raise ValueError('original actual primitive Q/P cache required')
    if a.low_memory is not True or refresher.evaluator.low_memory is not True or refresher.engine._low_mem is not True:raise ValueError('synchronous ordinary low-memory path required')
    if a.scale!=profile['scale']or refresher.evaluator.scale!=profile['scale']:raise ValueError('original canonical alignment target required')
    engine=refresher.engine;view=engine.ctx
    if view._param_cache is not cache or engine._cache is not cache:raise ValueError('bootstrap must borrow actual common cache')
    if typed((engine._degree,engine._num_slots,engine._gap,engine._sparse,engine._composite_degree,engine._level_budget,engine._dc_subtract))!=typed((131072,1024,64,True,2,[4,4],False)):raise ValueError('exact original sparse bootstrap geometry required')
    if typed((view.degree,view.num_slots,view.degree_layout,view.composite_degree))!=typed((131072,1024,(256,512),2)):raise ValueError('bootstrap context geometry changed')
    if set(cache.raw_rotation_keys)!=set(numerical_plan()['required_rotation_indices']):raise ValueError('literal100 rotation inventory required')
    if refresher.evaluator.weighted_sum is not refresher._weighted_observed:raise ValueError('original refresh alignment observer replaced')
    keys=[]
    for index in numerical_plan()['required_rotation_indices']:
        key=cache.raw_rotation_keys[index];mapping=cache.coef_maps[index]
        keys.append((index,id(key),id(key['a']),id(key['b']),id(mapping)))
    helpers=[graph,a,refresher,refresher.evaluator,*refresher.compiled_primitives]
    for helper in(a,refresher.evaluator):
        helpers.extend(x for x in(getattr(helper,'rotation',None),getattr(helper,'_rescale',None),getattr(helper,'_plaintext_multiply',None),getattr(helper,'_multiplication_compiler',None))if x is not None)
    methods=[]
    for helper in helpers:
        if hasattr(helper,'ctx')and helper.ctx is not ctx:raise ValueError('compiled helper context changed')
        for name in('refresh','_finish','_prepare','_lower','_ensure_capacity','_public_slots','_encode_public','_clear_working_caches',
                    'weighted_sum','multiply','add','rotate','rescale','multiply_encoded','relinearize','bind','clear','linear','normalize','attention','evaluate','checkpoint','normalize_heads','rope','transpose_pairs','_bound','_bootstrap','install_alignment_observer'):
            if hasattr(helper,name):methods.append((id(helper),name,accessor(helper,name)))
    for name in('bootstrap','meta_bootstrap','control_gen','setup_key','_stage_checkpoint'):
        methods.append((id(engine),name,accessor(engine,name)))
    return(id(ctx),id(cache),id(graph),id(a),id(refresher),id(provider),id(refresher.engine),id(refresher.engine.ctx),
        id(getattr(a,'ledger',None)),id(getattr(refresher,'ledger',None)),id(getattr(getattr(a,'ledger',None),'callback',None)),
        typed((ctx.degree,ctx.num_slots,ctx.degree_layout,ctx.composite_degree,ctx.max_level,cache.dnum,cache.degree_layout)),
        accessor(refresher,'_weighted_original'),id(refresher._weighted_observed),typed(refresher.plan),typed((refresher.meta_precision,refresher.engine._dc_subtract,a.strategy,a.progress,a.preserve_dispatch_cache,
                                  refresher.evaluator.strategy,refresher.evaluator.progress,refresher.evaluator.preserve_dispatch_cache)),tuple(keys),tuple(methods))

def exercise(ctx,value,weights,*,checkpoint,dependencies):
    """Recording/explicit-dependency body; actual entry is disabled.

    Caller owns the key scope. Any failure must cross that scope so its public
    backings remain retained. No setup/key generation or private codec exists.
    """
    if _FAILURE_QUARANTINE:raise RuntimeError('failed arithmetic owners retained; dedicated process must exit')
    ledger=Ledger(checkpoint);profile=ledger.profile;initial_state=state(value,profile)
    if metadata(value,profile)!=dict(level=49,scale=profile['scale'],nsd=1):raise ValueError('fresh originalL49 input required')
    # Explicit trusted dependencies keep this source-only component free of
    #automatic backend imports. A future actual source/grant entry must bind
    #these exact original199 definitions; no such loader is selected here.
    required={'BoundedArithmetic','BoundedPackedTransformer','BoundedSecureRefresh','PackedPairLayout','PolynomialApproximation','enable_scoped_ntt_subsets'}
    if type(dependencies)is not dict or set(dependencies)!=required:raise ValueError('explicit reviewed original dependency selection required')
    BoundedArithmetic=dependencies['BoundedArithmetic'];BoundedPackedTransformer=dependencies['BoundedPackedTransformer']
    BoundedSecureRefresh=dependencies['BoundedSecureRefresh'];PackedPairLayout=dependencies['PackedPairLayout']
    PolynomialApproximation=dependencies['PolynomialApproximation'];enable_scoped_ntt_subsets=dependencies['enable_scoped_ntt_subsets']
    dependency_snapshot=tuple(sorted((k,id(v))for k,v in dependencies.items()))
    class Arithmetic(ArithmeticInstrumentation,BoundedArithmetic):pass
    class Refresh(RefreshInstrumentation,BoundedSecureRefresh):pass
    class Graph(GraphInstrumentation,BoundedPackedTransformer):pass
    graph=refresher=provider=None;primary=None;entered=False;retire_calls=()
    try:
        entered=True
        provider=enable_scoped_ntt_subsets(ctx)
        refresher=Refresh(ctx,period=1024,level_budget=(4,4),low_memory=True,compiled=True,meta_precision=10,preserve_dispatch_cache=True)
        refresher.ledger=ledger;refresher.install_alignment_observer()
        graph=Graph.__new__(Graph);graph.ctx=ctx;graph.layout=PackedPairLayout(2,4,32,65536);graph.features=32;graph._head_contract=None
        graph.arithmetic=Arithmetic(ctx,refresher,low_memory=True,preserve_dispatch_cache=True);graph.arithmetic.ledger=ledger
        weights_snapshot=typed(weights)
        def controls():
            if tuple(sorted((k,id(v))for k,v in dependencies.items()))!=dependency_snapshot or typed(weights)!=weights_snapshot:raise ValueError('original dependency/weight bindings changed')
            return control_state(ctx,graph,refresher,provider)
        ledger.control=controls;ledger.control_state=ledger.control();ledger.track(value)
        def working():
            owners=[graph.arithmetic,refresher.evaluator,*refresher.compiled_primitives];roots=[]
            for owner in tuple(owners):
                owners.extend(x for x in(getattr(owner,'rotation',None),getattr(owner,'_rescale',None),getattr(owner,'_plaintext_multiply',None),getattr(owner,'_multiplication_compiler',None))if x is not None)
            for owner in owners:
                for name in('_multiplication_bindings','_constant_cache','binding','_binding','compiler','_compiler','_executables'):
                    if hasattr(owner,name):roots.append(getattr(owner,name))
            return roots
        ledger.working=working
        retire_calls=(('arithmetic controls',graph.arithmetic._clear_working_caches),
            *(( 'refresh helper',helper.clear)for helper in refresher.compiled_primitives),
            ('refresh evaluator',refresher.evaluator._clear_working_caches),
            ('refresh engine',refresher.engine.__dict__.clear),('NTT provider',provider.close))
        coefficients=ledger.plan['coefficients']
        approximations={name:PolynomialApproximation(name,tuple(row['domain']),tuple(row['chebyshev_coefficients']),row['input_power_basis'])for name,row in coefficients.items()}
        ledger.note('input','original_embeddings',value)
        result=graph.evaluate(value,'qwen',ledger.plan['case']['config'],weights,approximations)
        ledger.note('output','logits',result)
        if len(ledger.trace)!=47 or ledger.full_count!=216 or ledger.counts!=ledger.plan['ideal_counts']or len(ledger.refresh_observations)!=19 or ledger.counter!=dict(ledger.plan['semantic_counts'],refresh=19):raise ValueError('complete original fixed graph/count inventory differs')
        return dict(checkpoints=ledger.trace,counts=ledger.counts,refresh_observations=ledger.refresh_observations,complete=True,
                    no_decode_or_oracle=True,output_metadata=metadata(result,profile))
    except BaseException as error:
        primary=protect_body(error)
        if entered:_FAILURE_QUARANTINE.append((ctx,initial_state,graph,refresher,provider,ledger,
            tuple(v for ref in tuple(ledger.live.values())if(v:=ref())is not None)))
        raise
    finally:
        error=None
        def attempt(label,fn):
            nonlocal error
            try:fn()
            except BaseException as exc:
                error=select(error,[(label,mark(exc))])
        # Reachable arrays get barriers even on failure. Unknown pre-return
        #results cannot be proved ready; retain all local owners in that case.
        arrays={id(old[1]):old[1]for old in ledger.callback_held}
        for ref in tuple(ledger.live.values()):
            if(v:=ref())is not None:
                array=getattr(v,'polynomial',None)
                if array is not None:arrays[id(array)]=array
        # Keep original callback-held arrays even if a wrapper was replaced.
        actions=[]
        for array in arrays.values():
            try:actions.append(('array readiness',array.block_until_ready))
            except BaseException as exc:error=select(error,[('readiness accessor',mark(exc))])
        for label,action in actions:attempt(label,action)
        if primary is None and error is None:
            attempt('final original controls',ledger.guard)
        if primary is None and error is None:
            for label,action in retire_calls:attempt(label,action)
        if error is not None:
            _FAILURE_QUARANTINE.append((ctx,value,graph,refresher,provider,ledger))
            chosen=select(primary,[('arithmetic cleanup',error)])
            if chosen is not primary:raise chosen
