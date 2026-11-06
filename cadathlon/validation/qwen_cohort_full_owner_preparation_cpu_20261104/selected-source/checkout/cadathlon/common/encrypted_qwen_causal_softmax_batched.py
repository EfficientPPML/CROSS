"""Tile-local frozen-rule softmax cohorts; no change to the literal predecessor.

Only disjoint query lanes with the same original Newton count AND inverse
refresh bound share a schedule. Per-K coefficients, alpha, positions and domains
remain exact. This owner borrows the full-slot keyless synchronous evaluator;
it neither owns nor retires a context. Failed entered work is quarantined until
dedicated-process exit. Operation counts are not performance measurements.
"""
from dataclasses import dataclass, fields
import math
import numpy as np
from encrypted_qwen_causal_softmax import (
    QwenCausalSoftmax, QwenProbabilityTile, verify_contract, require_synchronous, DELTA,
    positive_scale, drop_divisor, weighted_sum_scale, multiply_scale,
)
from encrypted_qwen_head_routing import QwenHeadRouting, QwenHeadLayout, typed
from encrypted_qwen_rms_refresh import QwenInverseRefreshBinding, accessor

_QUARANTINE=[]

@dataclass(frozen=True)
class CausalQueryRule:
    k:int
    row:int
    position:int
    alpha:float
    domain:tuple
    first_weight:float
    first_bias:float
    iterations:int
    refresh_bound:float

@dataclass(frozen=True)
class CausalCohort:
    query_tile:int
    index:int
    iterations:int
    refresh_bound:float
    queries:tuple

def descriptor_state(plans):
    def value(v):
        if type(v)in(CausalCohort,CausalQueryRule):
            return type(v),tuple((f.name,value(getattr(v,f.name)))for f in fields(v))
        if type(v)is tuple:return tuple,tuple(value(x)for x in v)
        return typed(v)
    return value(plans)

def cohort_plan(contract,positions):
    verify_contract(contract);layout=QwenHeadLayout(positions)
    if type(positions)is not tuple:raise TypeError('original immutable positions required')
    plans=[]
    for tile in range(layout.tile_count):
        groups={}
        for row,position in enumerate(layout.tile_positions(tile)):
            k=tile*16+row+1
            if k==1:continue
            domain,w,b,n,bound=contract.key_rules[k-1]
            query=CausalQueryRule(k,row,position,float(1/k),domain,w,b,n,bound)
            groups.setdefault((n,bound),[]).append(query)
        plans.append(tuple(CausalCohort(tile,index,n,bound,tuple(queries))
            for index,((n,bound),queries)in enumerate(groups.items())))
    return tuple(plans)

def _vectors(cohort,squarings):
    """Immutable all-slot PT arrays; inactive denominators use original dummy1.

    Every inactive lane uses the cohort's first original affine rule. Its dummy
    value1 lies in that unchanged denominator domain. The per-query rule fills
    all32 key lanes of every active head, including masked future-key lanes.
    """
    mask=np.zeros((16,4096));anchors=np.zeros_like(mask);dummy=np.zeros_like(mask)
    alpha=np.zeros_like(mask);dummy[:,::32]=1.
    first=cohort.queries[0]
    weights=np.full_like(mask,first.first_weight);bias=np.full_like(mask,first.first_bias)
    for query in cohort.queries:
        if not query.domain[0]<=1.<=query.domain[1]:raise ValueError('dummy1 outside original reciprocal domain')
        weights[query.row,:512]=query.first_weight;bias[query.row,:512]=query.first_bias
        for head in range(16):
            start=head*32
            mask[query.row,start:start+query.k]=1.
            anchors[query.row,start]=1.;dummy[query.row,start]=0.
            alpha[query.row,start:start+query.k]=query.alpha
    # Alpha at an anchor is also the original per-K1/K mean multiplier.
    values=(mask,anchors,dummy,alpha,anchors*alpha,mask*2.**(1-squarings),weights,bias)
    return tuple(np.frombuffer(v.astype('<f8').tobytes(),dtype='<f8')for v in values)

class QwenCausalSoftmaxBatched(QwenCausalSoftmax):
    """Explicit successor; the original full-model join still selects literal K."""
    def __init__(self,arithmetic,contract,config_bytes,*,positions,refresher=None,progress=None):
        if _QUARANTINE:raise RuntimeError('failed cohort owner requires dedicated-process exit')
        self.cohorts=cohort_plan(contract,positions)
        self._cohort_original=self.cohorts;self._cohort_state=descriptor_state(self.cohorts)
        self._vector_factory=_vectors;self._descriptor_factory=descriptor_state
        self._cohort_method=accessor(self,'evaluate_tile')
        self._pending_return=();self._active_arrays=();self._failure=None
        # The inherited initialization sequence, with both wrapper accessors
        # captured before the final guard. No mutable initialization bypass.
        self.contract=contract;state=verify_contract(contract);self.refresher=refresher
        require_synchronous(arithmetic,refresher)
        QwenHeadRouting.__init__(self,arithmetic,contract.manifest_bytes,config_bytes,positions=positions,progress=progress)
        self._softmax_bound=(contract,refresher);self._contract=state;self._arithmetic_accessors=self._accessors()
        if len(self.layout.positions)==1:
            if refresher is not None:raise ValueError('K1 direct-V branch has no refresher')
            self.refresh_binding=None
        else:self.refresh_binding=QwenInverseRefreshBinding(self,refresher)
        self._refresh_binding=self.refresh_binding
        self._binding_accessors=()if self.refresh_binding is None else tuple(accessor(self.refresh_binding,n)for n in('refresh','check'))
        self._softmax_guard()
    def _softmax_guard(self):
        if _QUARANTINE:raise RuntimeError('failed cohort owner requires dedicated-process exit')
        if self.refresh_binding is not self._refresh_binding:
            raise ValueError('construction-bound refresh wrapper changed')
        methods=()if self.refresh_binding is None else tuple(accessor(self.refresh_binding,n)for n in('refresh','check'))
        if any(a is not b for new,old in zip(methods,self._binding_accessors)for a,b in zip(new,old))or len(methods)!=len(self._binding_accessors):
            raise ValueError('construction-bound refresh wrapper accessor changed')
        # The inherited guard calls binding.check(), so authenticate the two
        # wrapper accessors before any inherited dynamic invocation.
        super()._softmax_guard()
        if _vectors is not self._vector_factory or descriptor_state is not self._descriptor_factory:
            raise ValueError('construction-bound cohort vector/state factory changed')
        if (self.cohorts is not self._cohort_original or self._descriptor_factory(self.cohorts)!=self._cohort_state
                or accessor(self,'evaluate_tile')!=self._cohort_method):
            raise ValueError('construction-bound causal cohort descriptors/accessor changed')
    def evaluate_tile(self,scores,*,query_tile,capture_checkpoints=False):
        self._softmax_guard()
        if self._active:raise ValueError('serial softmax cannot be reentered')
        if type(capture_checkpoints)is not bool:raise TypeError('exact checkpoint bool required')
        positions=self.layout.tile_positions(query_tile);a=self.arithmetic;c=self.contract
        if len(self.layout.positions)==1:
            if scores is not None:raise ValueError('K1 bypass requires no score ciphertext')
            plan=dict(stages=[],refreshes=[],counts={},output_level=None,output_scale=None)
            self.last_run=plan
            return QwenProbabilityTile(None,positions,self.layout.positions,(0,),query_tile,c.layer,{}, {},plan)
        self._validate(scores,0)
        if a._level(scores)!=21:raise ValueError('actual score input level21 required')
        q=tuple(self.ctx._param_cache.q_towers);refs={};checkpoints={};stages=[];events=[]
        array_binding=();array_state=();pending_return=()
        counts=dict(rotations=0,ordinary_additions=0,weighted_sums=0,ciphertext_products=0,refreshes=0)
        def retain(name,value):refs[name]=state(value);return value
        def state(value):return (value,(value.polynomial,getattr(value,'moduli_array',None)),self._fingerprint(value))
        def check_states(values):
            if any(v.polynomial is not owners[0] or self._fingerprint(v)!=old for v,owners,old in values):raise ValueError('retained softmax ciphertext changed')
        retain('scores',scores)
        def returned(value):
            nonlocal pending_return
            pending_return=(value,())
            pending_return=(value,(getattr(value,'polynomial',None),getattr(value,'moduli_array',None)))
            self._pending_return=pending_return
            return value
        def arrays_state(values):
            return tuple((id(v),id(v.base),typed(v.shape),v.dtype.str,v.strides,v.flags.writeable)for v in values)
        def guard():
            self._softmax_guard();check_states(tuple(refs.values()))
            if (self._active_arrays is not array_binding or arrays_state(array_binding)!=array_state
                    or self._pending_return is not pending_return):raise ValueError('active cohort plaintext/return owners changed')
        def metadata(value,level,scale):
            self._validate(value,0)
            if (type(a._level(value))is not int or a._level(value)!=level or typed(value._ckks_scale)!=typed(scale) or
                type(value.num_moduli)is not int or value.num_moduli!=2*(level+1) or typed(tuple(value.moduli))!=typed(q[:2*(level+1)])):raise ValueError('softmax primitive metadata differs')
        def note(label,value,level,scale,kind,owners=()):
            returned(value)
            guard();check_states(owners);metadata(value,level,scale);self._forget(value);current=state(value)
            if self.progress is not None:self.progress(label,value,dict(level=level,scale=scale,counts=dict(counts)))
            guard();check_states((*owners,current));self._forget(value)
            for prior,_,_ in refs.values():
                if prior is not scores:self._forget(prior)
            stages.append(dict(label=label,kind=kind,level=level,scale=scale))
            if capture_checkpoints:checkpoints[label]=retain('checkpoint.'+label,value)
            return value
        def weighted(label,values,weights,bias=0.):
            guard();owners=tuple(state(v) for v in values);level=min(a._level(v) for v in values)
            scale=weighted_sum_scale([v._ckks_scale for v in values],DELTA,drop_divisor(q,level))['output_scale'];counts['weighted_sums']+=1
            return note(label,a.weighted_sum(values,weights,bias=bias),level-1,scale,'weighted',owners)
        def mul(label,x,y):
            guard();owners=(state(x),state(y));level=min(a._level(x),a._level(y));scale=multiply_scale(x._ckks_scale,y._ckks_scale,drop_divisor(q,level));counts['ciphertext_products']+=1
            return note(label,a.multiply(x,y),level-1,scale,'multiply',owners)
        def rotate(value,shift):
            guard();old=state(value);level=a._level(value);scale=value._ckks_scale;counts['rotations']+=1
            out=returned(a.rotate(value,shift%65536));metadata(out,level,scale);guard();check_states((old,));return out
        def add(x,y,subtract=False):
            guard();owners=(state(x),state(y));level=min(a._level(x),a._level(y));scale=x._ckks_scale;counts['ordinary_additions']+=1
            out=returned(a.add(x,y,subtract=subtract));metadata(out,level,scale);guard();check_states(owners);return out
        def reduce(value):
            for bit in range(5):value=add(value,rotate(value,1<<bit))
            return value
        def broadcast(value):
            for bit in range(5):value=add(value,rotate(value,-(1<<bit)))
            return value
        def refresh(label,value,bound,expected_level):
            guard()
            if a._level(value)!=expected_level:raise ValueError('frozen refresh entry level differs')
            retain('refresh_source',value);counts['refreshes']+=1
            out=self.refresh_binding.refresh(value,public_bound=float(bound))
            out=note(label,out,25,positive_scale(out._ckks_scale),'refresh',(state(value),))
            events.append(dict(label=label,input_level=expected_level,output_level=25,actual_output_scale=out._ckks_scale,public_bound=float(bound)))
            refs.pop('refresh_source');return out
        def inverse(prefix,denominator,rule,bound=None):
            _,w,b,n=rule;retain('denominator',denominator)
            y=retain('inverse',weighted(prefix+'.first',[denominator],[w],b))
            for update in range(2,n+1):
                if bound is not None and update==7:y=retain('inverse',refresh(prefix+'.refresh_before7',y,bound,6))
                product=mul(prefix+f'.{update}.product',denominator,y)
                correction=weighted(prefix+f'.{update}.correction',[product],[-1.],2.)
                y=retain('inverse',mul(prefix+f'.{update}.update',y,correction))
            return y
        self._active=True;self.last_run=None
        try:
            aggregate=None
            for cohort in self.cohorts[query_tile]:
                guard()
                array_binding=self._vector_factory(cohort,c.squarings);self._active_arrays=array_binding;array_state=arrays_state(array_binding)
                mask,anchors,dummy,alpha,anchor_alpha,mapped_mask,first_weight,first_bias=array_binding
                prefix='C'+str(cohort.index)+'.K'+','.join(str(query.k)for query in cohort.queries)
                masked=retain('masked_source',weighted(prefix+'.source_branch_mask',[scores],[mask]))
                mean=retain('mean',broadcast(weighted(prefix+'.mean_anchor_alpha',[reduce(masked)],[anchor_alpha])))
                centered=add(masked,mean,True)
                t=retain('mapped_t',weighted(prefix+'.normalized_t_branch_mask',[centered],[mapped_mask]))
                t=retain('mapped_t',refresh(prefix+'.normalized_t.refresh',t,2.,18));terms={1:t}
                def power(n):
                    if n not in terms:terms[n]=retain('power'+str(n),mul(prefix+'.power'+str(n),power(n//2),power((n+1)//2)))
                    return terms[n]
                base=retain('base',weighted(prefix+'.base_exp',[power(n) for n in range(1,9)],c.coefficients[1:],c.coefficients[0]))
                for name in tuple(refs):
                    if name.startswith('power') or name in ('mapped_t','mean','masked_source'):refs.pop(name)
                terms.clear()
                masked_exp=weighted(prefix+'.initial_mean_mask_alpha',[base],[alpha])
                denominator=broadcast(weighted(prefix+'.initial_mean_anchor_dummy1',[reduce(masked_exp)],[anchors],dummy))
                y=inverse(prefix+'.initial_inverse',denominator,c.initial_rule)
                y=retain('inverse',refresh(prefix+'.completed_initial_inverse.refresh',y,4.,3))
                probability=retain('probability',weighted(prefix+'.initial_probability_alpha_mask',[mul(prefix+'.initial_quotient',base,y)],[alpha]))
                refs.pop('base');refs.pop('denominator');refs.pop('inverse')
                for iteration in range(c.squarings):
                    tag=prefix+f'.round{iteration}'
                    probability=retain('probability',refresh(tag+'.probability.refresh',probability,2.,a._level(probability)))
                    masked_probability=weighted(tag+'.mask',[probability],[mask])
                    numerator=retain('numerator',mul(tag+'.square',masked_probability,masked_probability))
                    denominator=broadcast(weighted(tag+'.denominator_anchor_dummy1',[reduce(numerator)],[anchors],dummy))
                    krule=(None,first_weight,first_bias,cohort.iterations)
                    y=inverse(tag+'.inverse',denominator,krule,cohort.refresh_bound)
                    probability=retain('probability',mul(tag+'.quotient',numerator,y))
                    for name in ('numerator','denominator','inverse'):refs.pop(name)
                probability=retain('probability',refresh(prefix+'.final_probability.refresh',probability,2.,a._level(probability)))
                branch=weighted(prefix+'.final_branch_mask',[probability],[mask])
                aggregate=retain('aggregate',branch if aggregate is None else add(aggregate,branch));refs.pop('probability');guard()
            output=note('output',aggregate,24,aggregate._ckks_scale,'assembled');guard()
            plan=dict(cohorts=self.cohorts[query_tile],stages=stages,refreshes=events,counts=dict(counts),input_level=21,input_scale=scores._ckks_scale,output_level=24,output_scale=output._ckks_scale)
            self.last_run=plan
            return QwenProbabilityTile(output,positions,self.layout.positions,(0,) if query_tile==0 else (),query_tile,c.layer,dict(counts),checkpoints,plan)
        except BaseException as error:
            self.last_run=None
            # Capture local originals independently of mutable wrapper attributes.
            self._failure=(error,scores,tuple(refs.values()),array_binding,pending_return,self._cohort_original)
            _QUARANTINE.append((self,self._failure))
            try:error.add_note('entered cohort failure: original borrowed arrays/owners quarantined until process exit')
            except BaseException:pass
            raise
        finally:
            self._active=False
            if self._failure is None:self._active_arrays=();self._pending_return=()
