"""Frozen BERT-v2 initial softmax with an explicit normalized-t refresh boundary.

Prefix and tail preserve the selected degree8 power circuit and static six
Newton updates. The tail never recenters its input. Public domains, the B2
full-slot range/error contract and repeated composition are caller obligations.
No client codec, implicit refresh, exact exponential or inverse is used.
"""
from dataclasses import dataclass
import hashlib,json,math
import numpy as np
from encrypted_bert_initial_softmax import InitialSoftmaxContract,MANIFEST_SHA256
from encrypted_score_reduction import ScoreReduction,ScoreReductionLayout
from encrypted_fullslot_refresh_binding import FullSlotRefreshBinding
from encrypted_integer_affine import backward_scale_plan
from encrypted_scaled_newton import integer_newton_updates
from encrypted_scale_planning import positive_scale,drop_divisor,weighted_sum_scale,multiply_scale,static_newton_forward

FIELDS=('layer','centered_domain','squarings','normalized_divisor','normalized_domain',
        'exponential_coefficients','initial_domain','initial_seed','initial_updates','source_manifest_sha256')

@dataclass(frozen=True,init=False)
class LayerInitialContract:
    layer:int
    centered_domain:tuple
    squarings:int
    normalized_divisor:int
    normalized_domain:tuple
    exponential_coefficients:tuple
    initial_domain:tuple
    initial_seed:float
    initial_updates:int
    source_manifest_sha256:str
    def __init__(self,manifest_bytes,layer):
        if type(manifest_bytes)is not bytes or hashlib.sha256(manifest_bytes).hexdigest()!=MANIFEST_SHA256:
            raise ValueError('exact frozen BERT-v2 manifest bytes required')
        if type(layer)is not int or not 0<=layer<12:raise ValueError('public integer layer in0..11 required')
        manifest=json.loads(manifest_bytes);base=InitialSoftmaxContract.from_manifest_bytes(manifest_bytes)
        rule=manifest['softmax'][str(layer)];s=rule['squarings'];divisor=2**(s-1)
        domain=tuple(map(float,rule['centered_domain']))
        if type(s)is not int or s not in (5,6,7) or not (-divisor<=domain[0]<domain[1]<=divisor):
            raise ValueError('frozen centered-score map exceeds the base polynomial domain')
        values=dict(layer=layer,centered_domain=domain,squarings=s,normalized_divisor=divisor,
            normalized_domain=tuple(x/divisor for x in domain),exponential_coefficients=base.exponential_coefficients,
            initial_domain=base.initial_domain,initial_seed=base.initial_seed,initial_updates=base.initial_updates,
            source_manifest_sha256=MANIFEST_SHA256)
        for name,value in values.items():object.__setattr__(self,name,value)
    @property
    def power_coefficients(self):
        return InitialSoftmaxContract(self.exponential_coefficients).power_coefficients

def state(contract):return tuple(ScoreReduction._typed(getattr(contract,name,None)) for name in FIELDS)

def _plan_inputs(q_moduli,level,scale,input_scale,minimum):
    q=tuple(q_moduli)
    if (type(level)is not int or level<minimum or len(q)!=2*(level+1) or len(set(q))!=len(q)
            or any(type(x)is not int or x<=1 for x in q)):
        raise ValueError('exact ordered CD2 Q prefix with sufficient reserved levels required')
    canonical=positive_scale(scale)
    return q,canonical,canonical if input_scale is None else positive_scale(input_scale)

def prefix_scale_plan(q_moduli,*,input_level,canonical_scale,input_scale=None):
    q,canonical,source=_plan_inputs(q_moduli,input_level,canonical_scale,input_scale,5)
    rows=[]
    def ws(name,scales,level):
        row=weighted_sum_scale(scales,canonical,drop_divisor(q,level))
        row.update(name=name,input_level=level,output_level=level-1);rows.append(row)
        return row['output_scale']
    mask=ws('score_mean_mask',[source],input_level)
    mean=ws('score_mean_anchor',[mask],input_level-1)
    t=ws('normalized_t',[source,mean],input_level-2)
    return dict(input_level=input_level,input_scale=source,canonical_scale=canonical,consumed_levels=3,
        levels=dict(score_mean=input_level-2,normalized_t=input_level-3),
        scales=dict(score_mean=mean,normalized_t=t),weighted_sum_scale_stages=rows,final_scale=t)

def tail_scale_plan(q_moduli,*,input_level,canonical_scale,input_scale=None):
    q,canonical,source=_plan_inputs(q_moduli,input_level,canonical_scale,input_scale,21)
    terms={1:dict(power=1,level=input_level,scale=source)};weighted=[]
    def power(n):
        if n not in terms:
            a,b=power(n//2),power((n+1)//2);level=min(a['level'],b['level'])
            terms[n]=dict(power=n,level=level-1,scale=multiply_scale(a['scale'],b['scale'],drop_divisor(q,level)))
        return terms[n]
    def ws(name,scales,level):
        row=weighted_sum_scale(scales,canonical,drop_divisor(q,level))
        row.update(name=name,input_level=level,output_level=level-1);weighted.append(row)
        return row['output_scale']
    ordered=[power(i) for i in range(1,9)]
    base=ws('base_exp',[r['scale'] for r in ordered],min(r['level'] for r in ordered))
    masked=ws('mean_exp_mask',[base],input_level-4)
    denominator_scale=ws('mean_exp_anchor',[masked],input_level-5)
    denominator=input_level-6;inverse=denominator-11;divisor=drop_divisor(q,inverse)
    # Retain the original first-affine target policy; replay actual metadata
    # from the refreshed input rather than assigning a canonical scale.
    nominal=backward_scale_plan(q[:2*(denominator+1)],input_scale=canonical,
        first_output_level=denominator-1,updates=6,final_scale=float(divisor))
    newton=static_newton_forward(q,input_scale=denominator_scale,first_output_level=denominator-1,
        updates=6,first_target_scale=nominal['initial_target_scale'])
    product=multiply_scale(base,newton['rows'][-1]['scale'],divisor)
    final=ws('probabilities',[product],input_level-18)
    return dict(input_level=input_level,input_scale=source,canonical_scale=canonical,consumed_levels=19,
        levels=dict(base_exp=input_level-4,mean_exp=denominator,inverse=inverse,
                    base_times_inverse=input_level-18,probabilities=input_level-19),
        scales=dict(base_exp=base,mean_exp=denominator_scale,inverse=newton['rows'][-1]['scale'],
                    base_times_inverse=product,probabilities=final),
        weighted_sum_scale_stages=weighted,power_scale_stages=ordered,
        nominal_inverse_target_plan=nominal,static_inverse_scale_plan=newton,final_scale=final,
        recenter_input=False,extra_pre_power_mask=False,initial_updates=6,implemented_squaring_rounds=0)

class BertSplitInitialSoftmax:
    """Borrowed layer-bound prefix/tail with an explicit full-slot B2 adapter.

    Standalone prefix/tail calls bind their own input; evaluate_tile additionally
    preserves the original score input across refresh and the complete tail.
    Supplied original positions are public query identities, never new learned
    position indices. The operator remains position-agnostic.
    """
    def __init__(self,arithmetic,layout,manifest_bytes,refresher,*,layer,original_positions,progress=None):
        if not isinstance(layout,ScoreReductionLayout):raise TypeError('full-slot score layout required')
        if (type(original_positions)is not tuple or len(original_positions)!=layout.valid_rows
                or any(type(x)is not int or not 0<=x<128 for x in original_positions)
                or tuple(sorted(set(original_positions)))!=original_positions):
            raise ValueError('strict ordered original positions for every valid query required')
        if progress is not None and not callable(progress):raise TypeError('callable progress required')
        self.arithmetic,self.layout,self.refresher=arithmetic,layout,refresher
        self.contract=LayerInitialContract(manifest_bytes,layer);self.original_positions=original_positions;self.progress=progress
        self.reduction=ScoreReduction(arithmetic,layout,progress=self._callback)
        self.refresh_binding=FullSlotRefreshBinding(arithmetic,refresher)
        self._bound=(arithmetic,layout,refresher,self.contract,self.reduction,self.refresh_binding,progress)
        self._contract_state=state(self.contract);self._positions=ScoreReduction._typed(original_positions)
        self._callback_identity=self.reduction.progress
        self._active=False;self._sources=();self._source_states=();self._retained={};self._captured={};self._capture=False
        self.last_plan=None;self._guard()

    @staticmethod
    def rotation_indices():return ScoreReductionLayout.rotation_indices()
    def _fingerprint(self,value):return self.reduction._value_fingerprint(value)
    def _guard(self):
        now=(self.arithmetic,self.layout,self.refresher,self.contract,self.reduction,self.refresh_binding,self.progress)
        if any(a is not b for a,b in zip(now,self._bound)):raise ValueError('construction-bound split owner changed')
        if (state(self.contract)!=self._contract_state or ScoreReduction._typed(self.original_positions)!=self._positions
                or self.reduction.progress is not self._callback_identity):raise ValueError('split public contract or callback changed')
        self.reduction._guard_context();self.refresh_binding.check()
        if tuple(self._fingerprint(v) for v in self._sources)!=self._source_states:
            raise ValueError('split source ciphertext changed')
        for value,saved in (*self._retained.values(),*self._captured.values()):
            if self._fingerprint(value)!=saved:raise ValueError('split retained ciphertext changed')
    def _strip(self,value):
        for current in (value,*(v for v,_ in self._retained.values()),*(v for v,_ in self._captured.values())):
            if not any(current is source for source in self._sources):ScoreReduction._forget(current)
    def _callback(self,label,value,counts):
        self._guard();saved=self._fingerprint(value)
        if self.progress is not None:self.progress(label,value,dict(counts))
        self._guard()
        if self._fingerprint(value)!=saved:raise ValueError('split callback changed its result')
        self._strip(value)
    def _validate(self,value,minimum):
        self.reduction._validate(value,minimum)
        if ScoreReduction._typed(getattr(value,'degree_layout',None))!=ScoreReduction._typed(self.arithmetic.ctx.degree_layout):
            raise ValueError('split ciphertext layout differs from borrowed context')
    def _note(self,name,value,plan=None,*,retain=False):
        self._guard()
        self.refresh_binding._metadata._validate_ciphertext(value,minimum_levels=0,canonical=False)
        if ScoreReduction._typed(getattr(value,'degree_layout',None))!=ScoreReduction._typed(self.arithmetic.ctx.degree_layout):
            raise ValueError('split result layout differs from borrowed context')
        if plan is not None and (self.arithmetic._level(value)!=plan['levels'][name]
                or ScoreReduction._typed(value._ckks_scale)!=ScoreReduction._typed(plan['scales'][name])):
            raise ValueError('split primitive metadata differs from actual ordered scale plan')
        ScoreReduction._forget(value)
        if retain:self._retained[name]=(value,self._fingerprint(value))
        if self._capture:self._captured[name]=(value,self._fingerprint(value))
        self._callback('split initial '+name,value,{})
        return value
    def _begin(self,value,minimum,capture):
        self._guard()
        if self._active:raise ValueError('split initial softmax cannot be reentered')
        if type(capture)is not bool:raise ValueError('explicit boolean capture option required')
        self._validate(value,minimum)
        self._active=True;self._capture=capture;self._sources=(value,);self._source_states=(self._fingerprint(value),);self.last_plan=None
    def _end(self):
        self._active=False;self._sources=();self._source_states=();self._retained.clear();self._captured.clear();self._capture=False
    def _result(self,value):
        self._guard();self._strip(value)
        return (value,{name:current for name,(current,_) in self._captured.items()}) if self._capture else value
    def _prefix(self,scores):
        level=self.arithmetic._level(scores);q=self.arithmetic.ctx._param_cache.q_moduli_at_level(level)
        plan=prefix_scale_plan(q,input_level=level,canonical_scale=self.arithmetic.scale,input_scale=scores._ckks_scale)
        mean=self._note('score_mean',self.reduction.mean(scores),plan,retain=True)
        divisor=self.contract.normalized_divisor
        t=self._note('normalized_t',self.arithmetic.weighted_sum([scores,mean],
            [self.layout.input_mask/divisor,-self.layout.input_mask/divisor]),plan,retain=True)
        del self._retained['score_mean']
        plan.update(layer=self.contract.layer,centered_domain=self.contract.centered_domain,
            normalized_domain=self.contract.normalized_domain,normalized_divisor=divisor,original_positions=self.original_positions)
        return t,plan
    def _tail(self,t):
        level=self.arithmetic._level(t);q=self.arithmetic.ctx._param_cache.q_moduli_at_level(level)
        plan=tail_scale_plan(q,input_level=level,canonical_scale=self.arithmetic.scale,input_scale=t._ckks_scale)
        self._retained['tail_input']=(t,self._fingerprint(t));terms={1:t}
        def power(n):
            if n not in terms:
                self._guard();terms[n]=ScoreReduction._forget(self.arithmetic.multiply(power(n//2),power((n+1)//2)))
            return terms[n]
        c=self.contract.power_coefficients
        # Keep the original ordered masked polynomial terms. Bounded nonzero
        # invalid refreshed lanes are not assumed zero, nor recentered here.
        base=self._note('base_exp',self.arithmetic.weighted_sum([power(i) for i in range(1,9)],
            [c[i]*self.layout.input_mask for i in range(1,9)],bias=c[0]*self.layout.input_mask),plan,retain=True)
        terms.clear()
        denominator=self._note('mean_exp',self.reduction.mean_with_dummy(base,
            domain=self.contract.initial_domain,inactive_dummy=1.),plan,retain=True)
        inverse=None
        for update,inverse in integer_newton_updates(self.arithmetic,denominator,normalization_factor=1,
                initial_estimate=self.contract.initial_seed,updates=self.contract.initial_updates,
                first_output_scale=plan['static_inverse_scale_plan']['initial_target_scale']):
            expected=plan['static_inverse_scale_plan']['rows'][update-1]
            if self.arithmetic._level(inverse)!=expected['level'] or inverse._ckks_scale!=expected['scale']:
                raise ValueError('static inverse checkpoint differs from ordered metadata plan')
            self._note('inverse_update'+str(update),inverse)
        inverse=self._note('inverse',inverse,plan)
        del self._retained['mean_exp']
        product=self._note('base_times_inverse',self.arithmetic.multiply(base,inverse),plan)
        del self._retained['base_exp']
        output=self._note('probabilities',self.arithmetic.weighted_sum([product],[self.layout.mean_weight]),plan)
        plan.update(layer=self.contract.layer,normalized_domain=self.contract.normalized_domain,
            initial_domain=self.contract.initial_domain,future_normalized_squarings=self.contract.squarings,
            original_positions=self.original_positions)
        return output,plan
    def prefix(self,scores,*,capture_checkpoints=False):
        self._begin(scores,5,capture_checkpoints)
        try:
            output,plan=self._prefix(scores);self.last_plan=plan;return self._result(output)
        finally:self._end()
    def tail(self,normalized_t,*,capture_checkpoints=False):
        self._begin(normalized_t,21,capture_checkpoints)
        try:
            output,plan=self._tail(normalized_t);self.last_plan=plan;return self._result(output)
        finally:self._end()
    def evaluate_tile(self,scores,*,capture_checkpoints=False):
        self._begin(scores,5,capture_checkpoints)
        try:
            t,prefix=self._prefix(scores);saved=self._fingerprint(t)
            self._guard();refreshed=self.refresher.refresh(t,public_bound=2.);self._guard()
            if self._fingerprint(t)!=saved:raise ValueError('refresh changed normalized-t source')
            self._validate(refreshed,25)
            if self.arithmetic._level(refreshed)!=25:raise ValueError('declared full-slot refresh output level25 required')
            refreshed=self._note('refreshed_normalized_t',refreshed,retain=True)
            output,tail=self._tail(refreshed)
            self.last_plan=dict(layer=self.contract.layer,prefix=prefix,tail=tail,refresh_calls=1,
                refresh_public_bound=2.,actual_refresh_scale=refreshed._ckks_scale,
                range_verified=False,encrypted_refresh_error_verified=False)
            return self._result(output)
        finally:self._end()
