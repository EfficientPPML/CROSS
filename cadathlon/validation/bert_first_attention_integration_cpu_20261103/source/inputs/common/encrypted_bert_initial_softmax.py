"""Experimental frozen-v2 BERT initial softmax, before its five square rounds.

All operations use one borrowed keyless full-slot CD2 context. There is no
refresh, client codec, exact-softmax substitution, or ciphertext range test.
The declared centered-score/denominator domains are public caller contracts.
This component alone does not implement attention, a BERT block or full model.
"""
from dataclasses import dataclass
import hashlib,json,math
import numpy as np
from encrypted_integer_affine import backward_scale_plan
from encrypted_scale_planning import positive_scale,drop_divisor,weighted_sum_scale,multiply_scale,static_newton_forward
from encrypted_scaled_newton import integer_newton_updates
from encrypted_score_reduction import ScoreReduction,ScoreReductionLayout,SLOTS

MANIFEST_SHA256='4e2277beea7189aa1d77b168c500c8a175f6c630afa520fcc6b38cb5a179cff9'
EXP_SHA256='64c193dd47fbbc6f850b14a35dad72803d35a2353497aba4db8061b217fe41dd'
POWER_SHA256='5173fba9463c2962090f8ddeb64953ef60ad8ba700b2cd37573aba1e11b6c59a'

@dataclass(frozen=True)
class InitialSoftmaxContract:
    exponential_coefficients: tuple
    source_manifest_sha256: str=MANIFEST_SHA256
    centered_domain: tuple=(-16.,16.)
    exponential_domain: tuple=(-.5,.5)
    squarings: int=5
    initial_domain: tuple=(.5,2.)
    initial_seed: float=.8
    initial_updates: int=6

    def __post_init__(self):
        c=np.asarray(self.exponential_coefficients)
        if (c.dtype.kind not in 'fiu' or c.shape!=(9,) or not np.isfinite(c).all()
                or hashlib.sha256(c.astype('<f8').tobytes()).hexdigest()!=EXP_SHA256
                or self.source_manifest_sha256!=MANIFEST_SHA256
                or self.centered_domain!=(-16.,16.) or self.exponential_domain!=(-.5,.5)
                or type(self.squarings)is not int or self.squarings!=5
                or self.initial_domain!=(.5,2.) or type(self.initial_seed)is not float or self.initial_seed!=.8
                or type(self.initial_updates)is not int or self.initial_updates!=6):
            raise ValueError('exact frozen BERT v2 initial-softmax contract required')
        power=np.polynomial.chebyshev.cheb2poly(c.astype(np.float64))
        if hashlib.sha256(power.astype('<f8').tobytes()).hexdigest()!=POWER_SHA256:
            raise ValueError('power conversion differs from the frozen candidate arithmetic')
        object.__setattr__(self,'exponential_coefficients',tuple(map(float,c)))

    @classmethod
    def from_manifest_bytes(cls,raw):
        if not isinstance(raw,bytes) or hashlib.sha256(raw).hexdigest()!=MANIFEST_SHA256:
            raise ValueError('exact frozen v2 manifest bytes required')
        model=json.loads(raw)
        return cls(tuple(model['base_exp']['coefficients']))

    @property
    def power_coefficients(self):
        return tuple(map(float,np.polynomial.chebyshev.cheb2poly(self.exponential_coefficients)))


CONTRACT_FIELDS=('exponential_coefficients','source_manifest_sha256','centered_domain',
    'exponential_domain','squarings','initial_domain','initial_seed','initial_updates')

def _contract_state(contract):
    return tuple(ScoreReduction._typed(getattr(contract,name,None)) for name in CONTRACT_FIELDS)

def _verified_contract_state(contract):
    """Validate every field and its type against the canonical frozen defaults."""
    if not isinstance(contract,InitialSoftmaxContract):raise TypeError('expected frozen initial-softmax contract')
    expected=InitialSoftmaxContract(contract.exponential_coefficients)
    state=_contract_state(contract)
    if state!=_contract_state(expected):
        raise ValueError('initial-softmax contract fields differ from the exact frozen defaults')
    return state


def initial_softmax_plan(q_moduli,*,input_level,canonical_scale,input_scale=None):
    q=tuple(q_moduli)
    if (type(input_level)is not int or input_level<24 or len(q)!=2*(input_level+1)
            or len(set(q))!=len(q) or any(type(x)is not int or x<=1 for x in q)):
        raise ValueError('static full-slot CD2 plan needs22 levels plus two continuation levels')
    canonical_scale=positive_scale(canonical_scale)
    source=canonical_scale if input_scale is None else positive_scale(input_scale)
    weighted=[]
    def ws(name,scales,level):
        row=weighted_sum_scale(scales,canonical_scale,drop_divisor(q,level))
        row.update(name=name,input_level=level,output_level=level-1);weighted.append(row)
        return row['output_scale']
    mean_mask=ws('score_mean_mask',[source],input_level)
    mean=ws('score_mean_anchor',[mean_mask],input_level-1)
    t=ws('normalized_t',[source,mean],input_level-2)
    terms={1:dict(power=1,level=input_level-3,scale=t)}
    def power(n):
        if n not in terms:
            left=power(n//2);right=power((n+1)//2);level=min(left['level'],right['level'])
            terms[n]=dict(power=n,level=level-1,
                scale=multiply_scale(left['scale'],right['scale'],drop_divisor(q,level)))
        return terms[n]
    ordered=[power(i) for i in range(1,9)]
    base=ws('base_exp',[x['scale'] for x in ordered],min(x['level'] for x in ordered))
    exp_mask=ws('mean_exp_mask',[base],input_level-7)
    den_scale=ws('mean_exp_anchor',[exp_mask],input_level-8)
    denominator=input_level-9;inverse=denominator-11
    divisor=drop_divisor(q,inverse)
    # Preserve the historical REAL first-affine encoding target. Correct only
    # its predicted metadata, replayed using the actual denominator scale.
    nominal=backward_scale_plan(q[:2*(denominator+1)],input_scale=canonical_scale,
        first_output_level=denominator-1,updates=6,final_scale=float(divisor))
    scale=static_newton_forward(q,input_scale=den_scale,first_output_level=denominator-1,
        updates=6,first_target_scale=nominal['initial_target_scale'])
    product_scale=multiply_scale(base,scale['rows'][-1]['scale'],divisor)
    final=ws('probabilities',[product_scale],input_level-21)
    scales=dict(score_mean=mean,normalized_t=t,base_exp=base,mean_exp=den_scale,
                inverse=scale['rows'][-1]['scale'],base_times_inverse=product_scale,probabilities=final)
    return dict(input_level=input_level,input_scale=source,canonical_scale=canonical_scale,
        consumed_levels=22,reserved_output_levels=2,
        levels={'score_mean':input_level-2,'normalized_t':input_level-3,'base_exp':input_level-7,
                'mean_exp':denominator,'inverse':inverse,'base_times_inverse':input_level-21,'probabilities':input_level-22},
        scales=scales,weighted_sum_scale_stages=weighted,power_scale_stages=ordered,
        nominal_inverse_target_plan=nominal,static_inverse_scale_plan=scale,actual_product_scale=product_scale,
        final_scale=final,initial_updates=6,first_weight=-.8*.8,first_bias=2*.8,
        encoding_target_policy='Preserve original nominal target; replay actual binary64 metadata without relabeling.',
        future_normalized_squarings=5,implemented_squaring_rounds=0)


class BertInitialSoftmax:
    """One complete masked initial probability tile; full65536 physical slots.

    Each row may have its own public subset of the128 keys. A partial last
    query tile uses explicit valid_rows. Output key/head/query padding is zero
    in the ideal circuit. No implicit refresh or bound-aware owner is allowed.
    """
    def __init__(self,arithmetic,layout,contract,*,progress=None):
        contract_state=_verified_contract_state(contract)
        if not isinstance(layout,ScoreReductionLayout):raise TypeError('expected full-slot ScoreReductionLayout')
        self.arithmetic,self.layout,self.contract=arithmetic,layout,contract
        self.reduction=ScoreReduction(arithmetic,layout,progress=progress)
        self._bound_arithmetic=arithmetic;self._bound_layout=layout;self._bound_contract=contract
        self._bound_reduction=self.reduction;self._bound_contract_state=contract_state
        self.last_plan=None
        self._guard_bindings()

    @staticmethod
    def rotation_indices():return ScoreReductionLayout.rotation_indices()

    @staticmethod
    def _forget(value):
        for name in tuple(vars(value)):
            if name.startswith('_public_'):delattr(value,name)
        return value

    def _guard_bindings(self):
        if (self.arithmetic is not self._bound_arithmetic or self.layout is not self._bound_layout
                or self.contract is not self._bound_contract or self.reduction is not self._bound_reduction):
            raise ValueError('construction-bound initial-softmax owner was replaced')
        if _contract_state(self.contract)!=self._bound_contract_state:
            raise ValueError('construction-bound initial-softmax contract changed')
        self.reduction._guard_context()

    def evaluate_tile(self,value,*,capture_checkpoints=False):
        self._guard_bindings()
        if type(capture_checkpoints)is not bool:raise ValueError('checkpoint option must be an explicit bool')
        self.reduction._validate(value,24)
        arithmetic=self.arithmetic;context=arithmetic.ctx;cache=context._param_cache
        input_level=arithmetic._level(value);input_scale=value._ckks_scale
        q=tuple(cache.q_moduli_at_level(input_level))
        plan=initial_softmax_plan(q,input_level=input_level,canonical_scale=arithmetic.scale,input_scale=input_scale)
        contract=self.contract;layout=self.layout
        snapshots={};original=value
        fingerprint=self.reduction._value_fingerprint(value)
        def guard():
            self._guard_bindings()
            if (self.arithmetic is not arithmetic or arithmetic.ctx is not context or context._param_cache is not cache
                    or self.layout is not layout or self.contract is not contract
                    or tuple(cache.q_moduli_at_level(input_level))!=q
                    or self.reduction._value_fingerprint(original)!=fingerprint):
                raise ValueError('initial-softmax static input/context/contract changed')
        # The local callback wrapper validates the captured input immediately,
        # before the reduction may execute its next rotate/add primitive.
        callback=self.reduction.progress
        def observed_progress(*args):
            callback(*args);guard()
        reduction=ScoreReduction(arithmetic,layout,
            progress=observed_progress if callback is not None else None)
        def note(name,result):
            guard();self._forget(result)
            if (arithmetic._level(result)!=plan['levels'][name] or result._ckks_scale!=plan['scales'][name]
                    or type(result._ckks_nsd)is not int or result._ckks_nsd!=1):
                raise RuntimeError('initial-softmax primitive depth/NSD differs')
            if capture_checkpoints:snapshots[name]=result
            reduction._note('initial softmax '+name,result);guard();self._forget(result)
            return result
        mean=note('score_mean',reduction.mean(value))
        # Frozen center/32 then exp domain map*2 is exactly the power-of-two
        # coefficient1/16. Public masking is applied before polynomial powers.
        t=note('normalized_t',arithmetic.weighted_sum([value,mean],
            [layout.input_mask/16.,-layout.input_mask/16.]))
        del mean,value
        terms={1:t}
        def power(n):
            if n not in terms:terms[n]=self._forget(arithmetic.multiply(power(n//2),power((n+1)//2)))
            return terms[n]
        coefficients=contract.power_coefficients
        base=note('base_exp',arithmetic.weighted_sum([power(i) for i in range(1,9)],
            [coefficients[i]*layout.input_mask for i in range(1,9)],bias=coefficients[0]*layout.input_mask))
        terms.clear();del t
        denominator=note('mean_exp',reduction.mean_with_dummy(base,
            domain=contract.initial_domain,inactive_dummy=1.))
        inverse=None
        for update,inverse in integer_newton_updates(arithmetic,denominator,normalization_factor=1,
                initial_estimate=contract.initial_seed,updates=contract.initial_updates,
                first_output_scale=plan['static_inverse_scale_plan']['initial_target_scale']):
            guard();reduction._note(f'initial softmax inverse update{update}',inverse);guard()
        inverse=note('inverse',inverse);del denominator
        product=note('base_times_inverse',arithmetic.multiply(base,inverse));del base,inverse
        if product._ckks_scale!=plan['actual_product_scale']:
            raise RuntimeError('initial-softmax product scale differs from actual recurrence')
        output=note('probabilities',arithmetic.weighted_sum([product],[layout.mean_weight]));del product
        if output._ckks_scale!=plan['final_scale']:raise RuntimeError('initial-softmax output scale differs')
        self.last_plan=plan
        return (output,snapshots) if capture_checkpoints else output
