"""One frozen BERT-v2 normalized-squaring round, without refresh or client code.

The retained numerator and denominator share one borrowed full-slot context.
The tracked inverse has a separate component proof; its joined N131072/p60
precision and the complete repeated-round softmax remain unvalidated here.
"""
from dataclasses import dataclass,field
import hashlib,json,math
from encrypted_score_reduction import ScoreReduction,ScoreReductionLayout,validate_public_dummy
from tracked_reciprocal import tracked_scale_plan,tracked_reciprocal_updates,_forward as tracked_scale_forward
from encrypted_scale_planning import positive_scale,drop_divisor,weighted_sum_scale,multiply_scale
MANIFEST_SHA256='4e2277beea7189aa1d77b168c500c8a175f6c630afa520fcc6b38cb5a179cff9'

@dataclass(frozen=True)
class NormalizationRoundContract:
    manifest_bytes:bytes=field(repr=False)
    key_count:int
    domain:tuple=field(init=False)
    seed:float=field(init=False)
    updates:int=field(init=False)
    first_weight:float=field(init=False)
    first_bias:float=field(init=False)
    def __post_init__(self):
        if not isinstance(self.manifest_bytes,bytes) or hashlib.sha256(self.manifest_bytes).hexdigest()!=MANIFEST_SHA256:
            raise ValueError('exact frozen BERT-v2 manifest required')
        if type(self.key_count)is not int or not 1<=self.key_count<=128:raise ValueError('public integer K in1..128 required')
        rule=json.loads(self.manifest_bytes)['normalization_reciprocal_by_public_K'][str(self.key_count)]
        domain=tuple(rule['domain']);seed=rule['seed'];updates=rule['iterations']
        if domain!=(.5/self.key_count,2.) or type(seed)is not float or type(updates)is not int or updates<2:
            raise ValueError('frozen normalization rule differs')
        object.__setattr__(self,'domain',domain);object.__setattr__(self,'seed',seed);object.__setattr__(self,'updates',updates)
        object.__setattr__(self,'first_weight',-seed*seed);object.__setattr__(self,'first_bias',2*seed)
        validate_public_dummy(1.,domain)


CONTRACT_FIELDS=('manifest_bytes','key_count','domain','seed','updates','first_weight','first_bias')

def _contract_state(contract):
    return tuple(ScoreReduction._typed(getattr(contract,name,None)) for name in CONTRACT_FIELDS)

def _verified_contract_state(contract):
    """Reconstruct every derived field from the pinned bytes, including types."""
    if not isinstance(contract,NormalizationRoundContract):raise TypeError('frozen round contract required')
    expected=NormalizationRoundContract(contract.manifest_bytes,contract.key_count)
    state=_contract_state(contract)
    if state!=_contract_state(expected):
        raise ValueError('normalization contract fields differ from the exact manifest')
    return state


def normalization_round_plan(q_moduli,*,input_level,canonical_scale,contract,input_scale=None):
    _verified_contract_state(contract)
    q=tuple(q_moduli);used=contract.updates+5
    if (type(input_level)is not int or input_level<used+2 or len(q)!=2*(input_level+1)
            or any(type(x)is not int or x<=1 for x in q) or len(set(q))!=len(q)
            or isinstance(canonical_scale,bool) or not isinstance(canonical_scale,(int,float))
            or not math.isfinite(canonical_scale) or canonical_scale<=0):
        raise ValueError('round requires its static circuit plus two continuation levels')
    canonical_scale=positive_scale(canonical_scale)
    source=canonical_scale if input_scale is None else positive_scale(input_scale)
    masked=input_level-1;square=input_level-2;denominator=input_level-3
    mask_plan=weighted_sum_scale([source],canonical_scale,drop_divisor(q,input_level))
    mask_plan.update(name='masked_input',input_level=input_level,output_level=masked)
    numerator_scale=multiply_scale(mask_plan['output_scale'],mask_plan['output_scale'],drop_divisor(q,masked))
    denominator_plan=weighted_sum_scale([numerator_scale],canonical_scale,drop_divisor(q,square))
    denominator_plan.update(name='denominator',input_level=square,output_level=denominator)
    inverse=denominator-contract.updates-1
    final_divisor=drop_divisor(q,inverse)
    # Keep the original encoding target selection, including its binary64
    # operation order. The actual replay below changes no encoded operation.
    nominal_numerator=canonical_scale*canonical_scale/drop_divisor(q,masked)
    nominal_target=canonical_scale*final_divisor/nominal_numerator
    nominal=tracked_scale_plan(q[:2*(denominator+1)],input_scale=canonical_scale,
        input_level=denominator,updates=contract.updates,final_scale=nominal_target)
    inverse_plan=tracked_scale_forward(q[:2*(denominator+1)],input_scale=denominator_plan['output_scale'],
        input_level=denominator,updates=contract.updates,first_output_scale=nominal['first_output_scale'])
    output_scale=multiply_scale(numerator_scale,inverse_plan['rows'][-1]['scale'],final_divisor)
    if not math.isfinite(output_scale) or not math.isclose(output_scale,canonical_scale,rel_tol=1e-12):
        raise ValueError('tracked output scale is outside next reduction tolerance')
    return dict(key_count=contract.key_count,updates=contract.updates,normalization_factor=1,
        first_weight=contract.first_weight,first_bias=contract.first_bias,denominator_domain=list(contract.domain),
        input_level=input_level,input_scale=source,canonical_scale=canonical_scale,consumed_levels=used,reserved_output_levels=2,
        levels=dict(masked_input=masked,squared_numerator=square,denominator=denominator,inverse=inverse,probabilities=inverse-1),
        scales=dict(masked_input=mask_plan['output_scale'],squared_numerator=numerator_scale,denominator=denominator_plan['output_scale'],
                    inverse=inverse_plan['rows'][-1]['scale'],probabilities=output_scale),
        weighted_sum_scale_stages=[mask_plan,denominator_plan],nominal_inverse_target_plan=nominal,
        encoding_target_policy='Preserve original nominal target; replay actual binary64 metadata without relabeling.',
        tracked_inverse_scale_plan=inverse_plan,output_relative_scale_error=output_scale/canonical_scale-1,
        final_mask_or_scale_assignment=False,
        mask_omission_reason='Public mask precedes the numerator square; multiplying that retained ideal-zero padding by dummy-lane inverse preserves zero. The actual output scale is retained and checked against the next reduction tolerance.',
        logical_ciphertext_products=2*contract.updates,logical_plaintext_products=3,
        logical_rotations=14,actual_composite_rescales=2*contract.updates+3)


class BertNormalizationRound:
    """Uniform public K across live rows; per-row key positions may differ.

    Denominator-domain membership is a caller assumption, not a ciphertext
    branch or a claim derived from noisy previous probability values.
    """
    def __init__(self,arithmetic,layout,contract,*,progress=None):
        contract_state=_verified_contract_state(contract)
        if not isinstance(layout,ScoreReductionLayout):raise TypeError('full-slot score layout required')
        if not all(int(x)==contract.key_count for x in layout.counts[:layout.valid_rows]):
            raise ValueError('every live row must have the same declared public K')
        self.arithmetic,self.layout,self.contract=arithmetic,layout,contract
        self.reduction=ScoreReduction(arithmetic,layout,progress=progress)
        self._bound_arithmetic=arithmetic;self._bound_layout=layout;self._bound_contract=contract
        self._bound_reduction=self.reduction;self._bound_contract_state=contract_state
        self.last_plan=None
        self._guard_bindings()

    @staticmethod
    def rotation_indices():return ScoreReductionLayout.rotation_indices()

    def _guard_bindings(self):
        if (self.arithmetic is not self._bound_arithmetic or self.layout is not self._bound_layout
                or self.contract is not self._bound_contract or self.reduction is not self._bound_reduction):
            raise ValueError('construction-bound normalization owner was replaced')
        if _contract_state(self.contract)!=self._bound_contract_state:
            raise ValueError('construction-bound normalization contract changed')
        self.reduction._guard_context()

    def evaluate_tile(self,value,*,capture_checkpoints=False):
        self._guard_bindings()
        if type(capture_checkpoints)is not bool:raise ValueError('checkpoint option must be an explicit bool')
        contract=self.contract;arithmetic=self.arithmetic;layout=self.layout
        self.reduction._validate(value,contract.updates+7)
        original=value;fingerprint=self.reduction._value_fingerprint(value);retained_fingerprints={}
        plan=normalization_round_plan(arithmetic.ctx._param_cache.q_moduli_at_level(arithmetic._level(value)),
            input_level=arithmetic._level(value),canonical_scale=arithmetic.scale,contract=contract,input_scale=value._ckks_scale)
        def guard():
            self._guard_bindings()
            if (self.arithmetic is not arithmetic or self.contract is not contract or self.layout is not layout
                    or self.reduction._value_fingerprint(original)!=fingerprint
                    or any(self.reduction._value_fingerprint(saved)!=state for saved,state in retained_fingerprints.values())):
                raise ValueError('round input, retained value or owner changed')
        callback=self.reduction.progress
        def observed(*args):callback(*args);guard()
        reduction=ScoreReduction(arithmetic,layout,progress=observed if callback is not None else None)
        checkpoints={};inverse_steps=[]
        def note(name,result):
            guard();reduction._forget(result)
            if (arithmetic._level(result)!=plan['levels'][name] or result._ckks_scale!=plan['scales'][name]
                    or type(result._ckks_nsd)is not int or result._ckks_nsd!=1):raise RuntimeError('round metadata differs from the actual public schedule')
            reduction._note('normalization round '+name,result);guard();reduction._forget(result)
            if capture_checkpoints:checkpoints[name]=result
            return result
        masked=note('masked_input',reduction._weighted(value,layout.input_mask,label='normalization masked input'))
        numerator=note('squared_numerator',arithmetic.multiply(masked,masked));del masked,value
        retained_fingerprints['numerator']=(numerator,reduction._value_fingerprint(numerator))
        denominator=note('denominator',reduction._reduce_broadcast(numerator,dummy=1.,label='normalization denominator'))
        retained_fingerprints['denominator']=(denominator,reduction._value_fingerprint(denominator))
        estimate=None
        for update,estimate,residual in tracked_reciprocal_updates(arithmetic,denominator,normalization_factor=1,
                first_weight=contract.first_weight,first_bias=contract.first_bias,updates=contract.updates,
                first_output_scale=plan['tracked_inverse_scale_plan']['first_output_scale']):
            guard();reduction._note(f'normalization inverse y{update}',estimate);guard();reduction._forget(estimate)
            if residual is not None:
                reduction._note(f'normalization inverse e{update}',residual);guard();reduction._forget(residual)
            if capture_checkpoints:inverse_steps.append((update,estimate,residual))
        estimate=note('inverse',estimate);del denominator
        del retained_fingerprints['denominator']
        output=note('probabilities',arithmetic.multiply(numerator,estimate));del numerator,estimate
        retained_fingerprints.clear()
        self.last_plan=plan
        if capture_checkpoints:return output,checkpoints,inverse_steps
        return output
