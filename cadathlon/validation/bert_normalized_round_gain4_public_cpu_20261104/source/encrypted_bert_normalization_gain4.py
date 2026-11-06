"""Isolated fixed-gain4 normalized round; no refresh, codec or scale relabeling.

All approximation contracts and real operations retain the frozen predecessor.
Only actual public encoding targets change. This is not an accuracy guarantee.
Entered failures retain borrowed backing until the dedicated process retires.
"""
from collections import Counter
from collections.abc import Mapping
import hashlib,json,math
import numpy as np
from encrypted_bert_normalization_round import (
    NormalizationRoundContract,_verified_contract_state,_contract_state)
from encrypted_score_reduction import ScoreReduction,ScoreReductionLayout,SHIFTS
from encrypted_scale_planning import positive_scale,drop_divisor,weighted_sum_scale,multiply_scale
from encrypted_integer_affine import _encode_checked
from tracked_reciprocal import tracked_scale_plan,tracked_reciprocal_updates,_forward

Q_SHA='fc2b841869ef2b2b556209323ed033c5a177344d1de395fc94e8a5865df39a49'
P_SHA='c03a649ab504238a4f24d57d7fe8e49f2a445c5d842579494cd243a7043d1d49'
DELTA=1.1467305003699405e18
_QUARANTINE=[]

def typed(value):
    if isinstance(value,(tuple,list)):return type(value),tuple(typed(x)for x in value)
    if isinstance(value,Mapping):return type(value),tuple((typed(k),typed(v))for k,v in value.items())
    return type(value),value

def accessor(owner,name):
    method=getattr(owner,name);return getattr(method,'__self__',None),getattr(method,'__func__',method)

def gain4_round_plan(q_moduli,*,input_level,canonical_scale,contract,input_scale=None):
    _verified_contract_state(contract);q=tuple(q_moduli);used=contract.updates+5
    if type(input_level)is not int or input_level!=25 or len(q)!=52 or any(type(x)is not int or x<=1 for x in q)or len(set(q))!=len(q):raise ValueError('fixed level25 CD2 schedule required')
    if typed(canonical_scale)!=typed(DELTA):raise ValueError('fixed p60 canonical scale required')
    source=canonical_scale if input_scale is None else positive_scale(input_scale)
    if not math.isclose(source,canonical_scale,rel_tol=1e-12):raise ValueError('input scale outside original continuation tolerance')
    mask=weighted_sum_scale([source],4*canonical_scale,drop_divisor(q,25))
    numerator=multiply_scale(mask['output_scale'],mask['output_scale'],drop_divisor(q,24))
    anchor=weighted_sum_scale([numerator],16*canonical_scale,drop_divisor(q,23))
    inverse_level=22-contract.updates-1;divisor=drop_divisor(q,inverse_level)
    nominal_numerator=(4*canonical_scale)*(4*canonical_scale)/drop_divisor(q,24)
    nominal=tracked_scale_plan(q[:46],input_scale=16*canonical_scale,input_level=22,updates=contract.updates,final_scale=canonical_scale*divisor/nominal_numerator)
    inverse=_forward(q[:46],input_scale=anchor['output_scale'],input_level=22,updates=contract.updates,first_output_scale=nominal['first_output_scale'])
    final=multiply_scale(numerator,inverse['rows'][-1]['scale'],divisor)
    if not math.isclose(final,canonical_scale,rel_tol=1e-12)or inverse_level-1<2:raise ValueError('actual final scale/reserve differs')
    return dict(key_count=contract.key_count,gain=4,updates=contract.updates,input_scale=source,input_level=25,
        levels=dict(masked_input=24,squared_numerator=23,denominator=22,inverse=inverse_level,probabilities=inverse_level-1),
        scales=dict(masked_input=mask['output_scale'],squared_numerator=numerator,denominator=anchor['output_scale'],inverse=inverse['rows'][-1]['scale'],probabilities=final),
        weighted_sum_scale_stages=[mask,anchor],nominal_inverse_target_plan=nominal,tracked_inverse_scale_plan=inverse,
        consumed_levels=used,reserved_output_levels=2,logical_ciphertext_products=2*contract.updates,logical_plaintext_products=3,
        logical_rotations=14,actual_composite_rescales=2*contract.updates+3,final_mask_or_scale_assignment=False,
        real_polynomial_unchanged=True,denominator_domain=list(contract.domain),fixed_absolute_gate=1e-6)

def weighted_at_scale(arithmetic,value,weight,*,bias=0.,output_scale):
    """Original one-input weighted_sum order with an explicit real PT target."""
    level=arithmetic._level(value);weight=arithmetic._public_slots(weight);bias=arithmetic._public_slots(bias)
    arithmetic._prepare(level)
    divisor=math.prod(arithmetic.ctx._param_cache.q_moduli_at_level(level)[-2:])
    scale=positive_scale(output_scale)*divisor/value._ckks_scale
    plaintext=_encode_checked(arithmetic,weight,level,scale)
    result=arithmetic._plaintext_multiply.multiply(value,plaintext,level)
    if np.any(bias):
        constant=_encode_checked(arithmetic,bias,level,result._ckks_scale)
        result=arithmetic.ctx.he_add[level].add_plain(result,constant)
    result=arithmetic._rescale.rescale(result,level,level-1)
    result._ckks_nsd=1
    return arithmetic._finish(result)

class BertNormalizationRoundGain4:
    def __init__(self,arithmetic,layout,contract,*,progress=None):
        if _QUARANTINE:raise RuntimeError('entered failure requires dedicated-process retirement')
        self._contract=_verified_contract_state(contract)
        if type(layout)is not ScoreReductionLayout or any(int(k)!=contract.key_count for k in layout.counts[:layout.valid_rows]):raise ValueError('fixed uniform public K layout required')
        if progress is not None and not callable(progress):raise TypeError('callable progress required')
        self.arithmetic,self.layout,self.contract,self.progress=arithmetic,layout,contract,progress
        self.reduction=ScoreReduction(arithmetic,layout)
        self.ctx=arithmetic.ctx;self.cache=self.ctx._param_cache
        self._bound=(arithmetic,layout,contract,progress,self.reduction,self.ctx,self.cache)
        self._helpers=tuple(getattr(arithmetic,n)for n in ('rotation','_plaintext_multiply','_rescale','_multiplication_compiler'))
        self._functions=(weighted_at_scale,tracked_reciprocal_updates,_encode_checked,gain4_round_plan)
        self._methods=self._accessors();self._keys=self.cache.raw_rotation_keys;self._maps=self.cache.coef_maps
        self._key_state=self._inventory();self._active=False;self._failed=None;self.last_plan=None
        self._guard()

    def _accessors(self):
        a=self.arithmetic
        pairs=[accessor(a,n)for n in ('rotate','add','multiply','_level','_prepare','_finish','_public_slots','_encode_public','_clear_working_caches')]
        for h,names in ((a.rotation,('rotate','clear')),(a._plaintext_multiply,('multiply','clear')),(a._rescale,('rescale','clear')),(a._multiplication_compiler,('clear_executables',))):
            pairs.extend(accessor(h,n)for n in names)
        return tuple(pairs)

    def _inventory(self):
        cache=self.cache
        if cache.per_level_rotation_keys is not False or not isinstance(cache.raw_rotation_keys,Mapping)or not isinstance(cache.coef_maps,Mapping):raise ValueError('supplied key/map inventory required')
        rows=[]
        for key in ScoreReductionLayout.rotation_indices():
            pair=cache.raw_rotation_keys.get(key);permutation=cache.coef_maps.get(key)
            if not isinstance(pair,Mapping):raise ValueError('missing public key pair')
            ids=[]
            for name in ('a','b'):
                array=pair.get(name)
                if tuple(getattr(array,'shape',()))!=(8,114,131072)or str(getattr(array,'dtype',None))!='uint32'or isinstance(array,np.ndarray)and array.flags.writeable:raise ValueError('fixed read-only public switching key required')
                ids.append(id(array))
            if tuple(getattr(permutation,'shape',()))!=(131072,)or str(getattr(permutation,'dtype',None))!='int32'or isinstance(permutation,np.ndarray)and permutation.flags.writeable:raise ValueError('fixed public coefficient map required')
            rows.append((key,id(pair),*ids,id(permutation)))
        for name in ('eval_key_a_full','eval_key_b_full'):
            a=getattr(cache,name,None)
            if tuple(getattr(a,'shape',()))!=(8,131072,114)or str(getattr(a,'dtype',None))!='uint32':raise ValueError('supplied relin key required')
            rows.append((name,id(a)))
        return tuple(rows)

    def _guard(self):
        if self._failed is not None or _QUARANTINE:raise RuntimeError('failed owner/backing remains quarantined')
        a,c,k=self.arithmetic,self.ctx,self.cache
        from encrypted_packed_qwen import CompiledArithmetic
        from encrypted_polynomial import EncryptedPolynomialEvaluator
        from finite_field import BarrettContext
        if type(a)is not CompiledArithmetic or accessor(a,'_finish')!=(a,EncryptedPolynomialEvaluator._finish):raise ValueError('actual synchronous compiled arithmetic required')
        if k.ff_context_cls is not BarrettContext:raise ValueError('exact standard Barrett family required')
        if any(x is not y for x,y in zip((a,self.layout,self.contract,self.progress,self.reduction,c,k),self._bound)):raise ValueError('borrowed owner binding changed')
        self.reduction._guard_context()
        if _contract_state(self.contract)!=self._contract:raise ValueError('fixed contract changed')
        if a.ctx is not c or c._param_cache is not k:raise ValueError('context identity changed')
        for obj,n,v in ((a,'low_memory',True),(a,'scale',DELTA),(c,'scaling_factor',DELTA),(c,'degree_layout',(256,512)),(c,'dnum',8),(c,'max_level',49),(k,'degree_layout',(256,512)),(k,'dnum',8)):
            if typed(getattr(obj,n,None))!=typed(v):raise ValueError('fixed serial common profile required: '+n)
        if a.progress is not None or a.refresher is not None or callable(getattr(a,'declare_bound',None)):raise ValueError('no implicit callbacks, refresh or bounds')
        for name,h in (('q_towers',Q_SHA),('p_towers',P_SHA)):
            q=getattr(k,name)
            if any(type(x)is not int for x in q)or hashlib.sha256(json.dumps(list(q)).encode()).hexdigest()!=h or tuple(getattr(c,name))!=tuple(q):raise ValueError('fixed ordered common moduli required')
        if any(getattr(a,n)is not v for n,v in zip(('rotation','_plaintext_multiply','_rescale','_multiplication_compiler'),self._helpers))or any(h.ctx is not c for h in self._helpers[:3]):raise ValueError('compiled helper/context changed')
        if self._accessors()!=self._methods or (weighted_at_scale,tracked_reciprocal_updates,_encode_checked,gain4_round_plan)!=self._functions:raise ValueError('executable accessor changed')
        if k.raw_rotation_keys is not self._keys or k.coef_maps is not self._maps or self._inventory()!=self._key_state:raise ValueError('supplied key/map identity changed')

    def _fingerprint(self,value):
        n=value.ntt_ctx;f=n.ff_ctx;p=value.polynomial
        return (tuple(typed(getattr(value,k,None))for k in ('batch','num_elements','degree','r','c','degree_layout','precision','num_moduli','moduli','_ckks_nsd','_ckks_scale')),
            id(p),tuple(p.shape),str(p.dtype),frozenset(p.devices()),id(n),id(f),type(f),typed(tuple(n.moduli)),typed(tuple(f.moduli)),typed(f.computation_format_is_standard))

    def _validate(self,value,*,level=None,scale=None):
        from jaxite_word import Polynomial
        from finite_field import BarrettContext
        from support import require_secure_execution
        if not isinstance(value,Polynomial):raise TypeError('actual ciphertext required')
        require_secure_execution(self.ctx,value)
        for name,expected in (('batch',1),('num_elements',2),('degree',131072),('r',256),('c',512),('degree_layout',(256,512)),('precision',32),('_ckks_nsd',1)):
            if typed(getattr(value,name,None))!=typed(expected):raise ValueError('typed ciphertext metadata differs: '+name)
        L=self.arithmetic._level(value)
        if type(L)is not int or not 0<=L<=49 or level is not None and L!=level:raise ValueError('unexpected ciphertext level')
        q=tuple(self.cache.q_towers[:2*(L+1)])
        if type(value.num_moduli)is not int or value.num_moduli!=len(q)or tuple(value.moduli)!=q or any(type(x)is not int for x in value.moduli):raise ValueError('ordered Q prefix differs')
        n=value.ntt_ctx;f=n.ff_ctx
        if type(f)is not BarrettContext or f.computation_format_is_standard is not True or tuple(n.moduli)!=q or tuple(f.moduli)!=q:raise ValueError('NTT/field prefix or representation differs')
        if tuple(value.polynomial.shape)!=(1,2,256,512,len(q))or str(value.polynomial.dtype)!='uint32'or len(value.polynomial.devices())!=1:raise ValueError('single-device uint32 payload required')
        if type(value._ckks_scale)is not float or not math.isfinite(value._ckks_scale)or value._ckks_scale<=0 or scale is not None and value._ckks_scale!=scale:raise ValueError('actual scale differs')

    def clear(self):
        if self._active:raise RuntimeError('cannot clear an active borrowed round')
        if self._failed is not None:raise RuntimeError('failed backing cannot be released here')
        self.last_plan=None

    def evaluate_tile(self,value,*,capture_checkpoints=False):
        self._guard()
        if self._active or type(capture_checkpoints)is not bool:raise ValueError('serial exact-bool invocation required')
        self.last_plan=None
        self._validate(value,level=25)
        plan=gain4_round_plan(self.cache.q_moduli_at_level(25),input_level=25,canonical_scale=self.arithmetic.scale,contract=self.contract,input_scale=value._ckks_scale)
        held={};captures={};steps=[];counts=Counter();source=value
        def keep(role,v):
            held[role]=(v,v.polynomial,v.ntt_ctx,v.ntt_ctx.ff_ctx,self._fingerprint(v));return v
        keep('source',source)
        def guard():
            self._guard()
            if self._active is not True:raise ValueError('active ownership changed')
            for v,p,n,f,state in held.values():
                if v.polynomial is not p or v.ntt_ctx is not n or n.ff_ctx is not f or self._fingerprint(v)!=state:raise ValueError('retained metadata or original backing changed')
        def call(role,fn,*args,**kwargs):
            guard();v=fn(*args,**kwargs);self._validate(v);keep(role,v);guard();return v
        def note(name,v):
            guard()
            if name in plan['levels']:self._validate(v,level=plan['levels'][name],scale=plan['scales'][name])
            if capture_checkpoints:keep('capture:'+name,v);captures[name]=v
            if self.progress is not None:self.progress(name,v,dict(counts))
            guard()
            for row in held.values():
                owned=row[0]
                if owned is not source:ScoreReduction._forget(owned)
            return v
        def tree(v,reverse=False):
            for shift in SHIFTS:
                rotated=call('rotation',self.arithmetic.rotate,v,(-shift if reverse else shift)%65536)
                v=call('tree',self.arithmetic.add,v,rotated);counts['rotations']+=1;counts['ciphertext_additions']+=1
            held.pop('rotation',None);return v
        try:
            self._active=True
            masked=call('masked',weighted_at_scale,self.arithmetic,value,self.layout.input_mask,output_scale=4*DELTA);counts['plaintext_products']+=1;note('masked_input',masked)
            numerator=call('numerator',self.arithmetic.multiply,masked,masked);counts['ciphertext_products']+=1;note('squared_numerator',numerator);held.pop('masked',None)
            summed=tree(numerator)
            anchored=call('anchored',weighted_at_scale,self.arithmetic,summed,self.layout.anchors,bias=self.layout.dummy_anchors,output_scale=16*DELTA);counts['plaintext_products']+=1
            denominator=note('denominator',tree(anchored,True));keep('denominator',denominator);held.pop('tree',None);held.pop('anchored',None)
            # The unchanged generator performs the same rounded first affine
            # and original y/e recurrence, with its jointly solved first scale.
            generator=tracked_reciprocal_updates(self.arithmetic,denominator,normalization_factor=1,first_weight=self.contract.first_weight,first_bias=self.contract.first_bias,updates=self.contract.updates,first_output_scale=plan['tracked_inverse_scale_plan']['first_output_scale'])
            for update,y,e in generator:
                guard();self._validate(y);keep('y',y)
                if update==1:counts['plaintext_products']+=1;counts['ciphertext_products']+=1
                else:counts['ciphertext_products']+=1+(update<self.contract.updates)
                if e is not None:self._validate(e);keep('e',e)
                note(f'y{update:02}',y)
                if e is not None:note(f'e{update:02}',e)
                if capture_checkpoints:steps.append((update,y,e))
            inverse=note('inverse',y);held.pop('denominator',None);held.pop('e',None)
            output=call('output',self.arithmetic.multiply,numerator,inverse);counts['ciphertext_products']+=1;note('probabilities',output)
            guard();self.last_plan=plan
            return (output,captures,steps)if capture_checkpoints else output
        except BaseException as error:
            self._failed=error;self.last_plan=None
            _QUARANTINE.append((self,self.arithmetic,self.ctx,source,tuple(held.values()),error,error.__traceback__))
            raise
        finally:self._active=False
