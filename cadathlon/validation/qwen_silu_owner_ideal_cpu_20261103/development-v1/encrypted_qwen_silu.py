"""Literal candidate3 SiLU with explicit conditional tanh continuation.

Borrowed synchronous/keyless arithmetic only. No codecs, range inspection,
implicit refresh, context ownership or changed nonlinear rules.
"""
from dataclasses import dataclass
from collections.abc import Mapping
import json
import numpy as np
from encrypted_qwen_head_routing import QwenHeadRouting,typed
from encrypted_qwen_causal_softmax import (complete_fingerprint,validate_structure,require_synchronous,Q_SHA,P_SHA)
from encrypted_qwen_rms_refresh import QwenInverseRefreshBinding,accessor
from encrypted_scale_planning import positive_scale,drop_divisor,weighted_sum_scale,multiply_scale
from qwen_silu_contract import QwenSiLUContract,contract_state,verify_contract,silu_scale_plan,digest,DELTA
_QUARANTINE=[]

@dataclass(frozen=True)
class QwenSiLUResult:
    output:object
    original_positions:tuple
    layer:int
    checkpoints:dict
    plan:dict

class QwenSiLU(QwenHeadRouting):
    def _fingerprint(self,value):return complete_fingerprint(self,value)
    def _validate(self,value,depth):return validate_structure(self,value,depth)
    def required_rotation_indices(self):return ()
    def __init__(self,arithmetic,contract,config_bytes,*,positions,refresher=None,progress=None):
        if _QUARANTINE:raise RuntimeError('failed SiLU owner requires dedicated-process exit')
        self.contract=contract;state=verify_contract(contract);self.refresher=refresher
        require_synchronous(arithmetic,refresher)
        if bool(contract.doublings)!=(refresher is not None):raise ValueError('refresher supplied exactly for rational sites')
        super().__init__(arithmetic,contract.manifest_bytes,config_bytes,positions=positions,progress=progress)
        self._contract=state;self._objects=(contract,refresher);self._refs={}
        self.refresh_binding=QwenInverseRefreshBinding(self,refresher) if refresher is not None else None
        self._refresh_binding=self.refresh_binding;self._methods=self._accessors();self._compiler_state=self._compiler_configuration()
        self._silu_guard()
    def _compilers(self):
        a=self.arithmetic
        return (a._multiplication_compiler,a.rotation._compiler,a._plaintext_multiply.compiler,a._rescale.compiler)
    def _accessors(self):
        a=self.arithmetic
        targets=[(a,('weighted_sum','multiply','_level','_finish')),(a.rotation,('rotate',)),
            (a._plaintext_multiply,('multiply',)),(a._rescale,('rescale',))]
        targets.extend((c,('bind','clear_executables')) for c in self._compilers())
        return tuple(accessor(o,n) for o,names in targets for n in names)
    def _compiler_configuration(self):
        result=[]
        for c in self._compilers():
            if type(getattr(c,'max_executables',None))is not int or c.max_executables<1 or not isinstance(getattr(c,'_executables',None),Mapping):raise ValueError('explicit compiler configuration required')
            result.append((id(c),typed(c.max_executables),id(c._executables)))
        return tuple(result)
    def _cache_state(self):
        refs=[]
        def binding(v):
            refs.append(v)
            if isinstance(v,tuple):return (id(v),tuple(binding(x) for x in v))
            executable=getattr(v,'executable',None);controls=getattr(v,'controls',None)
            payload=getattr(v,'polynomial',None);moduli=getattr(v,'moduli_array',None);refs.extend((executable,controls,payload,moduli))
            meta=() if payload is None else tuple(typed(getattr(v,n,None)) for n in ('_ckks_scale','_ckks_nsd','moduli','num_moduli','degree_layout','precision','num_elements','degree','batch','r','c'))
            return (id(v),id(executable),id(controls),tuple(id(x) for x in controls) if isinstance(controls,tuple) else (),id(payload),id(moduli),id(getattr(v,'ntt_ctx',None)),meta)
        def mapping(v):
            if not isinstance(v,Mapping):raise ValueError('ordinary compiler/constant maps required')
            refs.append(v);return (id(v),tuple((id(k),binding(x)) for k,x in v.items()))
        a=self.arithmetic
        result=tuple(mapping(c._executables) for c in self._compilers())
        result+=(mapping(a._multiplication_bindings),mapping(a._constant_cache))
        result+=tuple(binding(getattr(h,n,None)) for h,n in ((a.rotation,'_binding'),(a._plaintext_multiply,'binding'),(a._rescale,'binding')))
        return result+(typed(getattr(a,'_active_level',None)),),tuple(refs)
    def _silu_guard(self):
        if _QUARANTINE:raise RuntimeError('failed SiLU owner requires dedicated-process exit')
        super()._guard();require_synchronous(self.arithmetic,self.refresher)
        if any(x is not y for x,y in zip((self.contract,self.refresher),self._objects)) or contract_state(self.contract)!=self._contract or self.refresh_binding is not self._refresh_binding:raise ValueError('construction-bound SiLU changed')
        if any(x is not y for now,old in zip(self._accessors(),self._methods) for x,y in zip(now,old)) or self._compiler_configuration()!=self._compiler_state:raise ValueError('construction-bound primitive/compiler changed')
        c=self.ctx;k=c._param_cache
        if (typed(c.max_level)!=typed(49) or typed(c.dnum)!=typed(8) or typed(c.scaling_factor)!=typed(DELTA) or
            digest(json.dumps(list(k.q_towers)).encode())!=Q_SHA or digest(json.dumps(list(k.p_towers)).encode())!=P_SHA):raise ValueError('fixed commonp60/d8 ordered profile required')
        if self.refresh_binding is not None:self.refresh_binding.check()
        self._check_states(tuple(self._refs.values()))
    def _state(self,v):return (v,(v.polynomial,getattr(v,'moduli_array',None)),self._fingerprint(v))
    def _check_states(self,states):
        if any(v.polynomial is not arrays[0] or self._fingerprint(v)!=old for v,arrays,old in states):raise ValueError('retained SiLU source/basis/checkpoint changed')
    def evaluate_tile(self,value,*,tile,capture_checkpoints=False):
        self._silu_guard()
        if self._active:raise ValueError('serial SiLU cannot be reentered')
        if type(capture_checkpoints)is not bool:raise TypeError('exact checkpoint boolean required')
        positions=self.layout.tile_positions(tile);self._validate(value,0)
        a=self.arithmetic;c=self.contract;q=tuple(self.ctx._param_cache.q_towers)
        if typed(a._level(value))!=typed(24):raise ValueError('SiLU requires actual inputL24')
        self._refs={'source':self._state(value)};self._active=True;self.last_run=None
        checkpoints={};stages=[];scales=[];counts=dict(weighted_sums=0,ciphertext_products=0,refreshes=0)
        mask=np.zeros((16,4096));mask[:len(positions),:3072]=1.;mask=mask.ravel()
        def retain(name,v):self._refs[name]=self._state(v);return v
        def metadata(v,level,scale):
            self._validate(v,0)
            if typed(a._level(v))!=typed(level) or typed(v._ckks_scale)!=typed(scale):raise ValueError('SiLU actual metadata differs from recurrence')
        def note(label,v,level,scale,kind,owners=()):
            self._silu_guard();self._check_states(owners);metadata(v,level,scale);self._forget(v);current=self._state(v)
            cache,cache_owners=self._cache_state()
            if self.progress is not None:self.progress(label,v,dict(level=level,scale=scale,counts=dict(counts)))
            self._silu_guard();self._check_states((*owners,current))
            if self._cache_state()[0]!=cache:raise ValueError('callback changed active compiler/constant cache')
            assert cache_owners
            self._forget(v)
            for prior,_,_ in self._refs.values():
                if prior is not value:self._forget(prior)
            stages.append(dict(label=label,kind=kind,level=level,scale=scale))
            if capture_checkpoints:checkpoints[label]=retain('checkpoint.'+label,v)
            return v
        def weighted(label,values,weights,bias=0.):
            self._silu_guard();owners=tuple(self._state(v) for v in values);level=min(a._level(v) for v in values)
            scale=weighted_sum_scale([v._ckks_scale for v in values],DELTA,drop_divisor(q,level))['output_scale'];counts['weighted_sums']+=1
            return note(label,a.weighted_sum(values,weights,bias=bias),level-1,scale,'weighted',owners)
        def mul(label,x,y):
            self._silu_guard();owners=(self._state(x),self._state(y));level=min(a._level(x),a._level(y))
            scale=multiply_scale(x._ckks_scale,y._ckks_scale,drop_divisor(q,level));counts['ciphertext_products']+=1
            return note(label,a.multiply(x,y),level-1,scale,'multiply',owners)
        entered=False
        try:
            entered=True;lo,hi=c.polynomial_domain
            t=weighted('map',[value],[2/(hi-lo)*c.input_factor*mask],-(hi+lo)/(hi-lo)*mask);terms={1:retain('basis1',t)}
            def term(n):
                if n not in terms:
                    product=mul(f'basis{n}.product',term(n//2),term((n+1)//2))
                    out=product if c.strategy=='power' else weighted(f'basis{n}',[product],[2.],-1.) if n%2==0 else weighted(f'basis{n}',[product,t],[2.,-1.])
                    terms[n]=retain(f'basis{n}',out)
                return terms[n]
            indices=[n for n in range(1,len(c.coefficients)) if c.strategy=='chebyshev' or c.coefficients[n]!=0]
            final_mask=mask if not c.doublings else 1.
            result=weighted('output' if not c.doublings else 'polynomial',[term(n) for n in indices],
                [c.coefficients[n]*final_mask for n in indices],c.coefficients[0]*final_mask)
            # All basis arrays survive the sum callback; retire only afterward.
            self._refs={n:s for n,s in self._refs.items() if not n.startswith('basis')};terms.clear();retain('result',result)
            for j,event in enumerate(c.events,1):
                self._silu_guard()
                if typed(a._level(result))!=typed(event[2]):raise ValueError('explicit SiLU refresh entry differs')
                old=self._state(result);fresh=self.refresh_binding.refresh(result,public_bound=event[1]);counts['refreshes']+=1
                result=note(f'double{j}.refresh',fresh,25,positive_scale(fresh._ckks_scale),'refresh',(old,))
                scales.append(result._ckks_scale);stages[-1]['public_bound']=event[1];retain('numerator',result)
                square=mul(f'double{j}.square',result,result);retain('square',square)
                denominator=weighted(f'double{j}.denominator',[square],[1.],1.);retain('denominator',denominator)
                inverse=weighted(f'double{j}.inverse.first',[denominator],[c.inverse_rule[1]],c.inverse_rule[2]);retain('inverse',inverse)
                for update in range(2,c.inverse_rule[3]+1):
                    product=mul(f'double{j}.inverse{update}.product',denominator,inverse)
                    correction=weighted(f'double{j}.inverse{update}.correction',[product],[-1.],2.)
                    inverse=mul(f'double{j}.inverse{update}',inverse,correction);retain('inverse',inverse)
                result=weighted(f'double{j}.output',[mul(f'double{j}.quotient',result,inverse)],[2.]);retain('result',result)
                for name in ('numerator','square','denominator','inverse'):self._refs.pop(name)
            if c.doublings:result=mul('output',value,weighted('sigmoid',[result],[.5*mask],.5*mask));retain('result',result)
            self._silu_guard()
            plan=silu_scale_plan(q,c,input_scale=value._ckks_scale,refresh_scales=tuple(scales))
            if stages!=plan['stages'] or counts!=plan['counts']:raise ValueError('complete actual SiLU operation/scale replay differs')
            self.last_run=plan
            return QwenSiLUResult(result,positions,c.layer,checkpoints,plan)
        except BaseException:
            if entered:_QUARANTINE.append((self,self.ctx,a,self.refresher,self.refresh_binding,dict(self._refs)))
            raise
        finally:
            self._active=False
            if not _QUARANTINE:self._refs={}
