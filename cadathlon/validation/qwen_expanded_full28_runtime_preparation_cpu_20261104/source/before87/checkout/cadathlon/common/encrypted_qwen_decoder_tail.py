"""Candidate3 decoder tail with explicit conditional continuation.

Consumes actual attended2048 tiles and their original L25 hidden1024 inputs.
The original residual is really masked before either residual addition. Public
norm/SiLU domains and refresh error envelopes remain caller obligations. This
owner borrows synchronous keyless objects and never owns a context or codecs.
"""
from dataclasses import dataclass,field
import json
import numpy as np
from encrypted_qwen_head_routing import QwenHeadRouting,typed,MANIFEST,CONFIG
from encrypted_qwen_rmsnorm import QwenRMSContract
from encrypted_qwen_rms_refresh import QwenInverseRefreshBinding,accessor,typed as typed_tree
from encrypted_qwen_causal_softmax import complete_fingerprint,validate_structure,require_synchronous
from encrypted_qwen_attention_join import (QwenAttentionJoin,QwenAttentionJoinResult,
    QwenOuterRefreshBinding,QwenOuterRefreshEvent,_CompleteRMS,JOIN_PLAN,CHECKPOINT,digest)
from encrypted_qwen_silu import QwenSiLU
from qwen_silu_contract import QwenSiLUContract
from encrypted_chained_streaming_linear import ChainedStreamingLinear
from encrypted_streaming_linear import RowTileLayout
from encrypted_scale_planning import drop_divisor,weighted_sum_scale,multiply_scale

PARAMETERS='60c556376e42358f227a2978edb6b2addf59dc9e694b65e17265b15e813d65c3'
_QUARANTINE=[]
ROLES=('o','gate','up','down')

@dataclass(frozen=True)
class QwenDecoderTailContract:
    manifest_bytes:bytes=field(repr=False)
    site_metadata_bytes:bytes=field(repr=False)
    certificate_bytes:bytes=field(repr=False)
    join_plan_bytes:bytes=field(repr=False)
    layer:int
    sequence_length:int
    outer_events:tuple=field(init=False)
    def __post_init__(self):
        if type(self.layer)is not int or not 0<=self.layer<28:raise ValueError('exact layer0..27 required')
        if type(self.sequence_length)is not int or not 1<=self.sequence_length<=32:raise ValueError('exact sequence length1..32 required')
        if type(self.join_plan_bytes)is not bytes or digest(self.join_plan_bytes)!=JOIN_PLAN:raise ValueError('exact public continuation plan required')
        p=json.loads(self.join_plan_bytes)
        if p['candidate_manifest_sha256']!=MANIFEST or p['passed']is not True:raise ValueError('continuation identity differs')
        norm=QwenRMSContract(self.manifest_bytes,self.site_metadata_bytes,f'{self.layer}.post.inverse_sqrt')
        QwenSiLUContract(self.manifest_bytes,self.certificate_bytes,self.layer)
        row=next(r for r in p['normalizations'] if r['site']==norm.site)
        if row['output_level']!=norm.output_level:raise ValueError('post RMS level differs')
        object.__setattr__(self,'outer_events',(
            QwenOuterRefreshEvent('raw_post',self.layer,21 if self.sequence_length==1 else 19,float(row['raw_input_bound'])),
            QwenOuterRefreshEvent('normalized_post',self.layer,norm.output_level,float(row['output_bound']))))

def contract_state(c):
    return (tuple(typed(getattr(c,n,None)) for n in ('manifest_bytes','site_metadata_bytes','certificate_bytes','join_plan_bytes','layer','sequence_length')),
        type(c.outer_events),tuple((type(e),tuple(typed(getattr(e,n,None)) for n in QwenOuterRefreshEvent.__dataclass_fields__)) for e in c.outer_events))
def verify_contract(c):
    if type(c)is not QwenDecoderTailContract:raise TypeError('typed frozen decoder-tail contract required')
    canonical=QwenDecoderTailContract(c.manifest_bytes,c.site_metadata_bytes,c.certificate_bytes,c.join_plan_bytes,c.layer,c.sequence_length)
    if contract_state(c)!=contract_state(canonical):raise ValueError('complete typed decoder-tail contract changed')
    return contract_state(c)

class QwenTailRefreshBinding(QwenOuterRefreshBinding):
    def __init__(self,owner,refresher):
        self._event_state=verify_contract(owner.contract)
        QwenInverseRefreshBinding.__init__(self,owner,refresher)
    def check(self):
        QwenInverseRefreshBinding.check(self);require_synchronous(self.arithmetic,self.refresher)
        if contract_state(self.owner.contract)!=self._event_state:raise ValueError('tail outer event contract changed')

@dataclass(frozen=True)
class QwenDecoderTailResult:
    outputs:tuple
    original_positions:tuple
    layer:int
    metadata:tuple
    refresh_events:tuple

class QwenDecoderTail(QwenHeadRouting):
    # Reuse the reviewed complete state and cache checks without instantiating
    # a second attention graph or replacing any of the child arithmetic.
    _compilers=QwenAttentionJoin._compilers
    _compiler_configuration=QwenAttentionJoin._compiler_configuration
    _callback_cache_state=QwenAttentionJoin._callback_cache_state
    _linear_public_state=QwenAttentionJoin._linear_public_state
    _linear_inventory=QwenAttentionJoin._linear_inventory
    _state=QwenAttentionJoin._state
    _check_refs=QwenAttentionJoin._check_refs
    _remember=QwenAttentionJoin._remember
    _call=QwenAttentionJoin._call
    _metadata=QwenAttentionJoin._metadata
    _refresh=QwenAttentionJoin._refresh
    def _fingerprint(self,value):return complete_fingerprint(self,value)
    def _validate(self,value,depth):return validate_structure(self,value,depth)
    def __init__(self,arithmetic,linear,contract,config_bytes,*,parameter_manifest_bytes,weights,gamma,positions,refresher,progress=None):
        if _QUARANTINE:raise RuntimeError('failed decoder tail requires dedicated-process exit')
        state=verify_contract(contract);require_synchronous(arithmetic,refresher)
        if type(positions)is not tuple or len(positions)!=contract.sequence_length:raise ValueError('contract sequence and original positions differ')
        if not isinstance(linear,ChainedStreamingLinear) or linear.ctx is not arithmetic.ctx:raise TypeError('borrowed same-context chained linear required')
        if not isinstance(linear.layout,RowTileLayout) or typed((linear.layout.slots,linear.layout.features))!=typed((65536,4096)):raise ValueError('fullslot F4096 linear required')
        require_synchronous(linear.arithmetic)
        if accessor(linear,'linear_tile')[1]is not ChainedStreamingLinear.linear_tile:raise ValueError('actual chained linear method required')
        if callable(getattr(linear.arithmetic,'declare_bound',None)) or getattr(linear.arithmetic,'refresher',None)is not None:raise ValueError('ordinary static linear arithmetic required')
        if type(parameter_manifest_bytes)is not bytes or digest(parameter_manifest_bytes)!=PARAMETERS:raise ValueError('authenticated decoder tensor manifest required')
        p=json.loads(parameter_manifest_bytes)
        if p['candidate_sha256']!=MANIFEST or p['config_sha256']!=CONFIG or p['checkpoint_sha256']!=CHECKPOINT:raise ValueError('decoder parameter identities differ')
        if type(weights)is not dict or set(weights)!=set(ROLES):raise ValueError('exact four decoder matrix roles required')
        self.contract=contract;self.linear=linear;self.refresher=refresher
        frozen=[]
        for role,name,shape in (('o','self_attn.o_proj',(1024,2048)),('gate','mlp.gate_proj',(3072,1024)),('up','mlp.up_proj',(3072,1024)),('down','mlp.down_proj',(1024,3072))):
            w=weights[role];row=p['matrices'][f'model.layers.{contract.layer}.{name}.weight']
            if type(w)is not np.ndarray or w.dtype!=np.dtype('float64') or w.shape!=shape or not np.isfinite(w).all():raise ValueError('exact public matrix shape/dtype required')
            raw=np.asarray(w,dtype='<f8').tobytes()
            if digest(raw)!=row['sha256'] or row['shape']!=list(shape):raise ValueError('public decoder matrix bytes differ')
            frozen.append(np.frombuffer(raw,dtype='<f8').reshape(shape))
        self.weights=tuple(frozen)
        self._planned_linear_keys=tuple(sorted({k for w in self.weights for k in linear.required_rotation_indices(w,linear.layout)}))
        self._planned_refresh_keys=tuple(refresher.plan['rotations'])
        super().__init__(arithmetic,contract.manifest_bytes,config_bytes,positions=positions,progress=progress)
        self._contract=state;self._refs={};self._phase_refs={};self._phase=None;self._events=[];self._attention=None;self._attention_state=None
        self._child_progress=self._subprogress
        norm=QwenRMSContract(contract.manifest_bytes,contract.site_metadata_bytes,f'{contract.layer}.post.inverse_sqrt')
        self.norm=_CompleteRMS(arithmetic,norm,config_bytes,gamma=gamma,positions=positions,refresher=refresher if norm.events else None,progress=self._child_progress)
        silu=QwenSiLUContract(contract.manifest_bytes,contract.certificate_bytes,contract.layer)
        self.silu=QwenSiLU(arithmetic,silu,config_bytes,positions=positions,refresher=refresher if silu.doublings else None,progress=self._child_progress)
        self.outer=QwenTailRefreshBinding(self,refresher)
        self._objects=(contract,linear,linear.arithmetic,linear.layout,refresher,self.weights,self.norm,self.silu,self.outer)
        self._compiler_state=self._compiler_configuration();self._method_refs=self._methods()
        self._linear_layout=typed(tuple(vars(linear.layout).items()));self._linear_state=self._linear_public_state()
        self._linear_keys=self._planned_linear_keys;self._linear_key_state=self._linear_inventory()
        self._weight_state=tuple((id(w),id(w.base),typed(w.shape),w.dtype.str,w.__array_interface__['data'][0]) for w in self.weights)
        self._weight_owners=tuple(w.base for w in self.weights)
        self._join_guard()
    def required_rotation_indices(self):
        powers={s%65536 for b in range(10) for s in (1<<b,-(1<<b))}
        return tuple(sorted(powers|set(self._planned_linear_keys)|set(self._planned_refresh_keys)))
    def _methods(self):
        targets=[(self.linear,('linear_tile','required_rotation_indices','_validate_ciphertext')),(self.linear.arithmetic,('_ensure_capacity','_prepare'))]
        for a in (self.arithmetic,self.linear.arithmetic):
            targets.append((a,('rotate','weighted_sum','multiply','add','_level','_finish')))
            targets.extend((getattr(a,n),(method,)) for n,method in (('rotation','rotate'),('_plaintext_multiply','multiply'),('_rescale','rescale')))
        targets.extend(((self.norm,('evaluate_tile','_rms_guard','_fingerprint','_validate')),(self.silu,('evaluate_tile','_silu_guard','_fingerprint','_validate')),(self.outer,('refresh_event','check'))))
        targets.extend((c,('bind','clear_executables')) for c in self._compilers())
        return tuple(accessor(o,n) for o,names in targets for n in names)
    def _join_guard(self):
        if _QUARANTINE:raise RuntimeError('failed decoder tail requires dedicated-process exit')
        super()._guard();require_synchronous(self.arithmetic,self.refresher);require_synchronous(self.linear.arithmetic)
        now=(self.contract,self.linear,self.linear.arithmetic,self.linear.layout,self.refresher,self.weights,self.norm,self.silu,self.outer)
        if any(x is not y for x,y in zip(now,self._objects)) or contract_state(self.contract)!=self._contract:raise ValueError('construction-bound decoder tail changed')
        if self.linear.ctx is not self.ctx or self.linear.arithmetic.ctx is not self.ctx or typed(tuple(vars(self.linear.layout).items()))!=self._linear_layout or callable(getattr(self.linear.arithmetic,'declare_bound',None)) or getattr(self.linear.arithmetic,'refresher',None)is not None:raise ValueError('borrowed linear layout/arithmetic changed')
        if any(x is not y for now,old in zip(self._methods(),self._method_refs) for x,y in zip(now,old)):raise ValueError('construction-bound tail accessor changed')
        if self._compiler_configuration()!=self._compiler_state or self._linear_public_state()!=self._linear_state or self._linear_inventory()!=self._linear_key_state:raise ValueError('construction-bound compiler/linear state changed')
        if any(w.flags.writeable or (id(w),id(w.base),typed(w.shape),w.dtype.str,w.__array_interface__['data'][0])!=s for w,s in zip(self.weights,self._weight_state)):raise ValueError('immutable decoder matrix storage changed')
        self.norm._rms_guard();self.silu._silu_guard();self.outer.check();self._check_refs()
        if self._attention is not None and self._result_state(self._attention)!=self._attention_state:raise ValueError('retained attention result structure changed')
    @staticmethod
    def _result_state(r):
        return (type(r),id(r.outputs),tuple(id(v) for v in r.outputs),typed_tree(r.original_positions),typed_tree(r.layer),typed_tree(r.metadata),typed_tree(r.refresh_events))
    def _subprogress(self,label,value,metadata):
        self._join_guard();self._validate(value,0)
        # Complete child owners retain their own basis/inverse original arrays;
        # parent owns its original residual, up branch and current callback.
        self._phase_refs={'current':self._state(value)}
        self._notify(f'{self._phase}.{label}',value,metadata)
    def _notify(self,label,value,metadata):
        self._join_guard();current=self._state(value)
        refs=self._refs;phase_refs=self._phase_refs
        retained=tuple(refs.items());phase_retained=tuple(phase_refs.items())
        borrowed=tuple(state[0] for name,state in retained if name.startswith(('input.','attention.')))
        def owned(v):return not any(v is supplied for supplied in borrowed)
        if owned(value):self._forget(value)
        cache,cache_owners=self._callback_cache_state()
        if self.progress is not None:self.progress(label,value,dict(metadata))
        self._join_guard()
        if (self._refs is not refs or self._phase_refs is not phase_refs or
            tuple(refs)!=tuple(n for n,_ in retained) or tuple(phase_refs)!=tuple(n for n,_ in phase_retained) or
            any(refs[n] is not state for n,state in retained) or any(phase_refs[n] is not state for n,state in phase_retained)):
            raise ValueError('callback changed retained tail owner map')
        if any(v.polynomial is not arrays[0] or self._fingerprint(v)!=state for v,arrays,state in
               (current,*(s for _,s in retained),*(s for _,s in phase_retained))):raise ValueError('callback changed retained tail payload')
        if self._callback_cache_state()[0]!=cache:raise ValueError('callback changed active compiled cache/binding')
        assert cache_owners
        if owned(value):self._forget(value)
        for prior,_,_ in (*refs.values(),*phase_refs.values()):
            if owned(prior):self._forget(prior)
    def _linear(self,value,kind,tile):
        self._join_guard();level=self.arithmetic._level(value)
        scale=weighted_sum_scale([value._ckks_scale],self.linear.arithmetic.scale,drop_divisor(self.ctx._param_cache.q_towers,level))['output_scale']
        result=self._call(f'tile{tile}.{kind}.linear',lambda:self.linear.linear_tile(value,self.weights[ROLES.index(kind)],valid_rows=self.layout.tile_rows(tile)))
        self._metadata(result,level-1,scale);self._notify(f'tile{tile}.{kind}.linear',result,dict(level=level-1,scale=scale));return result
    def evaluate(self,attention,inputs):
        self._join_guard()
        if self._active:raise ValueError('serial decoder tail cannot be reentered')
        if type(attention)is not QwenAttentionJoinResult or typed(attention.layer)!=typed(self.contract.layer) or typed(attention.original_positions)!=typed(self.layout.positions):raise ValueError('actual typed same-layer attention result required')
        if type(inputs)is not tuple or len(inputs)!=self.layout.tile_count or type(attention.outputs)is not tuple or len(attention.outputs)!=len(inputs):raise ValueError('exact ordered original and attention tile tuples required')
        if type(attention.metadata)is not tuple or len(attention.metadata)!=len(inputs) or type(attention.refresh_events)is not tuple:raise ValueError('typed attention result metadata required')
        attention_level=22 if self.contract.sequence_length==1 else 20
        for tile,(source,value) in enumerate(zip(inputs,attention.outputs)):
            self._metadata(source,25);self._metadata(value,attention_level)
            if typed_tree(attention.metadata[tile])!=typed_tree(dict(tile=tile,level=attention_level,scale=value._ckks_scale)):raise ValueError('attention public metadata differs from payload')
        self._active=True;self.last_run=None;self._refs={};self._phase_refs={};self._events=[]
        self._attention=attention;self._attention_state=self._result_state(attention);entered=False;stages=[];outputs=[];initial_states=()
        a=self.arithmetic;q=self.ctx._param_cache.q_towers
        def note(label,v,level,scale):
            self._metadata(v,level,scale);self._notify(label,v,dict(level=level,scale=scale));stages.append(dict(label=label,level=level,scale=scale));return v
        def add(label,x,y):
            self._join_guard();owners=(self._state(x),self._state(y));level=min(a._level(x),a._level(y));scale=x._ckks_scale
            out=a.add(x,y)
            if any(v.polynomial is not refs[0] or self._fingerprint(v)!=state for v,refs,state in owners):raise ValueError('add changed retained operand')
            return note(label,out,level,scale)
        try:
            for tile,(source,value) in enumerate(zip(inputs,attention.outputs)):
                self._remember(f'input.{tile}',source);self._remember(f'attention.{tile}',value)
            initial_states=tuple(self._refs.values())
            entered=True
            for tile,(source,attended) in enumerate(zip(inputs,attention.outputs)):
                mask=np.zeros((16,4096));mask[:self.layout.tile_rows(tile),:1024]=1.
                scale=weighted_sum_scale([source._ckks_scale],a.scale,drop_divisor(q,25))['output_scale']
                masked=self._remember('masked_original',note(f'tile{tile}.original_residual_mask',a.weighted_sum([source],[mask.ravel()]),24,scale))
                projected=self._remember('projected',self._linear(attended,'o',tile))
                residual=self._remember('raw_post',add(f'tile{tile}.attention_residual',masked,projected))
                fresh=self._remember('fresh_post',self._refresh(residual,'raw_post',tile))
                normalized=self._call(f'tile{tile}.post_norm',lambda:self.norm.evaluate_tile(fresh,tile=tile)).output
                self._remember('post_norm',normalized)
                fresh=self._remember('fresh_normalized',self._refresh(normalized,'normalized_post',tile))
                gate=self._remember('gate',self._linear(fresh,'gate',tile))
                up=self._remember('up',self._linear(fresh,'up',tile))
                activation=self._call(f'tile{tile}.silu',lambda:self.silu.evaluate_tile(gate,tile=tile)).output
                self._remember('activation',activation);self._join_guard()
                level=min(a._level(activation),a._level(up));scale=multiply_scale(activation._ckks_scale,up._ckks_scale,drop_divisor(q,level))
                product=self._remember('gated',note(f'tile{tile}.gated',a.multiply(activation,up),level-1,scale))
                down=self._remember('down',self._linear(product,'down',tile))
                output=add(f'tile{tile}.output',residual,down);self._metadata(output,4,residual._ckks_scale)
                outputs.append(self._remember(f'output.{tile}',output))
                keep={f'input.{i}' for i in range(len(inputs))}|{f'attention.{i}' for i in range(len(inputs))}|{f'output.{i}' for i in range(len(outputs))}
                self._refs={k:v for k,v in self._refs.items() if k in keep}
            self._join_guard();metadata=tuple(dict(tile=i,level=4,scale=v._ckks_scale) for i,v in enumerate(outputs))
            self.last_run=dict(layer=self.contract.layer,stages=stages,metadata=metadata,outer_refreshes=list(self._events))
            return QwenDecoderTailResult(tuple(outputs),self.layout.positions,self.contract.layer,metadata,tuple(self._events))
        except BaseException as error:
            self.last_run=None
            if entered:
                _QUARANTINE.append((self,inputs,attention,initial_states+tuple(self._refs.values()),tuple(self._phase_refs.values()),error))
                error.add_note('entered decoder-tail failure: borrowed owners/arrays quarantined; outer key loader must quarantine until process exit')
            raise
        finally:
            self._active=False
            if not _QUARANTINE:self._refs={};self._phase_refs={};self._attention=None;self._attention_state=None
