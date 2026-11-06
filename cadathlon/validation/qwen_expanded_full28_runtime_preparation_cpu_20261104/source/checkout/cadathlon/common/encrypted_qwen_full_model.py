"""Experimental full28 candidate3 owner; domains/refresh accuracy are conditional.

Only actual prior layer objects feed later layers. This borrows synchronous,
keyless arithmetic and never loads a model, opens files, encodes or decodes.
Parameter callbacks supply authenticated public arrays, never activations.
Entered failures quarantine original arrays and borrowed owners until process
exit; a failed primitive barrier also requires outer-loader quarantine.
"""
from dataclasses import dataclass
import hashlib,json,math
import numpy as np
from full_contract import FullQwenContract,EntryRefresh,state as contract_state,verify as verify_contract
from encrypted_qwen_head_routing import QwenHeadRouting,typed
from encrypted_qwen_rms_refresh import QwenInverseRefreshBinding,accessor,typed as typed_tree
from encrypted_qwen_rmsnorm import QwenRMSContract,rms_scale_plan
from encrypted_qwen_causal_softmax import complete_fingerprint,validate_structure,require_synchronous
from encrypted_qwen_attention_join import (QwenAttentionJoin,QwenAttentionJoinContract,QwenAttentionJoinResult,
    _CompleteRMS,digest)
from encrypted_qwen_decoder_tail import QwenDecoderTail,QwenDecoderTailContract,QwenDecoderTailResult
from encrypted_chained_streaming_linear import ChainedStreamingLinear
from encrypted_streaming_linear import RowTileLayout
from encrypted_scale_planning import weighted_sum_scale,drop_divisor

_QUARANTINE=[]

# Exact ordered public profile used by the accepted level/scale/key plan.
Q_SHA='fc2b841869ef2b2b556209323ed033c5a177344d1de395fc94e8a5865df39a49'
P_SHA='c03a649ab504238a4f24d57d7fe8e49f2a445c5d842579494cd243a7043d1d49'
DELTA=1.1467305003699405e18

def fixed_profile(arithmetic):
    c=arithmetic.ctx;k=c._param_cache
    for obj,name,expected in ((arithmetic,'scale',DELTA),(c,'scaling_factor',DELTA),(c,'max_level',49),
        (c,'dnum',8),(k,'dnum',8),(c,'degree_layout',(256,512)),(k,'degree_layout',(256,512))):
        if typed_tree(getattr(obj,name,None))!=typed_tree(expected):raise ValueError('fixed common p60/d8 profile differs: '+name)
    for name,count,pin in (('q_towers',100,Q_SHA),('p_towers',14,P_SHA)):
        values=tuple(getattr(k,name))
        if len(values)!=count or any(type(v)is not int for v in values) or hashlib.sha256(json.dumps(list(values)).encode()).hexdigest()!=pin or typed_tree(tuple(getattr(c,name)))!=typed_tree(values):raise ValueError('fixed ordered common Q/P differs')


class QwenParameterSource:
    """Construction-bound callbacks for public matrices/gammas only.

    Returned array bytes are checked against the fixed manifests and copied
    into immutable storage. Callbacks are ordinary trusted Python, not a
    sandbox. File/header/ownership authentication belongs to the caller.
    """
    def __init__(self,contract,*,matrix_loader,gamma_loader):
        self.contract=contract;self._contract=verify_contract(contract)
        if not callable(matrix_loader) or not callable(gamma_loader):raise TypeError('public parameter callbacks required')
        self.matrix_loader=matrix_loader;self.gamma_loader=gamma_loader
        self._objects=(contract,matrix_loader,gamma_loader)
        self._accessors=(accessor(self,'matrix'),accessor(self,'gamma'),accessor(self,'check'),accessor(self,'_freeze'))
    def check(self):
        if any(a is not b for a,b in zip((self.contract,self.matrix_loader,self.gamma_loader),self._objects)) or contract_state(self.contract)!=self._contract:raise ValueError('public parameter source changed')
        now=(accessor(self,'matrix'),accessor(self,'gamma'),accessor(self,'check'),accessor(self,'_freeze'))
        if any(a is not b for n,o in zip(now,self._accessors) for a,b in zip(n,o)):raise ValueError('public parameter accessor changed')
    @staticmethod
    def _freeze(value,shape,pin):
        if type(value)is not np.ndarray or value.dtype!=np.dtype('float64') or value.shape!=shape or not np.isfinite(value).all():raise ValueError('public array shape/dtype/finite contract differs')
        raw=np.asarray(value,dtype='<f8').tobytes()
        if digest(raw)!=pin:raise ValueError('public array digest differs')
        return np.frombuffer(raw,dtype='<f8').reshape(shape)
    def matrix(self,name):
        self.check()
        descriptors={**json.loads(self.contract.qkv_bytes)['matrices'],**json.loads(self.contract.decoder_bytes)['matrices']}
        if type(name)is not str or name not in descriptors:raise ValueError('unknown public matrix')
        row=descriptors[name];value=self.matrix_loader(name);self.check()
        return self._freeze(value,tuple(row['shape']),row['sha256'])
    def gamma(self,site):
        self.check();rule=QwenRMSContract(self.contract.manifest_bytes,self.contract.site_bytes,site)
        value=self.gamma_loader(site);self.check()
        return self._freeze(value,(rule.width,),rule.gamma_sha256)

class QwenContinuationBinding(QwenInverseRefreshBinding):
    def __init__(self,owner,refresher):
        self._contract_state=verify_contract(owner.contract)
        super().__init__(owner,refresher)
    def check(self):
        super().check();require_synchronous(self.arithmetic,self.refresher)
        if contract_state(self.owner.contract)!=self._contract_state:raise ValueError('full continuation contract changed')
    def refresh_event(self,value,*,event):
        self.check()
        if self._active:raise ValueError('continuation refresh cannot be reentered')
        if type(event)is not EntryRefresh or not any(event is e for e in self.owner.contract.entry_events):raise ValueError('exact construction-bound continuation event required')
        self.owner._metadata(value,event.input_level)
        original=self.owner._state(value);self._active=True
        try:
            result=self.owner._returned(self.refresher.refresh(value,public_bound=event.bound))
            self.check();self.owner._assert_states((original,))
            self.owner._metadata(result,25)
            if not math.isclose(result._ckks_scale,self.arithmetic.scale,rel_tol=1e-12):raise ValueError('actual continuation scale outside refresh contract')
            self.owner._forget(result);return result
        finally:self._active=False
    def refresh(self,*args,**kwargs):raise ValueError('typed continuation event required')

@dataclass(frozen=True)
class QwenFullModelResult:
    logit_blocks:tuple
    blocks:tuple
    original_positions:tuple
    last_position:int
    metadata:tuple
    continuation_events:tuple

class QwenFullModel(QwenHeadRouting):
    _compilers=QwenAttentionJoin._compilers
    _compiler_configuration=QwenAttentionJoin._compiler_configuration
    _callback_cache_state=QwenAttentionJoin._callback_cache_state
    _linear_public_state=QwenAttentionJoin._linear_public_state
    _linear_inventory=QwenAttentionJoin._linear_inventory
    _state=QwenAttentionJoin._state
    _check_refs=QwenAttentionJoin._check_refs
    _remember=QwenAttentionJoin._remember
    _metadata=QwenAttentionJoin._metadata
    def _fingerprint(self,value):return complete_fingerprint(self,value)
    def _validate(self,value,depth):return validate_structure(self,value,depth)
    def required_rotation_indices(self):return self.contract.required_keys
    def __init__(self,arithmetic,linear,contract,parameters,*,refresher,progress=None):
        if _QUARANTINE:raise RuntimeError('failed full owner requires dedicated-process exit')
        state=verify_contract(contract);require_synchronous(arithmetic,refresher);fixed_profile(arithmetic)
        if type(parameters)is not QwenParameterSource or parameters.contract is not contract:raise TypeError('exact bound public parameter source required')
        parameters.check()
        if not isinstance(linear,ChainedStreamingLinear) or linear.ctx is not arithmetic.ctx:raise TypeError('borrowed same-context chained linear required')
        if not isinstance(linear.layout,RowTileLayout) or typed((linear.layout.slots,linear.layout.features))!=typed((65536,4096)):raise ValueError('fullslot F4096 linear required')
        if accessor(linear,'linear_tile')[1]is not ChainedStreamingLinear.linear_tile:raise ValueError('actual chained linear method required')
        require_synchronous(linear.arithmetic)
        self.contract=contract;self.parameters=parameters;self.linear=linear;self.refresher=refresher
        super().__init__(arithmetic,contract.manifest_bytes,contract.config_bytes,positions=contract.positions,progress=progress)
        self._profile_function=fixed_profile
        self._contract_state=state;self._refs={};self._phase_refs={};self._phase=None;self._child=None;self._child_state=None
        self._borrowed=();self._callback_owners=();self._result=None;self._result_state=None;self._vocabulary=None;self._vocabulary_state=None;self._pending_return=()
        self._events=[];self._child_progress=self._subprogress;self._saved_child_progress=self._child_progress;self._pending_return=()
        self.outer=QwenContinuationBinding(self,refresher)
        self._objects=(contract,parameters,linear,linear.arithmetic,linear.layout,refresher,self.outer)
        self._class_objects=(QwenAttentionJoin,QwenDecoderTail,_CompleteRMS,QwenParameterSource)
        self._compiler_state=self._compiler_configuration();self._linear_layout=typed(tuple(vars(linear.layout).items()))
        self._linear_state=self._linear_public_state();self._linear_keys=tuple(json.loads(contract.key_plan_bytes)['actual_matrix_keys'])
        self._linear_key_state=self._linear_inventory();self._method_refs=self._methods();self._full_guard()
    def _methods(self):
        targets=[(self.linear,('linear_tile','required_rotation_indices','_validate_ciphertext')),(self.linear.arithmetic,('_ensure_capacity','_prepare')),
            (self.parameters,('check','matrix','gamma')),(self.outer,('check','refresh_event')),
            (self,('_make_attention','_make_tail','_make_final','_child_call','_subprogress','_notify','_returned'))]
        for a in (self.arithmetic,self.linear.arithmetic):
            targets.append((a,('rotate','weighted_sum','multiply','add','_level','_finish')))
            targets.extend((getattr(a,n),(m,)) for n,m in (('rotation','rotate'),('_plaintext_multiply','multiply'),('_rescale','rescale')))
        targets.extend((c,('bind','clear_executables')) for c in self._compilers())
        return tuple(accessor(o,n) for o,names in targets for n in names)
    def _full_guard(self):
        if _QUARANTINE:raise RuntimeError('failed full owner requires dedicated-process exit')
        super()._guard();require_synchronous(self.arithmetic,self.refresher);require_synchronous(self.linear.arithmetic)
        if fixed_profile is not self._profile_function:raise ValueError('fixed profile checker changed')
        fixed_profile(self.arithmetic)
        now=(self.contract,self.parameters,self.linear,self.linear.arithmetic,self.linear.layout,self.refresher,self.outer)
        if any(a is not b for a,b in zip(now,self._objects)) or contract_state(self.contract)!=self._contract_state:raise ValueError('construction-bound full owner changed')
        if any(a is not b for a,b in zip((QwenAttentionJoin,QwenDecoderTail,_CompleteRMS,QwenParameterSource),self._class_objects)):raise ValueError('child implementation identity changed')
        if self.linear.ctx is not self.ctx or self.linear.arithmetic.ctx is not self.ctx or typed(tuple(vars(self.linear.layout).items()))!=self._linear_layout or callable(getattr(self.linear.arithmetic,'declare_bound',None)) or getattr(self.linear.arithmetic,'refresher',None)is not None:raise ValueError('ordinary borrowed linear family changed')
        if any(a is not b for n,o in zip(self._methods(),self._method_refs) for a,b in zip(n,o)):raise ValueError('construction-bound full accessor changed')
        if self._compiler_configuration()!=self._compiler_state or self._linear_public_state()!=self._linear_state or self._linear_inventory()!=self._linear_key_state:raise ValueError('full compiler/linear/key state changed')
        if self._child_progress is not self._saved_child_progress:raise ValueError('saved child callback changed')
        self.parameters.check();self.outer.check();self._check_refs()
        if self._child is not None:
            owner,method,guard,callback=self._child_state
            if self._child is not owner or accessor(owner,'evaluate' if hasattr(owner,'evaluate') else 'evaluate_tile')!=method or accessor(owner,'_join_guard' if hasattr(owner,'_join_guard') else '_rms_guard')!=guard or owner.progress is not callback:raise ValueError('active child owner/accessor changed')
            getattr(owner,'_join_guard' if hasattr(owner,'_join_guard') else '_rms_guard')()
        if self._result is not None and self._structured_state(self._result)!=self._result_state:raise ValueError('retained child result metadata changed')
        if self._vocabulary is not None and self._array_state(self._vocabulary)!=self._vocabulary_state:raise ValueError('immutable vocabulary storage changed')
    @staticmethod
    def _array_state(value):
        return (id(value),id(value.base),typed(value.shape),value.dtype.str,value.__array_interface__['data'][0],value.flags.writeable)
    @staticmethod
    def _structured_state(result):
        return (type(result),id(result.outputs),tuple(id(v) for v in result.outputs),typed_tree(result.original_positions),typed_tree(result.layer),typed_tree(result.metadata),typed_tree(result.refresh_events))
    def _returned(self,result):
        # Retain returned objects and their arrays before metadata validation.
        # A pre-return barrier failure still needs whole-owner/loader quarantine.
        self._pending_return=(result,())
        values=getattr(result,'outputs',None)
        if type(values)is not tuple:values=(getattr(result,'output',result),)
        arrays=[]
        for v in values:
            arrays.append((v,getattr(v,'polynomial',None),getattr(v,'moduli_array',None)))
            self._pending_return=(result,tuple(arrays))
        return result
    def _assert_states(self,states):
        if any(v.polynomial is not arrays[0] or self._fingerprint(v)!=state for v,arrays,state in states):raise ValueError('retained full-model payload changed')
    def _notify(self,label,value,metadata):
        self._full_guard();current=self._state(value);refs=self._refs;phase=self._phase_refs
        saved=tuple(refs.items());phase_saved=tuple(phase.items());cache,cache_owners=self._callback_cache_state()
        states=(current,*(s for _,s in saved),*(s for _,s in phase_saved));pending=self._pending_return;borrowed=self._borrowed
        self._callback_owners=(states,cache_owners,pending,borrowed)
        def owned(v):return not any(v is supplied for supplied in self._borrowed)
        if owned(value):self._forget(value)
        if self.progress is not None:self.progress(label,value,dict(metadata))
        self._full_guard()
        if self._pending_return is not pending or self._borrowed is not borrowed or self._refs is not refs or self._phase_refs is not phase or tuple(refs)!=tuple(n for n,_ in saved) or tuple(phase)!=tuple(n for n,_ in phase_saved) or any(refs[n]is not s for n,s in saved) or any(phase[n]is not s for n,s in phase_saved):raise ValueError('callback changed full retained owner map')
        self._assert_states(states)
        if self._callback_cache_state()[0]!=cache:raise ValueError('callback changed compiled cache/binding')
        for v,_,_ in states:
            if owned(v):self._forget(v)
        self._callback_owners=()
    def _subprogress(self,label,value,metadata):
        self._full_guard();self._validate(value,0)
        self._phase_refs={'current':self._state(value)}
        self._notify(f'{self._phase}.{label}',value,metadata)
    def _make_attention(self,layer):
        c=self.contract
        weights={role:self.parameters.matrix(f'model.layers.{layer}.self_attn.{role}_proj.weight') for role in ('q','k','v')}
        gammas={role:self.parameters.gamma(f'{layer}.{role}.inverse_sqrt') for role in ('input','q','k')};self._full_guard()
        return QwenAttentionJoin(self.arithmetic,self.linear,QwenAttentionJoinContract(c.manifest_bytes,c.site_bytes,c.certificate_bytes,c.continuation_bytes,layer),c.config_bytes,
            parameter_manifest_bytes=c.qkv_bytes,weights=weights,gammas=gammas,positions=c.positions,refresher=self.refresher,progress=self._child_progress)
    def _make_tail(self,layer):
        c=self.contract
        weights={role:self.parameters.matrix(f'model.layers.{layer}.{name}.weight') for role,name in
            (('o','self_attn.o_proj'),('gate','mlp.gate_proj'),('up','mlp.up_proj'),('down','mlp.down_proj'))}
        gamma=self.parameters.gamma(f'{layer}.post.inverse_sqrt');self._full_guard()
        return QwenDecoderTail(self.arithmetic,self.linear,QwenDecoderTailContract(c.manifest_bytes,c.site_bytes,c.certificate_bytes,c.continuation_bytes,layer,len(c.positions)),c.config_bytes,
            parameter_manifest_bytes=c.decoder_bytes,weights=weights,gamma=gamma,positions=c.positions,refresher=self.refresher,progress=self._child_progress)
    def _make_final(self):
        c=self.contract;gamma=self.parameters.gamma('final.inverse_sqrt');self._full_guard()
        return _CompleteRMS(self.arithmetic,QwenRMSContract(c.manifest_bytes,c.site_bytes,'final.inverse_sqrt'),c.config_bytes,
            gamma=gamma,positions=(c.last_position,),refresher=None,progress=self._child_progress)
    def _child_call(self,label,owner,fn):
        self._full_guard();self._phase=label;self._phase_refs={};self._child=owner
        self._child_state=(owner,accessor(owner,'evaluate' if hasattr(owner,'evaluate') else 'evaluate_tile'),accessor(owner,'_join_guard' if hasattr(owner,'_join_guard') else '_rms_guard'),self._child_progress)
        self._full_guard();result=self._returned(fn());self._full_guard()
        self._child=None;self._child_state=None;self._phase_refs={};return result
    def _bind_result(self,result,inputs,layer,kind):
        expected_type=QwenAttentionJoinResult if kind=='attention' else QwenDecoderTailResult
        level=(22 if len(self.layout.positions)==1 else 20) if kind=='attention' else 4
        if type(result)is not expected_type or typed_tree(result.layer)!=typed_tree(layer) or typed_tree(result.original_positions)!=typed_tree(self.layout.positions) or type(result.outputs)is not tuple or len(result.outputs)!=len(inputs) or type(result.metadata)is not tuple or len(result.metadata)!=len(inputs) or type(result.refresh_events)is not tuple:raise ValueError('actual typed ordered child result required')
        for tile,value in enumerate(result.outputs):
            scale=None if kind=='attention' else weighted_sum_scale([inputs[tile]._ckks_scale],self.arithmetic.scale,drop_divisor(self.ctx._param_cache.q_towers,25))['output_scale']
            self._metadata(value,level,scale)
            if typed_tree(result.metadata[tile])!=typed_tree(dict(tile=tile,level=level,scale=value._ckks_scale)):raise ValueError('child result metadata differs from payload')
            self._remember(f'{kind}.{tile}',value)
        self._result=result;self._result_state=self._structured_state(result)
        for tile,value in enumerate(result.outputs):self._notify(f'layer{layer}.{kind}.output{tile}',value,result.metadata[tile])
        return result.outputs
    def _entry_refresh(self,value,event,tile):
        self._full_guard();original=self._state(value);out=self.outer.refresh_event(value,event=event)
        self._full_guard();self._assert_states((original,))
        row=dict(stage=event.stage,tile=tile,input_level=event.input_level,bound=event.bound,level=25,scale=out._ckks_scale)
        self._events.append(row);self._notify(f'{event.stage}.refresh{tile}',out,row);return out
    def evaluate(self,inputs):
        self._full_guard()
        if self._active:raise ValueError('serial full owner cannot be reentered')
        if type(inputs)is not tuple or len(inputs)!=self.layout.tile_count:raise ValueError('exact original embedding tile tuple required')
        for value in inputs:self._metadata(value,25)
        initial_states=tuple(self._state(v) for v in inputs)
        entered=False;blocks=[];metadata=[];layers=[]
        a=self.arithmetic;q=self.ctx._param_cache.q_towers
        try:
            self._active=True
            self.last_run=None;self._refs={};self._phase_refs={};self._borrowed=inputs;self._events=[]
            for i,value in enumerate(inputs):self._remember(f'original.{i}',value)
            current=inputs;entered=True
            for layer in range(28):
                if layer:
                    self._result=None;self._result_state=None
                    event=self.contract.entry_events[layer-1]
                    current=tuple(self._remember(f'current.{i}',self._entry_refresh(v,event,i)) for i,v in enumerate(current))
                else:
                    for i,v in enumerate(current):self._remember(f'current.{i}',v)
                attention_owner=self._make_attention(layer)
                attention=self._child_call(f'layer{layer}.attention',attention_owner,lambda:attention_owner.evaluate(current))
                self._bind_result(attention,current,layer,'attention')
                tail_owner=self._make_tail(layer)
                tail=self._child_call(f'layer{layer}.tail',tail_owner,lambda:tail_owner.evaluate(attention,current))
                current=self._bind_result(tail,current,layer,'block')
                layers.append(dict(layer=layer,metadata=tail.metadata))
                keep={f'original.{i}' for i in range(len(inputs))}|{f'block.{i}' for i in range(len(current))}
                self._refs={n:s for n,s in self._refs.items() if n in keep}
                del attention_owner,tail_owner,attention,tail
            self._result=None;self._result_state=None
            source=current[self.contract.last_tile];mask=np.zeros((16,4096));mask[self.contract.last_local_row,:1024]=1.
            scale=weighted_sum_scale([source._ckks_scale],a.scale,drop_divisor(q,4))['output_scale']
            selected=self._returned(a.weighted_sum([source],[mask.ravel()]));self._metadata(selected,3,scale);self._remember('selected',selected)
            self._notify('final.select_last',selected,dict(level=3,scale=scale,last_position=self.contract.last_position))
            shift=self.contract.last_local_row*4096
            if shift:
                selected=self._returned(a.rotate(selected,shift));self._metadata(selected,3,scale);self._remember('selected_rotated',selected)
                self._notify('final.row_to_zero',selected,dict(level=3,scale=scale,shift=shift))
            fresh=self._remember('final.fresh',self._entry_refresh(selected,self.contract.entry_events[-1],0))
            final_owner=self._make_final();plan=rms_scale_plan(q,final_owner.contract,input_scale=fresh._ckks_scale)
            normalized=self._child_call('final.norm',final_owner,lambda:final_owner.evaluate_tile(fresh,tile=0)).output
            self._metadata(normalized,self.contract.final_norm_level,plan['output_scale']);self._remember('final.norm',normalized)
            self._notify('final.norm.output',normalized,dict(level=self.contract.final_norm_level,scale=normalized._ckks_scale,last_position=self.contract.last_position))
            self._vocabulary=self.parameters.matrix('lm_head.weight');self._vocabulary_state=self._array_state(self._vocabulary);self._full_guard()
            for descriptor in self.contract.vocabulary_blocks:
                self._full_guard();weights=self._vocabulary[descriptor.start:descriptor.stop]
                expected=weighted_sum_scale([normalized._ckks_scale],self.linear.arithmetic.scale,drop_divisor(q,descriptor.input_level))['output_scale']
                value=self._returned(self.linear.linear_tile(normalized,weights,valid_rows=1))
                self._metadata(value,descriptor.output_level,expected);blocks.append(self._remember(f'logits.{descriptor.index}',value))
                row=dict(block=descriptor.index,start=descriptor.start,stop=descriptor.stop,level=descriptor.output_level,scale=expected,last_position=self.contract.last_position)
                metadata.append(row);self._notify(f'logits.{descriptor.index}',value,row)
            self._full_guard()
            self.last_run=dict(layers=layers,continuation_events=tuple(self._events),metadata=tuple(metadata),last_position=self.contract.last_position)
            return QwenFullModelResult(tuple(blocks),self.contract.vocabulary_blocks,self.layout.positions,self.contract.last_position,tuple(metadata),tuple(self._events))
        except BaseException as error:
            self.last_run=None
            if entered:
                _QUARANTINE.append((self,inputs,initial_states,tuple(self._refs.values()),tuple(self._phase_refs.values()),self._callback_owners,self._child,self._result,self._vocabulary,self._pending_return,error))
                error.add_note('entered full Qwen failure: borrowed owners/arrays retained; outer loader must quarantine until dedicated-process exit')
            raise
        finally:
            self._active=False
            if not _QUARANTINE:
                self._refs={};self._phase_refs={};self._borrowed=();self._callback_owners=();self._child=None;self._child_state=None;self._result=None;self._result_state=None;self._vocabulary=None;self._vocabulary_state=None;self._pending_return=()
