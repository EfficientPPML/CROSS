"""Experimental serial candidate3 RMS/QKV/RoPE/softmax/PV composition.

All nonlinear domains and refresh envelopes remain conditional public inputs.
This owner neither inspects clear ranges nor owns keys, codecs or a context.
It returns attended2048 rows; output projection and the decoder block are absent.
An entered failure quarantines borrowed owners until dedicated-process exit.
"""
from dataclasses import dataclass,field
import hashlib,json,math
import numpy as np
from encrypted_qwen_head_routing import QwenHeadRouting,typed,MANIFEST,CONFIG
from encrypted_qwen_rmsnorm import QwenRMSContract,QwenRMSNorm
from encrypted_qwen_rms_refresh import QwenInverseRefreshBinding,accessor
from encrypted_qwen_causal_softmax import (QwenCausalSoftmaxContract,QwenCausalSoftmax,
    complete_fingerprint,validate_structure,require_synchronous)
from encrypted_qwen_attention_bypass import QwenAttentionRoutingWithBypass
from encrypted_chained_streaming_linear import ChainedStreamingLinear
from encrypted_streaming_linear import RowTileLayout
from encrypted_scale_planning import drop_divisor,weighted_sum_scale

JOIN_PLAN='cf6071b33de34dd3973d34e85ec699380fbf2d67efe808443ab50339394355e6'
PARAMETERS='c707068e1a247f4ef14fda3493a3e2edf9dea2f11ec19e94531a64d80dbcf48b'
from qwen_candidate_bridge import verify_parameters
CHECKPOINT='f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b'
_QUARANTINE=[]
def digest(raw):return hashlib.sha256(raw).hexdigest()

@dataclass(frozen=True)
class QwenOuterRefreshEvent:
    stage:str
    layer:int
    input_level:int
    public_bound:float
    output_level:int=25

@dataclass(frozen=True)
class QwenAttentionJoinContract:
    manifest_bytes:bytes=field(repr=False)
    site_metadata_bytes:bytes=field(repr=False)
    certificate_bytes:bytes=field(repr=False)
    join_plan_bytes:bytes=field(repr=False)
    layer:int
    outer_events:tuple=field(init=False)
    def __post_init__(self):
        if type(self.layer)is not int or not 0<=self.layer<28:raise ValueError('exact layer0..27 required')
        if type(self.join_plan_bytes)is not bytes or digest(self.join_plan_bytes)!=JOIN_PLAN:raise ValueError('exact public continuation plan required')
        plan=json.loads(self.join_plan_bytes)
        if plan['candidate_manifest_sha256']!=MANIFEST or plan['passed']is not True:raise ValueError('public continuation binding differs')
        QwenCausalSoftmaxContract(self.manifest_bytes,self.certificate_bytes,self.layer)
        sites={r['site']:r for r in plan['normalizations']};rope={r['site']:r for r in plan['rope_bounds']}
        events=[]
        for kind in ('input','q','k'):
            site=f'{self.layer}.{kind}.inverse_sqrt';rule=QwenRMSContract(self.manifest_bytes,self.site_metadata_bytes,site)
            row=sites[site]
            if row['output_level']!=rule.output_level:raise ValueError('normalization level plan differs')
            if kind=='input':events.append(QwenOuterRefreshEvent('normalized_input',self.layer,rule.output_level,float(row['output_bound'])))
            else:
                events.append(QwenOuterRefreshEvent(kind+'_raw',self.layer,24,float(row['raw_input_bound'])))
                reserve=rule.output_level==2
                if rope[site]['requires_extra_pre_rope_refresh']is not reserve:raise ValueError('pre-RoPE reserve schedule differs')
                if reserve:events.append(QwenOuterRefreshEvent(kind+'_reserve',self.layer,2,float(row['output_bound'])))
                events.append(QwenOuterRefreshEvent(kind+'_rope',self.layer,24 if reserve else rule.output_level-1,float(rope[site]['public_refresh_bound'])))
        object.__setattr__(self,'outer_events',tuple(events))

def contract_state(c):
    return (tuple(typed(getattr(c,n,None)) for n in ('manifest_bytes','site_metadata_bytes','certificate_bytes','join_plan_bytes','layer')),
        type(c.outer_events),tuple((type(e),tuple(typed(getattr(e,n,None)) for n in QwenOuterRefreshEvent.__dataclass_fields__)) for e in c.outer_events))
def verify_contract(c):
    if type(c)is not QwenAttentionJoinContract:raise TypeError('typed frozen join contract required')
    rebuilt=QwenAttentionJoinContract(c.manifest_bytes,c.site_metadata_bytes,c.certificate_bytes,c.join_plan_bytes,c.layer)
    if contract_state(c)!=contract_state(rebuilt):raise ValueError('complete typed join contract changed')
    return contract_state(c)

class QwenOuterRefreshBinding(QwenInverseRefreshBinding):
    """Distinct stage-bound outer events; inherited checks only validate engine."""
    def __init__(self,owner,refresher):
        self._event_state=verify_contract(owner.contract)
        super().__init__(owner,refresher)
    def check(self):
        super().check();require_synchronous(self.arithmetic,self.refresher)
        if contract_state(self.owner.contract)!=self._event_state:raise ValueError('outer event contract changed')
    def refresh_event(self,value,*,event):
        self.check()
        if self._active:raise ValueError('outer refresh cannot be reentered')
        if type(event)is not QwenOuterRefreshEvent or not any(event is e for e in self.owner.contract.outer_events):raise ValueError('construction-bound outer event required')
        self.owner._validate(value,0)
        if typed(self.arithmetic._level(value))!=typed(event.input_level):raise ValueError('outer refresh entry level differs')
        original=(value.polynomial,getattr(value,'moduli_array',None),self.owner._fingerprint(value))
        self._active=True
        try:
            result=self.refresher.refresh(value,public_bound=event.public_bound)
            self.check()
            if value.polynomial is not original[0] or self.owner._fingerprint(value)!=original[2]:raise ValueError('outer refresh changed retained source')
            self.owner._validate(result,0)
            if typed(self.arithmetic._level(result))!=typed(25) or not math.isclose(result._ckks_scale,self.arithmetic.scale,rel_tol=1e-12):raise ValueError('outer refresh output differs')
            self.owner._forget(result)
            return result
        finally:self._active=False
    def refresh(self,*args,**kwargs):raise ValueError('outer refresh requires an explicit typed event')

class _CompleteRMS(QwenRMSNorm):
    def _fingerprint(self,value):return complete_fingerprint(self,value)
    def _validate(self,value,depth):return validate_structure(self,value,depth)
class _CompleteHead(QwenHeadRouting):
    def _fingerprint(self,value):
        # Preserve the predecessor's first two metadata entries: its rotation
        # check deliberately compares them while allowing new payload identity.
        return QwenHeadRouting._fingerprint(self,value)+complete_fingerprint(self,value)[1:]
    def _validate(self,value,depth):return validate_structure(self,value,depth)

@dataclass(frozen=True)
class QwenAttentionJoinResult:
    outputs:tuple
    original_positions:tuple
    layer:int
    metadata:tuple
    refresh_events:tuple

class QwenAttentionJoin(QwenHeadRouting):
    def _fingerprint(self,value):return complete_fingerprint(self,value)
    def _validate(self,value,depth):return validate_structure(self,value,depth)
    def __init__(self,arithmetic,linear,contract,config_bytes,*,parameter_manifest_bytes,
                 weights,gammas,positions,refresher,progress=None):
        if _QUARANTINE:raise RuntimeError('failed attention owner requires dedicated-process exit')
        state=verify_contract(contract);require_synchronous(arithmetic,refresher)
        if not isinstance(linear,ChainedStreamingLinear) or linear.ctx is not arithmetic.ctx:raise TypeError('borrowed same-context chained linear required')
        if typed((linear.layout.slots,linear.layout.features))!=typed((65536,4096)) or not isinstance(linear.layout,RowTileLayout):raise ValueError('full-slot F4096 linear required')
        require_synchronous(linear.arithmetic)
        if accessor(linear,'linear_tile')[1]is not ChainedStreamingLinear.linear_tile:raise ValueError('actual chained linear method required')
        if callable(getattr(linear.arithmetic,'declare_bound',None)) or getattr(linear.arithmetic,'refresher',None)is not None:raise ValueError('ordinary static linear arithmetic required')
        if type(parameter_manifest_bytes)is not bytes or digest(parameter_manifest_bytes)!=PARAMETERS:raise ValueError('authenticated84 public QKV tensor manifest required')
        parameters=verify_parameters(contract.manifest_bytes,parameter_manifest_bytes,'qkv')
        if (parameters['schema']!='qwen-qkv-public-parameters-v1' or
            parameters['config_sha256']!=CONFIG or parameters['checkpoint_sha256']!=CHECKPOINT):raise ValueError('public parameter identities differ')
        if type(weights)is not dict or set(weights)!={'q','k','v'} or type(gammas)is not dict or set(gammas)!={'input','q','k'}:raise ValueError('exact QKV and three norm parameters required')
        self.contract=contract;self.linear=linear;self.refresher=refresher
        frozen=[]
        for kind in ('q','k','v'):
            value=weights[kind];shape=(2048 if kind=='q' else 1024,1024)
            name=f'model.layers.{contract.layer}.self_attn.{kind}_proj.weight';row=parameters['matrices'][name]
            if type(value)is not np.ndarray or value.dtype!=np.dtype('float64') or value.shape!=shape or not np.isfinite(value).all():raise ValueError('exact public QKV shape/dtype required')
            raw=np.asarray(value,dtype='<f8').tobytes()
            if digest(raw)!=row['sha256'] or row['shape']!=list(shape):raise ValueError('public QKV bytes differ')
            frozen.append(np.frombuffer(raw,dtype='<f8').reshape(shape))
        self.weights=tuple(frozen)
        self._planned_linear_keys=tuple(sorted({key for w in self.weights for key in linear.required_rotation_indices(w,linear.layout)}))
        self._planned_refresh_keys=tuple(refresher.plan['rotations'])
        super().__init__(arithmetic,contract.manifest_bytes,config_bytes,positions=positions,progress=progress)
        self._contract=state;self._refs={};self._phase_refs={};self._phase=None;self._events=[]
        self._child_progress=self._subprogress
        self.norms=tuple(_CompleteRMS(arithmetic,QwenRMSContract(contract.manifest_bytes,contract.site_metadata_bytes,f'{contract.layer}.{kind}.inverse_sqrt'),
            config_bytes,gamma=gammas[kind],positions=positions,
            refresher=refresher if QwenRMSContract(contract.manifest_bytes,contract.site_metadata_bytes,f'{contract.layer}.{kind}.inverse_sqrt').events else None,
            progress=self._child_progress) for kind in ('input','q','k'))
        self.head=_CompleteHead(arithmetic,contract.manifest_bytes,config_bytes,positions=positions,progress=self._child_progress)
        self.softmax=QwenCausalSoftmax(arithmetic,QwenCausalSoftmaxContract(contract.manifest_bytes,contract.certificate_bytes,contract.layer),config_bytes,
            positions=positions,refresher=refresher if len(positions)>1 else None,progress=self._child_progress)
        self.pv=QwenAttentionRoutingWithBypass(arithmetic,contract.manifest_bytes,config_bytes,positions=positions,progress=self._child_progress)
        self.outer=QwenOuterRefreshBinding(self,refresher)
        self._join_objects=(contract,linear,linear.arithmetic,linear.layout,refresher,self.weights,self.norms,self.head,self.softmax,self.pv,self.outer)
        self._children=tuple(self.norms)+(self.head,self.softmax,self.pv)
        self._compiler_state=self._compiler_configuration()
        self._method_refs=self._methods()
        self._linear_layout=typed(tuple(vars(linear.layout).items()))
        self._linear_state=self._linear_public_state()
        self._linear_keys=self._planned_linear_keys
        self._linear_key_state=self._linear_inventory()
        self._weight_state=tuple((id(w),id(w.base),typed(w.shape),w.dtype.str,w.__array_interface__['data'][0]) for w in self.weights)
        self._weight_owners=tuple(w.base for w in self.weights)
        self._join_guard()
    def required_rotation_indices(self):
        geometry={s%65536 for s in (*range(1,len(self.layout.positions)),*[-96*h for h in range(1,16)])}
        powers={s%65536 for bit in range(10) for s in (1<<bit,-(1<<bit))}
        return tuple(sorted(set(super().required_rotation_indices())|geometry|powers|
            set(self._planned_linear_keys)|set(self._planned_refresh_keys)))
    def _methods(self):
        targets=[(self.linear,('linear_tile','required_rotation_indices','_validate_ciphertext')),
                 (self.linear.arithmetic,('_ensure_capacity','_prepare'))]
        for a in (self.arithmetic,self.linear.arithmetic):
            targets.append((a,('rotate','weighted_sum','multiply','add','_level','_finish')))
            targets.extend((getattr(a,n),(method,)) for n,method in (('rotation','rotate'),('_plaintext_multiply','multiply'),('_rescale','rescale')))
        targets.extend((n,('evaluate_tile','_rms_guard','_fingerprint','_validate')) for n in self.norms)
        targets.extend(((self.head,('rope','expand_kv','scores','_guard','_fingerprint','_validate')),
            (self.softmax,('evaluate_tile','_softmax_guard')),(self.pv,('attend_softmax','_bypass_guard')),
            (self.outer,('refresh_event','check'))))
        targets.extend((compiler,('bind','clear_executables')) for compiler in self._compilers())
        return tuple(accessor(obj,name) for obj,names in targets for name in names)
    def _compilers(self):
        return tuple(compiler for a in (self.arithmetic,self.linear.arithmetic) for compiler in
            (a._multiplication_compiler,a.rotation._compiler,a._plaintext_multiply.compiler,a._rescale.compiler))
    def _compiler_configuration(self):
        from collections.abc import Mapping
        result=[]
        for compiler in self._compilers():
            capacity=getattr(compiler,'max_executables',None);cache=getattr(compiler,'_executables',None)
            if type(capacity)is not int or capacity<1 or not isinstance(cache,Mapping):raise ValueError('explicit compiler configuration required')
            result.append((id(compiler),typed(capacity),id(cache)))
        return tuple(result)
    def _callback_cache_state(self):
        """Dynamic caches may evolve during primitives, never inside callbacks.

        Keep their original reachable objects alive while comparing identities;
        this is not an arbitrary-Python sandbox or a failed-DMA reclamation proof.
        """
        refs=[];state=[]
        def binding(value):
            refs.append(value)
            if isinstance(value,tuple):return (id(value),tuple(binding(v) for v in value))
            executable=getattr(value,'executable',None);controls=getattr(value,'controls',None)
            refs.extend((executable,controls))
            payload=getattr(value,'polynomial',None);moduli_array=getattr(value,'moduli_array',None)
            refs.extend((payload,moduli_array))
            metadata=() if payload is None else tuple(typed(getattr(value,n,None)) for n in
                ('_ckks_scale','_ckks_nsd','moduli','num_moduli','degree_layout','precision','num_elements','degree','batch','r','c'))
            return (id(value),id(executable),id(controls),tuple(id(v) for v in controls) if isinstance(controls,tuple) else (),
                id(payload),id(moduli_array),id(getattr(value,'ntt_ctx',None)),metadata)
        def mapping(value):
            refs.append(value)
            return (id(value),tuple((id(k),binding(v)) for k,v in value.items()))
        for compiler in self._compilers():state.append(mapping(compiler._executables))
        for a in (self.arithmetic,self.linear.arithmetic):
            state.append(mapping(a._multiplication_bindings));state.append(mapping(a._constant_cache))
            state.extend(binding(getattr(helper,name,None)) for helper,name in
                ((a.rotation,'_binding'),(a._plaintext_multiply,'binding'),(a._rescale,'binding')))
            state.append(typed(getattr(a,'_active_level',None)))
        return tuple(state),tuple(refs)
    def _linear_public_state(self):
        a=self.linear.arithmetic
        if typed(a.scale)!=typed(self.arithmetic.scale) or a.strategy!='power':raise ValueError('ordinary canonical-target linear facade required')
        objects=[]
        for name,method in (('rotation','rotate'),('_plaintext_multiply','multiply'),('_rescale','rescale')):
            helper=getattr(a,name,None)
            if helper is None or getattr(helper,'ctx',None)is not self.ctx:raise ValueError('linear helpers must share borrowed context')
            objects.append(id(helper))
        return (typed(a.scale),typed(a.strategy),id(getattr(a,'progress',None)),tuple(objects),id(getattr(a,'_multiplication_compiler',None)))
    def _linear_inventory(self):
        from collections.abc import Mapping
        cache=self.ctx._param_cache;rows=[]
        if cache.per_level_rotation_keys is not False:raise ValueError('reusable linear keys required')
        for key in self._linear_keys:
            row=cache.raw_rotation_keys.get(key);mapping=cache.coef_maps.get(key)
            if not isinstance(row,Mapping) or row.get('a')is None or row.get('b')is None or mapping is None:raise ValueError('missing actual matrix rotation pair/map')
            rows.append((key,id(row),id(row['a']),id(row['b']),id(mapping)))
        return tuple(rows)
    def _join_guard(self):
        if _QUARANTINE:raise RuntimeError('failed attention owner requires dedicated-process exit')
        super()._guard();require_synchronous(self.arithmetic,self.refresher);require_synchronous(self.linear.arithmetic)
        now=(self.contract,self.linear,self.linear.arithmetic,self.linear.layout,self.refresher,self.weights,self.norms,self.head,self.softmax,self.pv,self.outer)
        if any(a is not b for a,b in zip(now,self._join_objects)) or contract_state(self.contract)!=self._contract:raise ValueError('construction-bound attention join changed')
        if (self.linear.ctx is not self.ctx or self.linear.arithmetic.ctx is not self.ctx or
            typed(tuple(vars(self.linear.layout).items()))!=self._linear_layout or
            callable(getattr(self.linear.arithmetic,'declare_bound',None)) or getattr(self.linear.arithmetic,'refresher',None)is not None):raise ValueError('borrowed linear layout/arithmetic changed')
        if any(a is not b for current,old in zip(self._methods(),self._method_refs) for a,b in zip(current,old)):raise ValueError('construction-bound join accessor changed')
        if self._compiler_configuration()!=self._compiler_state:raise ValueError('construction-bound compiler configuration changed')
        if self._linear_public_state()!=self._linear_state or self._linear_inventory()!=self._linear_key_state:raise ValueError('construction-bound linear facade/inventory changed')
        if any(w.flags.writeable or (id(w),id(w.base),typed(w.shape),w.dtype.str,w.__array_interface__['data'][0])!=state for w,state in zip(self.weights,self._weight_state)):raise ValueError('immutable public QKV storage changed')
        for norm in self.norms:norm._rms_guard()
        self.head._guard();self.softmax._softmax_guard();self.pv._bypass_guard();self.outer.check()
        self._check_refs()
    def _state(self,value):return (value,(value.polynomial,getattr(value,'moduli_array',None)),self._fingerprint(value))
    def _check_refs(self):
        for value,arrays,state in (*self._refs.values(),*self._phase_refs.values()):
            if value.polynomial is not arrays[0] or self._fingerprint(value)!=state:raise ValueError('retained join source/result changed')
    def _remember(self,name,value):self._refs[name]=self._state(value);return value
    def _subprogress(self,label,value,metadata):
        self._join_guard();self._validate(value,0)
        if self._phase.endswith(('.softmax','.pv')):
            # These frozen successors already retain complete original array
            # owners for their bounded live working sets. Keep the callback's
            # current result here, not every retired K/update intermediate.
            self._phase_refs={'current':self._state(value)}
        elif self._phase.endswith('_norm'):
            if label=='map' or (label.startswith('T') and '.' not in label):key=label
            elif label=='variance':key='variance'
            elif label=='inverse' or (label.startswith('inverse') and '.' not in label):key='inverse'
            else:key='current'
            self._phase_refs[key]=self._state(value)
        else:self._phase_refs[id(value)]=self._state(value)
        self._notify(f'{self._phase}.{label}',value,metadata)
        if self._phase.endswith('_norm') and label=='inverse':
            # All balanced basis results survived the inverse callback. The
            # child retires them immediately afterward; mirror that boundary.
            self._phase_refs={n:s for n,s in self._phase_refs.items() if n!='map' and not str(n).startswith('T')}
    def _notify(self,label,value,metadata):
        self._join_guard();current=self._state(value);self._forget(value)
        cache_state,cache_owners=self._callback_cache_state()
        if self.progress is not None:self.progress(label,value,dict(metadata))
        self._join_guard()
        if self._callback_cache_state()[0]!=cache_state:raise ValueError('callback changed active compiled cache/binding')
        assert cache_owners
        if value.polynomial is not current[1][0] or self._fingerprint(value)!=current[2]:raise ValueError('join callback changed current result')
        self._forget(value)
        for prior,_,_ in (*self._refs.values(),*self._phase_refs.values()):
            if not any(prior is self._refs[n][0] for n in self._refs if n.startswith('input.')):self._forget(prior)
    def _call(self,label,fn):
        self._join_guard();self._phase=label;self._phase_refs={}
        result=fn();self._join_guard();self._phase_refs={};return result
    def _metadata(self,value,level,scale=None):
        self._validate(value,0)
        if typed(self.arithmetic._level(value))!=typed(level) or (scale is not None and typed(value._ckks_scale)!=typed(scale)):raise ValueError('joined result level/scale differs')
    def _refresh(self,value,stage,tile):
        event=next(e for e in self.contract.outer_events if e.stage==stage)
        result=self.outer.refresh_event(value,event=event);self._join_guard()
        row=dict(stage=stage,tile=tile,input_level=event.input_level,public_bound=event.public_bound,level=25,scale=result._ckks_scale)
        self._events.append(row);self._notify(f'tile{tile}.{stage}.refresh',result,row)
        return result
    def _linear(self,value,kind,tile):
        self._join_guard();level=self.arithmetic._level(value);q=self.ctx._param_cache.q_towers
        scale=weighted_sum_scale([value._ckks_scale],self.linear.arithmetic.scale,drop_divisor(q,level))['output_scale']
        index=('q','k','v').index(kind)
        result=self._call(f'tile{tile}.{kind}.linear',lambda:self.linear.linear_tile(value,self.weights[index],valid_rows=self.layout.tile_rows(tile)))
        self._metadata(result,level-1,scale);self._notify(f'tile{tile}.{kind}.linear',result,dict(level=level-1,scale=scale))
        return result
    def evaluate(self,inputs):
        self._join_guard()
        if self._active:raise ValueError('serial attention join cannot be reentered')
        if type(inputs)is not tuple or len(inputs)!=self.layout.tile_count:raise ValueError('exact ordered row-tile tuple required')
        for value in inputs:self._metadata(value,25)
        self._active=True;self.last_run=None;self._events=[];self._refs={};self._phase_refs={};entered=False
        try:
            for tile,value in enumerate(inputs):self._remember(f'input.{tile}',value)
            entered=True
            queries=[];keys=[];values=[]
            for tile,value in enumerate(inputs):
                norm=self._call(f'tile{tile}.input_norm',lambda:self.norms[0].evaluate_tile(value,tile=tile)).output
                norm=self._remember(f'tile{tile}.normalized',self._refresh(norm,'normalized_input',tile))
                q=self._remember(f'tile{tile}.q_raw',self._linear(norm,'q',tile))
                k=self._remember(f'tile{tile}.k_raw',self._linear(norm,'k',tile))
                v=self._remember(f'tile{tile}.v',self._linear(norm,'v',tile))
                for kind,raw,norm_owner,heads in (('q',q,self.norms[1],16),('k',k,self.norms[2],8)):
                    fresh=self._remember(f'tile{tile}.{kind}.fresh',self._refresh(raw,kind+'_raw',tile))
                    normalized=self._call(f'tile{tile}.{kind}_norm',lambda o=norm_owner,x=fresh:o.evaluate_tile(x,tile=tile)).output
                    self._remember(f'tile{tile}.{kind}.norm',normalized)
                    if norm_owner.contract.output_level==2:normalized=self._remember(f'tile{tile}.{kind}.reserved',self._refresh(normalized,kind+'_reserve',tile))
                    roped=self._call(f'tile{tile}.{kind}.rope',lambda x=normalized,h=heads:self.head.rope(x,tile=tile,heads=h))
                    fresh=self._remember(f'tile{tile}.{kind}.rope_refreshed',self._refresh(roped,kind+'_rope',tile))
                    if kind=='q':queries.append(fresh)
                    else:
                        expanded=self._call(f'tile{tile}.k.expand',lambda:self.head.expand_kv(fresh,tile=tile))
                        self._metadata(expanded,24);keys.append(self._remember(f'key.{tile}',expanded))
                expanded=self._call(f'tile{tile}.v.expand',lambda:self.head.expand_kv(v,tile=tile))
                self._metadata(expanded,23);values.append(self._remember(f'value.{tile}',expanded))
                # Completed synchronous Q/K/V payloads outlive temporary phase inputs.
                keep={f'input.{i}' for i in range(len(inputs))}|{f'key.{i}' for i in range(tile+1)}|{f'value.{i}' for i in range(tile+1)}|{f'query.{i}' for i in range(tile+1)}
                self._remember(f'query.{tile}',queries[-1]);self._refs={n:s for n,s in self._refs.items() if n in keep}
            outputs=[];metadata=[]
            for tile,query in enumerate(queries):
                scores=None if len(self.layout.positions)==1 else self._call(f'tile{tile}.scores',lambda:self.head.scores(query,tuple(keys),query_tile=tile)).output
                if scores is not None:self._metadata(scores,21);self._remember('scores',scores)
                probabilities=self._call(f'tile{tile}.softmax',lambda:self.softmax.evaluate_tile(scores,query_tile=tile))
                if probabilities.output is not None:self._remember('probabilities',probabilities.output)
                attended=self._call(f'tile{tile}.pv',lambda:self.pv.attend_softmax(probabilities,tuple(values),query_tile=tile)).output
                level=22 if len(self.layout.positions)==1 else 20;self._metadata(attended,level)
                outputs.append(self._remember(f'output.{tile}',attended));row=dict(tile=tile,level=level,scale=attended._ckks_scale)
                metadata.append(row);self._notify(f'tile{tile}.output',attended,row)
                self._refs.pop('scores',None);self._refs.pop('probabilities',None)
            self._join_guard();self.last_run=dict(layer=self.contract.layer,metadata=metadata,outer_refreshes=list(self._events))
            return QwenAttentionJoinResult(tuple(outputs),self.layout.positions,self.contract.layer,tuple(metadata),tuple(self._events))
        except BaseException as error:
            self.last_run=None
            if entered:
                _QUARANTINE.append((self,inputs,tuple(self._refs.values()),tuple(self._phase_refs.values()),error))
                error.add_note('entered attention failure: borrowed owners/arrays quarantined; outer key loader must also quarantine until process exit')
            raise
        finally:
            self._active=False
            if not _QUARANTINE:self._refs={};self._phase_refs={}
