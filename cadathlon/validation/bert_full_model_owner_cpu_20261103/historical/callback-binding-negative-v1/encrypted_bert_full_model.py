"""Joined twelve-layer BERT ciphertext owner; no codec or clear activation input.

All later activations come directly from the preceding ciphertext operation.
The caller supplies authenticated public identities, pinned public weights and
original-position embeddings. Public domain and repeated-refresh error premises
remain explicit obligations; this owner does not inspect encrypted ranges.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
import hashlib
import numpy as np
from encrypted_bert_layer_attention import BertLayerAttention
from encrypted_bert_all_ffn import BertFFNTile
from encrypted_bert_pooler_head import BertPoolerHead
from encrypted_bert_attention_tile import BertAttentionTile
from encrypted_bert_all_row_norm import BertRowLayerNorm
from encrypted_chained_streaming_linear import ChainedStreamingLinear, require_chained_rotation_keys
from encrypted_fullslot_refresh_binding import FullSlotRefreshBinding, typed
from layer_attention_contract import make_contract, parameters, PREFIX
from server_public_model import tensor_shapes, TENSOR_SHA256, CANDIDATE_SHA256


def freeze_weights(weights):
    shapes=tensor_shapes()
    if not isinstance(weights,Mapping) or set(weights)!=set(shapes):
        raise ValueError('all196 pinned public server tensors required')
    result={}
    for name,shape in shapes.items():
        value=np.asarray(weights[name])
        if value.dtype.kind!='f' or value.dtype.itemsize not in (4,8) or value.shape!=shape or not np.isfinite(value).all():
            raise ValueError('finite F32/exact promoted F64 public tensor required: '+name)
        with np.errstate(over='ignore',invalid='ignore'):raw=value.astype('<f4');lift=raw.astype('<f8')
        if (not np.isfinite(raw).all() or not np.array_equal(value,lift)
                or hashlib.sha256(raw.tobytes()).hexdigest()!=TENSOR_SHA256[name]):
            raise ValueError('public tensor differs from pinned checkpoint: '+name)
        result[name]=np.frombuffer(lift.tobytes(),dtype='<f8').reshape(shape)
    return MappingProxyType(result)


@dataclass(frozen=True)
class FullModelResult:
    output:object
    checkpoints:dict
    original_positions:tuple
    metadata:dict


class BertFullModel:
    def __init__(self,arithmetic,linear,refresher,weights,manifest_bytes,*,original_positions,context_id,request_id,progress=None):
        if not isinstance(linear,ChainedStreamingLinear):raise TypeError('actual chained streaming linear required')
        if progress is not None and not callable(progress):raise TypeError('progress must be callable or None')
        if type(manifest_bytes)is not bytes or hashlib.sha256(manifest_bytes).hexdigest()!=CANDIDATE_SHA256:
            raise ValueError('exact frozen candidate manifest required')
        # Also validates original CLS0, IDs and all static public profile pins.
        first=make_contract(original_positions,context_id,request_id,layer=0)
        if original_positions[0]!=0:raise ValueError('first compact row must be original CLS position0')
        self.arithmetic,self.linear,self.refresher=arithmetic,linear,refresher
        self.ctx=arithmetic.ctx;self.weights=freeze_weights(weights);self.manifest_bytes=manifest_bytes
        self.original_positions=original_positions;self.positions=tuple(original_positions[i:i+16] for i in range(0,len(original_positions),16))
        self.context_id,self.request_id,self.progress=context_id,request_id,progress
        self.contracts=tuple(make_contract(original_positions,context_id,request_id,layer=i) for i in range(12))
        self.binding=FullSlotRefreshBinding(arithmetic,refresher);self._metadata=self.binding._metadata
        self._active=False;self._inputs=();self._input_states=();self._live=[];self._captured=[];self._children=();self._child_state=()
        self._progress_stage=None;self.last_run=None
        self._callback_bound=self._callback
        self._parameters=parameters();cache=self.ctx._param_cache
        self._bound=(arithmetic,linear,refresher,self.ctx,cache,self.weights,self.manifest_bytes,self.contracts,self.binding,
                     self._metadata,progress,linear.arithmetic,linear.layout)
        self._fixed=(typed(original_positions),typed(self.positions),typed(context_id),typed(request_id),typed(self.contracts))
        self._weight_identity=self._weight_state()
        required=set(BertAttentionTile.required_rotation_indices())|set(BertRowLayerNorm.required_rotation_indices())|set(refresher.plan['rotations'])
        for name,value in self.weights.items():
            if value.ndim==2:required.update(linear.required_rotation_indices(value,linear.layout))
        self.required_keys=tuple(sorted(required));self._required=typed(self.required_keys)
        self._key_maps=(cache.raw_rotation_keys,cache.coef_maps)
        self._key_identity=self._key_state()
        self._eval_keys=(getattr(cache,'eval_key_a_full',None),getattr(cache,'eval_key_b_full',None))
        self._guard()

    def _weight_state(self):
        return tuple((n,id(v),v.dtype.str,v.shape,v.strides,v.flags.writeable) for n,v in self.weights.items())

    def _key_state(self):
        cache=self.ctx._param_cache;keys=cache.raw_rotation_keys;maps=cache.coef_maps
        result=[]
        for index in self.required_keys:
            row=keys.get(index)
            if not isinstance(row,Mapping) or row.get('a') is None or row.get('b') is None or maps.get(index) is None:
                raise ValueError('missing complete full-model public key inventory')
            result.append((index,id(row),id(row['a']),id(row['b']),id(maps[index])))
        return tuple(result)

    def _guard(self):
        now=(self.arithmetic,self.linear,self.refresher,self.ctx,self.ctx._param_cache,self.weights,self.manifest_bytes,self.contracts,
             self.binding,self._metadata,self.progress,self.linear.arithmetic,self.linear.layout)
        if any(a is not b for a,b in zip(now,self._bound)):raise ValueError('construction-bound full-model owner changed')
        if ((typed(self.original_positions),typed(self.positions),typed(self.context_id),typed(self.request_id),typed(self.contracts))!=self._fixed
                or self._weight_state()!=self._weight_identity or typed(self.required_keys)!=self._required):
            raise ValueError('full-model public weights/positions/identities changed')
        ctx=self.ctx;cache=ctx._param_cache;p=self._parameters
        if (self.arithmetic.ctx is not ctx or self.linear.ctx is not ctx or self.linear.arithmetic.ctx is not ctx
                or self.arithmetic.refresher is not None or self.linear.arithmetic.refresher is not None
                or any(callable(getattr(x,'declare_bound',None)) for x in (self.arithmetic,self.linear.arithmetic))
                or typed(self.linear.arithmetic.scale)!=typed(p['scaling_factor']) or typed(self.arithmetic.scale)!=typed(p['scaling_factor'])
                or typed((self.linear.layout.slots,self.linear.layout.features))!=typed((65536,4096))):
            raise ValueError('one static keyless full-model context required')
        for name in ('degree','num_slots','composite_degree','dnum','max_level','degree_layout'):
            expected=tuple(p[name]) if name=='degree_layout' else p[name]
            if typed(getattr(ctx,name,None))!=typed(expected):raise ValueError('full-model context profile changed: '+name)
        if (typed(tuple(cache.q_towers))!=typed(tuple(p['q_towers'])) or typed(tuple(cache.p_towers))!=typed(tuple(p['p_towers']))
                or cache.raw_rotation_keys is not self._key_maps[0] or cache.coef_maps is not self._key_maps[1]
                or self._key_state()!=self._key_identity
                or any(a is not b for a,b in zip((getattr(cache,'eval_key_a_full',None),getattr(cache,'eval_key_b_full',None)),self._eval_keys))):
            raise ValueError('full-model public key/ring identity changed')
        self.binding.check();require_chained_rotation_keys(ctx,self.required_keys)
        if tuple(self._fingerprint(v) for v in self._inputs)!=self._input_states:raise ValueError('original fresh input changed')
        if any(self._fingerprint(v)!=state for v,state in (*self._live,*self._captured)):
            raise ValueError('full-model live/captured ciphertext changed')
        if len(self._children)!=len(self._child_state):raise ValueError('borrowed child inventory changed')
        for child,(original,callback) in zip(self._children,self._child_state):
            if child is not original or child.progress is not callback:raise ValueError('borrowed child owner/callback changed')
            child._guard()

    def _fingerprint(self,value):return self._metadata._value_fingerprint(value)

    def _forget_owned(self,values):
        borrowed={id(v) for v in self._inputs}
        for value in values:
            if id(value) not in borrowed:
                for name in tuple(vars(value)):
                    if name.startswith('_public_'):delattr(value,name)

    def _callback(self,label,value,counts):
        self._guard();state=self._fingerprint(value)
        if self.progress is not None:self.progress(str(self._progress_stage)+' / '+label,value,dict(counts))
        self._guard()
        if self._fingerprint(value)!=state:raise ValueError('full-model callback changed current result')
        self._forget_owned((value,*(v for v,_ in self._live),*(v for v,_ in self._captured)))

    def _install_child(self,child):
        self._children=(child,);self._child_state=((child,child.progress),);self._guard()

    def _checkpoint(self,name,value,checkpoints,capture):
        self._metadata._validate_ciphertext(value,minimum_levels=0,canonical=False)
        if typed(value.degree_layout)!=typed(self.ctx.degree_layout):raise ValueError('checkpoint layout differs')
        self._callback(name,value,{})
        if capture:checkpoints[name]=value;self._captured.append((value,self._fingerprint(value)))
        return value

    def _attention_model(self,layer):
        prefix=f'bert.encoder.layer.{layer}.'
        return {role:{kind:self.weights[prefix+PREFIX[role]+'.'+kind] for kind in ('weight','bias')} for role in PREFIX}

    def evaluate(self,input_tiles,*,capture_checkpoints=False):
        self._guard()
        if self._active:raise ValueError('full-model owner cannot be reentered')
        if type(capture_checkpoints)is not bool:raise ValueError('capture_checkpoints must be bool')
        if type(input_tiles) not in (tuple,list) or len(input_tiles)!=8:raise ValueError('eight original fresh tiles required')
        sources=tuple(input_tiles)
        for value in sources:
            self._metadata._validate_ciphertext(value,minimum_levels=49,canonical=False)
            if self.arithmetic._level(value)!=49 or typed(value._ckks_scale)!=typed(self.arithmetic.scale) or typed(value.degree_layout)!=typed(self.ctx.degree_layout):
                raise ValueError('original input must be exact canonical level49')
        self._inputs=sources;self._input_states=tuple(self._fingerprint(v) for v in sources);self._active=True;self.last_run=None
        current=sources;checkpoints={};layers=[]
        try:
            for layer in range(12):
                self._guard();self._progress_stage=f'layer{layer}.attention';prefix=f'bert.encoder.layer.{layer}.'
                self._live=[(v,self._fingerprint(v)) for v in current]
                child=BertLayerAttention(self.arithmetic,self.linear,self.refresher,self._attention_model(layer),self.contracts[layer],self.manifest_bytes,
                    self.weights[prefix+'attention.output.LayerNorm.weight'],self.weights[prefix+'attention.output.LayerNorm.bias'],progress=self._callback_bound)
                self._install_child(child)
                attention_result=child.evaluate(current)
                if attention_result.original_positions!=self.positions or len(attention_result.outputs)!=len(self.positions):raise ValueError('attention changed original-position tile inventory')
                attention=attention_result.outputs;self._live=[(v,self._fingerprint(v)) for v in (*current,*attention)]
                for tile,value in enumerate(attention):self._checkpoint(f'layer{layer}.attention.{tile}',value,checkpoints,capture_checkpoints)
                attention_metadata=attention_result.metadata;self._children=();self._child_state=();del child,attention_result
                outputs=[];ffn_metadata=[]
                arrays={name:self.weights[prefix+name.replace('intermediate.','intermediate.dense.').replace('output.','output.dense.')]
                        for name in ('intermediate.weight','intermediate.bias','output.weight','output.bias')}
                for tile,(value,positions) in enumerate(zip(attention,self.positions)):
                    self._progress_stage=f'layer{layer}.ffn.{tile}'
                    child=BertFFNTile(self.arithmetic,self.linear,self.refresher,arrays,self.manifest_bytes,
                        self.weights[prefix+'output.LayerNorm.weight'],self.weights[prefix+'output.LayerNorm.bias'],
                        layer=layer,original_positions=positions,progress=self._callback_bound)
                    self._install_child(child);normalized=child.evaluate_tile(value)
                    if normalized.original_positions!=positions:raise ValueError('FFN changed original positions')
                    output=self._checkpoint(f'layer{layer}.block.{tile}',normalized.output,checkpoints,capture_checkpoints)
                    outputs.append(output);self._live.append((output,self._fingerprint(output)));ffn_metadata.append(child.last_run)
                    self._children=();self._child_state=();del child,normalized
                layers.append(dict(layer=layer,attention=attention_metadata,ffn=ffn_metadata,
                    input_levels=[self.arithmetic._level(v) for v in current],output_levels=[self.arithmetic._level(v) for v in outputs]))
                current=tuple(outputs);self._live=[(v,self._fingerprint(v)) for v in current]
                del attention
            self._progress_stage='head';self._guard()
            arrays={name:self.weights[('bert.' if name.startswith('pooler.') else '')+name.replace('pooler.','pooler.dense.')]
                    for name in ('pooler.weight','pooler.bias','classifier.weight','classifier.bias')}
            head=BertPoolerHead(self.arithmetic,self.linear,self.refresher,arrays,self.manifest_bytes,
                original_positions=self.positions[0],progress=self._callback_bound)
            self._install_child(head);result=head.evaluate_tile(current[0],capture_checkpoints=capture_checkpoints)
            if capture_checkpoints:
                for name,value in result.checkpoints.items():self._checkpoint('head.'+name,value,checkpoints,True)
            output=self._checkpoint('output',result.output,checkpoints,capture_checkpoints)
            if self.arithmetic._level(output)!=2:raise ValueError('full-model output must be level2')
            self._guard();metadata=dict(layers=layers,head=result.plan,output_level=2,original_positions=self.original_positions,
                explicit_direct_ciphertext_connections=True,range_verified=False,encrypted_error_verified=False)
            self.last_run=metadata;return FullModelResult(output,checkpoints,self.original_positions,metadata)
        finally:
            self._active=False;self._inputs=();self._input_states=();self._live.clear();self._captured.clear();self._children=();self._child_state=();self._progress_stage=None
