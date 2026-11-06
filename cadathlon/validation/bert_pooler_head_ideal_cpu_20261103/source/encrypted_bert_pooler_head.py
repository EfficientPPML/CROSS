"""Pinned BERT pooler/classifier, selecting original CLS through a real source mask.

Borrows initialized arithmetic, chained dense loops and explicit full-slot
refresh. The input is the first compact encoder-output tile with position0.
No client activations, codec, model loader or range inspection occurs here.
"""
from dataclasses import dataclass
from collections.abc import Mapping
from types import MappingProxyType
import hashlib
import numpy as np
from encrypted_bert_rational_tanh import BertRationalTanh,BertTanhContract
from encrypted_fullslot_refresh_binding import typed
from encrypted_scale_planning import weighted_sum_scale,drop_divisor
from encrypted_chained_streaming_linear import ChainedStreamingLinear,require_chained_rotation_keys
TENSORS={
 'pooler.weight':((768,768),'41fa4a1811934ec7009a4a94ccefd8c1834a15d3c863c744aa78135c0e0e5bd3'),
 'pooler.bias':((768,),'9950db2634ea18cb363ea8ca10e6fac1fc8ac485e82b270f9bea2804c4408847'),
 'classifier.weight':((2,768),'e691bd558c442e6a67f2ed611c25e378bdd88e8f3abd696659f0d36ceea76d3e'),
 'classifier.bias':((2,),'2a5f1ee5fdb9203159bdd9c08109829d2873ea8d24d3b75ccd7e7296edc58685')}
def freeze_weights(arrays):
    if not isinstance(arrays,Mapping) or set(arrays)!=set(TENSORS):raise ValueError('all four pinned head tensors required')
    result={}
    for name,(shape,digest) in TENSORS.items():
        a=np.asarray(arrays[name])
        if a.dtype.kind not in 'fiu' or a.shape!=shape or not np.isfinite(a).all():raise ValueError('invalid head tensor')
        with np.errstate(over='ignore',invalid='ignore'):raw=a.astype('<f4');lift=raw.astype('<f8')
        if not np.isfinite(raw).all() or not np.array_equal(a,lift) or hashlib.sha256(raw.tobytes()).hexdigest()!=digest:raise ValueError('head tensor differs from pinned FP32 lift')
        result[name]=np.frombuffer(lift.tobytes(),dtype='<f8').reshape(shape)
    return MappingProxyType(result)
@dataclass(frozen=True)
class BertHeadResult:
    output:object
    checkpoints:dict
    plan:dict
    original_position:int=0
class BertPoolerHead:
    def __init__(self,arithmetic,linear,refresher,arrays,manifest_bytes,*,original_positions,progress=None):
        if not isinstance(linear,ChainedStreamingLinear):raise TypeError('chained streaming linear required')
        if (type(original_positions)is not tuple or not 1<=len(original_positions)<=16 or original_positions[0]!=0
                or any(type(x)is not int or not 0<=x<128 for x in original_positions)
                or tuple(sorted(set(original_positions)))!=original_positions):raise ValueError('first compact tile must contain original CLS position0')
        if progress is not None and not callable(progress):raise TypeError('progress must be callable or None')
        self.arithmetic,self.linear,self.refresher,self.progress=arithmetic,linear,refresher,progress
        self.original_positions=original_positions;self.model=freeze_weights(arrays)
        self.tanh=BertRationalTanh(arithmetic,refresher,BertTanhContract(manifest_bytes),progress=self._callback)
        self._callback_bound=self.tanh.progress;self._positions=typed(original_positions)
        self._bound=(arithmetic,linear,refresher,progress,self.model,self.tanh,linear.arithmetic,linear.layout)
        self._weights=self._weight_state();self._active=False;self._source=None;self._source_state=None;self._retained={};self.last_run=None
        self.required_keys=tuple(sorted(set(linear.required_rotation_indices(self.model['pooler.weight'],linear.layout))|set(linear.required_rotation_indices(self.model['classifier.weight'],linear.layout))))
        self._keys=typed(self.required_keys);self._guard()
    def _weight_state(self):return tuple((n,id(v),v.dtype.str,v.shape,v.strides,v.flags.writeable) for n,v in self.model.items())
    def _fingerprint(self,v):return self.tanh._metadata._value_fingerprint(v)
    def _guard(self):
        if any(a is not b for a,b in zip((self.arithmetic,self.linear,self.refresher,self.progress,self.model,self.tanh,self.linear.arithmetic,self.linear.layout),self._bound)):raise ValueError('construction-bound head changed')
        if (typed(self.original_positions)!=self._positions or self._weight_state()!=self._weights or typed(self.required_keys)!=self._keys
                or self.tanh.progress is not self._callback_bound or self.tanh.arithmetic is not self.arithmetic or self.tanh.refresher is not self.refresher
                or self.linear.ctx is not self.arithmetic.ctx or self.linear.arithmetic.ctx is not self.arithmetic.ctx
                or self.linear.arithmetic.refresher is not None or callable(getattr(self.linear.arithmetic,'declare_bound',None))
                or typed(self.linear.arithmetic.scale)!=typed(self.arithmetic.scale)
                or typed((self.linear.layout.slots,self.linear.layout.features))!=typed((65536,4096))):raise ValueError('head model/position/arithmetic contract changed')
        self.tanh._guard();require_chained_rotation_keys(self.arithmetic.ctx,self.required_keys)
        if self._source is not None and self._fingerprint(self._source)!=self._source_state:raise ValueError('original head source changed')
        if any(self._fingerprint(v)!=state for v,state in self._retained.values()):raise ValueError('captured head value changed')
    def _callback(self,label,value,counts):
        self._guard();state=self._fingerprint(value)
        if self.progress is not None:self.progress('head '+label,value,dict(counts))
        self._guard()
        if self._fingerprint(value)!=state:raise ValueError('head callback changed current result')
        for current in (value,*(v for v,_ in self._retained.values())):
            for n in tuple(vars(current)):
                if n.startswith('_public_'):delattr(current,n)
    def _linear(self,value,name):
        self._guard();L=self.arithmetic._level(value)
        scale=weighted_sum_scale([value._ckks_scale],self.arithmetic.scale,drop_divisor(self.arithmetic.ctx._param_cache.q_moduli_at_level(L),L))['output_scale']
        # The actual diagonal source mask restricts both layers to row0's first
        #768 features. Other valid tokens and refreshed feature padding cannot
        #contribute to either projection; biases are also restricted to row0.
        result=self.linear.linear_tile(value,self.model[name+'.weight'],self.model[name+'.bias'],valid_rows=1)
        self.tanh._validate(result,0)
        if self.arithmetic._level(result)!=L-1 or typed(result._ckks_scale)!=typed(scale):raise ValueError('head dense metadata differs from plan')
        self._callback(name+' dense',result,{})
        return result
    def evaluate_tile(self,value,*,capture_checkpoints=False):
        self._guard()
        if self._active:raise ValueError('head owner cannot be reentered')
        if type(capture_checkpoints)is not bool:raise ValueError('capture_checkpoints must be bool')
        self.tanh._validate(value,9);self._active=True;self._source=value;self._source_state=self._fingerprint(value);self.last_run=None
        checkpoints={};stages=[]
        def remember(name,v):
            if capture_checkpoints:checkpoints[name]=v;self._retained[name]=(v,self._fingerprint(v))
            stages.append(dict(stage=name,level=self.arithmetic._level(v),scale=v._ckks_scale))
        try:
            pooled=self._linear(value,'pooler');remember('pooler_dense',pooled)
            activated,tanh_checkpoints=self.tanh.evaluate_tile(pooled,capture_checkpoints=capture_checkpoints)
            if capture_checkpoints:
                for name,v in tanh_checkpoints.items():checkpoints['tanh.'+name]=v;self._retained['tanh.'+name]=(v,self._fingerprint(v))
            self._callback('tanh output',activated,{});remember('tanh_output',activated)
            output=self._linear(activated,'classifier');remember('classifier',output);self._guard()
            plan=dict(stages=stages,tanh=self.tanh.last_plan,original_position=0,input_tile_positions=self.original_positions,
                      classifier_width=2,source_mask='Only compact row0,features0..767 at both genuine dense maps',range_verified=False,encrypted_error_verified=False)
            self.last_run=plan;return BertHeadResult(output,checkpoints,plan)
        finally:self._retained.clear();self._source=self._source_state=None;self._active=False
