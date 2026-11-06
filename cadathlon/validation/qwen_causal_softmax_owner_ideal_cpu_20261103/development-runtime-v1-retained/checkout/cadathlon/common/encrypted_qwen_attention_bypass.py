"""Versioned PV successor preserving the frozen Qwen query0 direct-V path.

The predecessor remains unchanged. This owner adds public result-layout
validation, a real final direct-V mask, and retained payload/accessor guards.
No normalization, refresh, codec or context ownership is introduced.
"""
from encrypted_qwen_attention_routing import QwenAttentionRouting,QwenAttendedTile
from encrypted_qwen_causal_softmax import QwenProbabilityTile
from encrypted_qwen_head_routing import typed
from encrypted_qwen_rms_refresh import accessor
from encrypted_scale_planning import drop_divisor,weighted_sum_scale,multiply_scale
import numpy as np

class QwenAttentionRoutingWithBypass(QwenAttentionRouting):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self._bypass_accessors=self._accessors()
    def _accessors(self):
        a=self.arithmetic
        return (*tuple(accessor(a,n) for n in ('rotate','weighted_sum','multiply','add','_level')),
            accessor(a.rotation,'rotate'),accessor(a._plaintext_multiply,'multiply'),accessor(a._rescale,'rescale'))
    def _bypass_guard(self):
        super()._guard()
        if any(a is not b for now,old in zip(self._accessors(),self._bypass_accessors) for a,b in zip(now,old)):raise ValueError('PV arithmetic accessor changed')
    def _execute(self,sources,depth,body,extra_guard=None):
        self._bypass_guard()
        if self._active:raise ValueError('serial PV owner cannot be reentered')
        for value in sources:self._validate(value,depth)
        def state(v):return (v,v.polynomial,self._fingerprint(v))
        originals=tuple(state(v) for v in sources);retained={};a=self.arithmetic;q=tuple(self.ctx._param_cache.q_towers)
        counts=dict(rotations=0,weighted_sums=0,ciphertext_products=0,additions=0)
        def check(rows):
            if any(v.polynomial is not payload or self._fingerprint(v)!=old for v,payload,old in rows):raise ValueError('PV retained source/result changed')
        def guard():
            self._bypass_guard();check((*originals,*retained.values()))
            if extra_guard is not None:extra_guard()
        def keep(name,value):retained[name]=state(value);return value
        def metadata(value,level,scale):
            self._validate(value,0)
            if (type(a._level(value))is not int or a._level(value)!=level or typed(value._ckks_scale)!=typed(scale) or
                type(value.num_moduli)is not int or value.num_moduli!=2*(level+1) or typed(tuple(value.moduli))!=typed(q[:2*(level+1)])):raise ValueError('PV primitive metadata differs')
        def note(label,value,level,scale,owners=()):
            guard();check(owners);metadata(value,level,scale);self._forget(value);current=state(value)
            if self.progress is not None:self.progress(label,value,dict(level=level,scale=scale,counts=dict(counts)))
            guard();check((*owners,current));self._forget(value)
            for prior,_,_ in retained.values():self._forget(prior)
            return value
        def rotate(value,shift):
            guard();shift%=65536
            if shift==0:return value
            old=state(value);level=a._level(value);scale=value._ckks_scale;counts['rotations']+=1
            out=a.rotate(value,shift);metadata(out,level,scale);guard();check((old,));return out
        def weighted(values,weights,label):
            guard();owners=tuple(state(v) for v in values);level=min(a._level(v) for v in values)
            scale=weighted_sum_scale([v._ckks_scale for v in values],a.scale,drop_divisor(q,level))['output_scale'];counts['weighted_sums']+=1
            return note(label,a.weighted_sum(values,weights),level-1,scale,owners)
        def multiply(x,y,label):
            guard();owners=(state(x),state(y));level=min(a._level(x),a._level(y));scale=multiply_scale(x._ckks_scale,y._ckks_scale,drop_divisor(q,level));counts['ciphertext_products']+=1
            return note(label,a.multiply(x,y),level-1,scale,owners)
        def add(x,y):
            guard();owners=(state(x),state(y));level=min(a._level(x),a._level(y));scale=x._ckks_scale;counts['additions']+=1
            out=a.add(x,y);metadata(out,level,scale);guard();check(owners);return out
        self._active=True;self.last_run=None
        try:
            output=body(rotate,weighted,multiply,add,keep,guard)
            output=note('output',output,a._level(output),output._ckks_scale);guard()
            self.last_run=dict(counts=counts,output_level=a._level(output),output_scale=output._ckks_scale)
            return output
        finally:self._active=False
    def attend_softmax(self,result,expanded_value_tiles,*,query_tile):
        self._bypass_guard();positions=self.layout.tile_positions(query_tile)
        if not isinstance(result,QwenProbabilityTile):raise TypeError('explicit causal-softmax result required')
        expected_direct=(0,) if query_tile==0 else ()
        if (typed(result.query_positions)!=typed(positions) or typed(result.key_positions)!=typed(self.layout.positions) or
            typed(result.query_tile)!=typed(query_tile) or typed(result.direct_value_queries)!=typed(expected_direct) or
            type(result.layer)is not int or not 0<=result.layer<28):raise ValueError('probability result public layout differs')
        if type(expanded_value_tiles)is not tuple or len(expanded_value_tiles)!=self.layout.tile_count:raise ValueError('exact ordered expanded V tiles required')
        probability=result.output;single=len(self.layout.positions)==1
        if single:
            if probability is not None:raise ValueError('K1 must preserve direct V bypass')
        elif probability is None or self.arithmetic._level(probability)!=24:raise ValueError('actual probability level24 required')
        def header():return (result.output,typed((result.query_positions,result.key_positions,result.direct_value_queries,result.query_tile,result.layer)))
        bound=header()
        def guard_header():
            now=header()
            if now[0]is not bound[0] or now[1]!=bound[1]:raise ValueError('probability result layout changed during PV')
        sources=expanded_value_tiles if single else (probability,*expanded_value_tiles)
        def body(rotate,weighted,multiply,add,keep,guard):
            mask=np.zeros((16,4096));mask[:len(positions),:2048]=1.
            if single:return weighted([expanded_value_tiles[0]],[mask.ravel()],'K1_direct_value_output_mask')
            total=None
            for key,keypos in enumerate(self.layout.positions):
                shifted=rotate(probability,key);values=[];weights=[]
                for head in range(16):
                    anchor=np.zeros((16,4096))
                    for row,pos in enumerate(positions):
                        if keypos<=pos and not(query_tile==0 and row==0):anchor[row,head*128]=1.
                    values.append(rotate(shifted,-96*head));weights.append(anchor.ravel())
                repeated=weighted(values,weights,f'key{key}_probability_anchors')
                for bit in range(7):repeated=add(repeated,rotate(repeated,-(1<<bit)))
                keep('probability',repeated)
                source=expanded_value_tiles[key//16];values=[];weights=[]
                for row in range(len(positions)):
                    target=np.zeros((16,4096));target[row,:2048]=1.
                    values.append(rotate(source,((key%16)-row)*4096));weights.append(target.ravel())
                broadcast=weighted(values,weights,f'key{key}_value_broadcast')
                product=multiply(repeated,broadcast,f'key{key}_weighted_value')
                total=product if total is None else add(total,product);keep('aggregate',total);guard()
            if query_tile==0:
                mask[0,:2048]=0.;direct=np.zeros((16,4096));direct[0,:2048]=1.
                return weighted([total,expanded_value_tiles[0]],[mask.ravel(),direct.ravel()],'attended_output_mask_with_K1_direct_value')
            return weighted([total],[mask.ravel()],'attended_output_mask')
        output=self._execute(sources,1 if single else 3,body,guard_header)
        return QwenAttendedTile(output,positions,self.layout.positions,dict(self.last_run['counts']),self.last_run['output_level'],self.last_run['output_scale'])
