"""Isolated serial, keyless layer-attention composition; no codec or file handler.

The caller supplies authenticated ciphertext inputs and public contracts/model
data. This owner never generates keys, refreshes through a client, closes its
borrowed context, clips slots, or substitutes native intermediate activations.
Public domains and repeated-refresh numerical error remain caller obligations.
"""
from dataclasses import dataclass
from types import MappingProxyType
import hashlib,json,math
import numpy as np
from encrypted_bert_attention_tile import BertAttentionTile
from encrypted_bert_layer_softmax import BertLayerSoftmax
from encrypted_bert_all_row_norm import BertRowLayerNorm
from encrypted_chained_streaming_linear import ChainedStreamingLinear
from encrypted_score_reduction import ScoreReduction,ScoreReductionLayout
from encrypted_scale_planning import weighted_sum_scale,multiply_scale,drop_divisor
from encrypted_fullslot_refresh_binding import FullSlotRefreshBinding
from layer_attention_contract import validate_contract,validate_projection_arrays,parameters


@dataclass(frozen=True)
class LayerAttentionResult:
    outputs:tuple
    original_positions:tuple
    metadata:dict


def _json(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


class BertLayerAttention:
    """Fresh layer0 has eight inputs; later layers use T actual low-level inputs.

    Layer0 valid rows retain their original learned-position embeddings before
    compaction. K/V still cover the padded128-row frame; only valid queries
    execute nonlinear stages. All work is serial. The adapter preserves the eight-tile frame by real ciphertext subtraction. Full BERT/file stages and
    common-profile numerical readiness are outside this isolated owner.
    """
    def __init__(self,arithmetic,linear,refresher,model,contract,manifest_bytes,gamma,beta,*,progress=None):
        if not isinstance(linear,ChainedStreamingLinear):raise TypeError('promoted chained streaming linear required')
        if any(callable(getattr(owner,'declare_bound',None)) for owner in (arithmetic,linear.arithmetic)):
            raise TypeError('bound-aware arithmetic is unsupported by layer attention')
        if progress is not None and not callable(progress):raise TypeError('progress must be callable or None')
        self.contract=validate_contract(contract)
        self.layer=self.contract['layer'];self._layer=ScoreReduction._typed(self.layer)
        checked=validate_projection_arrays(model,self.layer)
        # Exact float64 lifts backed by immutable bytes; callers cannot turn
        # writeability back on and mutate coefficients from a progress hook.
        frozen={}
        for name in ('query','key','value','output'):
            frozen[name]=MappingProxyType({key:np.frombuffer(np.asarray(checked[name][key],dtype='<f8').tobytes(),dtype='<f8').reshape(np.shape(checked[name][key]))
                for key in ('weight','bias')})
        self.model=MappingProxyType(frozen)
        self.arithmetic=arithmetic;self.linear=linear;self.refresher=refresher;self.progress=progress
        self.ctx=arithmetic.ctx;self._active=False;self._sources=();self._source_states=();self._retained=[];self.last_run=None
        self.refresh_binding=FullSlotRefreshBinding(arithmetic,refresher)
        self.attention=BertAttentionTile(arithmetic,progress=self._callback)
        positions=tuple(self.contract['original_positions']);K=self.contract['key_count']
        self.positions=tuple(tuple(positions[i:i+16]) for i in range(0,K,16))
        self.softmaxes=tuple(BertLayerSoftmax(arithmetic,ScoreReductionLayout(np.arange(128)<K,valid_rows=len(pos)),
            manifest_bytes,refresher,layer=self.layer,original_positions=pos,progress=self._callback) for pos in self.positions)
        self.norm=BertRowLayerNorm(arithmetic,manifest_bytes,gamma,beta,site=f'{self.layer}.attention',epsilon=1e-12,progress=self._callback)
        self._bound=(arithmetic,linear,refresher,self.ctx,self.ctx._param_cache,self.model,self.contract,
            self.attention,self.softmaxes,self.norm,progress,linear.arithmetic,linear.layout,self.refresh_binding)
        self._child_callbacks=(self.attention.progress,self.norm.progress,*(owner.progress for owner in self.softmaxes))
        self._contract_bytes=_json(self.contract);self._positions=self.positions;self._model_state=self._coefficient_state()
        self._expected_parameters=parameters();self._guard()

    def _coefficient_state(self):
        return tuple((name,key,id(value),value.dtype.str,value.shape,value.strides,value.flags.writeable)
            for name,row in self.model.items() for key,value in row.items())

    def _guard(self):
        now=(self.arithmetic,self.linear,self.refresher,self.ctx,self.ctx._param_cache,self.model,self.contract,
            self.attention,self.softmaxes,self.norm,self.progress,self.linear.arithmetic,self.linear.layout,self.refresh_binding)
        if any(a is not b for a,b in zip(now,self._bound)):
            raise ValueError('construction-bound layer-attention owner changed')
        if (ScoreReduction._typed(self.layer)!=self._layer or _json(self.contract)!=self._contract_bytes
                or ScoreReduction._typed(self.positions)!=ScoreReduction._typed(self._positions)
                or self._coefficient_state()!=self._model_state):raise ValueError('public layer-attention contract/model changed')
        if any(a is not b for a,b in zip((self.attention.progress,self.norm.progress,*(owner.progress for owner in self.softmaxes)),self._child_callbacks)):
            raise ValueError('construction-bound attention child callback changed')
        p=self._expected_parameters;c=self.ctx;k=c._param_cache
        if (self.arithmetic.ctx is not c or self.linear.ctx is not c or self.linear.arithmetic.ctx is not c
                or self.linear.layout.slots!=65536 or self.linear.layout.features!=4096
                or self.norm.arithmetic is not self.arithmetic or self.norm.contract.site!=f'{self.layer}.attention'
                or self.arithmetic.refresher is not None or self.linear.arithmetic.refresher is not None
                or any(callable(getattr(owner,'declare_bound',None)) for owner in (self.arithmetic,self.linear.arithmetic))):
            raise ValueError('one static borrowed context and attention normalization required')
        for name in ('degree','num_slots','composite_degree','dnum','max_level'):
            if ScoreReduction._typed(getattr(c,name,None))!=ScoreReduction._typed(p[name]):
                raise ValueError('fixed layer-attention context parameter differs: '+name)
        if (tuple(c.degree_layout)!=tuple(p['degree_layout']) or tuple(k.degree_layout)!=tuple(p['degree_layout'])
                or tuple(k.q_towers)!=tuple(p['q_towers']) or tuple(k.p_towers)!=tuple(p['p_towers'])
                or type(k.batch)is not int or k.batch!=1
                or self.arithmetic.scale!=p['scaling_factor'] or self.linear.arithmetic.scale!=p['scaling_factor']):
            raise ValueError('fixed layer-attention ring/scale/layout differs')
        self.refresh_binding.check();self.attention._guard();self.norm._guard()
        for owner in self.softmaxes:owner._guard()
        if self._sources and tuple(self._fingerprint(v) for v in self._sources)!=self._source_states:
            raise ValueError('layer-attention input changed')
        if any(self._fingerprint(v)!=state for v,state in self._retained):
            raise ValueError('layer-attention retained ciphertext changed')

    def _fingerprint(self,value):return self.attention._fingerprint(value)

    def _callback(self,label,value,counts):
        self._guard();state=self._fingerprint(value)
        if self.progress is not None:self.progress(label,value,counts)
        self._guard()
        if self._fingerprint(value)!=state:raise ValueError('layer-attention callback changed its result')
        # Callbacks cannot confer justified public bounds. Clear newly added
        # annotations on this result and every earlier owned retained result;
        # borrowed input metadata is deliberately left untouched.
        ScoreReduction._forget(value)
        for retained,_ in self._retained:ScoreReduction._forget(retained)

    def _note(self,label,value,*,level,scale=None):
        self._guard();self.attention._validate(value,0)
        if (self.arithmetic._level(value)!=level or type(value._ckks_nsd)is not int or value._ckks_nsd!=1
                or not math.isfinite(float(value._ckks_scale)) or value._ckks_scale<=0
                or (scale is not None and value._ckks_scale!=scale)):
            raise ValueError('layer-attention output differs from actual public metadata plan')
        ScoreReduction._forget(value)
        self._callback(label,value,dict(level=level,scale=value._ckks_scale))
        return value

    def _weighted_scale(self,scale,level):
        return weighted_sum_scale([scale],self.arithmetic.scale,drop_divisor(self.ctx._param_cache.q_towers,level))['output_scale']

    def _project(self,value,name,valid_rows):
        row=self.model[name];level=self.arithmetic._level(value)
        expected=self._weighted_scale(value._ckks_scale,level)
        result=self.linear.linear_tile(value,row['weight'],row['bias'],valid_rows=valid_rows)
        return self._note(name+' projection',result,level=level-1,scale=expected)

    def evaluate(self,input_tiles):
        self._guard()
        if self._active:raise ValueError('layer attention cannot be reentered')
        expected_inputs=8 if self.layer==0 else len(self.positions)
        if not isinstance(input_tiles,(tuple,list)) or len(input_tiles)!=expected_inputs:
            raise ValueError('input tile inventory differs from public layer contract')
        sources=tuple(input_tiles)
        for value in sources:
            self.attention._validate(value,49 if self.layer==0 else 2)
            level=self.arithmetic._level(value)
            if not 2<=level<=49:raise ValueError('input tile lacks declared refresh capacity')
            if self.layer==0 and (level!=49 or ScoreReduction._typed(value._ckks_scale)!=ScoreReduction._typed(self._expected_parameters['scaling_factor'])):
                raise ValueError('fresh layer0 input must be canonical level49')
        self._active=True;self._sources=sources;self._source_states=tuple(self._fingerprint(v) for v in sources);self.last_run=None
        keys=[];values=[];outputs=[];metadata=[];frame=[];entry_metadata=[]
        try:
            if self.layer==0:frame.extend(sources)
            else:
                # Actual previous-layer outputs enter unchanged. Each is
                # refreshed through the borrowed public server at B128.
                for source in sources:
                    self._guard()
                    refreshed=self.refresh_binding.refresh(source,public_bound=128.)
                    refreshed=self._note('layer input B128 refresh',refreshed,level=25)
                    frame.append(refreshed);self._retained.append((refreshed,self._fingerprint(refreshed)))
                    entry_metadata.append(dict(kind='refresh',level=25,scale=refreshed._ckks_scale))
                # Keep the original eight-tile key/value graph. Public padded
                # tiles are genuine same-level ciphertext subtractions, never
                # plaintext payloads or substituted native activations.
                while len(frame)<8:
                    zero=self.arithmetic.add(frame[0],frame[0],subtract=True)
                    zero=self._note('public padded ciphertext tile',zero,level=25,scale=frame[0]._ckks_scale)
                    frame.append(zero);self._retained.append((zero,self._fingerprint(zero)))
                    entry_metadata.append(dict(kind='subtract_zero',level=25,scale=zero._ckks_scale))
            projected_level=48 if self.layer==0 else 24
            score_level=projected_level-3
            for source in frame:
                key=self._project(source,'key',16);keys.append(key);self._retained.append((key,self._fingerprint(key)))
                value=self._project(source,'value',16);values.append(value);self._retained.append((value,self._fingerprint(value)))
            for index,positions in enumerate(self.positions):
                self._guard();source=frame[index];query=self._project(source,'query',len(positions))
                scores=self.attention.scores(query,keys)
                product_scale=multiply_scale(query._ckks_scale,keys[0]._ckks_scale,
                    drop_divisor(self.ctx._param_cache.q_towers,projected_level))
                score_scale=self._weighted_scale(self._weighted_scale(product_scale,projected_level-1),projected_level-2)
                scores=self._note('all-key scores',scores,level=score_level,scale=score_scale);del query
                probabilities=self.softmaxes[index].evaluate_tile(scores);del scores
                probabilities=self._note('complete masked softmax',probabilities,level=24)
                mixed=self.attention.mix_values(probabilities,values)
                routed_scale=self._weighted_scale(self._weighted_scale(probabilities._ckks_scale,24),23)
                mixed_scale=multiply_scale(routed_scale,values[0]._ckks_scale,
                    drop_divisor(self.ctx._param_cache.q_towers,22))
                mixed=self._note('all-key value mixture',mixed,level=21,scale=mixed_scale);del probabilities
                projected=self._project(mixed,'output',len(positions));del mixed
                residual=self.arithmetic.add(projected,source)
                residual=self._note('attention residual',residual,level=20,scale=projected._ckks_scale);del projected
                normalized=self.norm.normalize_tile(residual,valid_rows=len(positions),original_positions=positions)
                output_level=20-(self.norm.inverse_evaluator.polynomial_depth(self.norm.approximation)+5)
                if output_level!=self.contract['output_level']:raise ValueError('selected normalization differs from layer output contract')
                output=self._note('attention normalized output',normalized.output,level=output_level,
                    scale=normalized.plan['scales']['output']);del residual
                if normalized.original_positions!=positions:raise ValueError('normalization changed compact original positions')
                outputs.append(output);self._retained.append((output,self._fingerprint(output)))
                metadata.append(dict(tile=index,original_positions=positions,valid_rows=len(positions),
                    output_level=output_level,output_scale=output._ckks_scale,softmax=self.softmaxes[index].last_run,norm=normalized.plan))
            self._guard()
            self.last_run=dict(scope='Layer-attention ciphertext graph only; numerical/domain readiness not inferred.',
                layer=self.layer,valid_key_count=self.contract['key_count'],input_tiles=len(sources),internal_frame_tiles=len(frame),
                entry_metadata=entry_metadata,output_tiles=len(outputs),
                outputs=metadata,range_verified=False,repeated_refresh_error_verified=False)
            return LayerAttentionResult(tuple(outputs),self.positions,self.last_run)
        finally:
            self._active=False;self._sources=();self._source_states=();self._retained.clear()
            keys.clear();values.clear();outputs.clear();frame.clear()
