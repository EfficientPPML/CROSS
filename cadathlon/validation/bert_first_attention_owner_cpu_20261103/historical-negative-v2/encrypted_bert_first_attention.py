"""Isolated serial, keyless first-attention composition; no codec or file handler.

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
from encrypted_bert_softmax_tile import BertSoftmaxTile
from encrypted_bert_v2_row_norm import V2AttentionRowNorm
from encrypted_chained_streaming_linear import ChainedStreamingLinear
from encrypted_score_reduction import ScoreReduction,ScoreReductionLayout
from encrypted_scale_planning import weighted_sum_scale,multiply_scale,drop_divisor
from first_attention_contract import validate_contract,validate_projection_arrays,parameters


@dataclass(frozen=True)
class FirstAttentionResult:
    outputs:tuple
    original_positions:tuple
    metadata:dict


def _json(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


class BertFirstAttention:
    """Eight fixed input tiles, T compact query outputs, common public context.

    Input valid rows retain their original learned-position embeddings before
    compaction. K/V still cover the padded128-row frame; only valid queries
    execute nonlinear stages. All work is serial. Full BERT/file stages and
    common-profile numerical readiness are outside this isolated owner.
    """
    def __init__(self,arithmetic,linear,refresher,model,contract,manifest_bytes,gamma,beta,*,progress=None):
        if not isinstance(linear,ChainedStreamingLinear):raise TypeError('promoted chained streaming linear required')
        if progress is not None and not callable(progress):raise TypeError('progress must be callable or None')
        self.contract=validate_contract(contract)
        checked=validate_projection_arrays(model)
        # Exact float64 lifts backed by immutable bytes; callers cannot turn
        # writeability back on and mutate coefficients from a progress hook.
        frozen={}
        for name in ('query','key','value','output'):
            frozen[name]=MappingProxyType({key:np.frombuffer(np.asarray(checked[name][key],dtype='<f8').tobytes(),dtype='<f8').reshape(np.shape(checked[name][key]))
                for key in ('weight','bias')})
        self.model=MappingProxyType(frozen)
        self.arithmetic=arithmetic;self.linear=linear;self.refresher=refresher;self.progress=progress
        self.ctx=arithmetic.ctx;self._active=False;self._sources=();self._source_states=();self._retained=[];self.last_run=None
        self.attention=BertAttentionTile(arithmetic,progress=self._callback)
        positions=tuple(self.contract['original_positions']);K=self.contract['key_count']
        self.positions=tuple(tuple(positions[i:i+16]) for i in range(0,K,16))
        self.softmaxes=tuple(BertSoftmaxTile(arithmetic,ScoreReductionLayout(np.arange(128)<K,valid_rows=len(pos)),
            manifest_bytes,refresher,continuation_levels=17,progress=self._callback) for pos in self.positions)
        self.norm=V2AttentionRowNorm(arithmetic,manifest_bytes,gamma,beta,site='0.attention',epsilon=1e-12,progress=self._callback)
        self._bound=(arithmetic,linear,refresher,self.ctx,self.ctx._param_cache,self.model,self.contract,
            self.attention,self.softmaxes,self.norm,progress,linear.arithmetic,linear.layout)
        self._contract_bytes=_json(self.contract);self._positions=self.positions;self._model_state=self._coefficient_state()
        self._expected_parameters=parameters();self._guard()

    def _coefficient_state(self):
        return tuple((name,key,id(value),value.dtype.str,value.shape,value.strides,value.flags.writeable)
            for name,row in self.model.items() for key,value in row.items())

    def _guard(self):
        now=(self.arithmetic,self.linear,self.refresher,self.ctx,self.ctx._param_cache,self.model,self.contract,
            self.attention,self.softmaxes,self.norm,self.progress,self.linear.arithmetic,self.linear.layout)
        if any(a is not b for a,b in zip(now,self._bound)):
            raise ValueError('construction-bound first-attention owner changed')
        if (_json(self.contract)!=self._contract_bytes or self.positions!=self._positions
                or self._coefficient_state()!=self._model_state):raise ValueError('public first-attention contract/model changed')
        p=self._expected_parameters;c=self.ctx;k=c._param_cache
        if (self.arithmetic.ctx is not c or self.linear.ctx is not c or self.linear.arithmetic.ctx is not c
                or self.linear.layout.slots!=65536 or self.linear.layout.features!=4096
                or self.norm.arithmetic is not self.arithmetic or self.norm.contract.site!='0.attention'
                or self.arithmetic.refresher is not None or self.linear.arithmetic.refresher is not None):
            raise ValueError('one static borrowed context and attention normalization required')
        for name in ('degree','num_slots','composite_degree','dnum','max_level'):
            if ScoreReduction._typed(getattr(c,name,None))!=ScoreReduction._typed(p[name]):
                raise ValueError('fixed first-attention context parameter differs: '+name)
        if (tuple(c.degree_layout)!=tuple(p['degree_layout']) or tuple(k.degree_layout)!=tuple(p['degree_layout'])
                or tuple(k.q_towers)!=tuple(p['q_towers']) or tuple(k.p_towers)!=tuple(p['p_towers'])
                or type(k.batch)is not int or k.batch!=1
                or self.arithmetic.scale!=p['scaling_factor'] or self.linear.arithmetic.scale!=p['scaling_factor']):
            raise ValueError('fixed first-attention ring/scale/layout differs')
        self.attention._guard();self.norm._guard()
        for owner in self.softmaxes:owner._guard()
        if self._sources and tuple(self._fingerprint(v) for v in self._sources)!=self._source_states:
            raise ValueError('first-attention input changed')
        if any(self._fingerprint(v)!=state for v,state in self._retained):
            raise ValueError('first-attention retained ciphertext changed')

    def _fingerprint(self,value):return self.attention._fingerprint(value)

    def _callback(self,label,value,counts):
        self._guard();state=self._fingerprint(value)
        if self.progress is not None:self.progress(label,value,counts)
        self._guard()
        if self._fingerprint(value)!=state:raise ValueError('first-attention callback changed its result')

    def _note(self,label,value,*,level,scale=None):
        self._guard();self.attention._validate(value,0)
        if (self.arithmetic._level(value)!=level or type(value._ckks_nsd)is not int or value._ckks_nsd!=1
                or not math.isfinite(float(value._ckks_scale)) or value._ckks_scale<=0
                or (scale is not None and value._ckks_scale!=scale)):
            raise ValueError('first-attention output differs from actual public metadata plan')
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
        if self._active:raise ValueError('first attention cannot be reentered')
        if not isinstance(input_tiles,(tuple,list)) or len(input_tiles)!=8:
            raise ValueError('exactly eight authenticated full-slot input tiles required')
        sources=tuple(input_tiles)
        for value in sources:
            self.attention._validate(value,49)
            if self.arithmetic._level(value)!=49:raise ValueError('input tile must be at declared level49')
        scales=[ScoreReduction._typed(v._ckks_scale) for v in sources]
        if any(s!=scales[0] for s in scales[1:]):raise ValueError('input tiles must share their actual encoding scale')
        self._active=True;self._sources=sources;self._source_states=tuple(self._fingerprint(v) for v in sources);self.last_run=None
        keys=[];values=[];outputs=[];metadata=[]
        try:
            for source in sources:
                key=self._project(source,'key',16);keys.append(key);self._retained.append((key,self._fingerprint(key)))
                value=self._project(source,'value',16);values.append(value);self._retained.append((value,self._fingerprint(value)))
            for index,positions in enumerate(self.positions):
                self._guard();source=sources[index];query=self._project(source,'query',len(positions))
                scores=self.attention.scores(query,keys)
                product_scale=multiply_scale(query._ckks_scale,keys[0]._ckks_scale,
                    drop_divisor(self.ctx._param_cache.q_towers,48))
                score_scale=self._weighted_scale(self._weighted_scale(product_scale,47),46)
                scores=self._note('all-key scores',scores,level=45,scale=score_scale);del query
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
                output=self._note('attention normalized output',normalized.output,level=9,
                    scale=normalized.plan['scales']['output']);del residual
                if normalized.original_positions!=positions:raise ValueError('normalization changed compact original positions')
                outputs.append(output);self._retained.append((output,self._fingerprint(output)))
                metadata.append(dict(tile=index,original_positions=positions,valid_rows=len(positions),
                    output_level=9,output_scale=output._ckks_scale,softmax=self.softmaxes[index].last_run,norm=normalized.plan))
            self._guard()
            self.last_run=dict(scope='First-attention ciphertext graph only; numerical/domain readiness not inferred.',
                valid_key_count=self.contract['key_count'],input_tiles=8,output_tiles=len(outputs),
                outputs=metadata,range_verified=False,repeated_refresh_error_verified=False)
            return FirstAttentionResult(tuple(outputs),self.positions,self.last_run)
        finally:
            self._active=False;self._sources=();self._source_states=();self._retained.clear()
            keys.clear();values.clear();outputs.clear()
