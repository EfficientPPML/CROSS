"""Frozen BERT-v2 complete softmax for all twelve public layer contracts.

Layer0 preserves the existing fresh-score initial path. Later layers use the
explicit normalized-t B2 refresh split. Each layer keeps its selected5/6/7
tracked normalization rounds and a real mask after final continuation refresh.
All range/refresh-error assumptions remain public caller obligations.
"""
import json
from types import SimpleNamespace
from encrypted_bert_initial_softmax import BertInitialSoftmax,InitialSoftmaxContract
from encrypted_bert_split_initial import BertSplitInitialSoftmax,LayerInitialContract,state
from encrypted_bert_normalization_round import BertNormalizationRound,NormalizationRoundContract
from encrypted_score_reduction import ScoreReduction,ScoreReductionLayout
from encrypted_fullslot_refresh_binding import FullSlotRefreshBinding
from encrypted_polynomial import EncryptedPolynomialEvaluator,PolynomialApproximation
from encrypted_scale_planning import weighted_sum_scale,drop_divisor

class BertLayerSoftmax:
    def __init__(self,arithmetic,layout,manifest_bytes,refresher,*,layer,original_positions,progress=None):
        contract=LayerInitialContract(manifest_bytes,layer)
        if not isinstance(layout,ScoreReductionLayout):raise TypeError('full-slot score layout required')
        if (type(original_positions)is not tuple or len(original_positions)!=layout.valid_rows
                or any(type(x)is not int or not 0<=x<128 for x in original_positions)
                or tuple(sorted(set(original_positions)))!=original_positions):
            raise ValueError('strict original query positions required')
        if progress is not None and not callable(progress):raise TypeError('callable progress required')
        self.arithmetic,self.layout,self.refresher=arithmetic,layout,refresher
        self.contract,self.original_positions,self.progress=contract,original_positions,progress
        self.reduction=ScoreReduction(arithmetic,layout,progress=self._callback)
        self.refresh_binding=FullSlotRefreshBinding(arithmetic,refresher)
        raw=json.loads(manifest_bytes);rule=raw['norms'][f'{layer}.attention'];poly=rule['polynomial']
        strategy={'power':'power','balanced_chebyshev':'chebyshev'}[rule['method']]
        approximation=PolynomialApproximation('inverse_sqrt',tuple(poly['domain']),tuple(poly['coefficients']))
        self.attention_norm_depth=5+EncryptedPolynomialEvaluator.polynomial_depth(SimpleNamespace(strategy=strategy),approximation)
        self.continuation_levels=3+1+self.attention_norm_depth+2
        if self.continuation_levels not in (17,18):raise ValueError('frozen attention continuation depth differs')
        if layer==0:
            self.initial=BertInitialSoftmax(arithmetic,layout,InitialSoftmaxContract.from_manifest_bytes(manifest_bytes),progress=self._callback)
        else:
            self.initial=BertSplitInitialSoftmax(arithmetic,layout,manifest_bytes,refresher,
                layer=layer,original_positions=original_positions,progress=self._callback)
        self.round=BertNormalizationRound(arithmetic,layout,
            NormalizationRoundContract(manifest_bytes,int(layout.counts[0])),progress=self._callback)
        self._active=False;self._source=None;self._source_state=None;self._retained={};self._captured={};self._capture=False
        self.last_run=None
        self._bound=(arithmetic,layout,refresher,self.contract,self.reduction,self.refresh_binding,self.initial,self.round,progress)
        self._contract_state=state(contract);self._positions=ScoreReduction._typed(original_positions)
        self._continuation=(ScoreReduction._typed(self.attention_norm_depth),ScoreReduction._typed(self.continuation_levels))
        self._child_callbacks=(self.reduction.progress,self.initial.reduction.progress,self.round.reduction.progress)
        self._guard()

    @staticmethod
    def rotation_indices():return ScoreReductionLayout.rotation_indices()
    def _fingerprint(self,value):return self.reduction._value_fingerprint(value)
    def _guard(self):
        now=(self.arithmetic,self.layout,self.refresher,self.contract,self.reduction,self.refresh_binding,self.initial,self.round,self.progress)
        if any(a is not b for a,b in zip(now,self._bound)):raise ValueError('construction-bound layer softmax changed')
        if (state(self.contract)!=self._contract_state or ScoreReduction._typed(self.original_positions)!=self._positions
                or (ScoreReduction._typed(self.attention_norm_depth),ScoreReduction._typed(self.continuation_levels))!=self._continuation):
            raise ValueError('layer softmax public contract changed')
        if any(a is not b for a,b in zip((self.reduction.progress,self.initial.reduction.progress,self.round.reduction.progress),self._child_callbacks)):
            raise ValueError('layer softmax child callback changed')
        self.reduction._guard_context();self.refresh_binding.check();self.round._guard_bindings()
        if self.contract.layer==0:self.initial._guard_bindings()
        else:self.initial._guard()
        if self._source is not None and self._fingerprint(self._source)!=self._source_state:
            raise ValueError('layer softmax original score ciphertext changed')
        for value,saved in (*self._retained.values(),*self._captured.values()):
            if self._fingerprint(value)!=saved:raise ValueError('layer softmax retained ciphertext changed')
    def _strip(self,value):
        for current in (value,*(v for v,_ in self._retained.values()),*(v for v,_ in self._captured.values())):
            if current is not self._source:ScoreReduction._forget(current)
    def _callback(self,label,value,counts):
        self._guard();saved=self._fingerprint(value)
        if self.progress is not None:self.progress(label,value,dict(counts))
        self._guard()
        if self._fingerprint(value)!=saved:raise ValueError('layer softmax callback changed its result')
        self._strip(value)
    def _validate(self,value,minimum):
        self.reduction._validate(value,minimum)
        if ScoreReduction._typed(getattr(value,'degree_layout',None))!=ScoreReduction._typed(self.arithmetic.ctx.degree_layout):
            raise ValueError('layer softmax ciphertext layout differs from context')
    def _note(self,label,value,level,actual):
        self._guard();self._validate(value,level)
        if self.arithmetic._level(value)!=level:raise ValueError('layer softmax output level differs from public schedule')
        ScoreReduction._forget(value)
        if self._capture:self._captured[label]=(value,self._fingerprint(value))
        self._callback('layer softmax '+label,value,{})
        actual.append(dict(stage=label,level=level,scale=value._ckks_scale))
        return value
    def level_plan(self,input_level):
        self._guard();expected=45 if self.contract.layer==0 else 21
        if type(input_level)is not int or input_level!=expected:
            raise ValueError('fresh layer0 scores must beL45; later-layer scores must beL21')
        initial=23 if self.contract.layer==0 else 6;level=initial
        depth=self.round.contract.updates+5;rounds=[]
        for index in range(self.contract.squarings):
            before=level;refresh=level<depth+2;entry=25 if refresh else level;level=entry-depth
            if level<2:raise ValueError('frozen round schedule exhausted continuation reserve')
            rounds.append(dict(index=index,preceding_level=before,refreshed=refresh,input_level=entry,output_level=level))
        if level>=self.continuation_levels:raise ValueError('frozen final-refresh assumption changed')
        return dict(layer=self.contract.layer,input_level=input_level,initial_output_level=initial,
            initial_normalized_t_refresh=self.contract.layer!=0,
            key_count=self.round.contract.key_count,selected_squarings=self.contract.squarings,
            updates_per_round=self.round.contract.updates,normalized_rounds=rounds,
            final_refresh=True,final_refresh_mask=True,final_mask_levels=1,output_level=24,
            refresh_count=(self.contract.layer!=0)+sum(row['refreshed'] for row in rounds)+1,
            refresh_public_bound=2.,attention_norm_depth=self.attention_norm_depth,
            continuation_levels_after_mask=self.continuation_levels)
    def _refresh(self,value,label,actual):
        self._guard();self._validate(value,2);saved=self._fingerprint(value)
        result=self.refresher.refresh(value,public_bound=2.);self._guard()
        if self._fingerprint(value)!=saved:raise ValueError('probability refresh changed its source')
        return self._note(label,result,25,actual)
    def evaluate_tile(self,scores,*,capture_checkpoints=False):
        self._guard()
        if self._active:raise ValueError('layer softmax cannot be reentered')
        if type(capture_checkpoints)is not bool:raise ValueError('explicit boolean checkpoint flag required')
        expected=45 if self.contract.layer==0 else 21;self._validate(scores,expected)
        plan=self.level_plan(self.arithmetic._level(scores))
        self._active=True;self._capture=capture_checkpoints;self._source=scores;self._source_state=self._fingerprint(scores);self.last_run=None
        actual=[]
        try:
            value=self._note('initial',self.initial.evaluate_tile(scores),plan['initial_output_level'],actual)
            initial_metadata=self.initial.last_plan
            for row in plan['normalized_rounds']:
                self._retained['round_input']=(value,self._fingerprint(value))
                if row['refreshed']:
                    value=self._refresh(value,'before round'+str(row['index']),actual)
                    self._retained['round_input']=(value,self._fingerprint(value))
                self._guard()
                value=self._note('round'+str(row['index']),self.round.evaluate_tile(value),row['output_level'],actual)
                del self._retained['round_input']
            self._retained['final_input']=(value,self._fingerprint(value))
            value=self._refresh(value,'final continuation refresh',actual)
            self._retained['final_refreshed']=(value,self._fingerprint(value))
            expected_scale=weighted_sum_scale([value._ckks_scale],self.arithmetic.scale,
                drop_divisor(self.arithmetic.ctx._param_cache.q_towers,25))['output_scale']
            value=self.reduction._weighted(value,self.layout.input_mask,label='final refreshed probability mask')
            if ScoreReduction._typed(value._ckks_scale)!=ScoreReduction._typed(expected_scale):
                raise ValueError('final probability mask metadata differs from actual ordered recurrence')
            value=self._note('final masked probabilities',value,24,actual)
            self._guard();self._strip(value)
            self.last_run=dict(plan=plan,initial_metadata=initial_metadata,actual_metadata=actual,
                original_positions=self.original_positions,range_verified=False,repeated_refresh_error_verified=False)
            return (value,{name:current for name,(current,_) in self._captured.items()}) if self._capture else value
        finally:
            self._source=self._source_state=None;self._retained.clear();self._captured.clear();self._active=False;self._capture=False
