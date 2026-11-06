import json
import numpy as np
import pytest
from q5_test_fixtures import DENSE_REFRESH_KEYS,EXPECTED_SELECTED_TRACE_INDICES
from q5_checkpoint_oracle import mapping,references,diagnostics,original_final_gate
from q5_actual_metadata_replay import client_replay


def test_all47_new_literal_to_independent_dense_mapping(math_fixture):
    f=math_fixture
    actual_indices=tuple(i for i,_ in f.captures)
    assert actual_indices==EXPECTED_SELECTED_TRACE_INDICES
    assert len(f.arithmetic.trace)==216
    assert len(f.observations)==19 and len(f.rows)==9+2*len(f.observations)==47
    assert f.arithmetic.counter==f.dense['counts']==dict(linear=22,normalize=7,normalize_heads=6,rope=6,attention=3,polynomial=22)
    assert f.arithmetic.counts==dict(add=270,weighted=214,multiply=226)
    dense,info=references(np,f.plan['case']['config'],f.weights,f.embeddings,f.rules)
    assert info['complete_dense47']is True and info['new_mapping_focused_proof']is False
    assert len(dense)==47 and all(row['passed']for row in info['original_domains'])
    for index,((trace_index,literal),reference)in enumerate(zip(f.captures,dense,strict=True)):
        row=f.rows[index]
        key=DENSE_REFRESH_KEYS[int(row['label'])]if row['kind'].startswith('refresh_')else(row['kind'],row['label'])
        assert tuple(mapping()[index]['dense_key'])==key
        period=f.dense['records'][key]
        assert reference.shape==(65536,)and reference.dtype==np.dtype('float64')
        assert reference.flags.writeable is False
        # Explicit independent period indices, not packed-layout unpack/roll.
        expected=period[np.arange(65536)%1024]
        np.testing.assert_array_equal(reference,expected)
        assert float(np.max(np.abs(literal-reference)))<=1e-10
        np.testing.assert_array_equal(literal.reshape(64,1024),np.broadcast_to(literal[:1024],(64,1024)))
    assert f.rows[-2]['kind']=='normalize_after'and f.rows[-1]['kind']=='output'
    assert f.rows[-2]['level']==6 and f.rows[-1]['level']==5


def test_key_before_query_refreshes_have_distinguishable_oracles(math_fixture):
    f=math_fixture
    for refresh_id,key_label,query_label in [(5,'3','2'),(13,'5','4')]:
        key=f.dense['records'][('rope_after',key_label)]
        query=f.dense['records'][('rope_after',query_label)]
        assert float(np.max(np.abs(key-query)))>1e-6,'fixture must detect swapped attribution'
        for offset,wanted in [(0,key),(1,query)]:
            rows=[i for i,r in enumerate(f.rows)if r['kind'].startswith('refresh_')and int(r['label'])==refresh_id+offset]
            assert len(rows)==2
            for i in rows:
                assert float(np.max(np.abs(f.captures[i][1].reshape(64,1024)-wanted)))<=1e-10


def test_client_replay_consumes_all19_synthetic_observed_return_scales(math_fixture):
    f=math_fixture
    scales=[r['after']['scale']for r in f.observations]
    assert len(scales)==19 and len(set(scales))>=2
    assert any(scale!=f.plan['profile']['scale']for scale in scales)
    report=client_replay(np,f.reference_root,f.weights,f.embeddings,f.rows,f.observations)
    assert report==dict(complete=True,checkpoints=47,refresh_returns=19,primitive_counts=dict(add=270,weighted=214,multiply=226),actual_scales_consumed=True,nominal_refresh_substitution=False,ciphertext_accuracy_claim=False)


@pytest.mark.parametrize('case',['padded_other_replica','logical_first_period','final_selected_row'],ids=['padded_other_replica','logical_first_period','final_selected_row'])
def test_diagnostic_indices_do_not_hide_replica_or_padding_errors(case):
    reference=np.zeros(65536,dtype=np.float64);actual=reference.copy()
    if case=='padded_other_replica':
        actual[1024+31]=3.;report=diagnostics(np,actual,reference,0)
        assert report['first_period_max_abs']==0. and report['other_replicas_max_abs']==3.
        assert report['valid_feature_max_abs']==0. and report['padded_feature_max_abs']==3.
        assert report['replica_max_deviation']==3.
    elif case=='logical_first_period':
        actual[((1*4+2)*4+1)*32+7]=2.;report=diagnostics(np,actual,reference,0)
        assert report['valid_feature_max_abs']==2. and report['padded_feature_max_abs']==0.
        assert report['first_period_max_abs']==2. and report['other_replicas_max_abs']==0.
    else:
        actual[((1*4+3)*4)*32+9]=4.;report=diagnostics(np,actual,reference,46)
        assert report['dense_polynomial_final_logits_max_abs']==4.
    assert report['scientific_threshold_selected']is False


def test_original_final_gate_remains_separate_from_dense_diagnostics(math_fixture):
    f=math_fixture
    expected=np.asarray(json.loads((f.reference_root/'frozen/cadathlon/agent_pset/p4_qwen/test/test5.out').read_bytes())['logits'],dtype=np.float64)
    slots=np.zeros(65536,dtype=np.float64)
    for b in range(2):slots[((b*4+3)*4)*32:((b*4+3)*4)*32+32]=expected[b]
    assert original_final_gate(np,f.reference_root,slots)['passed']is True
    slots[(3*4)*32]+=0.201
    assert original_final_gate(np,f.reference_root,slots)['passed']is False
