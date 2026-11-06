"""Pure public plan assertions with independent original per-K grouping."""
import json
import pytest
from plan_full import derive
from test_softmax_cohorts import M,C,independent_cohorts

@pytest.fixture(scope='module')
def plan():return derive()

def test_selected_cohort_descriptors_counts_all32_and28(plan):
    for row in plan['workloads']:
        n=row['sequence'];assert row['softmax_strategy']=='tile-cohorts-v1'
        total=0
        for layer,record in enumerate(row['softmax_cohorts']):
            assert record['layer']==layer and record['squarings']==M['softmax'][str(layer)]['squarings']
            layer_total=0
            for tile,groups in enumerate(record['tiles']):
                expected=independent_cohorts(n,tile)
                assert [(x['newton_iterations'],x['refresh_bound'],tuple(q['k']for q in x['queries']))for x in groups]==list(expected)
                for group in groups:
                    layer_total+=3+2*record['squarings']
                    for query in group['queries']:
                        k=query['k'];rule=M['reciprocal_by_keys'][str(k)]
                        assert query['position']==row['original_positions'][k-1]and query['row']==(k-1)%16
                        assert query['alpha']==float(1/k)and query['domain']==rule['domain']
                        assert (query['first_weight'],query['first_bias'])==(rule['rounded_first_weight'],rule['rounded_first_bias'])
            assert layer_total==record['refreshes'];total+=layer_total
        assert total==row['refresh_counts']['softmax']
        literal=(n-1)*sum(3+2*x['causal_squarings']for x in plan['layers'])
        assert literal==row['literal_reference_softmax_refreshes']
        assert row['refresh_counts']['total']-total==row['refresh_counts']['literal_reference_total']-literal
    final=plan['workloads'][-1]
    assert(final['refresh_counts']['softmax'],final['refresh_counts']['total'],final['refresh_counts']['literal_reference_total'])==(3760,5035,12931)

def test_full_contract_head_profile_and_execution_scope_unchanged(plan):
    assert plan['resources']['rotation_keys']==166 and plan['resources']['all38_final_ciphertext_bytes']==228*1024**2
    assert len(plan['vocabulary']['blocks'])==38 and plan['vocabulary']['all_logits']==151936
    assert plan['final_norm']['output_level']==3 and plan['vocabulary']['output_level']==2
    assert plan['execution_authorized']is False and plan['reference']['executed']is False
