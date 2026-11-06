"""Public schema/count/policy tests only; no input body or selector execution."""
import copy,json,math
import pytest
import protocol as p
@pytest.fixture(scope='module')
def result():return p.derive()
def test_exact3200_and_original416_subset(result):
 rows=result['calibration']['workload'];assert len(rows)==3200 and sum(r['original416_member']for r in rows)==416
 assert len({(r['prompt_ordinal'],r['length'])for r in rows if r['kind']=='prefill'})==3072
 assert all(sum(r['kind']=='prefill' and r['length']==n for r in rows)==96 for n in range(1,33))
 assert sum(r['kind']=='teacher'for r in rows)==sum(r['kind']=='free'for r in rows)==64
 assert all(r['original416_member']==(r['length']in(8,16,32))for r in rows if r['kind']=='prefill')
def test_all_original_sites_and_rules_bound(result):
 inventory=result['observation']['selection_sites'];assert len(inventory['normalization'])==113 and len(inventory['silu'])==len(inventory['centered_softmax'])==28
 assert result['old_candidate_manifest_sha256']=='7f6478a6bc710f3d2a370198e075e6ba8b0e55c144d981bca0eed59eaf176333'
 assert result['selection']['degree_ladder']==[16,32,64,128,256,512] and result['selection']['maximum_candidates']==3
 assert result['observation']['full_metrics']==dict(operators_per_context=421,blocks_per_context=28,last_position_logits=151936)
def test_unchanged_gates_and_exact_aggregate_denominators(result):
 old=p.pinned('original_protocol.json')['gates'];new=result['acceptance']['unchanged_numeric_gates']
 for k,v in old.items():
  if k not in ('prefill_top1_aggregation','teacher_forced_required_matches'):assert new[k]==v
 assert result['acceptance']['calibration']['prefill_required_matches']==2765
 assert result['acceptance']['untouched_validation']['prefill_required_matches']==922
 assert result['acceptance']['calibration']['teacher_required_matches']==58 and result['acceptance']['untouched_validation']['teacher_required_matches']==29
 assert result['acceptance']['calibration']['complete_free_required']==12 and result['acceptance']['untouched_validation']['complete_free_required']==6
@pytest.mark.parametrize('n,k',[(3072,2765),(288,260),(64,58),(1024,922),(32,29),(48,44)])
def test_rate_rounding(n,k):assert p.required_matches(n)==k
@pytest.mark.parametrize('n',[True,0,-1,1.,float('nan')])
def test_rate_bad_type(n):
 with pytest.raises(ValueError):p.required_matches(n)
@pytest.mark.parametrize('values,expected',[((1.,10.,1e-6),[.2,48.]),((1e-6,1e-6,1e-6),[1e-6,4.8e-6]),((100.,1000.,1e-6),[20.,4800.])])
def test_uniform_whole_site_factor4_rule(values,expected):assert p.uniform_domain(*values)==expected
@pytest.mark.parametrize('values',[(True,1.,1e-6),(0.,1.,1e-6),(-1.,1.,1e-6),(1.,float('nan'),1e-6),(2.,1.,1e-6),(1.,float('inf'),1e-6),(1.,1.,0.),(1.,1e308,1e-6)])
def test_no_coercion_clip_or_nonfinite_domain(values):
 with pytest.raises(ValueError):p.uniform_domain(*values)
def test_validation_is_new_recipe_with_pending_hard_gate(result):
 v=result['untouched_validation'];d=v['design'];assert not v['measured']and not v['tokenized']and not v['token_disjointness_verified']
 assert len(d['prompts'])==len({x['text']for x in d['prompts']})==32
 assert d['prefix_lengths']==list(range(1,33))and len(d['decode_seed_prompt_ordinals'])==8 and d['total']==1088
 assert len(set(d['decode_seed_prompt_ordinals']))==8
 assert result['row0_regression']['excluded_from_selection']is True and result['row0_regression']['original_failure_remains']is True
 assert result['execution_authorized']is False and result['distinct_from_original_three_candidate_protocol']is True

def test_public_pin_mutation_fails_before_policy(tmp_path,monkeypatch):
 (tmp_path/'inputs').mkdir();(tmp_path/'input_pins.json').write_bytes((p.P/'input_pins.json').read_bytes());(tmp_path/'inputs/original_protocol.json').write_text('{}');monkeypatch.setattr(p,'P',tmp_path)
 with pytest.raises(ValueError,match='hash'):p.derive()


def test_short_prefix_collision_against_expanded_inventory_is_rejected():
 r=p.context_accounting([[11],[11,22],[11,22]],[[11],[33],[33]])
 assert not r['disjoint'] and r['overlap_by_length']=={'1':1}
 assert r['calibration']['2']==dict(entries=2,unique=1)and r['validation']['1']==dict(entries=3,unique=2)
 assert r['calibration_entries']==r['validation_entries']==3 and r['denominators_preserved']
def test_different_lengths_are_different_full_contexts_but_row0_is_excluded():
 assert p.context_accounting([[11,22]],[[11],[33]])['disjoint']
 assert not p.context_accounting([[11,22]],[[0]])['disjoint']
 assert not p.context_accounting([[11]],[[99,88]],[[99,88]])['disjoint']
@pytest.mark.parametrize('bad',[[True],[1.0],[],[-1]])
def test_context_no_token_coercion(bad):
 with pytest.raises(ValueError):p.context_accounting([bad],[[2]])
