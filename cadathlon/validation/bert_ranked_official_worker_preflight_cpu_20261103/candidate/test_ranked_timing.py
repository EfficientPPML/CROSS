"""Public-only timing semantics; no model, keys, inputs or runtime execution."""
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
import hashlib
import json
import pytest
import ranked_timing as t


def policy(**changes):
    return t.TimingPolicy(**(dict(kind='upstream_harness', source_inventory_sha256='a'*64,
        hardware_policy_sha256='b'*64,cache_policy_sha256='c'*64,
        fresh_setup_per_invocation=True)|changes))


def stages(values=(1.,2.,3.,4.,5.)):
    return tuple(t.StageTiming(n,name,f'{round(v,4)}s') for (n,name),v in zip(t.STAGES,values))


def run(**changes):
    return t.RunTiming(**(dict(size=1,seed=42,invocation_id='1'*32,key_context_id='2'*64,
        input_sha256='3'*64,policy=policy(),stages=stages(),he_correct=80,official_correct=82)|changes))


def fresh_runs(**changes):
    return tuple(run(seed=seed,invocation_id=f'{i:032x}',key_context_id=f'{i:064x}',**changes)
                 for i,seed in enumerate(t.SEEDS,1))


def test_original_public_source_pins():
    root=Path(__file__).resolve().parent/'timing_upstream'
    assert t.verify_harness_sources({name:(root/name).read_bytes() for name,_ in t.SOURCE_PINS})
    handout=root/'problem.md'
    assert hashlib.sha256(handout.read_bytes()).hexdigest()==t.HANDOUT_SHA256
    with pytest.raises(ValueError):t.verify_harness_sources({name:b'changed' for name,_ in t.SOURCE_PINS})


def test_sum_each_run_then_median_not_sum_of_stage_medians():
    # Each complete run costs 102; taking each stage's median would yield 3.
    values=((100.,1.,1.,0.,0.),(1.,100.,1.,0.,0.),(1.,1.,100.,0.,0.))
    runs=[run(stages=stages(v),run_number=i,num_runs=3) for i,v in enumerate(values,1)]
    report=t.repeat_summary(runs,num_runs=3)
    assert Decimal(report['median_online_seconds'])==102
    assert report['per_run_online_seconds']==['102.0']*3
    assert report['setup_shared_within_repeats'] and report['seed_selection_repeated_by_upstream']
    assert report['ranked_recipe_complete'] is False


def test_even_median_and_default_one_run():
    assert t.DEFAULT_NUM_RUNS==1
    assert Decimal(t.repeat_summary([run()])['median_online_seconds'])==15
    runs=[run(run_number=i,num_runs=2,stages=stages((float(i),0.,0.,0.,0.))) for i in (1,2)]
    assert Decimal(t.repeat_summary(runs,num_runs=2)['median_online_seconds'])==Decimal('1.5')


def test_parse_only_original_online_labels_not_total_or_server():
    data={'model_name':'bert-base','dataset_name':'mrpc','Timing':
          {name:f'{i}.0s' for i,(_,name) in enumerate(t.STAGES,1)}}
    data['Timing'].update({'Total':'99999.0s','Key generation':'600.0s'})
    data['Server Reported']={'runtime':'.001s'}
    assert run(stages=t.stages_from_harness_json(json.dumps(data).encode())).online_seconds==15
    del data['Timing']['Input encryption']
    with pytest.raises(ValueError):t.stages_from_harness_json(json.dumps(data).encode())


@pytest.mark.parametrize('raw',[b'{"Timing":{},"Timing":{}}',b'[]',b'{"x":NaN}',b' '*((1<<20)+1)])
def test_json_rejects_duplicate_invalid_or_unbounded(raw):
    with pytest.raises(ValueError):t.stages_from_harness_json(raw)


@pytest.mark.parametrize('value',['NaNs','Infinitys','-0.0s','-1.0s','1s','1.0000s','0.00001s','1e999s',1.0,True])
def test_original_seconds_representation(value):
    with pytest.raises(ValueError):t.StageTiming(5,'Input preprocessing',value)


@pytest.mark.parametrize('value',['0.0s','0.0001s','1.2s','1234567.8912s'])
def test_valid_harness_seconds(value):
    assert t.StageTiming(5,'Input preprocessing',value).seconds_text==value


@pytest.mark.parametrize('bad',[
    stages()[:-1],stages()[::-1],(stages()[0],)*5,list(stages()),
])
def test_missing_duplicate_reordered_or_mutable_stage_inventory(bad):
    with pytest.raises(ValueError):run(stages=bad)


@pytest.mark.parametrize('changes',[{'size':True},{'seed':True},{'he_correct':True},{'official_correct':82.0},
    {'he_correct':101},{'num_runs':True},{'run_number':0},{'passed':1},{'key_context_id':'short'}])
def test_strict_typed_run_fields(changes):
    with pytest.raises(ValueError):run(**changes)


@pytest.mark.parametrize('case',[
    {'size':1,'he_correct':75,'official_correct':77},
    {'size':2,'he_correct':150,'official_correct':154},
    {'size':3,'he_correct':306,'official_correct':314},
])
def test_exact_integer_accuracy_boundary_accepts(case):
    assert t.repeat_summary([run(**case)])['num_runs']==1


@pytest.mark.parametrize('case',[
    {'size':1,'he_correct':74,'official_correct':74},
    {'size':2,'he_correct':149,'official_correct':149},
    {'size':3,'he_correct':305,'official_correct':305},
    {'size':1,'he_correct':75,'official_correct':78},
    {'size':2,'he_correct':150,'official_correct':155},
    {'size':3,'he_correct':306,'official_correct':315},
])
def test_exact_integer_accuracy_boundary_rejects(case):
    with pytest.raises(ValueError):t.repeat_summary([run(**case)])


def test_failed_run_or_stage_not_filtered_from_aggregate():
    for bad in (run(passed=False),run(stages=(replace(stages()[0],passed=False),)+stages()[1:])):
        with pytest.raises(ValueError):t.repeat_summary([bad])


@pytest.mark.parametrize('change',[
    {'input_sha256':'4'*64},{'seed':43},{'invocation_id':'4'*32},
    {'policy':policy(cache_policy_sha256='d'*64)},{'run_number':1},
])
def test_mixed_or_duplicate_repeated_runs_rejected(change):
    first=run(num_runs=2,run_number=1)
    second=run(**(dict(num_runs=2,run_number=2)|change))
    with pytest.raises(ValueError):t.repeat_summary([first,second],num_runs=2)
    with pytest.raises(ValueError):t.repeat_summary([first],num_runs=2)


def test_three_fresh_one_run_seeds_and_throughput():
    report=t.seed_summary(fresh_runs())
    assert report['seeds']==[42,43,44] and report['count']==100
    assert Decimal(report['median_online_seconds'])==15
    assert report['accuracy_counts_validated'] is True
    assert report['authentication_is_external'] is True
    assert report['full_benchmark_acceptance_not_implied'] is True
    assert abs(Decimal(report['throughput_samples_per_second'])-Decimal(100)/15)<Decimal('1e-26')


@pytest.mark.parametrize('mode',['diagnostic','retained_setup','same_key','same_invocation','repeated_run','mixed_size','mixed_policy','reorder','partial','zero'])
def test_seed_aggregate_cannot_promote_incomplete_or_nonofficial_records(mode):
    runs=list(fresh_runs())
    if mode=='diagnostic':runs=list(fresh_runs(policy=policy(kind='component_diagnostic')))
    elif mode=='retained_setup':runs=list(fresh_runs(policy=policy(fresh_setup_per_invocation=False)))
    elif mode=='same_key':runs[1]=replace(runs[1],key_context_id=runs[0].key_context_id)
    elif mode=='same_invocation':runs[1]=replace(runs[1],invocation_id=runs[0].invocation_id)
    elif mode=='repeated_run':runs[1]=replace(runs[1],num_runs=2)
    elif mode=='mixed_size':runs[1]=replace(runs[1],size=2,he_correct=160,official_correct=164)
    elif mode=='mixed_policy':runs[1]=replace(runs[1],policy=policy(hardware_policy_sha256='d'*64))
    elif mode=='reorder':runs.reverse()
    elif mode=='partial':runs.pop()
    elif mode=='zero':runs=list(fresh_runs(stages=stages((0.,)*5)))
    with pytest.raises(ValueError):t.seed_summary(runs)


def test_stage_number_and_policy_are_exact_typed():
    with pytest.raises(ValueError):t.StageTiming(5.,'Input preprocessing','1.0s')
    with pytest.raises(ValueError):t.StageTiming(5,'Input preprocessing','1.0s',1)
    with pytest.raises(ValueError):policy(fresh_setup_per_invocation=1)
    with pytest.raises(ValueError):policy(kind='server_reported')
