"""Public/metadata/failure checks only; no encrypted execution."""
from pathlib import Path
import json,math,sys,types
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parent))
import collect,proof,proof_support as support,fixtures

def valid(before):
    return {'passed':True,'all_twelve_outputs_before_client_decode':True,'no_refresh':True,'inputs_unchanged':True,
        'retained_outputs_unchanged':True,'actual_security_and_P_coverage_checked_before_keygen':True,
        'source_hashes_start':before,'source_hashes_end':before,
        'checks':{case+':'+name:{'passed':True,'slots_checked':65536,'absolute_error_gate':1e-6} for case in ('canonical','noncanonical') for name in fixtures.LABELS}}

def fake(monkeypatch):
    monkeypatch.setattr(collect.os,'sched_getaffinity',lambda pid:set(proof.CPU_AFFINITY))
    monkeypatch.setattr(collect,'source_bytes',lambda:{str(collect.LAB/'prototype/fixture.py'):b'fixed'})
    return collect.hashes(collect.source_bytes())

@pytest.mark.parametrize('fault',('none','missing','error','unreaped','keys','handle','earlydecode','mutatedinput','mutatedoutput','failure','slots','gate','source'))
def test_collector_complete_schedule_cleanup_and_identity(tmp_path,monkeypatch,fault):
    before=fake(monkeypatch)
    def child(command,directory,**options):
        assert options=={'timeout':None,'cleanup_grace_s':60.}
        record=dict(returncode=0,process_reaped=True,public_key_store_removed=True,owned_directory_handle_closed=True)
        raw=valid(before);first=next(iter(raw['checks']))
        if fault=='missing':raw['checks'].pop(first)
        if fault=='error':record['error']='failure'
        if fault=='unreaped':record['process_reaped']=False
        if fault=='keys':record['public_key_store_removed']=False
        if fault=='handle':record['owned_directory_handle_closed']=False
        if fault=='earlydecode':raw['all_twelve_outputs_before_client_decode']=False
        if fault=='mutatedinput':raw['inputs_unchanged']=False
        if fault=='mutatedoutput':raw['retained_outputs_unchanged']=False
        if fault=='failure':raw['checks'][first]['passed']=False
        if fault=='slots':raw['checks'][first]['slots_checked']=128
        if fault=='gate':raw['checks'][first]['absolute_error_gate']=1e-3
        if fault=='source':raw['source_hashes_end']={}
        (directory.parent/'numerical.json').write_text(json.dumps(raw));return record
    monkeypatch.setattr(collect.process,'run_case',child)
    assert collect.run(tmp_path/'attempt')==(0 if fault=='none' else 1)


def test_initial_report_failure_is_preserved(tmp_path,monkeypatch):
    fake(monkeypatch);original=collect.process.write_json;calls=[]
    def writer(path,value):
        calls.append(1)
        if len(calls)==1:raise OSError('initial write failed')
        original(path,value)
    monkeypatch.setattr(collect.process,'write_json',writer)
    assert collect.run(tmp_path/'attempt')==1
    assert 'initial write failed' in json.loads((tmp_path/'attempt/report.json').read_text())['error']


def test_cancellation_survives_final_report_failure(tmp_path,monkeypatch):
    fake(monkeypatch);original=collect.process.write_json;calls=[]
    def writer(path,value):
        calls.append(1)
        if len(calls)>=3:raise OSError('publication failed')
        original(path,value)
    monkeypatch.setattr(collect.process,'write_json',writer)
    def cancel(*args,**kwargs):raise KeyboardInterrupt('primary cancel')
    monkeypatch.setattr(collect.process,'run_case',cancel)
    with pytest.raises(KeyboardInterrupt,match='primary cancel') as caught:collect.run(tmp_path/'attempt')
    assert any('Final collector report' in note for note in caught.value.__notes__)


def test_all_six_public_stage_oracles_independent_coordinates():
    values=fixtures.values();assert len(np.unique(values))==65536
    assert np.min(values)<-39 and np.max(values)>39
    for gain in (1.,.125):
        refs=fixtures.oracle(values,gain)
        assert tuple(refs)==fixtures.LABELS
        for row in (0,1,7,15):
            for head in (0,5,11):
                source=row*4096+64*head;local=row*4096+16*((row+15)%16)+head;target=local+1792
                assert refs['head compaction'][row*4096+head]==values[source]*gain
                assert refs['local key placement'][local]==values[source]*gain
                assert refs['key placement'][target]==values[source]*gain
                assert refs['head expansion'][source]==values[source]*gain
        np.testing.assert_array_equal(np.roll(refs['key placement'],-1792),refs['key tile extraction'])
        assert all(np.count_nonzero(v)==192 for v in refs.values())


@pytest.mark.parametrize('kind',('nan','inf','shape','complex','wide','overflow'))
def test_numerical_failures_stay_json_serializable(kind):
    actual=np.zeros(65536);expected=actual.copy()
    if kind=='nan':actual[-1]=np.nan
    if kind=='inf':actual[-1]=np.inf
    if kind=='shape':actual=actual[:-1]
    if kind=='complex':actual=actual.astype(complex)
    if kind=='wide':
        if np.finfo(np.longdouble).max<=np.finfo(float).max:pytest.skip('no wider longdouble')
        actual=actual.astype(np.longdouble);actual[-1]=np.longdouble('1e4000')
    if kind=='overflow':actual[-1]=1e308;expected[-1]=-1e308
    with np.errstate(all='ignore'):result=support.metrics(actual,expected)
    assert not result['passed'];json.dumps(result,allow_nan=False)
