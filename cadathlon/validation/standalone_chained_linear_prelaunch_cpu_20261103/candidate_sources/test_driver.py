"""No-encryption preflight and cancellation/publication checks."""
from pathlib import Path
import json,sys
import pytest
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import check_component as driver
from lifecycle import CleanupTracker


def test_actual_public_profile_without_keygen(monkeypatch):
    def forbidden(*a,**k):pytest.fail('preflight must not create keys/contexts')
    for name in ('gen_pke_pair','gen_evaluation_key','gen_rotation_key','key_switch_gen'):
        monkeypatch.setattr(driver.key_gen,name,forbidden)
    monkeypatch.setattr(driver.CKKSContext,'__init__',forbidden)
    import numpy as np
    matrices={name:(np.ones(shape),np.zeros(shape[0])) for name,shape in
        [('query',(768,768)),('intermediate',(3072,768)),('output',(768,3072))]}
    r=driver.parameters(matrices)
    assert len(r['q_towers'])==4 and len(r['p_towers'])==3 and r['depth']==1 and r['dnum']==2
    assert r['actual_log2_qp']<438 and set(r['rotation_indices'])=={1,64,4416,6720}
    assert r['raw_key_pair_bytes']==1835008 and r['raw_all_owned_public_key_bytes']==9699328
    assert [x['nonzero_rotation_calls'] for x in r['matrix_plans'].values()]==[86,122,122]


@pytest.mark.parametrize('primary', [None, KeyboardInterrupt('primary')])
def test_attempt_all_cleanup_and_cancellation(primary):
    tracker=CleanupTracker(primary);events=[]
    def fail():events.append('first');raise RuntimeError('failure')
    def cancel():events.append('cancel');raise SystemExit(143)
    tracker.attempt('first',fail);tracker.attempt('cancel',cancel)
    tracker.attempt('last',lambda:events.append('last'))
    assert events==['first','cancel','last'] and len(tracker.errors)==2
    if primary is None:
        with pytest.raises(SystemExit) as caught:tracker.finish()
        assert caught.value.code==143
    else:tracker.finish();assert len(primary.__notes__)==2


@pytest.mark.parametrize('failure', ['initial-write','sources','metadata'])
def test_preflight_early_failure_publishes_failed_attempt(tmp_path,monkeypatch,failure):
    output=tmp_path/'out';monkeypatch.setattr(sys,'argv',['driver','--preflight','--output',str(output)])
    monkeypatch.setattr(driver,'sources',lambda:{'fixed.py':b'fixed'})
    def fail(*args):raise OSError('injected '+failure)
    if failure=='initial-write':
        original=driver.write;calls=[]
        def first_fails(path,value):
            calls.append(1)
            if len(calls)==1:fail()
            return original(path,value)
        monkeypatch.setattr(driver,'write',first_fails)
    elif failure=='sources':monkeypatch.setattr(driver,'sources',fail)
    else:monkeypatch.setattr(driver.importlib.metadata,'version',fail)
    monkeypatch.setattr(driver,'parameters',lambda *a,**k:pytest.fail('must stop before ring/key generation'))
    with pytest.raises(OSError,match='injected'):driver.main()
    report=json.loads((output/'report.json').read_text())
    assert not report['passed'] and 'injected' in report['error']
    assert not report['key_generation_started'] and not report['encryption_started']


def test_pinned_public_weights_fixture_oracles_and_two_distinct_rows():
    import numpy as np
    from pinned_assets import load_matrices,make_fixtures
    matrices,record=load_matrices('/tmp/bert-layer0-linear-weights.npz',
        '/home/jianming_gatech/cross_validation_assets/bert/f150c1d609d1e50dd5e2e5408661cfac8339277c/model.safetensors')
    layout=driver.RowTileLayout(8192)
    fixtures=make_fixtures(matrices,layout)
    assert record['export_verified_against_actual_checkpoint_offsets']
    for name,(weight,bias) in matrices.items():
        item=fixtures[name];valid=item['valid_rows'];raw=item['input'].reshape(2,4096)
        expected=np.zeros((2,4096));active=raw[:valid,:weight.shape[1]]
        if item['input_mask'] is not None:active=active*item['input_mask'].reshape(2,4096)[:valid,:weight.shape[1]]
        expected[:valid,:weight.shape[0]]=np.einsum('ri,oi->ro',active,weight)+bias
        np.testing.assert_allclose(expected.ravel(),item['expected'],rtol=0,atol=2e-14)
        assert np.all(raw[:valid,weight.shape[1]:]>=8)
        if valid==1:assert np.all(raw[1]>=8) and not item['expected'].reshape(2,4096)[1].any()
        else:assert not np.array_equal(raw[0],raw[1])
    assert fixtures['output']['contract']['signed_public_source_mask']


def test_observer_only_counts_real_rotations_and_restores_public_primitive_methods():
    import numpy as np
    sys.path.insert(0,str(HERE/'checkout/cadathlon/tests'))
    from test_chained_streaming_linear import Arithmetic,Value
    ops=Arithmetic(64,[]);counts=dict(rotations=0,weighted_sums=0,public_multiplications=0,ciphertext_additions=0)
    original=driver.observe_arithmetic(ops,counts)
    value=Value(np.ones(64))
    # Counter intentionally uses the proof's full physical slot count.
    ops.rotate(value,0);ops.rotate(value,8192);ops.rotate(value,64)
    result=ops.weighted_sum([value,value],[2.,3.]);ops.add(result,result)
    ops.rotate,ops.weighted_sum,ops.add=original
    assert counts==dict(rotations=1,weighted_sums=1,public_multiplications=2,ciphertext_additions=1)
    np.testing.assert_array_equal(result.slots,5.)
