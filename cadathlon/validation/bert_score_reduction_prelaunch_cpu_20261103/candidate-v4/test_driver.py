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
    r=driver.parameters()
    assert len(r['q_towers'])==10 and len(r['p_towers'])==3 and r['depth']==4 and r['dnum']==4
    assert r['actual_log2_qp']<3523 and len(r['rotation_indices'])==14
    assert r['raw_key_pair_bytes']==54525952 and r['raw_all_owned_public_key_bytes']==828375040
    assert [x['K'] for x in r['public_fixtures']]==[128,15]
    assert all(x['domain'][0]<=x['inactive_dummy']<=x['domain'][1]<1 for x in r['public_fixtures'])


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
    monkeypatch.setattr(driver,'parameters',lambda:pytest.fail('must stop before ring/key generation'))
    with pytest.raises(OSError,match='injected'):driver.main()
    report=json.loads((output/'report.json').read_text())
    assert not report['passed'] and 'injected' in report['error']
    assert not report['key_generation_started'] and not report['encryption_started']
