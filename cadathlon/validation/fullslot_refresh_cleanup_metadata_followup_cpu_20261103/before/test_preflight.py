"""No-keygen/no-encryption contract, gating and lifecycle tests for the prepared proof."""
from pathlib import Path
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import numpy as np
import pytest

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import proof
import collect


def test_independent_full_slot_pattern_is_fixed():
    values=proof.input_pattern()
    assert values.shape==(65536,) and values.dtype==np.float64
    assert len(np.unique(values))==65536
    assert values.min()==-16 and values.max()==16
    assert 0 in values and -14.2 in values and 14.2 in values
    for size in (128,4096,32768):assert not np.array_equal(values[:size],values[size:2*size])
    np.testing.assert_array_equal(values,proof.input_pattern())


def test_metrics_all_slots_and_zero_relative_denominator():
    values=proof.input_pattern();row=proof.metrics(values,values)
    assert row['passed'] and row['max_abs_error']==0 and row['max_relative_error_with_1e_12_floor']==0
    changed=values.copy();changed[-1]+=.002
    row=proof.metrics(changed,values)
    assert not row['passed'] and row['worst_slot_index']==65535
    assert len(row['per_4096_slot_block_max_abs_error'])==16


@pytest.mark.parametrize('kind',('nan','inf','shape','complex'))
def test_metrics_malformed_result_fails(kind):
    values=proof.input_pattern();actual=values.copy()
    if kind=='nan':actual[100]=np.nan
    if kind=='inf':actual[100]=np.inf
    if kind=='shape':actual=actual[:-1]
    if kind=='complex':actual=actual.astype(complex)
    assert not proof.metrics(actual,values)['passed']


def test_metrics_fixed_strict_boundary():
    x=np.zeros(65536);actual=x.copy();actual[42]=.001
    assert not proof.metrics(actual,x)['passed']
    actual[42]=np.nextafter(.001,0.)
    assert proof.metrics(actual,x)['passed']


def test_owned_tmpfs_path_only(tmp_path):
    for path in ('/dev/shm',tmp_path):
        with pytest.raises(ValueError):proof.validate_key_directory(path)
    with tempfile.TemporaryDirectory(prefix='cross-fullslot-preflight-',dir='/dev/shm') as name:
        assert proof.validate_key_directory(name)==Path(name)
        link=tmp_path/'link';link.symlink_to(name)
        with pytest.raises(ValueError,match='symlink'):proof.validate_key_directory(link)


def test_actual_secure_full_slot_public_plan_no_keygen(monkeypatch):
    from jaxite_word import CKKSContext,key_gen,he_params
    from jaxite_word.bootstrapping import Bootstrap
    from encrypted_refresh import RefreshProfile,refresh_plan
    def forbidden(*args,**kwargs):raise AssertionError('crypto construction forbidden in preflight')
    monkeypatch.setattr(CKKSContext,'__init__',forbidden);monkeypatch.setattr(Bootstrap,'__init__',forbidden)
    for name in ('gen_pke_pair','gen_rotation_key','gen_evaluation_key'):monkeypatch.setattr(key_gen,name,forbidden)
    params=RefreshProfile(131072,49,56,61,4).parameters();plan=refresh_plan(131072,65536,(4,4),max_level=49)
    assert len(params['q_towers'])==100 and len(params['p_towers'])==24
    assert he_params.qp_is_secure(params['q_towers'],params['p_towers'],131072,he_params.SecurityTarget(128,'classical'))
    he_params.assert_p_coverage(params['q_towers'],params['p_towers'],4)
    assert len(plan['rotations'])==106 and 262143 in plan['rotations']
    assert plan['output_level']==25 and plan['period']==65536
    assert proof.required_key_space(params,plan['rotations'])==55197908992


@pytest.mark.parametrize('primary',(False,True))
def test_cleanup_attempts_every_owner_even_after_failure(monkeypatch,primary):
    import jax
    from jaxite_word import ckks_ctx
    events=[]
    def fail(name):
        def call(*args):events.append(name);raise RuntimeError(name)
        return call
    owner=types.SimpleNamespace(close=fail('owner'));store=types.SimpleNamespace(close=fail('store'))
    monkeypatch.setattr(ckks_ctx,'evict_context_caches',fail('context'))
    monkeypatch.setattr(jax,'clear_caches',lambda:events.append('jax'))
    monkeypatch.setattr(proof.gc,'collect',lambda:events.append('gc'))
    active=KeyboardInterrupt('primary') if primary else None
    if primary:proof.release_all([object(),object()],owner,store,active);assert active.__notes__
    else:
        with pytest.raises(RuntimeError,match='owner'):proof.release_all([object(),object()],owner,store,active)
    assert events==['owner','context','context','jax','gc','store']


def test_import_does_not_initialize_backend():
    script="import sys;sys.path.insert(0,"+repr(str(HERE))+");import proof;assert 'jax' not in sys.modules;assert len(proof.input_pattern())==65536"
    result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,env=dict(os.environ,JAX_PLATFORMS='cpu'))
    assert result.returncode==0,result.stderr


def fake_snapshot(monkeypatch):
    monkeypatch.setattr(collect.os,'sched_getaffinity',lambda pid:{132,133,134,135})
    monkeypatch.setattr(collect,'source_bytes',lambda:{str(collect.LAB/'prototype/fixed.py'):b'fixed'})


def test_collector_initial_source_failure_preserved(tmp_path,monkeypatch):
    monkeypatch.setattr(collect.os,'sched_getaffinity',lambda pid:{132,133,134,135})
    def failure():raise OSError('source unavailable')
    monkeypatch.setattr(collect,'source_bytes',failure)
    assert collect.run(tmp_path/'out')==1
    report=json.loads((tmp_path/'out/report.json').read_text());assert not report['passed'] and 'source unavailable' in report['error']


@pytest.mark.parametrize('failure',('returncode','unreaped','retainedkeys','openhandle','wronglevel','partialslots','noend'))
def test_collector_cannot_claim_partial_or_unclean_pass(tmp_path,monkeypatch,failure):
    fake_snapshot(monkeypatch)
    def run_case(command,directory,**options):
        assert options=={'timeout':None,'cleanup_grace_s':60.}
        raw={'passed':True,'full_slots_checked':65536,'output_level':25,'square_level':24,'server_completed_before_client_decode':True}
        record={'returncode':0,'process_reaped':True,'public_key_store_removed':True,'owned_directory_handle_closed':True}
        if failure=='returncode':record['returncode']=1
        if failure=='unreaped':record['process_reaped']=False
        if failure=='retainedkeys':record['public_key_store_removed']=False
        if failure=='openhandle':record['owned_directory_handle_closed']=False
        if failure=='wronglevel':raw['output_level']=24
        if failure=='partialslots':raw['full_slots_checked']=128
        if failure=='noend':raw['server_completed_before_client_decode']=False
        (directory.parent/'numerical.json').write_text(json.dumps(raw));return record
    monkeypatch.setattr(collect.process,'run_case',run_case)
    assert collect.run(tmp_path/'out')==1
    report=json.loads((tmp_path/'out/report.json').read_text());assert not report['passed'] and report['source_files_unchanged']


def test_collector_cancellation_survives_final_write_failure(tmp_path,monkeypatch):
    fake_snapshot(monkeypatch);writes=0;original=collect.process.write_json
    def writer(path,value):
        nonlocal writes
        writes+=1
        if writes>=3:raise OSError('write failed')
        original(path,value)
    def cancel(*args,**kwargs):raise KeyboardInterrupt('primary cancel')
    monkeypatch.setattr(collect.process,'write_json',writer);monkeypatch.setattr(collect.process,'run_case',cancel)
    with pytest.raises(KeyboardInterrupt,match='primary cancel') as caught:collect.run(tmp_path/'out')
    assert any('Final collector' in note for note in caught.value.__notes__)


def test_every_codec_guard_and_server_identity():
    class Context:pass
    for name in proof.CODEC_METHODS:setattr(Context,name,lambda self,*args:'allowed')
    client,server=Context(),Context();phase=['client-encoding'];restorations=[]
    proof.install_codec_guards(Context,lambda:server,lambda:phase[0],restorations)
    for name in proof.CODEC_METHODS:
        assert getattr(client,name)()=='allowed'
        with pytest.raises(AssertionError):getattr(server,name)()
    phase[0]='server'
    for name in proof.CODEC_METHODS:
        with pytest.raises(AssertionError):getattr(client,name)()
    phase[0]='client-validation'
    for name in proof.CODEC_METHODS:assert getattr(client,name)()=='allowed'
    for restore in restorations:restore()
    assert server.decrypt_slots_batch()=='allowed'


def test_keygen_all_entrypoints_both_aliases_deduplicated():
    module=types.ModuleType('fixture_key_gen')
    for name in proof.KEYGEN_METHODS:setattr(module,name,lambda:'allowed')
    restorations=[]
    proof.install_generator_guards({'key_gen':module,'jaxite_word.key_gen':module},restorations)
    assert len(restorations)==4
    for name in proof.KEYGEN_METHODS:
        with pytest.raises(AssertionError):getattr(module,name)()
    for restore in restorations:restore()
    for name in proof.KEYGEN_METHODS:assert getattr(module,name)()=='allowed'
    with pytest.raises(AssertionError,match='both historical'):
        proof.install_generator_guards({'key_gen':module},[])


def test_cleanup_cancellation_and_restore_failure_attempt_all(monkeypatch):
    import jax
    from jaxite_word import ckks_ctx
    events=[]
    def fail_restore():events.append('restore');raise RuntimeError('restore failed')
    def cancel():events.append('cancel');raise KeyboardInterrupt('cleanup cancelled')
    owner=types.SimpleNamespace(close=lambda:events.append('owner'))
    store=types.SimpleNamespace(close=lambda:events.append('store'))
    monkeypatch.setattr(ckks_ctx,'evict_context_caches',lambda ctx:events.append('context'))
    monkeypatch.setattr(jax,'clear_caches',lambda:events.append('jax'));monkeypatch.setattr(proof.gc,'collect',lambda:events.append('gc'))
    callbacks=[fail_restore,cancel]
    with pytest.raises(KeyboardInterrupt,match='cleanup cancelled'):
        proof.release_all([object()],owner,store,None,callbacks)
    assert not callbacks and events==['restore','cancel','owner','context','jax','gc','store']


def test_input_fingerprint_detects_payload_and_scale_mutation():
    value=types.SimpleNamespace(polynomial=np.zeros((1,2,2,2,4),dtype=np.uint32),degree=4,batch=1,num_elements=2,
        num_moduli=4,moduli=[17,41,73,89],degree_layout=(2,2),precision=32,_ckks_scale=16.,_ckks_nsd=1,
        ntt_ctx=types.SimpleNamespace(ff_ctx=types.SimpleNamespace()))
    original=proof.ciphertext_fingerprint(value);value.polynomial[0,0,0,0,0]=1
    assert proof.ciphertext_fingerprint(value)!=original
    value.polynomial[0,0,0,0,0]=0;value._ckks_scale=32.
    assert proof.ciphertext_fingerprint(value)!=original


def test_insecure_actual_ring_stops_before_any_key_generation(monkeypatch):
    from jaxite_word import key_gen,he_params
    events=[]
    monkeypatch.setattr(he_params,'qp_is_secure',lambda *a,**k:False)
    for name in proof.KEYGEN_METHODS:
        monkeypatch.setattr(key_gen,name,lambda *a,**k:events.append('keygen'))
    with tempfile.TemporaryDirectory(prefix='cross-fullslot-preflight-',dir='/dev/shm') as directory:
        monkeypatch.setenv('CROSS_PUBLIC_KEY_DIRECTORY',directory)
        monkeypatch.setattr(proof.os,'sched_getaffinity',lambda pid:{132,133,134,135})
        with pytest.raises((ValueError,AssertionError),match='128-bit|insecure'):
            proof.run({})
    assert not events


@pytest.mark.parametrize('which',('collector','proof'))
def test_initial_publication_failure_can_record_failed_attempt(tmp_path,monkeypatch,which):
    calls=[];original=collect.process.write_json
    def first_fails(path,value):
        calls.append(1)
        if len(calls)==1:raise OSError('initial write failed')
        return original(path,value)
    monkeypatch.setattr(collect.process,'write_json',first_fails)
    if which=='collector':
        fake_snapshot(monkeypatch);assert collect.run(tmp_path/'out')==1;path=tmp_path/'out/report.json'
    else:
        monkeypatch.setattr(sys,'argv',['proof','--report',str(tmp_path/'report.json')])
        monkeypatch.setattr(proof,'sources',lambda:{'fixture':'hash'})
        monkeypatch.setattr(proof,'run',lambda r:pytest.fail('must not launch after failed initial record'))
        assert proof.main()==1;path=tmp_path/'report.json'
    result=json.loads(path.read_text());assert not result['passed'] and 'initial write failed' in result['error']


def test_archive_readback_rejects_changed_member(tmp_path):
    import tarfile,io
    source={str(collect.LAB/'prototype/a.py'):b'expected'};path=tmp_path/'source.tar.gz'
    with tarfile.open(path,'w:gz') as archive:
        info=tarfile.TarInfo('prototype/a.py');info.size=3;archive.addfile(info,io.BytesIO(b'bad'))
    with pytest.raises(ValueError,match='bytes differ'):collect.verify_archive(path,source)


@pytest.mark.parametrize('changed',('start','end','neither'))
def test_collector_binds_child_source_closure(tmp_path,monkeypatch,changed):
    fake_snapshot(monkeypatch);before=collect.hashes(collect.source_bytes())
    def child(command,directory,**options):
        raw={'passed':True,'full_slots_checked':65536,'output_level':25,'square_level':24,'server_completed_before_client_decode':True,
            'source_hashes_start':before,'source_hashes_end':before,'input_payload_and_metadata_unchanged':True,
            'actual_security_and_P_coverage_checked_before_keygen':True}
        if changed!='neither':raw['source_hashes_'+changed]={'changed':'hash'}
        (directory.parent/'numerical.json').write_text(json.dumps(raw))
        return {'returncode':0,'process_reaped':True,'public_key_store_removed':True,'owned_directory_handle_closed':True}
    monkeypatch.setattr(collect.process,'run_case',child)
    assert collect.run(tmp_path/'out')==(0 if changed=='neither' else 1)


def test_software_metadata_failure_preserves_attempt(tmp_path,monkeypatch):
    fake_snapshot(monkeypatch)
    def fail():raise OSError('package metadata unreadable')
    monkeypatch.setattr(collect.process,'software_versions',fail)
    assert collect.run(tmp_path/'out')==1
    result=json.loads((tmp_path/'out/report.json').read_text())
    assert not result['passed'] and 'package metadata unreadable' in result['error']
