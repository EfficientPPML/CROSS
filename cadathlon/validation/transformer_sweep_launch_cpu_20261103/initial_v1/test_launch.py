"""Metadata/process tests with fake ciphertext/device classes; no JAX or TPU."""
import importlib.util,json,os,signal,sys,tarfile
from pathlib import Path
from types import SimpleNamespace
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('sweep_launch',HERE/'launch.py')
launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)
COLLECTOR=Path('/tmp/transformer-collector-outer-failure-20261103/checkout')/launch.COLLECTOR

@pytest.fixture
def prepared(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir()
    for directory in launch.SOURCE_DIRS:(source/directory).mkdir(parents=True)
    target=source/launch.COLLECTOR;target.parent.mkdir(parents=True);target.write_bytes(COLLECTOR.read_bytes())
    (source/'jaxite_word/__init__.py').write_text('''class Device:
    platform='tpu';id=0;device_kind='FAKE metadata test device';process_index=0;local_hardware_id=0;coords=None;slice_index=None
    def __init__(self): self.client=self; self.platform_version='not a real accelerator'
    def memory_stats(self): return {'bytes_in_use':8,'peak_bytes_in_use':16}
class Payload:
    nbytes=8
    def devices(self): return {Device()}
class Polynomial:
    shape=(1,2,4,2,2);degree=8;moduli=[17,97];_ckks_scale=100.;_ckks_nsd=1
    def __init__(self): self.polynomial=Payload()
''')
    (source/'cadathlon/common/transformer_runner.py').write_text('''def require_secure_execution(owner,bundle):
    return [{'degree':8,'q_towers':[17,97],'p_towers':[113],'security_bits':128,'scope':'FAKE protocol-test manifest, not secure parameters'}]
''')
    for model,name in launch.MODELS.items():
        problem=source/'cadathlon/agent_pset'/name;(problem/'src').mkdir();(problem/'test').mkdir()
        (problem/f'src/{model}_optimized_reference.py').write_text('# fake protocol fixture\n')
        for number in range(1,7):
            for ext in ('in','out'):(problem/f'test/test{number}.{ext}').write_text('{}\n')
        (problem/'src/runner.py').write_text('''import argparse,json,time
from pathlib import Path
from jaxite_word import Polynomial
import transformer_runner
p=argparse.ArgumentParser();p.add_argument('--case');p.add_argument('--implementation');p.add_argument('--require-tpu',action='store_true');p.add_argument('--warmups',type=int);p.add_argument('--repeats',type=int);a=p.parse_args()
for i in range(a.warmups+a.repeats):
    transformer_runner.require_secure_execution(None,Polynomial())
    transformer_runner.require_secure_execution(None,Polynomial())
time.sleep(.04)
model='''+repr(model)+'''
row=dict(passed=True,case=Path(a.case).name,mode='HE-synthetic-smoke' if model=='bert' else 'HE-toy-smoke',
    implementation=a.implementation,requested_platforms=['tpu'],requested_device_count=1,requested_device_kinds=['FAKE'],
    cold_start_included=a.warmups==0,samples=[dict(server_s=.01,end_to_end_s=.02)]*a.repeats,
    warmup_samples=[dict(server_s=.01,end_to_end_s=.02)]*a.warmups,setup_s=.01,server_s=.01,end_to_end_s=.02,
    total_s=1.,max_abs_error=.001,nrmse=.001,top1_agreement=1.,full_model_evaluated=False)
row['full_'+model+'_evaluated']=False
if model=='bert':row['mrpc_accuracy_evaluated']=False
print(json.dumps(row))
''')
    snapshot=tmp_path/'snapshot'
    assert launch.prepare(SimpleNamespace(source_root=source,snapshot=snapshot))==0
    monkeypatch.setattr(launch,'__file__',str(snapshot/launch.PINNED_WRAPPER))
    monkeypatch.setattr(launch,'KEY_STORE_PREFLIGHT_BYTES',0)
    return source,snapshot

def args(tmp_path,snapshot):
    return SimpleNamespace(root=snapshot,output=tmp_path/'evidence',key_parent=tmp_path/'owned-keys',
        order='interleaved',warmups=0,repeats=1,case_timeout_s=None,cleanup_grace_s=.1,sample_interval_s=.01)

def test_snapshot_exact_bytes_readonly_and_detects_drift(prepared):
    source,snapshot=prepared;manifest,raw=launch.verified_sources(snapshot)
    assert all(not (p.stat().st_mode & 0o222) for p in snapshot.rglob('*'))
    assert manifest['instrumentation_sha256']==launch.sha(raw[launch.PINNED_WRAPPER])
    changed=snapshot/'demos/extra.py';changed.parent.chmod(0o755);changed.write_text('# new unreviewed dependency\n')
    with pytest.raises(ValueError,match='inventory/hash'):launch.verified_sources(snapshot)

def test_all_twelve_fake_children_archive_gate_devices_resources_and_cleanup(prepared,tmp_path):
    _,snapshot=prepared;arguments=args(tmp_path,snapshot)
    assert launch.run(arguments)==0
    report=json.loads((arguments.output/'report.json').read_text())
    assert report['passed'] and report['source_files_unchanged']
    assert not report['scoring_calibrated'] and not report['full_model_evaluated']
    assert report['owned_key_parent_removed'] and report['owned_key_parent_handle_closed']
    assert not arguments.key_parent.exists()
    for model in launch.MODELS:
        entry=report['models'][model]
        assert entry['complete'] and entry['all_six_passed'] and len(entry['cases'])==6
        assert not entry['steady_measurement_schedule']
        for name,record in entry['cases'].items():
            assert record['passed'] and record['resources']['samples']>0
            directory=arguments.output/model/Path(name).stem
            security=json.loads((directory/'security_device.json').read_text())
            assert security['passed'] and [e['boundary'] for e in security['events']]==['input','output']
            assert security['events'][0]['actual_rings'][0]['scope'].startswith('FAKE')
            assert security['events'][0]['ciphertexts'][0]['devices'][0]['device_kind'].startswith('FAKE')
            assert not any('secret' in key for key in security)
    with tarfile.open(arguments.output/'sources.tar.gz') as archive:
        blobs={m.name:archive.extractfile(m).read() for m in archive.getmembers()}
    assert {name:launch.sha(data) for name,data in blobs.items()}==report['source_hashes_start']
    with pytest.raises(FileExistsError):launch.run(arguments)

@pytest.mark.parametrize('mode',['missing','corrupt','incomplete'])
def test_unknown_child_cleanup_retains_owned_parent(prepared,tmp_path,monkeypatch,mode):
    _,snapshot=prepared;arguments=args(tmp_path,snapshot)
    def interrupted(suite,command,directory,**kwargs):
        kwargs['state']['collector_entered']=True
        (arguments.key_parent/'keep').write_text('unknown child ownership')
        if mode=='corrupt':(directory/'process.json').write_text('not-json')
        if mode=='incomplete':launch.write_json(directory/'process.json',{'process_reaped':False})
        raise KeyboardInterrupt('primary interruption')
    monkeypatch.setattr(launch,'collect_case',interrupted)
    with pytest.raises(KeyboardInterrupt,match='primary interruption'):launch.run(arguments)
    report=json.loads((arguments.output/'report.json').read_text())
    assert not report['passed'] and not report['owned_key_parent_removed']
    assert not report['inner_cleanup_complete']
    assert (arguments.key_parent/'keep').exists()

def test_initial_source_failure_records_failure_without_key_parent(prepared,tmp_path):
    _,snapshot=prepared;arguments=args(tmp_path,snapshot)
    path=snapshot/'demos/new.py';path.parent.chmod(0o755);path.write_text('# drift\n')
    with pytest.raises(ValueError,match='inventory/hash'):launch.run(arguments)
    report=json.loads((arguments.output/'report.json').read_text())
    assert not report['passed'] and report['errors'] and not arguments.key_parent.exists()

def test_process_tree_selection_keeps_group_and_nested_children_only():
    table={1:{'ppid':0,'pgid':1},10:{'ppid':1,'pgid':10},11:{'ppid':10,'pgid':11},
           12:{'ppid':11,'pgid':11},13:{'ppid':1,'pgid':10},20:{'ppid':1,'pgid':20}}
    assert set(launch.owned_processes(table,10))=={10,11,12,13}

def test_source_failure_and_final_publication_preserve_cancellation(prepared,tmp_path,monkeypatch):
    _,snapshot=prepared;arguments=args(tmp_path,snapshot);primary=KeyboardInterrupt('stop')
    def interrupted(suite,command,directory,**kwargs):
        kwargs['state']['collector_entered']=True
        launch.write_json(directory/'process.json',dict(process_reaped=True,public_key_store_removed=True,owned_directory_handle_closed=True))
        raise primary
    real_write=launch.write_json
    def write(path,value):
        if value.get('phase')=='finished':raise OSError('final publication failed')
        return real_write(path,value)
    monkeypatch.setattr(launch,'collect_case',interrupted);monkeypatch.setattr(launch,'write_json',write)
    with pytest.raises(KeyboardInterrupt) as observed:launch.run(arguments)
    assert observed.value is primary and any('Final sweep report publication failed' in note for note in primary.__notes__)
    assert not arguments.key_parent.exists()

def test_existing_key_parent_is_never_removed(prepared,tmp_path):
    _,snapshot=prepared;arguments=args(tmp_path,snapshot)
    arguments.key_parent.mkdir();(arguments.key_parent/'unrelated').write_text('keep')
    with pytest.raises(FileExistsError):launch.run(arguments)
    assert (arguments.key_parent/'unrelated').read_text()=='keep'
