"""Real dummy-process lifecycle tests only; no encryption or TPU claims."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time

import pytest

LAB=Path(__file__).resolve().parents[1]/"validation"

@pytest.fixture
def tree(tmp_path):
    root=tmp_path/'repo';validation=root/'cadathlon/validation';validation.mkdir(parents=True)
    for name in ('check_transformer_suite.py','run_bounded_qwen2_proof.py'):
        shutil.copy2(LAB/name,validation/name)
    for name in ('jaxite_word','demos','cadathlon/common'):
        path=root/name;path.mkdir(parents=True);(path/'fixture.py').write_text('# dummy fixture source\n')
    problem=root/'cadathlon/agent_pset/p4_qwen';src=problem/'src';src.mkdir(parents=True)
    (src/'qwen_bounded_reference.py').write_text("IMPLEMENTATION_STATUS='experimental-unvalidated'\n")
    tests=problem/'test';tests.mkdir()
    for i in range(1,7):
        for suffix in ('in','out'):(tests/f'test{i}.{suffix}').write_text('{}\n')
    keys=tmp_path/'keys';keys.mkdir()
    return root,validation/'run_bounded_qwen2_proof.py',src/'runner.py',keys,tmp_path/'proof'


def runner_source(root,mode):
    return f'''import json,os,sys,time
from pathlib import Path
ROOT=Path({str(root)!r})
mode={mode!r}
implementation=ROOT/'cadathlon/agent_pset/p4_qwen/src/qwen_bounded_reference.py'
keydir=Path(os.environ['CROSS_PUBLIC_KEY_DIRECTORY'])
(keydir/'public-dummy-key').write_text('dummy public bytes')
(ROOT/'childpid').write_text(str(os.getpid()))
if mode=='signal':
    try:
        (ROOT/'ready').write_text('ready')
        time.sleep(60)
    finally:
        time.sleep(.25)
        (ROOT/'graceful-cleanup').write_text('done')
else:
    time.sleep(.04)
    if mode=='mutation':(ROOT/'cadathlon/common/fixture.py').write_text('# changed')
    for _ in range(3):print('bounded homomorphic refresh starting',file=sys.stderr)
    for _ in range(2 if mode=='refresh' else 3):print('bounded homomorphic refresh complete',file=sys.stderr)
    result=dict(case='test2.in',passed=mode!='failed',mode='HE-toy-smoke',full_qwen_evaluated=False,
        implementation=str(implementation),implementation_status='experimental-unvalidated',
        requested_platforms=['tpu'],requested_device_count=1,requested_device_kinds=['TPU dummy-unit-fixture'],
        cold_start_included=True,samples=[dict(server_s=.005,end_to_end_s=.01)],warmup_samples=[],
        server_s=.005,end_to_end_s=.01,setup_s=0.,total_s=.02,max_abs_error=.01,nrmse=.01,top1_agreement=1.)
    if mode=='device':result.pop('requested_device_kinds')
    if mode=='numeric':result['max_abs_error']=.21
    print(json.dumps(result))
'''


def command(tree):
    root,wrapper,runner,keys,output=tree
    return [sys.executable,str(wrapper),'--output',str(output),'--key-directory',str(keys),'--cleanup-grace-s','.8']


@pytest.mark.parametrize('mode',['success','failed','numeric','device','mutation','refresh'])
def test_proof_is_bound_to_source_actual_gates_and_cleanup(tree,mode):
    root,wrapper,runner,keys,output=tree
    runner.write_text(runner_source(root,mode))
    process=subprocess.run(command(tree),capture_output=True,text=True,timeout=15)
    report=json.loads((output/'report.json').read_text())
    assert (process.returncode==0)==(mode=='success'),process.stderr
    assert report['passed'] is (mode=='success')
    assert report['complete'] is False and report['all_six_passed'] is False
    assert report['full_model_evaluated'] is False and report['scoring_calibrated'] is False
    assert report['process_reaped'] and report['public_key_store_removed']
    assert not list(keys.iterdir())
    assert report['source_files_unchanged'] is (mode!='mutation')
    assert report['case_timeout_s'] is None
    assert report['case']['cleanup_grace_s']==.8
    with tarfile.open(output/'sources.tar.gz') as archive:
        for path,member in report['archive_members'].items():
            assert hashlib.sha256(archive.extractfile(member).read()).hexdigest()==report['source_hashes_start'][path]
    assert report['source_archive_sha256']==hashlib.sha256((output/'sources.tar.gz').read_bytes()).hexdigest()
    if mode=='success':
        assert report['refresh_progress']=={'started':3,'completed':3}
    else:
        assert report.get('error')


def test_stale_output_is_rejected_without_overwrite(tree):
    root,wrapper,runner,keys,output=tree
    output.mkdir();(output/'sentinel').write_text('keep')
    runner.write_text(runner_source(root,'success'))
    process=subprocess.run(command(tree),capture_output=True,text=True,timeout=10)
    assert process.returncode!=0
    assert (output/'sentinel').read_text()=='keep'
    assert not (root/'childpid').exists()


@pytest.mark.parametrize('sig',[signal.SIGTERM,signal.SIGHUP])
def test_external_cancellation_allows_graceful_child_cleanup_and_preserves_failure(tree,sig):
    root,wrapper,runner,keys,output=tree
    runner.write_text(runner_source(root,'signal'))
    process=subprocess.Popen(command(tree),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
    try:
        deadline=time.monotonic()+10
        while not (root/'ready').exists():
            assert process.poll() is None
            if time.monotonic()>deadline:raise AssertionError('dummy child did not start')
            time.sleep(.02)
        process.send_signal(sig)
        stdout,stderr=process.communicate(timeout=10)
        assert process.returncode==128+sig,(stdout,stderr)
        report=json.loads((output/'report.json').read_text())
        assert not report['passed'] and report['process_reaped']
        assert report['public_key_store_removed'] and not list(keys.iterdir())
        assert (root/'graceful-cleanup').read_text()=='done'
        assert report['case']['returncode']==128+signal.SIGTERM
        assert report['source_files_unchanged']
    finally:
        if process.poll() is None:
            if (root/'childpid').exists():
                try:os.killpg(int((root/'childpid').read_text()),signal.SIGKILL)
                except ProcessLookupError:pass
            os.killpg(process.pid,signal.SIGKILL);process.wait()


def test_shared_child_shim_rejects_untracked_runner(tree):
    root,wrapper,runner,keys,output=tree
    unknown=root/'untracked.py'
    marker=root/'executed'
    unknown.write_text('from pathlib import Path\nPath('+repr(str(marker))+').write_text("bad")\n')
    result=subprocess.run([sys.executable,str(wrapper.with_name('check_transformer_suite.py')),
                           '--child-runner',str(unknown)],capture_output=True,text=True,timeout=10)
    assert result.returncode!=0 and not marker.exists()


def test_nested_wrapper_retains_replaced_child_directory(tree):
    root,wrapper,runner,keys,output=tree
    foreign=root/'unrelated-directory';foreign.mkdir();(foreign/'sentinel').write_text('must survive')
    code=runner_source(root,'success')
    needle="(keydir/'public-dummy-key').write_text('dummy public bytes')"
    code=code.replace(needle,"keydir.rmdir()\n(ROOT/'unrelated-directory').rename(keydir)\n"+needle)
    runner.write_text(code)
    process=subprocess.run(command(tree),capture_output=True,text=True,timeout=15)
    report=json.loads((output/'report.json').read_text())
    assert process.returncode==1 and not report['passed']
    assert report['process_reaped'] and not report['public_key_store_removed']
    assert report['owned_directory_handle_closed']
    assert not report['case']['public_key_store_removed']
    retained=Path(report['case']['owned_public_key_directory'])
    assert (retained/'sentinel').read_text()=='must survive'
    assert 'child cleanup incomplete' in report['cleanup_error']
    assert Path(report['owned_public_key_directory']).is_dir()


def test_wrapper_launch_failure_without_child_can_remove_outer_parent(tree,monkeypatch):
    import importlib.util
    from types import SimpleNamespace
    root,wrapper,runner,keys,output=tree
    runner.write_text(runner_source(root,'success'))
    monkeypatch.syspath_prepend(str(wrapper.parent))
    suite_spec=importlib.util.spec_from_file_location('check_transformer_suite',wrapper.with_name('check_transformer_suite.py'))
    owned_suite=importlib.util.module_from_spec(suite_spec);suite_spec.loader.exec_module(owned_suite)
    monkeypatch.setitem(sys.modules,'check_transformer_suite',owned_suite)
    spec=importlib.util.spec_from_file_location('launch_failure_wrapper',wrapper)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def missing(*args,**kwargs):raise FileNotFoundError('simulated launch failure')
    monkeypatch.setattr(module.suite.subprocess,'Popen',missing)
    assert module.run(SimpleNamespace(output=output,key_directory=keys,cleanup_grace_s=.1))==1
    report=json.loads((output/'report.json').read_text())
    assert not report['passed'] and not report['process_reaped']
    assert report['case']['process_reaped'] and report['case']['public_key_store_removed']
    assert 'pid' not in report['case']
    assert report['public_key_store_removed'] and report['owned_directory_handle_closed']
    assert not list(keys.iterdir())


def load_wrapper_for_faults(tree,monkeypatch):
    import importlib.util
    root,wrapper,runner,keys,output=tree
    runner.write_text(runner_source(root,'success'))
    monkeypatch.syspath_prepend(str(wrapper.parent))
    suite_spec=importlib.util.spec_from_file_location('check_transformer_suite',wrapper.with_name('check_transformer_suite.py'))
    owned_suite=importlib.util.module_from_spec(suite_spec);suite_spec.loader.exec_module(owned_suite)
    monkeypatch.setitem(sys.modules,'check_transformer_suite',owned_suite)
    spec=importlib.util.spec_from_file_location('fault_wrapper',wrapper)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('mode',['missing','corrupt','incomplete'])
def test_unknown_child_cleanup_retains_wrapper_parent(tree,monkeypatch,mode):
    from types import SimpleNamespace
    root,wrapper,runner,keys,output=tree
    module=load_wrapper_for_faults(tree,monkeypatch)
    def incomplete(*args,**kwargs):
        owned=Path(os.environ['CROSS_PUBLIC_KEY_DIRECTORY'])
        (owned/'sentinel').write_text('unknown child state; keep')
        if mode=='corrupt':(output/'test2/process.json').write_text('{')
        if mode=='incomplete':return {'returncode':1,'process_reaped':True,'public_key_store_removed':True}
        raise RuntimeError('collector did not return durable cleanup metadata')
    monkeypatch.setattr(module.suite,'run_case',incomplete)
    assert module.run(SimpleNamespace(output=output,key_directory=keys,cleanup_grace_s=.1))==1
    report=json.loads((output/'report.json').read_text())
    assert not report['passed'] and report['case_collector_entered']
    assert not report['public_key_store_removed'] and report['owned_directory_handle_closed']
    assert (Path(report['owned_public_key_directory'])/'sentinel').read_text()=='unknown child state; keep'
    assert 'incomplete or unknown' in report['cleanup_error']


def test_failure_before_case_collector_can_remove_owned_parent(tree,monkeypatch):
    from types import SimpleNamespace
    root,wrapper,runner,keys,output=tree
    module=load_wrapper_for_faults(tree,monkeypatch)
    def fail(*args,**kwargs):raise RuntimeError('command construction failed')
    monkeypatch.setattr(module.suite,'runner_command',fail)
    assert module.run(SimpleNamespace(output=output,key_directory=keys,cleanup_grace_s=.1))==1
    report=json.loads((output/'report.json').read_text())
    assert not report['passed'] and not report['case_collector_entered']
    assert report['public_key_store_removed'] and report['owned_directory_handle_closed']
    assert not list(keys.iterdir())


@pytest.mark.parametrize('error_type',[KeyboardInterrupt,SystemExit])
def test_final_wrapper_evidence_failure_preserves_primary_cancellation(tree,monkeypatch,error_type):
    from types import SimpleNamespace
    root,wrapper,runner,keys,output=tree
    module=load_wrapper_for_faults(tree,monkeypatch)
    original_write=module.suite.write_json;primary=error_type('primary cancellation')
    def cancelled(*args,**kwargs):
        original_write(output/'test2/process.json',{'passed':False,'process_reaped':True,
            'public_key_store_removed':True,'owned_directory_handle_closed':True})
        raise primary
    report_writes=0
    def write(path,record):
        nonlocal report_writes
        if path==output/'report.json':
            report_writes+=1
            if report_writes==2:raise OSError('final evidence unavailable')
        original_write(path,record)
    monkeypatch.setattr(module.suite,'run_case',cancelled);monkeypatch.setattr(module.suite,'write_json',write)
    with pytest.raises(error_type) as caught:
        module.run(SimpleNamespace(output=output,key_directory=keys,cleanup_grace_s=.1))
    assert caught.value is primary and report_writes==2
    assert any('final evidence unavailable' in note for note in primary.__notes__)
    assert not list(keys.iterdir())
    assert json.loads((output/'report.json').read_text())['passed'] is False
