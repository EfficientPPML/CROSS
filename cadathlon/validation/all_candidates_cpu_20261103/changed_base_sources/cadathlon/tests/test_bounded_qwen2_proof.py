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
