"""Real audit/wrapper integration in fresh owned children, not capacity tests."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import pytest

HERE=Path(__file__).parent


@pytest.mark.parametrize('case',[
    'read_status','read_cgroup','read_mountinfo',
    'builtin_status','builtin_cgroup','builtin_mountinfo',
    'flag_plain','flag_cloexec','flag_nofollow','flag_nonblock','flag_all',
    'write_wronly','write_rdwr','write_creat','write_trunc','write_append',
    'flag_directory','flag_path','flag_bool','builtin_write',
    'bytes','pathlike','dot','double_slash','explicit_pid','unselected',
    'relative_dirfd','absolute_dirfd','symlink','protected',
    'body_error','swallowed_then_read'])
def test_real_proc_self_audit_boundary(case,tmp_path):
    own=tmp_path/'owned';own.mkdir(mode=0o700)
    command=[sys.executable,'-I','-B',str(HERE/'proc_self_probe.py'),case,
        str(HERE/'runtime'),str(HERE/'runtime_sources.json'),str(own)]
    child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        start_new_session=True,env={'PATH':'/usr/bin:/bin','HOME':str(tmp_path),
        'PYTHONDONTWRITEBYTECODE':'1','PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1'})
    try:
        # Ownership begins immediately after Popen, before any observation.
        stdout,stderr=child.communicate()
    except BaseException:
        try:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        finally:child.wait()
        raise
    assert child.returncode==0,(case,child.returncode,stderr.decode(errors='replace'))
    assert stderr==b''
    observed=json.loads(stdout)
    assert observed['schema']=='qwen-proc-self-integration-observation-v1'
    assert observed['case']==case and observed['passed']is True
    assert observed['pid']==child.pid and observed['pgid']==child.pid
    assert observed['scope_restored']is True and observed['runtime_sources']==85
    assert observed['source_manifest_sha256']=='01e5f8681ff5731d8d19d8f488b692bcf38025b83a0dbb3036b536a3d9f1726e'
    assert observed['backend_attempts']==[] and observed['backend_modules']==[]
    assert observed['resource_capacity_evaluated']is False
    assert observed['actual_generation_authorized']is False
    assert observed['denial_count']>0 if observed['sticky_denial_expected']else observed['denial_count']==0
    with pytest.raises(ProcessLookupError):os.kill(child.pid,0)
    with pytest.raises(ProcessLookupError):os.killpg(child.pid,0)
    # Preserve bounded child observations only. No proc contents are published.
    (tmp_path/'child-observation.json').write_bytes(stdout)
