"""Failure-publication and owned-worker cleanup regressions, without HE."""
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import pytest
import collect_file_stages as collector
import client_keys


def test_snapshot_failure_preserves_primary_writes_report_and_restores_signals(tmp_path,monkeypatch):
    output=tmp_path/'attempt';primary=OSError('original snapshot construction failure')
    original=Path.mkdir;signals={n:signal.getsignal(n) for n in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    def fail(path,*args,**kwargs):
        if path.name=='snapshot':raise primary
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'mkdir',fail)
    monkeypatch.setattr(collector,'sources',lambda:{'unit.py':b'# fixture\n'})
    monkeypatch.setattr(sys,'argv',['collector','--output',str(output)])
    with pytest.raises(OSError) as caught:collector.main()
    assert caught.value is primary
    report=json.loads((output/'report.json').read_text())
    assert report['passed'] is False and report['source_archive_sha256'] is None
    assert report['error']==repr(primary)
    assert {n:signal.getsignal(n) for n in signals}==signals


def test_cleanup_only_cancel_propagates_and_repeated_signals_are_shielded(tmp_path,monkeypatch):
    output=tmp_path/'attempt';primary=KeyboardInterrupt('cleanup cancellation')
    signals={n:signal.getsignal(n) for n in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    monkeypatch.setattr(collector,'sources',lambda:{'unit.py':b'# fixture\n'})
    monkeypatch.setattr(sys,'argv',['collector','--output',str(output)])
    class Child:
        pid=123;returncode=0
        def __init__(self,*args,**kwargs):
            (output/'tests.xml').write_text('<testsuites><testsuite tests="112" failures="0" errors="0" skipped="0"/></testsuites>')
        def wait(self,**kwargs):return 0
        def poll(self):return 0
    monkeypatch.setattr(collector.subprocess,'Popen',Child)
    monkeypatch.setattr(collector,'reap',lambda child:dict(reaped=True,group_gone=True))
    def fail(self):
        for number in signals:os.kill(os.getpid(),number)
        raise primary
    monkeypatch.setattr(client_keys._OwnedDirectory,'remove',fail)
    with pytest.raises(KeyboardInterrupt) as caught:collector.main()
    assert caught.value is primary
    report=json.loads((output/'report.json').read_text())
    assert report['passed'] is False and report['owned_test_handles_closed'] is True
    assert any('cleanup cancellation' in x for x in report['cleanup_errors'])
    assert {n:signal.getsignal(n) for n in signals}==signals


def test_reap_rejects_surviving_descendant_after_normal_leader_exit(tmp_path):
    marker=tmp_path/'helper'
    code="import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);open(sys.argv[1],'w').write(str(p.pid))"
    child=subprocess.Popen([sys.executable,'-c',code,str(marker)],start_new_session=True)
    child.wait(timeout=10)
    try:
        assert collector.group_alive(child.pid)
        with pytest.raises(RuntimeError,match='owned'):collector.reap(child)
    finally:
        if collector.group_alive(child.pid):
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
