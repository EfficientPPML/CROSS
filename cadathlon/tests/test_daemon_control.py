"""A stale PID and a broken client must not disrupt other competition jobs."""
import importlib.util
from pathlib import Path
import signal
import sys
from unittest.mock import Mock

import pytest

SUITE = Path(__file__).resolve().parents[1] / 'non_agent_pset/p1_benchmark/src/suite'
sys.path.insert(0, str(SUITE))
import daemon_control
import service_daemon


def test_unrelated_pid_is_never_signalled(monkeypatch):
    monkeypatch.setattr(daemon_control, 'is_our_daemon', lambda pid: False)
    kill = Mock()
    monkeypatch.setattr(daemon_control.os, 'kill', kill)
    assert not daemon_control.stop_daemon(12345)
    kill.assert_not_called()


def test_stuck_owned_process_has_bounded_cleanup(monkeypatch):
    identity = iter([True, True, True, False, False])
    monkeypatch.setattr(daemon_control, 'is_our_daemon', lambda pid: next(identity))
    kill = Mock()
    monkeypatch.setattr(daemon_control.os, 'kill', kill)
    assert daemon_control.stop_daemon(12345, timeout=0)
    assert kill.call_args_list == [((12345, signal.SIGTERM),), ((12345, signal.SIGKILL),)]


def test_reused_pid_is_not_killed(monkeypatch):
    identity = iter([True, False, False])
    monkeypatch.setattr(daemon_control, 'is_our_daemon', lambda pid: next(identity))
    kill = Mock()
    monkeypatch.setattr(daemon_control.os, 'kill', kill)
    daemon_control.stop_daemon(12345, timeout=0)
    kill.assert_called_once_with(12345, signal.SIGTERM)


def test_daemon_identity_uses_exact_argument(monkeypatch):
    monkeypatch.setattr(Path, 'read_bytes', lambda p: b'python\0' + bytes(daemon_control.DAEMON) + b'\0')
    assert daemon_control.is_our_daemon(12345)
    monkeypatch.setattr(Path, 'read_bytes', lambda p: b'sh\0-c\0python ' + bytes(daemon_control.DAEMON) + b'\0')
    assert not daemon_control.is_our_daemon(12345)


@pytest.mark.parametrize('parts', [
    [b'{"cmd":', b'"ping"}\n'],
    [b'{"cmd":"ping"}\n '],
])
def test_fragmented_request(parts):
    conn = Mock()
    conn.recv.side_effect = parts
    assert service_daemon.read_message(conn) == {'cmd': 'ping'}


@pytest.mark.parametrize('payload', [b'[]\n', b'{"cmd":"ping"}\n{}\n'])
def test_invalid_request(payload):
    conn = Mock()
    conn.recv.return_value = payload
    with pytest.raises(ValueError):
        service_daemon.read_message(conn)


def test_truncated_request():
    conn = Mock()
    conn.recv.side_effect = [b'{', b'']
    with pytest.raises(ConnectionError):
        service_daemon.read_message(conn)


def test_sigkill_does_not_claim_success_until_process_exits(monkeypatch):
    monkeypatch.setattr(daemon_control, 'is_our_daemon', lambda pid: True)
    kill = Mock()
    monkeypatch.setattr(daemon_control.os, 'kill', kill)
    with pytest.raises(TimeoutError, match="did not exit"):
        daemon_control.stop_daemon(12345, timeout=0, kill_timeout=0)
    assert kill.call_args_list == [((12345, signal.SIGTERM),), ((12345, signal.SIGKILL),)]


@pytest.fixture
def launcher(monkeypatch):
    monkeypatch.setenv('P3_SUITE_DIR', str(SUITE))
    spec = importlib.util.spec_from_file_location('p1_startup_test', SUITE / 'stages/server_preprocess_model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dead_daemon_ready_marker_is_rejected(launcher, tmp_path):
    from types import SimpleNamespace
    ready = tmp_path / 'ready'
    ready.write_text('ready')
    paths = SimpleNamespace(ready=ready, failed=tmp_path / 'failed', daemon_log=tmp_path / 'log')
    daemon = Mock()
    daemon.poll.return_value = 1
    with pytest.raises(SystemExit, match='exited with 1'):
        launcher.wait_for_ready(daemon, paths, started=0)


def test_failed_constructor_is_reaped_before_stage_returns(launcher, tmp_path, monkeypatch):
    from types import SimpleNamespace
    failed = tmp_path / 'failed'
    failed.write_text('constructor failed')
    paths = SimpleNamespace(ready=tmp_path / 'ready', failed=failed, daemon_log=tmp_path / 'log')
    daemon = Mock()
    daemon.poll.return_value = None
    with pytest.raises(SystemExit, match='build failed'):
        launcher.wait_for_ready(daemon, paths, started=0)
    daemon.wait.assert_called_once_with(timeout=5.)


def test_stalled_constructor_teardown_is_killed_then_reaped(launcher, monkeypatch):
    import subprocess
    daemon = Mock(pid=12345)
    daemon.wait.side_effect = [subprocess.TimeoutExpired('daemon', 5), 1]
    stop = Mock(return_value=True)
    monkeypatch.setattr(launcher, 'stop_daemon', stop)
    launcher.reap_failed_daemon(daemon)
    stop.assert_called_once_with(12345, timeout=5.)
    assert daemon.wait.call_count == 2


def test_ready_is_published_after_socket_is_listening(tmp_path, monkeypatch):
    import socket
    from types import SimpleNamespace
    server = object.__new__(service_daemon.Server)
    server.paths = SimpleNamespace(socket_path=tmp_path / 'socket', ready=tmp_path / 'ready')
    server._stop = True
    server.shutdown = Mock()
    original_write = Path.write_text
    publication = []

    def write_ready(path, text, *args, **kwargs):
        if path == server.paths.ready:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(str(server.paths.socket_path))
            publication.append(True)
        return original_write(path, text, *args, **kwargs)

    monkeypatch.setattr(Path, 'write_text', write_ready)
    server.serve()
    assert publication == [True]
    server.shutdown.assert_called_once()
