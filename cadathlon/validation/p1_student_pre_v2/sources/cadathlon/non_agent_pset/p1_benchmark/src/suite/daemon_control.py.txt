"""Stop only this suite's daemon; stale PID files are not process identities."""
from pathlib import Path
import os
import signal
import time

DAEMON = Path(__file__).resolve().with_name('service_daemon.py')


def is_our_daemon(pid):
    if pid <= 1:
        return False
    try:
        args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        # The launcher passes an absolute script path. Check an argument,
        # never a substring of a shell command or another process's name.
        return os.fsencode(DAEMON) in args
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False


def stop_daemon(pid, timeout=30.0, kill_timeout=5.0):
    """Bound cleanup time and recheck identity before each signal.

    Never remove libtpu's process-wide lock: a different job may own the TPU.
    The driver releases ownership when the actual owning process exits.
    """
    if not is_our_daemon(pid):
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + timeout
    while is_our_daemon(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    if is_our_daemon(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return True
        # SIGKILL is asynchronous. Do not launch another TPU process until
        # the old owner has exited and closed its device descriptors.
        deadline = time.monotonic() + kill_timeout
        while is_our_daemon(pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        if is_our_daemon(pid):
            raise TimeoutError(f"resident service pid {pid} did not exit after SIGKILL")
    return True
