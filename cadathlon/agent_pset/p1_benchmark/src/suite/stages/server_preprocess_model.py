"""Suite stage 3: build the contestant service on the accelerator and keep it.

The harness invokes this with no arguments, so the instance size and chip count
come from the handshake client_key_generation wrote.
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.environ["P3_SUITE_DIR"])
import stage_lib as lib

SUITE_DIR = Path(os.environ["P3_SUITE_DIR"])
STARTUP_TIMEOUT = float(os.environ.get("P3_STARTUP_TIMEOUT", "3600"))


def stop_existing(paths):
    """A TPU admits one process; sweep every instance's leftover daemon."""
    run_root = paths.rundir.parent
    if not run_root.exists():
        return
    for pidfile in sorted(run_root.glob("*/service.pid")):
        try:
            pid = int(pidfile.read_text().strip())
        except (ValueError, OSError):
            pidfile.unlink(missing_ok=True)
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pidfile.unlink(missing_ok=True)
            continue
        print(f"         [p3] stopping resident service pid {pid}")
        for _ in range(150):
            time.sleep(0.2)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
        else:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        pidfile.unlink(missing_ok=True)
    time.sleep(3)
    lock = Path("/tmp/libtpu_lockfile")
    try:
        if lock.exists():
            lock.unlink()
    except OSError:
        pass


def main():
    active = lib.read_active()
    size, devices = active["size"], active["devices"]
    paths = lib.Paths(size)
    paths.iodir.mkdir(parents=True, exist_ok=True)
    paths.ensure_rundir()
    stop_existing(paths)
    for marker in (paths.ready, paths.failed):
        marker.unlink(missing_ok=True)

    batch = len(__import__("numpy").load(paths.interm / "normalized.npy")) \
        if (paths.interm / "normalized.npy").exists() else None
    if batch is None:
        import json as _json
        batch = _json.loads((SUITE_DIR / "suite_config.json").read_text())
        batch = next(c["batch"] for c in batch["cases"] if c["instance"] == size)

    log = open(paths.daemon_log, "w")
    started = time.perf_counter()
    daemon = subprocess.Popen(
        [sys.executable, str(SUITE_DIR / "service_daemon.py"), str(size),
         "--devices", str(devices), "--batch", str(batch)],
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        env=dict(os.environ))
    while True:
        if paths.ready.exists():
            break
        if paths.failed.exists():
            print(paths.failed.read_text(), file=sys.stderr)
            sys.exit("[p3] service build failed")
        code = daemon.poll()
        if code is not None:
            try:
                tail = "".join(paths.daemon_log.read_text(errors="replace")
                               .splitlines(keepends=True)[-25:])
                print(tail, file=sys.stderr)
            except OSError:
                pass
            sys.exit(f"[p3] service exited with {code} before ready "
                     f"(see {paths.daemon_log})")
        if time.perf_counter() - started > STARTUP_TIMEOUT:
            daemon.terminate()
            sys.exit(f"[p3] timed out after {STARTUP_TIMEOUT:.0f}s building the service")
        time.sleep(0.5)

    elapsed = time.perf_counter() - started
    manifest = json.loads(paths.manifest.read_text())
    print(f"         [p3] service resident on {devices} chip(s), batch={batch}; "
          f"warm-up {manifest['setup']['warmup_s']:.1f}s")
    print(f"         [p3] ring degree={manifest['degree']} slots={manifest['num_slots']} "
          f"max_level={manifest['max_level']} log2(QP)={manifest['log2_QP']:.1f} "
          f"security={manifest['security_bits']}-bit")
    lib.report_steps(paths, {
        "server_setup_s": elapsed,
        "server_service_build_s": manifest["setup"]["service_build_s"],
        "server_warmup_s": manifest["setup"]["warmup_s"]})


if __name__ == "__main__":
    main()
