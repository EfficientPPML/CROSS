"""Resident server holding the contestant's service on the accelerator.

A compiled JAX executable cannot cross a process boundary and a TPU admits one
process at a time, but the harness runs every stage as its own process. So
stage 3 starts this daemon, it builds the service once, and stage 7 hands it
ciphertext bundles over a Unix socket. Rebuilding per request would charge
compilation to inference, which is the number the benchmark is measuring.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bundle_io
import stage_lib as lib

ACCEPT_TIMEOUT = 5.0
IDLE_TIMEOUT = float(os.environ.get("P3_IDLE_TIMEOUT", "3600"))
MAX_MESSAGE = 4 << 20


def read_message(conn):
    chunks, size = [], 0
    while True:
        chunk = conn.recv(65536)
        if not chunk:
            if not chunks:
                return None
            raise ConnectionError("peer closed mid-request")
        chunks.append(chunk)
        size += len(chunk)
        if size > MAX_MESSAGE:
            raise ValueError("request too large")
        if chunk.endswith(b"\n"):
            break
    return json.loads(b"".join(chunks).decode())


class Server:
    def __init__(self, size, devices):
        self.paths = lib.Paths(size)
        self.devices = devices
        self.service = None
        self.batch = None
        self.ready = False
        self._stop = False
        self.setup = {}

    def materialize(self, batch):
        import jax
        started = time.perf_counter()
        available = jax.devices()
        if len(available) < self.devices:
            raise RuntimeError(f"need {self.devices} chips, found {len(available)}")
        chips = tuple(available[: self.devices])
        keys = lib.load_keys(self.paths)
        keys_at = time.perf_counter()
        self.service = lib.build_service(batch, chips, keys)
        built = time.perf_counter()
        self.batch = batch

        # Force every lazy cost -- Mapping construction, constant encoding and
        # the XLA compile -- to happen here, inside the stage the harness
        # attributes to model preprocessing. Without this the first measured
        # batch pays for compilation and stage 7 stops meaning "inference".
        # The warm-up runs on zeros through the contestant's own service, so
        # an optimized service is warmed exactly the same way. This daemon
        # already holds the key pair (the local package is a test harness, not
        # a production isolation boundary; the judge audits the evaluator
        # path separately), so calling encrypt here reveals nothing.
        import numpy as np
        warm_started = time.perf_counter()
        warm_in = self.service.encrypt(np.zeros((batch, 784), dtype=np.float64))
        jax.block_until_ready(warm_in)
        warm_out = self.service.execute(warm_in)
        jax.block_until_ready(warm_out)
        del warm_in, warm_out
        warmed = time.perf_counter()
        packed = lib.pack_program()
        ring = packed.ring_config
        import math
        Q = math.prod(ring.q_towers)
        P = math.prod(ring.p_towers)
        self.setup = {"key_load_s": keys_at - started,
                      "service_build_s": built - keys_at,
                      "warmup_s": warmed - warm_started,
                      "total_s": warmed - started}
        self.paths.pubkeydir.mkdir(parents=True, exist_ok=True)
        self.paths.manifest.write_text(json.dumps({
            "degree": int(ring.degree), "num_slots": int(ring.num_slots),
            "num_q": len(ring.q_towers), "num_p": len(ring.p_towers),
            "dnum": int(ring.dnum), "depth": int(packed.depth),
            "log2_Q": math.log2(Q), "log2_QP": math.log2(Q * P),
            "security_bits": int(ring.target.bits),
            "security_model": str(ring.target.cost_model.value),
            "devices": [str(d) for d in chips], "batch": batch,
            "setup": self.setup}, indent=2))
        self.ready = True

    def infer(self, indir, outdir):
        import jax
        load_started = time.perf_counter()
        bundle = bundle_io.load_bundle(indir)
        jax.block_until_ready(bundle)
        load_s = time.perf_counter() - load_started

        execute_started = time.perf_counter()
        result = self.service.execute(bundle)
        jax.block_until_ready(result)
        execute_s = time.perf_counter() - execute_started

        store_started = time.perf_counter()
        count = bundle_io.save_bundle(result, outdir)
        store_s = time.perf_counter() - store_started
        return {"execute_s": execute_s, "load_s": load_s, "store_s": store_s,
                "result_ciphertexts": count, "batch": self.batch,
                "devices": self.devices,
                "per_inference_s": execute_s / self.batch,
                "images_per_s": self.batch / execute_s if execute_s > 0 else None}

    def handle(self, request):
        cmd = request.get("cmd")
        if cmd == "ping":
            return {"ok": True, "ready": self.ready}
        if cmd == "infer":
            if not self.ready:
                return {"ok": False, "error": "service not ready"}
            return {"ok": True, "result": self.infer(request["indir"], request["outdir"])}
        if cmd == "stop":
            return {"ok": True, "stopping": True}
        return {"ok": False, "error": f"unknown command {cmd!r}"}

    def shutdown(self):
        if self.service is not None:
            try:
                self.service.release()
            except Exception:
                traceback.print_exc()
            self.service = None
        self.ready = False
        for path in (self.paths.socket_path, self.paths.pidfile, self.paths.ready):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def serve(self):
        path = self.paths.socket_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(path))
            listener.listen(8)
            listener.settimeout(ACCEPT_TIMEOUT)
            print("[p3-service] listening", flush=True)
            last = time.time()
            while not self._stop:
                try:
                    conn, _ = listener.accept()
                except socket.timeout:
                    if time.time() - last > IDLE_TIMEOUT:
                        print("[p3-service] idle timeout", flush=True)
                        break
                    continue
                except OSError:
                    if self._stop:
                        break
                    continue
                last = time.time()
                with conn:
                    try:
                        request = read_message(conn)
                    except Exception as error:
                        print(f"[p3-service] bad request: {error}", flush=True)
                        continue
                    if request is None:
                        continue
                    try:
                        response = self.handle(request)
                    except Exception as error:
                        traceback.print_exc()
                        response = {"ok": False,
                                    "error": f"{type(error).__name__}: {error}"}
                    if response.pop("stopping", False):
                        self._stop = True
                    try:
                        conn.sendall((json.dumps(response) + "\n").encode())
                    except OSError:
                        pass
                last = time.time()
        finally:
            listener.close()
            self.shutdown()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("size", type=int)
    ap.add_argument("--devices", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True)
    args = ap.parse_args()

    server = Server(args.size, args.devices)
    paths = server.paths
    paths.ensure_rundir()

    def on_signal(signum, _frame):
        print(f"[p3-service] signal {signum}", flush=True)
        server._stop = True

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, on_signal)
        except (ValueError, OSError):
            pass

    paths.pidfile.write_text(f"{os.getpid()}\n")
    try:
        server.materialize(args.batch)
        print(f"[p3-service] ready {json.dumps(server.setup)}", flush=True)
        paths.ready.write_text("ready\n")
        server.serve()
    except BaseException:
        detail = traceback.format_exc()
        traceback.print_exc()
        try:
            paths.failed.write_text(detail)
        except OSError:
            pass
        server.shutdown()
        raise
    finally:
        paths.pidfile.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
