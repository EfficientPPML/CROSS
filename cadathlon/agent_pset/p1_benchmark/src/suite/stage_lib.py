"""Shared plumbing for the suite-backed stages.

The stages are separate processes that talk through files, exactly as the
benchmark harness expects. This module owns everything they share: where the
suite puts things, how the key pair is persisted, how the fixed network is
loaded, and how the contestant's `build_service` is constructed on each side.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np

SUITE_DIR = Path(__file__).resolve().parent
PROBLEM_DIR = SUITE_DIR.parent.parent                      # agent_pset/p1_benchmark/
SRC_DIR = PROBLEM_DIR / "src"
CONFIG = json.loads((SUITE_DIR / "suite_config.json").read_text())
INSTANCE_NAMES = ("single", "small", "medium", "large")

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
INPUT_SHAPE = (784,)


def cross_root():
    sys.path.insert(0, str(PROBLEM_DIR.parents[1] / "common"))
    from support import cross_root as _root
    return _root()


def _looks_like_suite(path: Path) -> bool:
    return (path / "harness" / "run_submission.py").is_file()


def suite_root() -> Path:
    """The ml-inference checkout this run is wired into.

    The harness derives its own paths from `Path.cwd()` (see
    `harness/params.py`), so the stages must agree with that first. Taking
    MLINF_ROOT ahead of the working directory silently sends stage output to a
    different checkout than the one the harness reads, which looks like the
    stages ran and produced nothing.
    """
    candidates = [Path.cwd()]
    env = os.environ.get("MLINF_ROOT")
    if env:
        candidates.append(Path(env))
    candidates.append(Path.home() / "ml-inference")
    for candidate in candidates:
        if _looks_like_suite(candidate):
            return candidate
    raise SystemExit(
        "no ml-inference checkout found; run the harness from the suite root "
        "or set MLINF_ROOT. Searched: "
        + ", ".join(str(c) for c in candidates))


class Paths:
    """The suite's directory contract, mirrored (harness/params.py owns it)."""

    def __init__(self, size: int):
        self.size = size
        self.name = INSTANCE_NAMES[size]
        self.root = suite_root()

    @property
    def iodir(self): return self.root / "io" / self.name

    @property
    def datadir(self): return self.root / "datasets" / self.name

    @property
    def pubkeydir(self): return self.iodir / "public_keys"

    @property
    def seckeydir(self): return self.iodir / "secret_key"

    @property
    def upload(self): return self.iodir / "ciphertexts_upload"

    @property
    def download(self): return self.iodir / "ciphertexts_download"

    @property
    def interm(self): return self.iodir / "intermediate"

    @property
    def test_pixels(self): return self.datadir / "intermediate" / "test_pixels.txt"

    @property
    def predictions(self): return self.iodir / "encrypted_model_predictions.txt"

    @property
    def manifest(self): return self.pubkeydir / "ring_manifest.json"

    @property
    def rundir(self): return SUITE_DIR / "run" / self.name

    def ensure_rundir(self) -> Path:
        self.rundir.mkdir(parents=True, exist_ok=True)
        return self.rundir

    @property
    def socket_path(self): return self.rundir / "service.sock"

    @property
    def pidfile(self): return self.rundir / "service.pid"

    @property
    def daemon_log(self): return self.rundir / "service.log"

    @property
    def ready(self): return self.rundir / "service.ready"

    @property
    def failed(self): return self.rundir / "service.failed"


ACTIVE = SUITE_DIR / "run" / "active.json"


def record_active(size: int, devices: int) -> None:
    """The harness calls server_preprocess_model with no arguments."""
    ACTIVE.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE.write_text(json.dumps({"size": int(size), "devices": int(devices)}))


def read_active() -> dict:
    if not ACTIVE.exists():
        raise SystemExit(f"{ACTIVE} missing; client_key_generation must run first")
    return json.loads(ACTIVE.read_text())


def device_count(default: int = 1) -> int:
    return int(os.environ.get("P3_DEVICES", default))


def implementation():
    """Load baseline.py or the contestant's bench_opt.py."""
    sys.path.insert(0, str(PROBLEM_DIR.parents[1] / "common"))
    from support import load_module
    name = "baseline.py" if os.environ.get("P3_BASELINE") else "bench_opt.py"
    sys.path.insert(0, str(SRC_DIR))          # so a golden's `from baseline import` works
    return load_module(SRC_DIR / name)


def load_model():
    """The fixed p3 network with the organizer weight asset."""
    cross_root()
    import torch
    sys.path.insert(0, str(SRC_DIR))
    from reference import build_model

    model = build_model(50)
    weights = PROBLEM_DIR / "weights" / "mlp_784_50_10.pth"
    if not weights.is_file():
        raise SystemExit(f"{weights} missing; see weights/MANIFEST.json")
    state = torch.load(weights, map_location="cpu")
    expected = set(model.state_dict().keys())
    if set(state.keys()) != expected:
        raise SystemExit(
            f"weight keys {sorted(state.keys())} != model keys {sorted(expected)}")
    model.load_state_dict(state)
    return model.double().eval()


# ---------------------------------------------------------------- key pair
def generate_keys(packed):
    import key_gen as kg
    runtime = packed.ring_config.runtime_kwargs
    kwargs = dict(runtime() if callable(runtime) else runtime)
    return kg.gen_pke_pair([int(v) for v in kwargs["q_towers"]],
                           [int(v) for v in kwargs["p_towers"]],
                           int(kwargs["degree"]),
                           noise_std=float(kwargs["sigma"]))


def _restore(array):
    if array.ndim == 2:
        return [[int(v) for v in row] for row in array]
    if array.ndim == 3:
        return [[[int(v) for v in row] for row in block] for block in array]
    raise ValueError(f"unexpected key rank {array.ndim}")


def save_keys(keys, paths: "Paths"):
    paths.pubkeydir.mkdir(parents=True, exist_ok=True)
    paths.seckeydir.mkdir(parents=True, exist_ok=True)
    np.save(paths.pubkeydir / "pk.npy", np.asarray(keys["public_key"], dtype=np.int64))
    np.save(paths.seckeydir / "sk.npy", np.asarray(keys["secret_key"], dtype=np.int64))


def load_keys(paths: "Paths") -> dict:
    return {"public_key": _restore(np.load(paths.pubkeydir / "pk.npy")),
            "secret_key": _restore(np.load(paths.seckeydir / "sk.npy"))}


def pack_program(model=None):
    """The packed program, identical on both sides of the split."""
    cross_root()
    from jaxite_word import nn, packing
    if model is None:
        model = load_model()
    return packing.pack(nn.vectorize(model, INPUT_SHAPE, warn=False),
                        packing.PackingPolicy(lazy_constants=True))


def build_service(batch: int, devices, keys):
    """Construct the contestant service with the shared key pair."""
    return implementation().build_service(
        load_model(), INPUT_SHAPE, batch=batch, devices=tuple(devices), keys=keys)


def read_pixels(path: Path) -> np.ndarray:
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            values = [float(v) for v in line.split()]
            if len(values) != 784:
                raise ValueError(f"expected 784 pixels per line, got {len(values)}")
            rows.append(values)
    return np.asarray(rows, dtype=np.float64)


def report_steps(paths: "Paths", steps: dict) -> None:
    """The harness folds io/<size>/server_reported_steps.json into results-N.json."""
    path = paths.iodir / "server_reported_steps.json"
    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            existing = {}
    existing.update({k: round(float(v), 6) for k, v in steps.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2))
