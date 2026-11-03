"""Local encrypted transformer checks and repeatable timing.

Interface preflight never certifies encryption. Every real request still goes
through ciphertext, live-ring, finite-output and numerical correctness gates.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from support import load_module, metrics, require_ciphertexts, require_secure_execution, synchronize


BERT_STAGES = (
    "client_key_generation", "server_preprocess_model", "client_encode_encrypt_input",
    "server_encrypted_compute", "client_decrypt_decode",
)


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def positive_int(value):
    number = nonnegative_int(value)
    if number == 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def add_arguments(parser, default_implementation):
    parser.add_argument("--case", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--reference", action="store_true", help="check plaintext fixtures only")
    mode.add_argument("--preflight", action="store_true", help="check interfaces without initializing JAX")
    parser.add_argument("--implementation", type=Path, default=default_implementation,
                        help="encrypted implementation module (defaults to the contestant file)")
    parser.add_argument("--warmups", type=nonnegative_int, default=0)
    parser.add_argument("--repeats", type=positive_int, default=1)
    parser.add_argument("--require-tpu", action="store_true",
                        help="reject a CPU/GPU fallback before creating the encrypted session")


def implementation_preflight(path, *, bert=False):
    module = load_module(path.resolve())
    names = ("create_session",) + (BERT_STAGES if bert else ())
    missing = [name for name in names if not callable(getattr(module, name, None))]
    status = getattr(module, "IMPLEMENTATION_STATUS", "unspecified")
    ready = not missing and status not in ("unimplemented", "plaintext-oracle")
    return module, {
        "implementation": str(path.resolve()),
        "implementation_status": status,
        "missing_interfaces": missing,
        "ready_for_execution": ready,
        "encryption_validated": False,
        "note": "Interface inspection only; encrypted correctness and TPU timings require actual execution.",
    }


def load_encrypted_implementation(path):
    module, info = implementation_preflight(path)
    status = info["implementation_status"]
    if status == "plaintext-oracle":
        raise ValueError("This .golden is a plaintext numerical oracle, not an encrypted baseline. "
                         "Use --reference for fixture checks; an encrypted create_session is still required.")
    if info["missing_interfaces"]:
        raise TypeError("Encrypted implementation is missing callable interface(s): "
                        + ", ".join(info["missing_interfaces"]))
    return module


def benchmark_session(implementation, *, config, weights, embeddings, expected,
                      devices, evaluate_kwargs, check_output, warmups=0, repeats=1,
                      require_tpu=False):
    """Benchmark fresh encrypted requests, validate every warmup and sample.

    Existing scalar server_s/end_to_end_s fields become sample medians. Raw
    samples and cold/warm distinctions are explicit. Caller measures total_s
    including fixture construction and cleanup. Ciphertext/ring checks stay
    outside the server interval; synchronization stays inside it.
    """
    if isinstance(warmups, bool) or not isinstance(warmups, int) or warmups < 0:
        raise ValueError("warmups must be a nonnegative integer")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    platforms = sorted({device.platform for device in devices})
    if not devices:
        raise ValueError("No JAX devices available")
    if require_tpu and platforms != ["tpu"]:
        raise RuntimeError(f"TPU required, but selected backend is {platforms}")
    started = time.perf_counter()
    session = implementation.create_session(
        config, weights, batch=len(embeddings), sequence_length=embeddings.shape[1], devices=devices)
    setup = time.perf_counter() - started
    samples, warmup_samples = [], []
    errors = []
    primary_error = None
    try:
        missing = [name for name in ("encrypt", "evaluate", "decrypt", "close")
                   if not callable(getattr(session, name, None))]
        if missing:
            raise TypeError("Session missing callable method(s): " + ", ".join(missing))
        for index in range(warmups + repeats):
            started = time.perf_counter()
            encrypted = session.encrypt(embeddings)
            require_ciphertexts(encrypted)
            require_secure_execution(session, encrypted)
            synchronize(encrypted)
            server_started = time.perf_counter()
            result = session.evaluate(encrypted, **evaluate_kwargs)
            synchronize(result)
            server = time.perf_counter() - server_started
            require_ciphertexts(result)
            require_secure_execution(session, result)
            # Do not retain a full input ciphertext set through decryption or
            # the next request. Large transformer inputs can occupy GiB, and
            # assignment evaluates its right-hand side before replacing the
            # previous binding on the next loop iteration.
            del encrypted
            output = np.asarray(session.decrypt(result), dtype=np.float64)
            elapsed = time.perf_counter() - started
            error = metrics(output, expected)
            check_output(output, expected, error)
            del result, output
            errors.append(error)
            sample = {"server_s": server, "end_to_end_s": elapsed}
            (warmup_samples if index < warmups else samples).append(sample)
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        close = getattr(session, "close", None)
        if callable(close):
            try:
                close()
            except BaseException as exc:
                if primary_error is None:
                    raise
                # Preserve the actual numerical/security failure if cleanup also fails.
                primary_error.add_note(f"Session cleanup also failed: {exc}")
    return {
        "passed": True, "setup_s": setup,
        "implementation": getattr(implementation, "__file__", None),
        "implementation_status": getattr(implementation, "IMPLEMENTATION_STATUS", "unspecified"),
        "server_s": float(np.median([sample["server_s"] for sample in samples])),
        "end_to_end_s": float(np.median([sample["end_to_end_s"] for sample in samples])),
        "samples": samples, "warmup_samples": warmup_samples,
        "cold_start_included": warmups == 0,
        # These are the devices passed to create_session, not proof that a
        # custom implementation used every device in the requested tuple.
        "requested_platforms": platforms, "requested_device_count": len(devices),
        "requested_device_kinds": [str(getattr(device, 'device_kind', 'unknown')) for device in devices],
        "max_abs_error": max(error["max_abs_error"] for error in errors),
        "nrmse": max(error["nrmse"] for error in errors),
        "top1_agreement": min(error["top1_agreement"] for error in errors),
    }
