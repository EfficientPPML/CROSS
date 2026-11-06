"""Reproducible CPU regression checkpoint after bounded-runtime integration."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
SCOPES = [
    "cadathlon/tests", "jaxite_word/ckks_ctx_test.py",
    "jaxite_word/codec_preparation_test.py", "jaxite_word/uniform_encoding_test.py",
    "jaxite_word/mapping_compile_test.py", "jaxite_word/bsgs_test.py",
    "jaxite_word/packing_test.py", "jaxite_word/composite_prime_gen_test.py",
    "jaxite_word/bootstrapping_test.py",
]


def source_bytes():
    paths = {Path(__file__).resolve(), ROOT / "jaxite_word/API_REFERENCE.md"}
    for folder in ("jaxite_word", "demos", "cadathlon/common", "cadathlon/tests"):
        paths.update((ROOT / folder).rglob("*.py"))
    for group in ("non_agent_pset", "agent_pset"):
        for problem in (ROOT / "cadathlon" / group).iterdir():
            if not problem.is_dir():
                continue
            paths.update((problem / "src").rglob("*.py"))
            paths.update((problem / "src").rglob("*.py.golden"))
            paths.update((problem / "test").glob("*.in"))
            paths.update((problem / "test").glob("*.out"))
    paths.update((ROOT / "cadathlon/validation").glob("*.py"))
    paths.add(ROOT / "cadathlon/non_agent_pset/p1_benchmark/src/suite/suite_config.json")
    return {
        str(p.relative_to(ROOT)): p.read_bytes() for p in sorted(paths)
        if p.is_file() and not p.is_symlink() and "__pycache__" not in p.parts
        and "/suite/run/" not in str(p)
    }


def hashes(sources):
    return {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpus", default="136-139")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = source_bytes()
    before = hashes(sources)
    write_json(out / "source-before.json", before)
    with tarfile.open(out / "sources.tar.gz", "w:gz") as archive:
        for name, data in sources.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
    environment = {
        "JAX_PLATFORMS": "cpu", "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
        "CROSS_ROOT": str(ROOT), "CROSS_TEST_SECURE_REFRESH": "0",
        "CROSS_TEST_BOUNDED_REFRESH": "0", "CROSS_TEST_SECURE_POLYNOMIAL": "0",
        "BOOTSTRAP_TEST_N4096": "0", "BOOTSTRAP_TEST_N8192": "0",
    }
    command = ["taskset", "-c", args.cpus, sys.executable, "-m", "pytest", *SCOPES,
               "-q", "--junitxml=" + str(out / "junit.xml")]
    report = {
        "scope": "CPU regression after integration; expensive opt-in secure tests have separate evidence.",
        "he_tpu_validation": False, "command": command, "environment": environment,
        "source_file_count": len(before), "source_archive": "sources.tar.gz",
        "review_base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "versions": {name: importlib.metadata.version(name)
                     for name in ("jax", "jaxlib", "numpy", "torch", "libtpu", "pytest")},
        "python": sys.version, "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    start = time.monotonic()
    with (out / "pytest.log").open("wb") as stream:
        process = subprocess.run(
            ["/usr/bin/time", "-v", "-o", str(out / "time.txt"), *command],
            cwd=ROOT, env=dict(os.environ, **environment), stdout=stream,
            stderr=subprocess.STDOUT, check=False)
    report.update(returncode=process.returncode, parent_elapsed_s=time.monotonic() - start,
                  completed_utc=datetime.now(timezone.utc).isoformat())
    after = hashes(source_bytes())
    write_json(out / "source-after.json", after)
    report["source_unchanged"] = before == after
    report["changed_sources"] = [name for name in sorted(before.keys() | after.keys())
                                 if before.get(name) != after.get(name)]
    if (out / "junit.xml").is_file():
        report["pytest_suites"] = [dict(suite.attrib)
                                  for suite in ET.parse(out / "junit.xml").iter("testsuite")]
    report["passed"] = process.returncode == 0 and before == after
    write_json(out / "report.json", report)
    write_json(out / "artifact_manifest.json", {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.iterdir()) if p.is_file()
    })
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
