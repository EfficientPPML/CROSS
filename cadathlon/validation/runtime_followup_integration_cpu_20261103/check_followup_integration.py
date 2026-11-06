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

ROOT = Path(__file__).resolve().parents[2]
SCOPES = ['cadathlon/tests/test_lazy_bconv.py', 'cadathlon/tests/test_scoped_ntt.py', 'cadathlon/tests/test_scoped_mul_ntt.py', 'cadathlon/tests/test_scoped_rescale_ntt.py', 'cadathlon/tests/test_scoped_nested_rescale.py', 'cadathlon/tests/test_optimized_transformer_wiring.py', 'cadathlon/tests/test_dispatch_cache_policy.py', 'cadathlon/tests/test_packed_transformer.py', 'cadathlon/tests/test_packed_qwen.py', 'cadathlon/tests/test_bounded_transformer_plan.py', 'cadathlon/tests/test_encrypted_reference_lifecycle.py', 'cadathlon/tests/test_bounded_reference_preflight.py', 'cadathlon/tests/test_bert_staging.py', 'cadathlon/tests/test_package.py', 'cadathlon/tests/test_transformer_owned_keys.py', 'cadathlon/tests/test_transformer_suite.py', 'cadathlon/tests/test_bounded_qwen2_proof.py', 'cadathlon/tests/test_encrypted_refresh.py', 'jaxite_word/ckks_level_ops_test.py', 'jaxite_word/rescale_test.py', 'jaxite_word/hemul_test.py', 'jaxite_word/hemul_double_rescaling_test.py']


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
    command = ["taskset", "-c", args.cpus, sys.executable, str(ROOT / "cadathlon/validation/followup_pytest_child.py"),
               str(out / "secure-records.json"), *SCOPES,
               "-q", "--junitxml=" + str(out / "junit.xml")]
    report = {
        "scope": "Isolated integration of scoped facade/nested rescale and serialization-safe lazy BConv after the five reviewed candidates. CPU component and lifecycle checks only.",
        "he_tpu_validation": False, "command": command, "environment": environment,
        "source_file_count": len(before), "source_archive": "sources.tar.gz",
        "review_base_commit": "98ca6833397069e4228c265d75fef64884218a28",
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
