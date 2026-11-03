"""Submission archives retain the paths needed by shared checks and suite stages."""
from pathlib import Path
import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile


def test_submit_includes_shared_security_gate_and_suite_configuration(tmp_path):
    root = tmp_path / "cadathlon"
    problem = root / "non_agent_pset" / "p1_benchmark"
    suite = problem / "src" / "suite"
    suite.mkdir(parents=True)
    common = root / "common"
    common.mkdir()
    for name in ("REPORT.md", "cross.patch"):
        (problem / name).write_text("")
    (problem / "src" / "Makefile").write_text("")
    (problem / "src" / "runner.py").write_text("")
    (suite / "suite_config.json").write_text("{}")
    (suite / "run").mkdir()
    (suite / "run" / "active.json").write_text("{}")
    (suite / "run" / "secrets.py").write_text("private = 1")
    (suite / "secret_keys.json").write_text("{}")
    (suite / "private.npy").write_bytes(b"private")
    for name in ("support.py", "Makefile.problem", "package.py", "transformer_runner.py",
                 "encrypted_polynomial.py", "encrypted_transformer.py", "encrypted_reference.py", "encrypted_refresh.py",
                 "encrypted_rotation.py", "encrypted_compilation.py", "encrypted_primitives.py", "encrypted_ntt.py",
                 "encrypted_packed_qwen.py", "encrypted_packed_transformer.py",
                 "encrypted_bounded_arithmetic.py", "encrypted_bounded_transformer.py"):
        (common / name).write_text("")
    script = Path(__file__).parents[1] / "common" / "package.py"
    subprocess.run([sys.executable, str(script), "submit", str(problem)], check=True)
    with tarfile.open(problem / "submission.tar.gz") as archive:
        names = set(archive.getnames())
    assert "common/support.py" in names
    assert "common/transformer_runner.py" in names
    assert "common/encrypted_reference.py" in names
    assert "common/encrypted_refresh.py" in names
    assert "common/encrypted_rotation.py" in names
    assert "common/encrypted_compilation.py" in names
    assert "common/encrypted_primitives.py" in names
    assert "common/encrypted_ntt.py" in names
    assert "common/encrypted_packed_qwen.py" in names
    assert "common/encrypted_packed_transformer.py" in names
    assert "common/encrypted_bounded_arithmetic.py" in names
    assert "common/encrypted_bounded_transformer.py" in names
    assert "common/encrypted_transformer.py" in names
    assert "common/encrypted_polynomial.py" in names
    assert "non_agent_pset/p1_benchmark/src/runner.py" in names
    assert "non_agent_pset/p1_benchmark/src/suite/suite_config.json" in names
    assert not any("/run/" in name for name in names)
    assert not any("secret" in name or name.endswith(".npy") for name in names)


def test_missing_shared_runtime_fails_before_creating_an_archive(tmp_path):
    root = tmp_path / "cadathlon"
    problem = root / "agent_pset" / "p4_qwen"
    (problem / "src").mkdir(parents=True)
    for name in ("REPORT.md", "cross.patch"):
        (problem / name).write_text("")
    script = Path(__file__).parents[1] / "common" / "package.py"
    result = subprocess.run([sys.executable, str(script), "submit", str(problem)],
                            text=True, capture_output=True)
    assert result.returncode != 0
    assert "Missing declared archive dependency" in result.stderr
    assert not (problem / "submission.tar.gz").exists()


def test_shared_runtime_manifest_contains_local_import_dependencies():
    """A helper added inside a lazy method must also reach the archive."""
    common = Path(__file__).parents[1] / "common"
    spec = importlib.util.spec_from_file_location("package_manifest_test", common / "package.py")
    package = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(package)
    shipped = set(package.COMMON_RUNTIME_FILES)
    for filename in shipped:
        if not filename.endswith(".py"):
            continue
        for node in ast.walk(ast.parse((common / filename).read_text())):
            modules = ([node.module] if isinstance(node, ast.ImportFrom)
                       else [name.name for name in node.names] if isinstance(node, ast.Import) else [])
            for module in modules:
                dependency = module.split(".")[0] + ".py" if module else None
                if dependency and (common / dependency).is_file():
                    assert dependency in shipped, f"{filename} imports unshipped {dependency}"


def test_extracted_release_runs_fixtures_and_experimental_preflight(tmp_path):
    """Test shipped paths, without relying on the source tree's import paths."""
    source = Path(__file__).parents[1]
    root = tmp_path / "cadathlon"
    problems = ("non_agent_pset/p1_benchmark", "agent_pset/p2_bert",
                "non_agent_pset/p3_cnn", "agent_pset/p4_qwen")
    for problem in problems:
        shutil.copytree(source / problem, root / problem,
                        ignore=shutil.ignore_patterns("__pycache__", "run", "results"))
    shutil.copytree(source / "common", root / "common",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (root / "validation").mkdir()
    for filename in ("p1_student_suite.json", "p3_compact_baseline.json"):
        shutil.copy2(source / "validation" / filename, root / "validation" / filename)
    evidence = root / problems[0] / "src/suite/results/C3/evidence/labels.npy"
    evidence.parent.mkdir(parents=True)
    evidence.write_bytes(b"organizer-only labels")
    scratch = root / problems[2] / "src/suite/run/active.json"
    scratch.parent.mkdir(parents=True)
    scratch.write_text('{"private": true}')
    # New student calibration dependencies live outside ephemeral run/.
    public_calibration = root / problems[0] / 'src/suite/calibration/student-baseline/test-id'
    public_calibration.mkdir(parents=True)
    (public_calibration/'report.json').write_text('{"aggregate_only": true}')
    (public_calibration/'archives/C3').mkdir(parents=True)
    (public_calibration/'archives/C3/case.json').write_text('{"images": 100}')
    subprocess.run([sys.executable, str(root / "common/package.py"), "release", str(root)], check=True)
    extracted = tmp_path / "extracted"
    with tarfile.open(root / "contestant-release.tar.gz") as archive:
        names = set(archive.getnames())
        assert "common/encrypted_rotation.py" in names
        assert "common/encrypted_compilation.py" in names
        assert "common/encrypted_primitives.py" in names
        assert "common/encrypted_ntt.py" in names
        assert "common/encrypted_packed_qwen.py" in names
        assert "common/encrypted_packed_transformer.py" in names
        assert "common/encrypted_bounded_arithmetic.py" in names
        assert "common/encrypted_bounded_transformer.py" in names
        assert "validation/p1_student_suite.json" in names
        assert "validation/p3_compact_baseline.json" in names
        assert str((public_calibration/'report.json').relative_to(root)) in names
        assert str((public_calibration/'archives/C3/case.json').relative_to(root)) in names
        assert 'non_agent_pset/p1_benchmark/src/suite/p1_evidence.py' in names
        assert not any(name.endswith(".golden") or "/run/" in name or "/evidence/" in name
                       for name in names)
        archive.extractall(extracted, filter="data")
    # An invalid backend makes accidental JAX initialization fail immediately.
    # Neither plaintext fixture checks nor interface preflight needs a backend.
    env = dict(os.environ, PYTHONPATH="", CROSS_ROOT=str(source.parent),
               JAX_PLATFORMS="packaging_test_no_backend")
    for problem in problems:
        src = extracted / problem / "src"
        result = subprocess.run([sys.executable, "runner.py", "--reference", "--case", "../test/test1.in"],
                                cwd=src, env=env, text=True, capture_output=True)
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(result.stdout)["mode"] == "plaintext-reference"
    for problem, reference in ((problems[1], "bert_reference.py"), (problems[1], "bert_packed_reference.py"),
                               (problems[1], "bert_bounded_reference.py"),
                               (problems[3], "qwen_reference.py"),
                               (problems[3], "qwen_packed_reference.py"),
                               (problems[3], "qwen_packed_pair_reference.py"),
                               (problems[3], "qwen_bounded_reference.py")):
        src = extracted / problem / "src"
        result = subprocess.run([sys.executable, "runner.py", "--preflight", "--implementation", reference],
                                cwd=src, env=env, text=True, capture_output=True)
        report = json.loads(result.stdout)
        assert report["implementation_status"] == "experimental-unvalidated", result.stderr
        assert report["encryption_validated"] is False
        assert "Interface inspection only" in report["note"]
        if problem == problems[1]:
            # The toy session is present; the full-model file harness is not.
            assert "create_session" not in report["missing_interfaces"]
            assert "client_key_generation" in report["missing_interfaces"]
            assert result.returncode == 1
        else:
            assert not report["missing_interfaces"]
            assert result.returncode == 0
