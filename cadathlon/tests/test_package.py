"""Submission archives retain the paths needed by shared checks and suite stages."""
from pathlib import Path
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
    for name in ("support.py", "Makefile.problem", "package.py"):
        (common / name).write_text("")
    script = Path(__file__).parents[1] / "common" / "package.py"
    subprocess.run([sys.executable, str(script), "submit", str(problem)], check=True)
    with tarfile.open(problem / "submission.tar.gz") as archive:
        names = set(archive.getnames())
    assert "common/support.py" in names
    assert "non_agent_pset/p1_benchmark/src/runner.py" in names
    assert "non_agent_pset/p1_benchmark/src/suite/suite_config.json" in names
    assert not any("/run/" in name for name in names)
    assert not any("secret" in name or name.endswith(".npy") for name in names)
