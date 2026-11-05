"""The portable P5 package includes its actual runtime dependencies."""
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


def test_wise_submission_has_runtime_and_excludes_execution_scratch(tmp_path):
    original = Path(__file__).parents[1]
    root = tmp_path / "cadathlon"
    shutil.copytree(original / "common", root / "common",
                    ignore=shutil.ignore_patterns("__pycache__"))
    problem = root / "agent_pset/p5_wise"
    shutil.copytree(original / "agent_pset/p5_wise", problem,
                    ignore=shutil.ignore_patterns("__pycache__", "run", "*.tar.gz"))
    scratch = problem / "src/run"
    scratch.mkdir()
    (scratch / "private.py").write_text("secret = 1")
    script = root / "common/package.py"
    command = [sys.executable, str(script), "submit", str(problem)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    with tarfile.open(problem / "submission.tar.gz") as archive:
        names = set(archive.getnames())
    for name in ("support", "encrypted_polynomial", "encrypted_compilation",
                 "encrypted_primitives", "encrypted_rotation", "encrypted_ntt",
                 "encrypted_refresh"):
        assert f"common/{name}.py" in names
    assert "agent_pset/p5_wise/src/evaluate.py" in names
    assert "agent_pset/p5_wise/src/native_runner.py" in names
    assert not any("/run/" in name for name in names)
    (root / "common/encrypted_rotation.py").unlink()
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Missing WISE runtime dependency" in result.stderr
