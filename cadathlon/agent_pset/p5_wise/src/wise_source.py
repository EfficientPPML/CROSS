"""Load the exact upstream WISE planner without importing its CPU backend."""
from pathlib import Path
import hashlib
import os
import subprocess
import sys

WISE_URL = "https://github.com/JianmingTONG/WISE"
WISE_COMMIT = "485655782dc331e6ebb48b159acac8ba924a9933"


def load_wise(root=None):
    root = Path(root or os.environ.get("WISE_ROOT", "")).resolve()
    if not (root / "src/sagittarius/core/utils.py").is_file():
        raise FileNotFoundError("Set WISE_ROOT to the pinned WISE checkout; see make install-wise")
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != WISE_COMMIT:
        raise ValueError(f"WISE must be pinned to {WISE_COMMIT}, found {commit}")
    changed = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no", "--", "src"], text=True)
    if changed.strip():
        raise ValueError("WISE planner source differs from the pinned commit")
    # Detect untracked Python files too: they must not shadow pinned imports.
    tracked = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "src"], text=True).splitlines()
    files = {p.relative_to(root).as_posix() for p in (root / "src").rglob("*.py")}
    if files != {p for p in tracked if p.endswith(".py")}:
        raise ValueError("WISE contains untracked or missing Python source")
    sys.path.insert(0, str(root / "src"))
    import sagittarius.core as wise
    if not Path(wise.__file__).resolve().is_relative_to(root):
        raise ValueError("another sagittarius installation shadows the pinned WISE planner")
    provenance = {"url": WISE_URL, "commit": commit, "source_sha256": {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sorted(files)}}
    return wise, provenance


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("destination", type=Path)
    args = p.parse_args()
    if args.destination.exists():
        raise FileExistsError("Choose a fresh destination; existing checkouts are never replaced")
    subprocess.run(["git", "clone", WISE_URL + ".git", str(args.destination)], check=True)
    subprocess.run(["git", "-C", str(args.destination), "checkout", "--detach", WISE_COMMIT], check=True)
    load_wise(args.destination)
