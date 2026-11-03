"""Check a pinned upstream checkout and stage a fresh CROSS benchmark run.

No upstream submissions are executed or copied. The original checkout is
read-only; an existing destination is always rejected.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

from benchmark import HARNESS_SHA256, UPSTREAM_COMMIT, UPSTREAM_URL
from p2_runtime import common_directory


def shared_runtime_files():
    """Use the release's declared helper closure, including lazy imports."""
    common = common_directory(__file__)
    spec = importlib.util.spec_from_file_location('p2_shared_package_manifest', common / 'package.py')
    package = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(package)
    return package.required_files(common, [name for name in package.COMMON_RUNTIME_FILES
                                           if name.endswith('.py')])


def check_upstream(root):
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"Expected BERT commit {UPSTREAM_COMMIT}, got {commit}")
    actual_files = {p.relative_to(root).as_posix() for p in (root / "harness").rglob("*.py")}
    if actual_files != set(HARNESS_SHA256):
        raise ValueError("Upstream harness file list differs from the pinned manifest")
    for name, expected in HARNESS_SHA256.items():
        path = root / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Modified upstream harness: {name}")
    return {"upstream": UPSTREAM_URL, "commit": commit, "harness_files": len(HARNESS_SHA256)}


def stage(root, destination):
    manifest = check_upstream(root)
    source = Path(__file__).resolve().parent
    license_path = root / "LICENSE.md"
    if not license_path.is_file():
        raise FileNotFoundError(f"Missing upstream license: {license_path}")
    if destination.exists():
        raise FileExistsError(f"Use a new destination; refusing to replace {destination}")
    helpers = shared_runtime_files()
    destination.mkdir(parents=True)
    for name in HARNESS_SHA256:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
    shutil.copy2(license_path, destination / "LICENSE.upstream.md")
    submissions = destination / "submissions"
    submissions.mkdir()
    for path in source.rglob("*.py"):
        if not {"__pycache__", "_cross_common"}.intersection(path.relative_to(source).parts):
            target = submissions / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    manifest["harness_sha256"] = HARNESS_SHA256
    manifest["starter_sha256"] = {
        p.relative_to(submissions).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(submissions.rglob("*.py"))}
    helper_directory = submissions / '_cross_common'
    helper_directory.mkdir()
    for path in helpers:
        shutil.copy2(path, helper_directory / path.name)
    manifest['common_directory'] = 'submissions/_cross_common'
    manifest['common_sha256'] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(helper_directory.glob('*.py'))}
    (destination / "cross_staging_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return dict(manifest, destination=str(destination))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--destination", type=Path)
    args = parser.parse_args()
    try:
        root = args.upstream.resolve()
        result = check_upstream(root) if args.check else stage(root, args.destination.resolve())
        print(json.dumps({k: v for k, v in result.items() if not k.endswith("sha256")}, indent=2))
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
