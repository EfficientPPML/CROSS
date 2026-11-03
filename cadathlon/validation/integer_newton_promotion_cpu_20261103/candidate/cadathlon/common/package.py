"""Local archives only; EXCLUDE applies to contestant releases."""
import argparse
import fnmatch
from pathlib import Path
import shutil
import tarfile


COMMON_RUNTIME_FILES = (
    "support.py", "Makefile.problem", "package.py", "transformer_runner.py",
    "encrypted_polynomial.py", "encrypted_transformer.py", "encrypted_reference.py",
    "encrypted_refresh.py", "encrypted_rotation.py", "encrypted_compilation.py",
    "encrypted_primitives.py", "encrypted_ntt.py", "encrypted_packed_qwen.py", "encrypted_packed_transformer.py",
    "encrypted_bounded_arithmetic.py", "encrypted_bounded_transformer.py",
    "encrypted_streaming_linear.py",
    "encrypted_integer_affine.py", "encrypted_scaled_newton.py",
)
PUBLIC_RELEASE_EVIDENCE = (
    "validation/p1_student_suite.json", "validation/p3_compact_baseline.json",
)


def required_files(root, names):
    files = [root / name for name in names]
    missing = [str(path) for path in files if not path.is_file() or path.is_symlink()]
    if missing:
        raise FileNotFoundError("Missing declared archive dependency: " + ", ".join(missing))
    return files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("clean", "submit", "release"))
    p.add_argument("directory", type=Path)
    args = p.parse_args()
    directory = args.directory.resolve()
    if args.action == "clean":
        for path in directory.rglob("__pycache__"):
            shutil.rmtree(path)
        # Sockets, pidfiles and daemon logs from a suite run; scratch, not source.
        for path in directory.rglob("suite/run"):
            if path.is_dir():
                shutil.rmtree(path)
        (directory / "submission.tar.gz").unlink(missing_ok=True)
        return
    if args.action == "submit":
        for required in ("REPORT.md", "cross.patch"):
            if not (directory / required).is_file():
                raise FileNotFoundError(f"Create {directory / required} before submission")
        # Preserve the non_agent_pset/agent_pset level: runners resolve common
        # three parents above src, just as they do in the source checkout.
        root = directory.parents[1]
        candidates = [directory / "REPORT.md", directory / "cross.patch"]
        candidates += [path for path in (directory / "src").rglob("*.py")
                       if not {"run", "results"}.intersection(
                           path.relative_to(directory / "src").parts)]
        candidates += [directory / "src" / "Makefile"]
        candidates += [directory / "src" / "suite" / "suite_config.json"]
        candidates += list(directory.glob("results*.json"))
        candidates += required_files(root / "common", COMMON_RUNTIME_FILES)
        output = directory / "submission.tar.gz"
    else:
        root = directory
        output = root / "contestant-release.tar.gz"
        candidates = []
        for problem in ("non_agent_pset/p1_benchmark", "agent_pset/p2_bert",
                        "non_agent_pset/p3_cnn", "agent_pset/p4_qwen"):
            folder = root / problem
            patterns = [line.strip()
                        for line in (folder / "EXCLUDE").read_text().splitlines()
                        if line.strip() and not line.lstrip().startswith("#")]
            for path in folder.rglob("*"):
                rel = path.relative_to(folder).as_posix()
                if not any(fnmatch.fnmatchcase(rel, pattern) for pattern in patterns):
                    candidates.append(path)
        candidates += required_files(root / "common", COMMON_RUNTIME_FILES)
        # Only the public aggregate reports cited by student documentation.
        # Per-image organizer verification evidence remains excluded above.
        candidates += required_files(root, PUBLIC_RELEASE_EVIDENCE)
    files = sorted({p for p in candidates if p.is_file() and not p.is_symlink()
                    and "__pycache__" not in p.parts and p.suffix not in (".gz", ".golden", ".pyc")})
    with tarfile.open(output, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(root), recursive=False)
    print(f"Created {output} ({len(files)} files); no upload performed")


if __name__ == "__main__":
    main()
