"""Verify retained public evidence without importing the encrypted runtime."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name: str):
    return json.loads((ROOT / name).read_text())


def check(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def archive(name: str, expected: dict[str, str], expected_digest=None):
    path = ROOT / name
    if expected_digest is not None:
        check(digest(path) == expected_digest, f"archive digest differs: {name}")
    with tarfile.open(path, "r:gz") as bundle:
        members = bundle.getmembers()
        check(len(members) == len(expected), f"member count differs: {name}")
        check(len({m.name for m in members}) == len(members), "duplicate members")
        check({m.name for m in members} == set(expected), f"inventory differs: {name}")
        for member in members:
            rel = PurePosixPath(member.name)
            check(member.isfile() and not rel.is_absolute() and ".." not in rel.parts,
                  f"nonregular or unsafe member: {member.name}")
            check(rel.suffix in {".py", ".json"} or rel.name == "Makefile.problem",
                  f"unexpected source suffix: {member.name}")
            content = bundle.extractfile(member)
            check(content is not None, "missing source content")
            with content:
                check(hashlib.sha256(content.read()).hexdigest() == expected[member.name],
                      f"source digest differs: {member.name}")
    return {"source_files": len(expected), "sha256": digest(path)}


def verify():
    result = {}
    prelaunch = read("prelaunch-v1/report.json")
    check(prelaunch["passed"] is True and prelaunch["returncode"] == 0, "tests failed")
    check(prelaunch["source_hashes_before"] == prelaunch["source_hashes_after"],
          "prelaunch source drift")
    expected = {}
    for path, sha in prelaunch["source_hashes_before"].items():
        name = ("repository/" + path.split("/checkout/", 1)[1]
                if "/checkout/" in path else "prototype/" + Path(path).name)
        check(name not in expected, "source name collision")
        expected[name] = sha
    check(len(expected) == 111, "final source count differs")
    result["prelaunch-v1/sources.tar.gz"] = archive(
        "prelaunch-v1/sources.tar.gz", expected, prelaunch["source_archive_sha256"])
    suites = ET.parse(ROOT / "prelaunch-v1/junit.xml").getroot().findall("testsuite")
    check(sum(int(s.attrib["tests"]) for s in suites) == 155, "test count differs")
    check(all(int(s.attrib[k]) == 0 for s in suites
              for k in ("errors", "failures", "skipped")), "nonpassing final tests")

    for version in (1, 2):
        name = f"native-preflight-v{version}"
        report = read(name + "/report.json")
        check(report["passed"] is True and report["returncode"] == 0, "native failure")
        check(report["private_references_removed"] is True, "private arrays retained")
        check(report["source_hashes_start"] == report["source_hashes_end"], "native source drift")
        expected = {Path(p).name: sha for p, sha in report["source_hashes_start"].items()}
        check(len(expected) == len(report["source_hashes_start"]), "native name collision")
        result[name + "/sources.tar.gz"] = archive(
            name + "/sources.tar.gz", expected, report["source_archive_sha256"])

    transport = read("norm-transport-v2.json")
    check(transport["passed"] is True and transport["exit_code"] == 0, "transport failure")
    check(transport["source_before"] == transport["source_after"], "transport source drift")
    result["norm-transport-v2-sources.tar.gz"] = archive(
        "norm-transport-v2-sources.tar.gz", transport["source_before"],
        transport["source_archive_sha256"])

    old = read("norm-private-validator-review-v1.json")
    fixed = read("norm-private-validator-review-v2.json")
    check(len(old["results"]) == len(fixed["results"]) == 2, "review count differs")
    check(all(r["accepted"] is True for r in old["results"]), "original repro differs")
    check(all(r["rejected"] is True for r in fixed["results"]), "repair repro differs")
    check([r["change"] for r in old["results"]] ==
          [r["change"] for r in fixed["results"]], "repair changed attacks")
    result["norm-private-validator-review-v1-sources.tar.gz"] = archive(
        "norm-private-validator-review-v1-sources.tar.gz",
        {Path(p).name: sha for p, sha in old["source_sha256"].items()})
    for path, sha in fixed["source_sha256"].items():
        check(prelaunch["source_hashes_before"][path] == sha, "final validator source differs")
    public = read("public-preflight-v1.json")
    check(public["passed"] is True and public["encryption_performed"] is False,
          "public preflight scope differs")
    check(public["actual_log2_qp"] < 881 and public["actual_switch_partitions"] == 7,
          "public profile differs")
    forbidden = {".npy", ".npz", ".bin", ".safetensors"}
    check(not any(p.suffix in forbidden for p in ROOT.rglob("*") if p.is_file()),
          "unexpected array or payload file")
    return {"passed": True, "tests": 155, "encrypted_execution_claimed": False,
            "source_archives": result}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
