"""Offline verification of mocked handler evidence; never run HE or overwrite."""
from pathlib import Path
import hashlib
import json
import tarfile
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent
sha=lambda data:hashlib.sha256(data).hexdigest()


def verify_run(path):
    report=json.loads((path/'report.json').read_text())
    terminal=json.loads((path/'terminal.json').read_text())
    assert report['passed'] is True and report['tests']==dict(tests=112,errors=0,failures=0,skipped=0)
    assert not report['cleanup_errors']
    assert all(report[k] is True for k in ('source_files_unchanged','source_archive_verified',
        'test_child_reaped','owned_test_tree_removed','owned_test_handles_closed'))
    assert report['process_cleanup']['reaped'] and report['process_cleanup']['group_gone']
    assert report['test_returncode']==report['process_cleanup']['returncode']==0
    assert terminal['observed_exit_code']==0 and terminal['session_reaped'] and terminal['test_pid_absent']
    assert terminal['test_pid']==report['test_pid']
    assert report['source_hashes_before']==report['source_hashes_after']==report['snapshot_hashes_after']
    assert len(report['source_hashes_before'])==24
    with tarfile.open(path/'sources.tar.gz') as archive:
        members=archive.getmembers()
        assert len(members)==len({m.name for m in members})==24 and all(m.isfile() for m in members)
        assert {m.name:sha(archive.extractfile(m).read()) for m in members}==report['source_hashes_before']
    assert sha((path/'sources.tar.gz').read_bytes())==report['source_archive_sha256']
    assert sum(int(s.attrib['tests']) for s in ET.parse(path/'tests.xml').getroot())==112
    assert report['actual_key_generation'] is False and report['encrypted_execution'] is False
    assert report['joined_numerical_model_evaluated'] is False
    return report


def main():
    pins=json.loads((HERE/'artifact_sha256.json').read_text())
    assert all(sha((HERE/name).read_bytes())==expected for name,expected in pins.items())
    final=verify_run(HERE/'source-bound-v2');verify_run(HERE/'development-v1')
    for name,expected in final['source_hashes_before'].items():
        assert sha((HERE/'source'/name).read_bytes())==expected
    assert not list(HERE.rglob('*.npy')) and not list(HERE.rglob('*.npz'))
    print(json.dumps(dict(passed=True,tests=112,exact_sources=24,
        scope='mocked handler bindings and lifecycle only; no HE or full-model numerical claim')))


if __name__=='__main__':main()
