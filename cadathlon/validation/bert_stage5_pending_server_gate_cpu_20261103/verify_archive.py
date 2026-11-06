"""Offline pending-gate evidence check; never reads external/private paths."""
from pathlib import Path
import hashlib,json,tarfile

ROOT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    inventory=json.loads((ROOT/'artifact_hashes.json').read_bytes())
    assert {p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json' and '__pycache__' not in p.parts}==inventory
    report=json.loads((ROOT/'report.json').read_bytes())
    expected=report['source_hashes_before']
    assert report['passed'] is True and report['actual_client_decode'] is False
    assert expected==report['source_hashes_after'] and len(expected)==report['source_count']==90
    assert not report['forbidden_private_or_payload_read_attempts'] and report['later_client_gate_calls']==0
    assert report['expected_rejection']=='completed actual server proof required before client reads'
    assert sha(ROOT/'sources.tar.gz')==report['source_archive_sha256']
    assert sha(ROOT/'incomplete-stage4-report.json')==report['incomplete_stage4_report_sha256']
    incomplete=json.loads((ROOT/'incomplete-stage4-report.json').read_bytes())
    assert incomplete['passed'] is False and incomplete['actual_execution_attempted'] is True
    with tarfile.open(ROOT/'sources.tar.gz') as archive:
        members=archive.getmembers()
        assert len(members)==len({m.name for m in members})==90
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        assert {m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in members}==expected
    assert sha(ROOT/'check_stage5_actual_pending_gate_20261103.py')==expected['proof/check_stage5_actual_pending_gate_20261103.py']
    terminal=json.loads((ROOT/'terminal.json').read_bytes())
    assert terminal['observed_tool_exit_code']==0 and terminal['returned_synchronously'] is True
    print(json.dumps(dict(passed=True,source_count=90,private_read_attempts=0,actual_client_decode=False)))

if __name__=='__main__':main()
