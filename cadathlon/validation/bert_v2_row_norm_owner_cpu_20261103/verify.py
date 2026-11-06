"""Verify public source-bound row-norm evidence without initializing a backend."""
from pathlib import Path
import hashlib,json,tarfile
HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    manifest=json.loads((HERE/'artifact_sha256.json').read_text())
    assert {name:sha(HERE/name) for name in manifest}==manifest
    for name,count in (('source-bound-v1',113),('source-bound-v2',114)):
        run=HERE/name;report=json.loads((run/'report.json').read_text())
        assert report['passed'] and report['source_files_unchanged'] and report['source_archive_verified']
        assert report['source_hashes_before']==report['source_hashes_after']
        assert len(report['source_hashes_before'])==count
        assert sha(run/'sources.tar.gz')==report['source_archive_sha256']
        with tarfile.open(run/'sources.tar.gz') as archive:
            members=archive.getmembers()
            assert len(members)==len({member.name for member in members})==count
            assert all(member.isfile() for member in members)
            assert {member.name:hashlib.sha256(archive.extractfile(member).read()).hexdigest() for member in members}==report['source_hashes_before']
        plan=json.loads((run/'public_plan.json').read_text())
        assert plan['passed'] and not plan['encrypted_execution'] and not plan['backend_initialized']
        assert sum(len(site['schedules']) for site in plan['sites'].values())==228
        assert all(row['consumed_levels']==11 for site in plan['sites'].values() for row in site['schedules'])
    print('Verified original113 and corrected114 source/public-fixture files,228 scale plans each, and all artifact hashes; no encrypted claim.')
if __name__=='__main__':main()
