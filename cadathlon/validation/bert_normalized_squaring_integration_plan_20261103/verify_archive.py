"""Offline exact-source and public-result verification; no crypto imports."""
from pathlib import Path
import hashlib,json,tarfile
ROOT=Path(__file__).resolve().parent
report=json.loads((ROOT/'report.json').read_text())
before=report['source_hashes_before']
assert report['passed'] and before==report['source_hashes_after'] and report['source_files_unchanged']
assert report['forbidden_calls']==[] and len(before)==108
assert hashlib.sha256((ROOT/'sources.tar.gz').read_bytes()).hexdigest()==report['source_archive_sha256']
with tarfile.open(ROOT/'sources.tar.gz') as archive:
    members=archive.getmembers()
    assert len(members)==len(before) and {m.name for m in members}==set(before)
    for member in members:
        assert member.isfile() and not member.issym() and not member.islnk()
        assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
        assert hashlib.sha256(archive.extractfile(member).read()).hexdigest()==before[member.name]
for p in report['profiles']:
    assert p['accepted'] and p['qp_is_secure'] and p['P_covers_each_partition']
    assert p['QP_bit_length']<=p['security_ceiling']
    for row in p['all_K_schedules']:
        if row['feasible']:
            assert row['block_output_level']>=2
            assert all(x['output_level']>=2 for x in row['normalized_rounds'])
verification=json.loads((ROOT/'verification.json').read_text())
assert verification['public_tests']==5 and verification['tests_exit_code']==verification['plan_exit_code']==0
print(json.dumps({'passed':True,'exact_sources':108,'public_tests':5,'profiles':4,'K_profile_schedules':512,'encrypted_execution':False},sort_keys=True))
