"""Offline verification of public preparation evidence only."""
from pathlib import Path
import hashlib,json,tarfile,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
for n,h in json.loads((ROOT/'artifact_hashes.json').read_text()).items():assert digest(ROOT/n)==h,n

def archive(path,expected,tarhash=None):
    if tarhash is not None:assert digest(path)==tarhash
    with tarfile.open(path) as tar:
        members=tar.getmembers()
        assert len(members)==len(expected)==len({m.name for m in members})
        assert all(m.isfile() for m in members)
        assert {m.name:hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in members}==expected
r=json.loads((ROOT/'preflight/report.json').read_text())
assert r['passed'] and r['tests_passed']==69 and r['source_count']==42 and r['candidate_source_count']==41
assert r['source_hashes_before']==r['source_hashes_after'] and r['source_files_unchanged'] and r['assets_unchanged']
assert r['asset_hashes_before']==r['asset_hashes_after'] and r['all_children_reaped']
assert all(x['returncode']==0 and x['reaped'] for x in r['processes'])
archive(ROOT/'preflight/sources.tar.gz',r['source_hashes_before'],r['source_archive_sha256'])
for sub in ['installed','public-gate']:
    p=ROOT/'preflight'/sub;v=json.loads((p/'report.json').read_text())
    assert v['passed'] and v['source_hashes_before']==v['source_hashes_after']==r['candidate_source_hashes']
    archive(p/'sources.tar.gz',v['source_hashes_before'],v['source_archive_sha256'])
installed=json.loads((ROOT/'preflight/installed/report.json').read_text())
assert installed['direct_entrypoints']==9 and installed['denied_calls']==12 and installed['all_restored']
assert installed['no_jax_or_he_modules_loaded'] and not installed['forbidden_reads']
public=json.loads((ROOT/'preflight/public-gate/report.json').read_text())
assert public['execute']is False and public['private_fixture_read']is False and public['terminal_scope_completed']
suites=ET.parse(ROOT/'preflight/tests.xml').getroot()
assert sum(int(s.get('tests')) for s in suites)==69
assert all(int(s.get('failures'))==int(s.get('errors'))==int(s.get('skipped'))==0 for s in suites)
for directory in ['development-v2-preserved','development-v3-preserved','development-v5-preserved']:
    p=ROOT/'history'/directory;v=json.loads((p/'report.json').read_text());archive(p/'sources.tar.gz',v['archive_time_hashes'])
p=ROOT/'history/installed-development-v2';v=json.loads((p/'report.json').read_text());assert v['passed'];archive(p/'sources.tar.gz',v['source_hashes_before'],v['source_archive_sha256'])
p=ROOT/'history/installed-development-v1';v=json.loads((p/'failure.json').read_text());assert not v['passed'] and digest(p/'sources.tar.gz')==v['exact_source_archive_sha256']
print('PASS:42 exact preparation sources;69 mocked checks;12 installed denials; public gates; no native/HE claim')
