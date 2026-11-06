"""Verify public synthetic provenance without opening actual data."""
from pathlib import Path
import hashlib,json,tarfile,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_bytes())
def archive(path,expected):
    with tarfile.open(path)as t:
        members=t.getmembers()
        assert len(members)==len(expected)==len({m.name for m in members})
        assert all(m.isfile()and m.name in expected for m in members)
        assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest()for m in members}==expected
def xml(path,tests,failures):
    suites=list(ET.parse(path).getroot().iter('testsuite'))
    counts={k:sum(int(s.get(k,0))for s in suites)for k in ('tests','failures','errors','skipped')}
    assert counts==dict(tests=tests,failures=failures,errors=0,skipped=0)
index=read(P/'public_files.json')
assert set(index)=={p.relative_to(P).as_posix()for p in P.rglob('*')if p.is_file()and p.name!='public_files.json'}
assert all(sha(P/n)==h for n,h in index.items())
proof=P/'preflight';r=read(proof/'report.json');outer=read(proof/'outer_receipt.json');sources=read(proof/'source_manifest.json')
assert r['passed']is True and r['tests']==17 and r['failures']==r['errors']==r['skipped']==0 and r['audit_denials']==[]
assert r['actual_verifier_main_executed']is r['actual_model_or_private_body_read']is False
assert sources==r['source_hashes']and len(sources)==3
assert all(sha(proof/'source'/n)==h for n,h in sources.items())
assert type(outer['exit_code'])is int and outer['exit_code']==0 and outer['reaped']is outer['group_absent']is True
assert outer['report_sha256']==sha(proof/'report.json')and outer['source_archive_sha256']==sha(proof/'sources.tar.gz')
archive(proof/'sources.tar.gz',sources)
for name,count,failures in (('source-bound-negative-v1',3,3),('source-bound-successor-v2',7,0)):
    directory=P/'independent-review'/name;row=read(directory/'report.json')
    assert row['source_hashes_before']==row['source_hashes_after']
    assert row['source_archive_sha256']==sha(directory/'sources.tar.gz')
    assert all(sha(directory/n)==h for n,h in row['source_hashes_before'].items())
    archive(directory/'sources.tar.gz',row['source_hashes_before'])
    xml(directory/('tests.xml'if failures else 'pytest.xml'),count,failures)
    if failures:
        assert row['child_returncode']==1 and row['direct_child_waited']is row['group_absent']is True
    else:
        assert row['passed']is row['child_waited']is row['child_group_absent']is True and row['child_exit_code']==0
        assert row['current_reviewer_sha256']==sources['root_qwen_initial3200_terminal_review_20261104.py']
print(json.dumps(dict(verified_public_files=len(index),synthetic_root_tests=17,independent_regression_tests=7,preserved_negative_failures=3,actual_initial_reconstruction=False)))
