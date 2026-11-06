"""Read-only public/source review; does not import decoder or run tests."""
from pathlib import Path
import hashlib,json,tarfile,xml.etree.ElementTree as ET
P=Path('/tmp/bert-toy5-overflow-diagnosis-v2-20261104')
OLD=Path('/tmp/bert-toy5-overflow-diagnosis-20261104')
OUT=Path('/tmp/bert-toy5-conversion-root-review-20261104')
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_bytes())
expected=read(P/'proof-sources-before.json');assert len(expected)==9
assert expected==read(P/'proof-sources-after.json')=={n:sha((P/n).read_bytes())for n in expected}
archive=P/'proof-sources.tar.gz';assert sha(archive.read_bytes())=='897fac4778eb1febbe7efa063f391009a5decddd9301b2a6456154512a9ffea5'
with tarfile.open(archive)as tar:
    members=tar.getmembers();assert len(members)==len({m.name for m in members})==9
    assert all(m.isfile()and m.name in expected for m in members)
    assert {m.name:sha(tar.extractfile(m).read())for m in members}==expected
proof=read(P/'report.json');assert sha((P/'report.json').read_bytes())=='57b40310daad8ee528666f837ca9aaefaada15a3c7a9b9786fb225de3643505c'
assert proof['source_pins']==expected and proof['synthetic_tests']==19 and proof['direct_tool_exit_code']==0
assert proof['actual_case_passed']is False and proof['actual_throw_site_proven']is False
assert proof['isolated_candidate_installed']is False and proof['frozen199_changed']is False
tree=ET.parse(P/'tests.xml');assert len(tree.findall('.//testcase'))==19
assert not any(tree.findall('.//'+n)for n in ('failure','error','skipped'))
old=read(OLD/'proof-sources-before.json');assert old==read(OLD/'proof-sources-after.json')=={n:sha((OLD/n).read_bytes())for n in old}
assert {n for n in expected if expected[n]!=old[n]}=={'sources/test_conversion.py'}
old_tree=ET.parse(OLD/'tests.xml');assert len(old_tree.findall('.//testcase'))==18 and len(old_tree.findall('.//failure'))==1
assert sha((OLD/'proof-sources.tar.gz').read_bytes())=='aac1e8fbf11629160eb19ca236c3d2d2a89ee5031ac07160ceb3e87d283d3d1c'
snapshot=read(Path('/tmp/cross-optimized-twelve-20261103/evidence/snapshot_manifest.json'))['source_hashes']
origins=read(P/'sources/source_origin.json');assert len(snapshot)==199 and len(origins)==5
for name,row in origins.items():assert expected['sources/'+name]==row['sha256']==snapshot[row['frozen_path']]
actual=Path('/tmp/cross-optimized-twelve-20261103/evidence/bert/test5/security_device.json')
assert expected['failure-preserved/security_device.json']==sha(actual.read_bytes())
peer=Path('/tmp/bert-toy5-conversion-security-review-20261104/review.json')
assert sha(peer.read_bytes())=='58266dbc1867677563c9988ab93affb3b8dab886d3d29e4b46709067d30c1f22'
assert read(peer)['source_hashes']==expected
report=dict(schema='bert-toy5-conversion-independent-root-review-v1',review_passed=True,
    scope='Public source/proof readback and static helper review; no diagnosis import, test rerun, actual body, decryption or TPU execution',
    sources=9,synthetic_tests=19,current_before_after_archive_equal=True,source_archive_sha256=sha(archive.read_bytes()),
    source_manifest_sha256=sha((P/'proof-sources-before.json').read_bytes()),proof_report_sha256=sha((P/'report.json').read_bytes()),
    original_five_runtime_copies_bound_to_frozen199=True,only_test_assumption_changed_between_18_and_19=True,
    preserved_predecessor=dict(tests=18,passed=17,failed=1,reason='Symmetric real constant was incorrectly expected to violate unchanged noise gate'),
    owner_reported_direct_exit_code=0,owner_reported_direct_tool_chunk='e5d48b',persistent_session=None,pid=None,pgid=None,
    process_scope='Direct tool completion reported by owner; no PID/PGID was captured, so no independent process-group absence claim',
    security_review_sha256=sha(peer.read_bytes()),
    conversion_scope='Exact legacy finite conversion preserved; fallback scales by exact binary64 ratio before conversion; true scaled overflow still fails',
    original_noise_limit=0.03125,crt_and_fft_are_synthetic_seams=True,numpy_cpu_guard_arithmetic=True,
    actual_case_still_failed=True,actual_throw_site_proven=False,encrypted_precision_or_application_success_claim=False,
    candidate_installed=False,frozen199_modified=False,
    review_script_sha256=sha(Path(__file__).read_bytes()))
OUT.mkdir(mode=0o700)
with (OUT/'review.json').open('x')as f:json.dump(report,f,indent=2);f.write('\n')
with (OUT/'review.py').open('xb')as f:f.write(Path(__file__).read_bytes())
print(json.dumps(dict(review_passed=True,review=str(OUT/'review.json'),sha256=sha((OUT/'review.json').read_bytes()),tests=19)))
