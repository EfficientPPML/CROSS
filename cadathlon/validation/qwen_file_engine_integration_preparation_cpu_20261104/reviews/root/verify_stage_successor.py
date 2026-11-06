"""Public terminal/source evidence readback; no runtime imports or test reruns."""
from pathlib import Path
import datetime,hashlib,json,os,tarfile,xml.etree.ElementTree as ET
sha=lambda b:hashlib.sha256(b).hexdigest()
load=lambda p:json.loads(p.read_bytes())
source=Path('/tmp/qwen-organizer-file-engine-integration-v2-20261104')
proof=Path('/tmp/qwen-stage-integration-successor-proof-20261104/development-v1')
old=Path('/tmp/qwen-stage-boundary-negative-proof-20261104/development-v1')
manifest=Path('/tmp/qwen-stage-integration-v2-sources-20261104.json')
assert sha(manifest.read_bytes())=='225cc51ab57696ad69b6f973f2402d8e2851db27b26fc35e66b5c610032e9f5e'
expected=load(manifest);assert len(expected)==366
for name in ('sources-before.json','sources-after.json','execution-after.json'):
    assert (proof/name).read_bytes()==manifest.read_bytes()
for tree in (source,proof/'execution'):
    actual={p.relative_to(tree).as_posix():sha(p.read_bytes())for p in tree.rglob('*')if p.is_file()}
    assert actual==expected
archive=proof/'sources.tar.gz'
assert sha(archive.read_bytes())=='baa4cb01e6966ec7746c26ad719eee8570c4589ee85794b2229957556e15114a'
with tarfile.open(archive)as tar:
    members=tar.getmembers();assert len(members)==len({m.name for m in members})==366 and all(m.isfile()for m in members)
    assert {m.name:sha(tar.extractfile(m).read())for m in members}==expected
before=load(old/'sources-before.json')
assert len(before)==365 and set(before)<=set(expected)
assert {n for n in before if before[n]!=expected[n]}=={'INTEGRATION_DESIGN.md','qwen_stage_composition.py'}
assert set(expected)-set(before)=={'stage_ownership_successor.json'}
lineage=load(source/'stage_ownership_successor.json')
assert lineage['predecessor_archive_sha256']==sha((old/'sources.tar.gz').read_bytes())
assert lineage['predecessor_source_map_sha256']==sha((old/'sources-before.json').read_bytes())
for n,row in lineage['overlays'].items():assert row==dict(before=before[n],after=expected[n])
assert expected['qwen_stage_composition.py']=='67b2899474dd75973a771ed04b5d8f76a54bcc8e61e1efdc0250948122c502a3'
assert before['test_stage_boundaries.py']==expected['test_stage_boundaries.py']=='0a0954ea4986efb4bd1f647fa1e4501b79185f5c1e586e468974a68ee4fea66d'
cases=list(ET.parse(proof/'tests.xml').getroot().iter('testcase'))
oldcases=list(ET.parse(old/'tests.xml').getroot().iter('testcase'))
assert len(cases)==12 and [c.attrib['name']for c in cases]==[c.attrib['name']for c in oldcases]
assert all(c.find(k)is None for c in cases for k in ('failure','error','skipped'))
report=load(proof/'report.json')
assert sha((proof/'report.json').read_bytes())=='0ece19736783039924fec6217686de59aff976f32ecc9414276b8ccaac6931ba'
assert report['passed']is True and report['source_unchanged']is True and report['test_count']==12 and report['test_failures_errors_skips']==0
assert type(report['exit_code'])is int and report['exit_code']==0 and report['child_reaped']is True and report['child_group_absent']is True
assert report['source_archive_sha256']==sha(archive.read_bytes())
assert sha((proof/'run.py').read_bytes())==load(proof/'driver-before.json')['sha256']=='ddfff53eff4589db689180e2de10c62d8de35888d108fab589235f0b00ce47d2'
live=load(proof/'live.json')
assert live['wrapper_pid']==live['wrapper_pgid']==3916699 and live['worker_pid']==live['worker_pgid']==3916724
for pid in (3916699,3916724):
    assert not Path('/proc',str(pid)).exists()
    try:os.killpg(pid,0)
    except ProcessLookupError:pass
    else:raise AssertionError(('group present',pid))
peer=Path('/tmp/qwen-stage-ownership-security-review-20261104/review.json')
assert sha(peer.read_bytes())=='bf658d1c7fa8e91badd2388872033b36c86c89feeb857c05b2993f1d991ad95e'
review=load(peer);assert review['scoped_source_review_passed']is True and review['source_map_sha256']==sha(manifest.read_bytes())
for n,digest in review['source_pins'].items():assert expected[n]==digest
prior=Path(__file__).with_name('stage-negative-review.json')
assert sha(prior.read_bytes())=='4c551942587a5ff829119883c1d486ee2dad4e3800ab18698ae918fdd11f01ea'
selection=Path(__file__).with_name('selection-review.json')
assert sha(selection.read_bytes())=='d6e7986fad11fd90b1060e074d5993fff59845e4355b0f09aa1dcf89834fb1c2'
out=Path(__file__).with_name('stage-successor-review.json');assert not out.exists()
result=dict(schema='qwen-stage-successor-root-review-v1',passed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_root=str(source),source_count=366,source_manifest_sha256=sha(manifest.read_bytes()),source_archive_sha256=sha(archive.read_bytes()),report_sha256=sha((proof/'report.json').read_bytes()),driver_sha256=sha((proof/'run.py').read_bytes()),security_receipt_sha256=sha(peer.read_bytes()),prior_negative_review_sha256=sha(prior.read_bytes()),selection_review_sha256=sha(selection.read_bytes()),stage_sha256=expected['qwen_stage_composition.py'],test_sha256=expected['test_stage_boundaries.py'],tests=12,failures_errors_skips=0,elapsed_seconds=report['elapsed_seconds'],owner_terminal='68008 / 17d040 exit0 and reaped, direct owner tool report received; no external live-observation claim',independently_observed_absent_pids_groups=[3916699,3916724],script_sha256=sha(Path(__file__).read_bytes()),scope='Source-bound recording-only joined335 key-mapping/engine38-output synthetic file boundaries. Three preserved callback/late-identity regressions repaired; exact12 cases pass. Backend imports denied by unchanged conftest; context/maps/security/signal/thread paths recording doubles. No actual key/model/CT body access, real HE, learned accuracy, fullmodel/runtime admission or performance claim.')
out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(passed=True,path=str(out),sha256=sha(out.read_bytes()),sources=366,tests=12)))
