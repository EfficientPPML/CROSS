"""Public-only full150 selector proof review; no initial outcome/snapshot reads."""
from pathlib import Path
import hashlib,json,os,tarfile,xml.etree.ElementTree as ET
P=Path('/tmp/qwen-expanded-prefix-selector-admission-preflight-v2-20261104')
LAB=Path('/tmp/qwen-expanded-prefix-selector-admission-v2-20261104')
OUT=Path('/tmp/qwen-expanded-prefix-selector-root-preflight-review-20261104')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_bytes())
def inventory(root):
    rows={}
    for p in root.rglob('*'):
        assert not p.is_symlink(),str(p)
        if p.is_file():rows[p.relative_to(root).as_posix()]=sha(p)
        else:assert p.is_dir()
    return rows
assert sha(P/'report.json')=='804fea6b0fbacaaaa226d50962e3e535ce6bdff62e0348c9febc896bfa120c3d'
r=read(P/'report.json');outer=read(P/'terminal_observation.json')
assert all(r.get(k)is True for k in ('passed','complete','child_reaped','child_group_absent','source_archive_verified','source_files_unchanged'))
assert all(r.get(k)is False for k in ('actual_coefficient_selection','actual_observer_admission','actual_observer_evidence_read','model_or_HE_executed','root_grant_issued'))
assert type(r['child_returncode'])is int and r['child_returncode']==0 and r['tests']==150 and r['xml_bad']==0
assert r['candidate_source_count']==36 and r['proof_source_count']==37
assert outer['session_id']==61520 and outer['observed_exit_code']==0 and outer['authoritatively_reaped']is True
assert outer['collector_pid']==r['pid'] and outer['child_pid']==r['child_pid']
assert all(outer[k]is True for k in ('collector_pid_absent','child_pid_absent','child_group_absent'))
external=Path('/tmp/qwen-expanded-prefix-selector-admission-v2-sources-20261104.json')
assert sha(external)=='f3d0bf68c503c2baba2cc34b7806c152efae153bbc180a4ebb85aaeae2d833f0'
expected=read(external)
assert expected==r['source_hashes_before']==r['source_hashes_after']==r['execution_hashes']
assert inventory(LAB)==inventory(P/'execution_source')==expected and len(expected)==36
assert read(P/'source_manifest.json')==expected
old=read(Path('/tmp/qwen-expanded-prefix-selector-preflight-v4-20261104/report.json'))['source_hashes_before']
assert len(old)==18 and all(expected[n]==h for n,h in old.items())
assert sha(P/'sources.tar.gz')==r['source_archive_sha256']=='4e76d85e6a3ce0f8f2570a225f86a63f22a1b101e7039b2fb900988f248e0a13'
driver=Path('/tmp/check_qwen_selector_admission_preflight_v2_20261104.py')
with tarfile.open(P/'sources.tar.gz')as tar:
    members=tar.getmembers();assert len(members)==len({m.name for m in members})==37 and all(m.isfile()for m in members)
    actual={m.name:hashlib.sha256(tar.extractfile(m).read()).hexdigest()for m in members}
assert actual==dict(expected,external_preflight=sha(driver))if False else actual=={**expected,'external_preflight.py':sha(driver)}
audit=read(P/'audit.json');assert r['audit']==audit and audit['passed']is True and audit['denials']==audit['model_backend_modules']==[]
assert len(audit['loaded_local_modules'])==18 and len(audit['synthetic_child_launches'])==8
for row in audit['loaded_local_modules'].values():assert expected[row['path']]==row['sha256']
suite=ET.parse(P/'tests.xml').getroot();assert len(suite.findall('.//testcase'))==150
assert not any(suite.findall('.//'+k)for k in ('failure','error','skipped'))
peer=Path('/tmp/qwen-selector-admission-v2-transformer-review-20261104/review.json')
assert sha(peer)=='0b83bca5742044910dab2026f0a8b4564252243b96a49eaa7aa06be3f7d93955'
assert read(peer)['source_hashes']==expected
for pid in (r['pid'],r['child_pid']):
    assert not Path('/proc',str(pid)).exists()
    try:os.killpg(pid,0)
    except ProcessLookupError:pass
    else:raise AssertionError('preflight process group alive')
OUT.mkdir(mode=0o700)
receipt=dict(passed=True,scope='Independent public source/terminal150 preflight review; no actual initial observation, private snapshot, model or coefficient selection',
    report_sha256=sha(P/'report.json'),source_archive_sha256=sha(P/'sources.tar.gz'),source_manifest_sha256=sha(external),sources=36,proof_sources=37,tests=150,
    original18_sources_unchanged=True,current_execution_start_end_archive_equal=True,audit_denials=[],model_backend_modules=[],loaded_origins=18,owned_synthetic_children=8,
    parent_child_independently_absent=[r['pid'],r['child_pid']],process_groups_absent=True,outer_session=61520,outer_exit_code=0,outer_reaped=True,
    peer_review_sha256=sha(peer),review_script_sha256=sha(Path(__file__)),actual_selection_grant_issued=False)
(OUT/'review.json').write_text(json.dumps(receipt,indent=2)+'\n')
(OUT/'review.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(passed=True,review=str(OUT/'review.json'),sha256=sha(OUT/'review.json'))))
