"""Read-only terminal/source verification of the synthetic88 issuer preflight."""
from pathlib import Path
import hashlib,json,os,tarfile,xml.etree.ElementTree as ET
P=Path('/tmp/qwen-candidate1-freeze-issuer-preflight-v2-20261104')
LAB=Path('/tmp/qwen-candidate1-freeze-issuer-v2-20261104')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def inventory(root):
    result={}
    for p in root.iterdir():
        assert p.is_file() and not p.is_symlink()
        result[p.name]=sha(p.read_bytes())
    return result
raw=(P/'source_manifest.json').read_bytes();expected=json.loads(raw)
assert sha(raw)=='35df92d84625d2943a1366de82e48686fa5f16513cefeb6f75820c60c583cc17' and len(expected)==6
r=json.loads((P/'report.json').read_bytes())
assert r['passed'] is True and r['tests']==88 and r['pytest_returncode']==r['xml_bad']==0
assert r['source_hashes_before']==r['source_hashes_after']==inventory(LAB)==inventory(P/'execution_source')==expected
assert r['source_manifest_sha256']==sha(raw) and r['audit_denials']==r['model_backend_modules']==[]
assert all(r[k] is False for k in ('actual_model_or_private_body_read','actual_candidate_outcome_read','actual_freeze_published','execution_authorized'))
assert r['requires_independent_final_exit0_reap_and_report_hash'] is True
assert set(r['loaded_local_origins'])=={'freeze_admission','freeze_issuer','test_freeze_issuer'}
assert all(expected[v['path']]==v['sha256'] for v in r['loaded_local_origins'].values())
xml=ET.parse(P/'tests.xml');assert len(xml.findall('.//testcase'))==88
assert not any(xml.findall('.//'+k)for k in ('failure','error','skipped'))
assert sha((P/'sources.tar.gz').read_bytes())=='da1291bcef3b9d2ac58379e12d493015c57c0ed7f120687c95e2522807a00453'
with tarfile.open(P/'sources.tar.gz') as tar:
    members=tar.getmembers();assert len(members)==len({m.name for m in members})==6 and all(m.isfile() for m in members)
    assert {m.name:sha(tar.extractfile(m).read())for m in members}==expected
assert not Path('/proc',str(r['pid'])).exists()
try:os.killpg(r['pgid'],0)
except ProcessLookupError:pass
else:raise AssertionError('proof process group still exists')
receipt=dict(schema='qwen-freeze-issuer-root-terminal-v1',passed=True,tool_terminal_chunk='56947c',exit_code=0,reaped=True,
    persistent_session=None,pid=r['pid'],pgid=r['pgid'],process_group_absent=True,
    report_sha256=sha((P/'report.json').read_bytes()),source_manifest_sha256=sha(raw),
    source_archive_sha256=sha((P/'sources.tar.gz').read_bytes()),sources=6,tests=88,
    current_execution_archive_start_end_equal=True,audit_denials=[],model_backend_modules=[],
    actual_model_or_private_body_read=False,actual_freeze_published=False,execution_authorized=False,
    review_script_sha256=sha(Path(__file__).read_bytes()))
out=(json.dumps(receipt,sort_keys=True,indent=2)+'\n').encode()
with (P/'root_terminal.json').open('xb') as stream:stream.write(out)
with (P/'root_review.py').open('xb') as stream:stream.write(Path(__file__).read_bytes())
print(json.dumps(receipt))
