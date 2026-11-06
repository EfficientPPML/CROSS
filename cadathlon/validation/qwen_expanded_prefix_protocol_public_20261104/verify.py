"""Portable public proposal verification; no tokenization or model inputs."""
from pathlib import Path
import hashlib,json,tarfile,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def capsule(name,passed,expected):
 d=P/name;r=json.loads((d/'report.json').read_bytes());before=r['source_hashes_before']
 assert len(before)==23 and before==r['source_hashes_after']==r['execution_source_hashes_before']==r['execution_source_hashes_after']
 assert r['passed']is passed and r['source_files_unchanged']is True and r['execution_files_unchanged']is True
 with tarfile.open(d/'sources.tar.gz')as t:
  ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==23 and all(m.isfile()for m in ms)
  raw={m.name:t.extractfile(m).read()for m in ms};assert {n:sha(b)for n,b in raw.items()}==before
 assert sha((d/'sources.tar.gz').read_bytes())==r['source_archive_sha256']
 tests=ET.parse(d/'tests.xml').getroot().findall('testsuite');assert {n:sum(int(x.get(n,0))for x in tests)for n in ('tests','errors','failures','skipped')}==dict(tests=68,errors=0,failures=0,skipped=0)==r['tests']
 assert json.loads(raw['inventory.json'])['expected_tests']==expected
 assert all(x['direct_returncode']==0 and all(x[n]is True for n in ('direct_child_reaped','group_empty_verified','cleanup_completed'))for x in r['children'])
 for n in ('audit.json','plan-audit.json'):
  audit=json.loads((d/n).read_bytes());assert audit['passed']and not audit['denied_attempts']and not audit['backend_modules']
 if passed:
  assert all(r[n]is True for n in ('complete','cleanup_completed','terminal_scope_completed','termination_scope_completed','zero_audited_private_model_body_reads'))
  assert sha((d/'plan.json').read_bytes())==r['plan_sha256']
 return raw

def main():
 inventory=json.loads((P/'artifact_hashes.json').read_bytes());assert {n:sha((P/n).read_bytes())for n in inventory}==inventory
 old=capsule('preserved-public-v2',False,67);new=capsule('public-v3',True,68)
 assert {n for n in old if old[n]!=new[n]}=={'inventory.json'}
 r=json.loads((P/'public-v3/plan.json').read_bytes());assert r['execution_authorized']is False
 w=r['calibration']['workload'];assert len(w)==3200 and sum(x['original416_member']for x in w)==416
 assert sum(x['kind']=='prefill'for x in w)==3072 and sum(x['kind']=='teacher'for x in w)==sum(x['kind']=='free'for x in w)==64
 assert all(sum(x['kind']=='prefill'and x['length']==n for x in w)==96 for n in range(1,33))
 v=r['untouched_validation'];assert not any(v[x]for x in ('measured','tokenized','token_disjointness_verified'))
 assert v['design']['total']==1088 and len(v['design']['prompts'])==32
 assert r['row0_regression']['excluded_from_selection']is True and r['row0_regression']['original_failure_remains']is True
 assert r['acceptance']['calibration']['prefill_required_matches']==2765 and r['acceptance']['untouched_validation']['prefill_required_matches']==922
 assert r['old_candidate_manifest_sha256']=='7f6478a6bc710f3d2a370198e075e6ba8b0e55c144d981bca0eed59eaf176333'
 print(json.dumps(dict(passed=True,sources=23,tests=68,calibration_entries=3200,old_subset=416,proposed_validation_entries=1088,tokenizer_model_selection_executed=False)))
if __name__=='__main__':main()
