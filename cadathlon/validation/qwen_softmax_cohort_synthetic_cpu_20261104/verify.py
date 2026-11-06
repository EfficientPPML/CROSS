"""Public-byte verifier only: no runtime imports, arithmetic, or private paths."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,io,json,tarfile,xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent
CURRENT=Path('/tmp/qwen-causal-softmax-cohort-successor-20261104')
RUNTIME='checkout/cadathlon/common/encrypted_qwen_causal_softmax_batched.py'
sha=lambda b:hashlib.sha256(b).hexdigest()
def raw(name):return(HERE/name).read_bytes()
def load(name):return json.loads(raw(name))
def xml(name,tests,failures):
 root=ET.fromstring(raw(name));cases=list(root.iter('testcase'))
 assert len(cases)==tests and len(list(root.iter('failure')))==failures
 assert not list(root.iter('error'))and not list(root.iter('skipped'))
def archive(name,mapping):
 out={};total=0
 with tarfile.open(fileobj=io.BytesIO(raw(name)),mode='r:gz')as tar:
  for member in tar:
   path=PurePosixPath(member.name)
   assert member.isfile()and not path.is_absolute()and '..'not in path.parts
   assert member.name in mapping and member.name not in out and 0<member.size<=8<<20
   total+=member.size;assert total<=64<<20
   body=tar.extractfile(member).read(member.size+1);assert len(body)==member.size
   assert sha(body)==mapping[member.name];out[member.name]=body
 assert set(out)==set(mapping)
 return out

def main(current=False):
 index=load('SHA256SUMS.json');paths={}
 for p in HERE.rglob('*'):
  assert not p.is_symlink()
  if p.is_file():paths[p.relative_to(HERE).as_posix()]=p
 assert set(paths)==set(index)|{'SHA256SUMS.json'}
 assert all(sha(paths[n].read_bytes())==h for n,h in index.items())
 whitelist=load('public_whitelist.json')
 assert set(paths)==set(whitelist['copied_files'])|set(whitelist['generated_files'])
 m=load('proof/sources-before.json')
 assert len(m)==197 and sha(raw('proof/sources-before.json'))=='69bbd4561761a400fc676dee63ab31b50499096d100a6752cee1567d6819b1fd'
 assert load('proof/sources-after.json')==load('proof/execution-sources-after.json')==m
 r=load('proof/report.json');assert sha(raw('proof/report.json'))=='7c94effe3dc5490d3499e66ecc7ce0bae8bb5ec72b4e6cb0a9c109a5016d5a61'
 assert r['schema']=='qwen-softmax-cohort-public-proof-v1'and r['passed']is True and r['synthetic_only']is True
 assert r['model_forward']is False and r['encrypted_execution']is False
 assert r['source_count']==197 and r['tests']==136 and r['bad_tests']==0
 assert r['sources_before']==r['sources_after']==r['execution_sources_after']==m
 assert r['parent191_unchanged']is True and r['archive_matches']is True
 assert type(r['child_returncode'])is int and r['child_returncode']==0 and r['child_reaped']is True and r['child_group_absent']is True
 assert sha(raw('proof/sources.tar.gz'))==r['source_archive_sha256']=='023cb45bfa7e966846c54e6e1244d53be0094699185a14ce1e7db55988e7d212'
 assert sha(raw('proof/executed_driver.py'))==r['driver_sha256']
 files=archive('proof/sources.tar.gz',m);xml('proof/tests.xml',136,0)
 parent=load('parent191_sources.json');assert len(parent)==191
 assert sha(raw('parent191_sources.json'))=='d0846e053f3a369a18768ee69328568be46355f6952a06ef89587902a235f79a'
 assert all(m[n]==h for n,h in parent.items())
 assert set(m)-set(parent)=={'cohort_parent_sources.json','COHORT_DESIGN.md',RUNTIME,'test_softmax_cohorts.py','test_cohort_binding_review.py','test_binding_call_order.py'}
 assert files['cohort_parent_sources.json']==raw('parent191_sources.json')and files['COHORT_DESIGN.md']==raw('COHORT_DESIGN.md')
 t=load('proof/terminal_observation.json');assert t['session']==31548 and type(t['exit_code'])is int and t['exit_code']==0 and t['reaped']is True
 assert t['report_sha256']==sha(raw('proof/report.json'))and t['source_archive_sha256']==r['source_archive_sha256']and t['source_manifest_sha256']==sha(raw('proof/sources-before.json'))
 assert t['current_execution_archive_197_match']is True and t['xml136_no_fail_error_skip']is True
 observed=load('proof/process-observed.json');assert observed==r['processes']
 for role,pid in(('collector',3175372),('child',3175385)):
  assert observed[role+'_pid']==observed[role+'_pgid']==pid
  row=t['observed_owned_processes'][role];assert row==dict(pid=pid,pgid=pid,pid_absent=True,group_absent=True)
 assert t['wrapper']['pid']==t['wrapper']['pgid']==16918
 histories={}
 for name,count,tests,fail in(('negative-v1',195,8,6),('successor-v1',195,8,0),('guard-order-negative-v1',196,3,2),('successor-v2',196,11,0)):
  prefix='history/'+name+'/'
  h=load(prefix+'sources-before.json');assert len(h)==count and load(prefix+'sources-after.json')==h
  archive(prefix+'sources.tar.gz',h);xml(prefix+'tests.xml',tests,fail);histories[name]=h
 assert histories['negative-v1']['test_cohort_binding_review.py']==histories['successor-v1']['test_cohort_binding_review.py']==m['test_cohort_binding_review.py']
 assert histories['guard-order-negative-v1']['test_binding_call_order.py']==histories['successor-v2']['test_binding_call_order.py']==m['test_binding_call_order.py']
 assert histories['successor-v2'][RUNTIME]==m[RUNTIME]==t['runtime_sha256']=='d50cabad91574d4446ad563954ced1d90c9cf984ac6f2db3bf5f49aad1058e97'
 for name,session,code in(('negative-v1',73093,1),('successor-v1',92322,0),('successor-v2',39604,0)):
  x=load('history/'+name+'/terminal.json');assert x['session']==session and x['exit_code']==code and x['reaped']is True
 old=load('history/guard-order-negative-v1/terminal.json');assert old['authoritative_tool_terminal_observation_available']is False
 replay=load('history/guard-order-negative-v1/terminal-replay.json');assert replay['direct_tool_chunk']=='7ae838'and replay['exit_code']==1 and replay['reaped']is True and replay['same_frozen_source']is True
 xml('history/guard-order-negative-v1/tests-replay.xml',3,2)
 review=load('peer-review/security-source-review.json');assert sha(raw('peer-review/security-source-review.json'))=='4a185650eb00d379efb31ec10306a541150fa45a94bf61ecf35f0845f227d994'
 assert review['no_further_concrete_blocker_in_reviewed_current_source']is True and review['parent191_exact']is True
 assert all(m[n]==h for n,h in review['source_hashes'].items())
 if current:
  assert not any(p.is_symlink()for p in CURRENT.rglob('*'))
  assert {p.relative_to(CURRENT).as_posix():sha(p.read_bytes())for p in CURRENT.rglob('*')if p.is_file()}==m
 print(json.dumps(dict(passed=True,indexed_public_files=len(index),source_count=197,parent191_unchanged=True,synthetic_tests=136,history_cases=30,current_checked=current,private_descriptors_followed=False,tests_rerun=False),sort_keys=True))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--current',action='store_true');main(p.parse_args().current)
