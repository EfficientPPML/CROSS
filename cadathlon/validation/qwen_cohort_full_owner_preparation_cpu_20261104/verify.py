"""Verify public source/metadata only; never import the measured runtime."""
from pathlib import Path,PurePosixPath
import argparse,ast,difflib,hashlib,io,json,tarfile,xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent
CURRENT=Path('/tmp/qwen-softmax-cohort-full-owner-integration-v2-20261104')
MANIFEST='ff8d902f71497517609f3629c3adfe12f9fffa5b720ec22aa0db61b5b5bd5d5e'
TAR='ffc73bde207c6007258abacfcb464e7a919614f3b41facf4822650c55ad77d16'
WORKER='5d7aae1a5a19166192ebc34ddf9eac52e33351d8bb44aa296049ead66d1517fc'
sha=lambda b:hashlib.sha256(b).hexdigest()
def raw(n):return(HERE/n).read_bytes()
def obj(n):return json.loads(raw(n))
def xml(n,tests,failures=0):
 root=ET.fromstring(raw(n));cases=list(root.iter('testcase'))
 assert len(cases)==tests and len(list(root.iter('failure')))==failures
 assert not list(root.iter('error'))and not list(root.iter('skipped'))
 return cases

def main(current=False):
 index=obj('SHA256SUMS.json');paths={}
 for p in HERE.rglob('*'):
  assert not p.is_symlink()
  if p.is_file():paths[p.relative_to(HERE).as_posix()]=p
 assert set(paths)==set(index)|{'SHA256SUMS.json'}
 assert all(sha(paths[n].read_bytes())==h for n,h in index.items())
 white=obj('public_whitelist.json')
 assert set(paths)==set(white['copied_files'])|set(white['generated_files'])
 for n,v in white['copied_files'].items():assert sha(raw(n))==v['sha256']and len(raw(n))==v['bytes']
 assert sha(raw('proof/sources.json'))==MANIFEST and sha(raw('proof/sources.tar.gz'))==TAR
 m=obj('proof/sources.json');assert len(m)==208
 assert obj('proof/original-sources-after.json')==obj('proof/execution-sources-after.json')==m
 files={};total=0
 with tarfile.open(fileobj=io.BytesIO(raw('proof/sources.tar.gz')),mode='r:gz')as t:
  for x in t:
   path=PurePosixPath(x.name)
   assert x.isfile()and not path.is_absolute()and '..'not in path.parts and x.name in m and x.name not in files
   assert 0<x.size<=16<<20;total+=x.size;assert total<=128<<20
   b=t.extractfile(x).read(x.size+1);assert len(b)==x.size and sha(b)==m[x.name];files[x.name]=b
 assert set(files)==set(m)
 parent=obj('provenance/parent199/sources.json');assert len(parent)==199
 assert sha(raw('provenance/parent199/sources.json'))=='c9b9dfd19f6514d6c43fd39378d1ea54a188b3b9d57085985dfc4ce8ae0dfe83'
 origin=json.loads(files['integration_origin.json']);changes=origin['changed_parent_files']
 assert len(changes)==6 and {n for n in parent if parent[n]!=m[n]}==set(changes)
 additions=set(m)-set(parent);assert len(additions)==9 and not set(parent)-set(m)
 reconstructed_parent={}
 for n,h in parent.items():
  if n in changes:
   old=raw('provenance/parent199-overlays/'+n);assert changes[n]==dict(old=h,new=m[n]);assert sha(old)==h
   patch=''.join(difflib.unified_diff(old.decode().splitlines(keepends=True),files[n].decode().splitlines(keepends=True),fromfile='parent199/'+n,tofile='current208/'+n)).encode()
   assert raw('patches/'+n+'.patch')==patch;reconstructed_parent[n]=sha(old)
  else:reconstructed_parent[n]=sha(files[n])
 assert reconstructed_parent==parent
 for n in set(changes)|additions:assert raw('selected-source/'+n)==files[n]
 assert files['cohort_parent199_sources.json']==raw('provenance/parent199/sources.json')
 assert m['checkout/cadathlon/common/encrypted_qwen_causal_softmax_batched.py']=='d50cabad91574d4446ad563954ced1d90c9cf984ac6f2db3bf5f49aad1058e97'
 old=ast.parse(raw('provenance/parent199-overlays/prepare_expanded.py'));new=ast.parse(files['prepare_expanded.py'])
 def variable(tree,name):return next(n for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id==name for t in n.targets))
 old_tests=variable(old,'TESTS');new_tests=variable(new,'TESTS');selected=ast.literal_eval(new_tests.value);assert len(selected)==15
 new_tests.value=old_tests.value;variable(new,'EXPECTED_TESTS').value=variable(old,'EXPECTED_TESTS').value
 class Revert(ast.NodeTransformer):
  def visit_Constant(self,node):
   if isinstance(node.value,str):node.value=node.value.replace('330','180')
   return node
 new=Revert().visit(new);assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False)
 r=obj('proof/worker/report.json');assert sha(raw('proof/worker/report.json'))==WORKER
 assert r['passed']is True and r['tests']==dict(tests=330,failures=0,errors=0,skipped=0)
 assert r['test_exit_code']==0 and r['denied_attempts']==[] and r['actions_completed']is True and r['synthetic_component_tests_completed']is True
 for n in ('actual_execution_authorized','learned_wholegraph_executed','encrypted_execution','scientific_calibration_verified'):assert r[n]is False
 for n in ('execution_sources_before','execution_sources_after','original_sources_before','original_sources_after'):assert r[n]==m
 assert r['source_manifest_sha256']==MANIFEST and r['cpu_affinity']==[112,113,114,115]
 assert r['import_cache']['dont_write_bytecode']is True and r['import_cache']['empty_at_terminal']is True
 assert len(r['fresh_derivation'])==5 and all(m[n]==h for n,h in r['fresh_derivation'].items())
 assert len(r['loaded_local_modules'])==45 and all(m[v['path']]==v['sha256']for v in r['loaded_local_modules'].values())
 xml('proof/worker/tests.xml',330)
 o=obj('proof/outer-report.json');assert sha(raw('proof/outer-report.json'))=='f958ec01e6a730a2f1b94c67888d0d06497ba158ed74a7aa3373a783bb8fb8ee'
 assert o['passed']is True and type(o['child_returncode'])is int and o['child_returncode']==0 and o['child_reaped']is True and o['child_group_absent']is True
 assert o['source_manifest_sha256']==MANIFEST and o['source_archive_sha256']==TAR and o['worker_report_sha256']==WORKER
 assert o['archive_matches']is True and o['current_execution_source_exact']is True and o['synthetic_only']is True
 assert sha(raw('proof/executed_outer_driver.py'))==o['outer_driver_sha256']
 terminal=obj('proof/owner-terminal.json');assert sha(raw('proof/owner-terminal.json'))=='1a6ab51a26857112ee4417e51c0c5e0f7a84ac69de9ffc55ecdd288bba31ca0e'
 assert terminal['session_id']==90010 and terminal['terminal_tool_chunk']=='6e7331'and type(terminal['exit_code'])is int and terminal['exit_code']==0 and terminal['reaped']is True
 assert terminal['passed']is True and terminal['worker_report_sha256']==WORKER and terminal['outer_report_sha256']==sha(raw('proof/outer-report.json'))
 for role,pid in [('outer',3397120),('worker',3397122)]:assert terminal['owned_processes'][role]==dict(pid=pid,pgid=pid,pid_absent=True,group_absent=True)
 assert terminal['shared_wrapper']['owned']is False and terminal['shared_wrapper']['retirement_claim']is False
 launch=obj('proof/launch.json');assert launch['actual_mode']is False
 assert launch['explicitly_unset']==['SOFTMAX_REVIEW_SOURCE','PYTEST_ADDOPTS','PYTEST_PLUGINS','PYTHONPATH','PYTHONHOME']
 assert all(n not in launch['environment']for n in launch['explicitly_unset'])
 assert launch['command'][:3]==['taskset','-c','112-115']and '-I'in launch['command']and '-B'in launch['command']
 plan=obj('proof/worker/public_plan.json');assert sha(raw('proof/worker/public_plan.json'))==r['public_plan_sha256']
 assert plan['execution_authorized']is False and plan['scientific_calibration_verified']is False and plan['reference']['executed']is False
 assert len(plan['workloads'])==32 and len(plan['layers'])==28 and plan['resources']['rotation_keys']==166
 assert plan['final_norm']['output_level']==3 and plan['vocabulary']['output_level']==2 and len(plan['vocabulary']['blocks'])==38
 assert plan['final_norm']['degree']==256 and plan['final_norm']['raw_input_bound']==8192 and plan['vocabulary']['all_logits']==151936
 rules=json.loads(files['inputs/candidate_manifest.json']);cert=json.loads(files['inputs/certificates.json'])
 for n,row in enumerate(plan['workloads'],1):
  groups=[]
  for tile in range((n+15)//16):
   grouped={}
   for k in range(tile*16+1,min(n,(tile+1)*16)+1):
    if k==1:continue
    key=(rules['reciprocal_by_keys'][str(k)]['iterations'],cert['softmax_K'][k-1]['existing_inverse_refresh_bound']);grouped.setdefault(key,[]).append(k)
   groups.append(grouped)
  per_query=sum(3+2*rules['softmax'][str(i)]['squarings']for i in range(28));count=sum(map(len,groups))
  assert row['sequence']==n and row['refresh_counts']['softmax']==count*per_query and row['literal_reference_softmax_refreshes']==(n-1)*per_query
  assert row['refresh_counts']['total']-count*per_query==row['refresh_counts']['literal_reference_total']-(n-1)*per_query
  for layer in row['softmax_cohorts']:
   assert len(layer['tiles'])==len(groups)
   for actual,required in zip(layer['tiles'],groups):assert {(g['newton_iterations'],g['refresh_bound']):[q['k']for q in g['queries']]for g in actual}==required
 assert plan['workloads'][-1]['refresh_counts']['total']==5035 and plan['workloads'][-1]['refresh_counts']['literal_reference_total']==12931
 peer=obj('peer-review/review.json');assert sha(raw('peer-review/review.json'))=='671583d13090c1aae1b4f5ee653b7a5ab96d9eada5d5f70b3966dd2d7239c1b9'
 assert peer['changed_parent_files']==changes and set(peer['added_files'])==additions
 assert sha(raw('root-review/review.json'))=='1dc7ee89be0816d49db792a6891faef5eff7fcaeadfda7d977a015419355fd90'
 assert obj('root-review/review.json')['passed']is True
 assert sha(raw('root-review/terminal-admission.json'))=='194033d2bffc732769dc3a25f7de697c94dcbb43eebf7694bd13c7439446f7a0'
 collection=obj('collection/report.json');assert collection['passed']is True and collection['tests_collected']==330 and collection['tests_executed']==0
 assert collection['child_returncode']==0 and collection['child_reaped']is True and collection['child_group_absent']is True
 assert sha(raw('collection/report.json'))==origin['collection_report_sha256']and sha(raw('collection/executed_driver.py'))==collection['driver_sha256']
 historical=obj('history/development-v1/post_terminal_sources.json');reconstruction=obj('history/development-v1/reconstruction.json');assert len(historical)==205
 assert reconstruction['base_source_manifest_sha256']==MANIFEST and reconstruction['captured_after_terminal']is True and reconstruction['startup_source_snapshot_available']is False
 sparse=reconstruction['sparse_files'];assert len(sparse)==8
 assert {n:sha(raw('history/development-v1/sparse-source/'+n)if n in sparse else files[n])for n in historical}==historical
 assert {n:h for n,h in historical.items()if m.get(n)!=h}==sparse
 xml('history/development-v1/tests.xml',12,1);xml('history/development-v2/tests.xml',8);xml('history/plan-development-v1/tests.xml',2)
 for path,session,code in [('development-v1',32608,1),('development-v2',60755,0),('plan-development-v1',60759,0)]:
  term=obj('history/'+path+'/terminal.json');assert term['session']==session and term['exit_code']==code and term['reaped']is True
 assert sha(raw('history/development-v2/test_cohort_full_integration.py'))==obj('history/development-v2/terminal.json')['source_sha256']
 for prefix,reportpin in [('parent199','275d977ece3e23f7525dfb3863f7431ca5443c0ed08060a01052ab070a6c3350'),('cohort197','7c94effe3dc5490d3499e66ecc7ce0bae8bb5ec72b4e6cb0a9c109a5016d5a61')]:
  assert sha(raw('provenance/'+prefix+'/report.json'))==reportpin
  assert obj('provenance/'+prefix+'/report.json')['passed']is True
 if current:
  assert not any(p.is_symlink()for p in CURRENT.rglob('*'))
  assert {p.relative_to(CURRENT).as_posix():sha(p.read_bytes())for p in CURRENT.rglob('*')if p.is_file()}==m
 print(json.dumps(dict(passed=True,indexed_public_files=len(index),source_count=208,unchanged_parent_files=193,tests=330,history205_reconstructed=True,all32_public_counts_verified=True,current_checked=current,private_descriptors_followed=False,tests_rerun=False),sort_keys=True))
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--current',action='store_true');main(parser.parse_args().current)
