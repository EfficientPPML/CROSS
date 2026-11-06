"""Narrow source/metadata readback; never imports selected code or executes tests."""
from pathlib import Path
import ast,hashlib,json,tarfile,datetime
R=Path('/tmp/qwen-softmax-cohort-full-owner-integration-v2-20261104')
P=Path('/tmp/qwen-expanded-full-model-runtime-final-v3-20261104')
F=Path('/tmp/qwen-cohort-full-owner-combined-preflight-20261104/preflight-v1')
M=Path('/tmp/qwen-cohort-full-integration-sources-v2-20261104.json')
PM=Path('/tmp/qwen-expanded-full-model-preflight-v3-20261104/sources.json')
OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def mapping(root):
 assert not any(p.is_symlink()for p in root.rglob('*'))
 return {str(p.relative_to(root)):sha(p)for p in root.rglob('*')if p.is_file()}
m=json.loads(M.read_bytes());parent=json.loads(PM.read_bytes());start=mapping(R)
assert sha(M)=='ff8d902f71497517609f3629c3adfe12f9fffa5b720ec22aa0db61b5b5bd5d5e'
assert sha(PM)=='c9b9dfd19f6514d6c43fd39378d1ea54a188b3b9d57085985dfc4ce8ae0dfe83'
assert len(m)==208 and len(parent)==199 and start==m and mapping(F/'execution')==m and mapping(P)==parent
assert sha(F/'sources.tar.gz')=='ffc73bde207c6007258abacfcb464e7a919614f3b41facf4822650c55ad77d16'
with tarfile.open(F/'sources.tar.gz')as t:
 members=t.getmembers();assert len(members)==len({v.name for v in members})==208 and all(v.isfile()for v in members)
 assert {v.name:hashlib.sha256(t.extractfile(v).read()).hexdigest()for v in members}==m
origin=json.loads((R/'integration_origin.json').read_bytes());selection=json.loads((R/'selected_preparation.json').read_bytes());assembly=json.loads((F/'assembly.json').read_bytes())
changes={n:{'old':parent[n],'new':m[n]}for n in parent if parent[n]!=m[n]}
assert changes==origin['changed_parent_files']==assembly['changed_parent_files']and len(changes)==6
assert set(changes)=={'PREPARATION.md','checkout/cadathlon/common/encrypted_qwen_attention_join.py','plan_full.py','prepare_expanded.py','selected_preparation.json','test_full_plan.py'}
added=sorted(set(m)-set(parent));assert len(added)==9 and len(parent)-len(changes)==193
assert sha(R/'cohort_parent199_sources.json')==sha(PM)
assert m['checkout/cadathlon/common/encrypted_qwen_causal_softmax_batched.py']=='d50cabad91574d4446ad563954ced1d90c9cf984ac6f2db3bf5f49aad1058e97'
new=(R/'prepare_expanded.py').read_text();old=(P/'prepare_expanded.py').read_text()
extra=",\n       'test_softmax_cohorts.py','test_cohort_binding_review.py','test_binding_call_order.py','test_cohort_full_integration.py','test_cohort_full_plan.py'"
assert new.count(extra)==1
reversed_text=new.replace(extra,'').replace('EXPECTED_TESTS=330','EXPECTED_TESTS=180').replace('all330 selected checks required','all180 selected checks required')
assert reversed_text==old and hashlib.sha256(reversed_text.encode()).hexdigest()=='3497a46a743adaf7cbd81eae507ab4dfcaf45f64d67f85fd611d6203a75ed1e9'
constants={}
for v in ast.parse(new).body:
 if isinstance(v,ast.Assign)and len(v.targets)==1 and isinstance(v.targets[0],ast.Name)and v.targets[0].id in ('TESTS','EXPECTED_TESTS','HISTORICAL'):
  constants[v.targets[0].id]=ast.literal_eval(v.value)
assert list(constants['TESTS'])==selection['test_files']and len(constants['TESTS'])==15 and constants['EXPECTED_TESTS']==selection['expected_tests']==330
assert list(constants['HISTORICAL'])==selection['blocked_historical_imports']
assert selection['cpu_affinity']==[112,113,114,115]and selection['python_flags']==['-I','-B']and selection['actual_mode_exists']is False and selection['external_owner_receipt_required']is True
assert set(constants['TESTS']).isdisjoint({'test_join.py','test_parameter_source.py','test_tail.py','test_tail_guards.py','test_tail_borrowed.py','test_full_reference.py','test_full_metrics.py','test_plan.py'})
extra_tests=list(constants['TESTS'])[-5:]
for name in extra_tests:
 tree=ast.parse((R/name).read_bytes())
 for node in ast.walk(tree):
  if isinstance(node,ast.Import):assert all(v.name not in ('test_join','test_tail','test_parameter_source')for v in node.names)
  if isinstance(node,ast.ImportFrom):assert node.module not in ('test_join','test_tail','test_parameter_source')
  if isinstance(node,ast.Call):assert ast.unparse(node.func)not in ('np.load','numpy.load')
# All normalized rules, public certificates, parameters and original helper sources remain exactly inherited.
protected=[n for n in parent if n.startswith(('inputs/','qkv-export-v1/','before87/','reference_sources/'))]+['full_contract.py','checkout/cadathlon/common/encrypted_qwen_full_model.py','checkout/cadathlon/common/encrypted_qwen_rmsnorm.py','checkout/cadathlon/common/encrypted_qwen_rms_refresh.py','checkout/cadathlon/common/encrypted_qwen_decoder_tail.py','checkout/cadathlon/common/qwen_candidate_bridge.py','execution_authorization.json','derive_expanded.py']
assert all(m[n]==parent[n]for n in protected)
auth=json.loads((R/'execution_authorization.json').read_bytes());assert all(auth[k]is False for k in ('scientific_calibration_pass_verified','model_forward_enabled','heldout_enabled','row0_enabled','he_enabled'));assert auth['first_pass_freeze_sha256']is None and auth['actual_execution_grant_sha256']is None
collection_path=F.parent/'collection-v1/report.json';collection=json.loads(collection_path.read_bytes())
assert sha(collection_path)==origin['collection_report_sha256']and collection['passed']is True and collection['tests_collected']==330 and collection['tests_executed']==0
assert collection['child_returncode']==0 and collection['child_reaped']is True and collection['child_group_absent']is True
assert collection['source_start']==collection['source_end']and len(collection['source_end'])==208
collection_changes=[n for n in m if collection['source_end'][n]!=m[n]];assert collection_changes==['integration_origin.json']
end=mapping(R);assert end==start
selected_pins={n:m[n]for n in ['prepare_expanded.py','selected_preparation.json','integration_origin.json','INTEGRATION_SCOPE.md','PREPARATION.md',*extra_tests]}
report=dict(schema='qwen-cohort-full-owner-preparation-closure-review-v1',passed=True,scope='Independent source/metadata-only closure and selected-entry review; no measured runtime import, test execution, model/private/input/parameter/key/CT body read, or new execution grant.',reviewed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_root=str(R),source_count=208,source_manifest_sha256=sha(M),source_archive_sha256=sha(F/'sources.tar.gz'),current_execution_archive_exact=True,current_start_end_exact=True,parent199_exact=True,parent_manifest_sha256=sha(PM),parent_files_unchanged=193,changed_parent_files=changes,added_files=added,reviewed_entry_parent_sha256=sha(P/'prepare_expanded.py'),entry_reverse_delta_byte_exact=True,selected_tests=list(constants['TESTS']),selected_count=330,collected_count=330,tests_executed_by_reviewer=0,collection_report_sha256=sha(collection_path),collection_to_final_changes=collection_changes,protected_file_count=len(set(protected)),all_protected_files_unchanged=True,actual_execution_flags_disabled=True,fixed_affinity=[112,113,114,115],historical_body_helpers_unselected=True,findings=[],reviewed_source_pins=selected_pins,limitations=['Root and Security own prior runtime/cohort arithmetic reviews; this review does not repeat them.','Collection330 is not a combined330 execution result. Final exact-map source/test/audit/publication/outer-reap closure remains required.','Synthetic integration replaces parameter authentication, RMS/head/linear stages explicitly; it is not a learned wholegraph or HE result.'])
(OUT/'review.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n');print(json.dumps(dict(passed=True,report_sha256=sha(OUT/'review.json'),sources=208,selected=330,reviewer_tests_executed=0)))
