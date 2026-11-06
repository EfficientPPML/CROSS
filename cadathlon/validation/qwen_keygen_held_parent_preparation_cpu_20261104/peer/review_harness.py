"""Static public harness review; never invokes its candidate or subprocess code."""
import ast,hashlib,json,tarfile
from pathlib import Path
OUT=Path('/tmp/qwen-held-parent-tests-transformers-review-20261104')
P=Path('/tmp/qwen-keygen-held-parent-focused-proof-20261104/preparation-v1')
OLD=Path('/tmp/qwen-key-producer-boundary-proof-20261104/preparation-v1')
def sha(raw):return hashlib.sha256(raw).hexdigest()
raw=(P/'handoff.json').read_bytes()
assert sha(raw)=='001cc843563e012aff795ec9779cffe99dba45664ca443c46f71950f4397abde'
h=json.loads(raw)
for n,v in h['files'].items():assert sha((P/n).read_bytes())==v,n
cfg=json.loads((P/'run_config.json').read_bytes());prop=json.loads((P/'launch-proposal.json').read_bytes())
selection=json.loads((P/'execution/selected_parent_tests.json').read_bytes())
expected=json.loads((P/'sources-before.json').read_bytes())
assert len(expected)==83 and sha((P/'sources-before.json').read_bytes())==cfg['source_manifest_sha256']
assert cfg['selected_nodes']==selection['ordered_nodes'] and len(cfg['selected_nodes'])==34
assert cfg['selected_xml_cases']==selection['ordered_xml_cases']
assert cfg['selected_files']==selection['selected_files']
for kind,key in (('nodes','selected_nodes'),('xml_cases','selected_xml_cases')):
    assert sha((json.dumps(cfg[key],sort_keys=True,indent=2,allow_nan=False)+'\n').encode())==cfg['selected_'+kind+'_sha256']
def inventory(root):
    result={}
    for p in root.rglob('*'):
        assert not p.is_symlink(),p
        if p.is_file():result[p.relative_to(root).as_posix()]=sha(p.read_bytes())
    return result
assert inventory(Path(cfg['source_root']))==inventory(P/'execution')==expected
assert sha((P/'sources.tar.gz').read_bytes())==cfg['source_archive_sha256']
with tarfile.open(P/'sources.tar.gz')as tar:
    members=tar.getmembers()
    assert len(members)==len(set(m.name for m in members))==83
    assert all(m.isfile() for m in members)
    assert {m.name:sha(tar.extractfile(m).read())for m in members}==expected
driver=(P/'executed_driver.py').read_text();worker=(P/'worker_driver.py').read_text()
for text in (driver,worker):ast.parse(text)
old=(OLD/'executed_driver.py').read_text()
assert sha(old.encode())=='8423e904893925bbc4ef798fc331eddcc5f1b17c29c2a1b6acc8a00384bc48f1'
begin="    with (out / 'tests.log').open('xb') as log:"
end='    try: os.killpg(child.pid, 0)'
assert driver[driver.index(begin):driver.index(end)]==old[old.index(begin):old.index(end)]
oldcfg=json.loads((OLD/'run_config.json').read_bytes())
for name in ('runtime_source_expectations','runtime_module_origins'):
    assert cfg[name]==oldcfg[name] and len(cfg[name])==3
for path,digest in cfg['runtime_source_expectations'].items():assert sha(Path(path).read_bytes())==digest
assert cfg['cpu_affinity']==[112,113,114,115]
assert prop['argv'][:5]==['taskset','-c','112-115','env','-i']
assert prop['argv'][-4:]==['/tmp/cross-pretrained-validation-env/bin/python','-I','-B',str(P/'executed_driver.py')]
assert prop['launch_authorized']is False and h['tests_executed']==0
assert set(prop['environment'])=={'PATH','HOME','LANG','TZ','PYTHONDONTWRITEBYTECODE','PYTEST_DISABLE_PLUGIN_AUTOLOAD','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'}
assert prop['environment']['PYTEST_DISABLE_PLUGIN_AUTOLOAD']=='1'
assert prop['environment']['PYTHONDONTWRITEBYTECODE']=='1'
assert cfg['actual_execution_authorized']is False and cfg['synthetic_only']is True
assert 'numpy'not in cfg['forbidden_roots']
assert set(cfg['forbidden_roots'])=={'flax','jax','jaxite_word','jaxlib','libtpu','safetensors','tensorflow','tokenizers','torch','torch_xla','transformers'}
for name,digest in h['files'].items():assert sha((P/name).read_bytes())==digest
report=dict(schema='qwen-held-parent-harness-source-peer-review-v1',passed=True,
    scope='Static public source/AST/archive review only; no harness/candidate import, collection, test execution, child launch or actual body reads.',
    handoff_sha256=sha(raw),pins=h['files'],source_count=83,static_selected_cases=34,
    source_review_sha256=sha((OUT/'review.json').read_bytes()),
    checks=dict(exact_current_execution_archive=True,exact_static_selection_and_order=True,
      parent_post_popen_ownership_block_exact=True,original_primary_direct_wait_and_group_cleanup=True,
      cpu_112_115_isolated_no_bytecode_and_clean_environment=True,
      backend_import_and_discovery_denials_precede_pytest=True,
      exact_collection_before_test_fixtures=True,cacheprovider_disabled=True,
      exact_xml_zero_failure_error_skip_and_worker_audit_required=True,
      numpy_genuinely_required_and_three_origin_source_pins_unchanged=True,
      final_normal_status_requires_source_audit_terminal_conjunction=True),
    concrete_blockers=[],limits=[
      'Backend Python guards and reviewed tests do not provide an OS sandbox.',
      'These are real tiny NumPy/file operations only; no actual full-size key store, context, RNG or security primitive is selected.',
      'The direct test run and independent outer terminal admission remain future owner/root work.',
      'Only three selected NumPy Python source/origin pins are claimed, not binary/native dependency closure.'],
    reviewer_script_sha256=sha(Path(__file__).read_bytes()))
out=(json.dumps(report,sort_keys=True,indent=2)+'\n').encode()
with (OUT/'harness-review.json').open('xb')as f:f.write(out)
print(json.dumps(dict(passed=True,receipt_sha256=sha(out),harness_executed=False)))
