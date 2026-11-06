import ast,difflib,hashlib,json,shutil,tarfile
from pathlib import Path
out=Path('/tmp/qwen-full28-device-focused-proof-20261104/preparation-v1');source=Path('/tmp/qwen-full28-device-focused-tests-20261104')
base=Path('/tmp/qwen-toy5-role-focused-proof-20261104/preparation-v1')
handoff=Path('/tmp/qwen-full28-device-focused-test-handoff-20261104')
sha=lambda b:hashlib.sha256(b).hexdigest()
pretty=lambda v:(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
compact=lambda v:(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
manifest=(handoff/'sources.json').read_bytes();expected=json.loads(manifest);assert len(expected)==458
assert {p.relative_to(source).as_posix():sha(p.read_bytes())for p in source.rglob('*')if p.is_file()}==expected
assert not(out/'execution').exists();(out/'execution').mkdir()
for name,h in expected.items():
 raw=(source/name).read_bytes();assert sha(raw)==h;p=out/'execution'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
(out/'sources-before.json').write_bytes(manifest);(out/'sources.tar.gz').write_bytes((handoff/'sources.tar.gz').read_bytes());(out/'source-handoff.json').write_bytes((handoff/'handoff.json').read_bytes())
selection=json.loads((source/'selected_tests.json').read_bytes());runtime=json.loads((source/'runtime_sources.json').read_bytes());plan=json.loads((source/'ARTIFACT_SELECTION.json').read_bytes())
worker=(base/'worker_driver.py').read_text();driver=(base/'executed_driver.py').read_text()
(out/'before_worker_driver.py').write_text(worker);(out/'before_executed_driver.py').write_text(driver)
def replace(text,old,new):
 assert old in text,old;return text.replace(old,new)
worker=replace(worker,'exact 55 prefixture selection','exact59 prefixture selection')
worker=replace(worker,'{108,109,110,111}','{112,113,114,115}')
worker=replace(worker,"def numpy_hashes():\n    return {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in config['numpy_pins']['sources']}\n",'')
worker=replace(worker,"start_ticks=ticks(); runtime_before=hashes(source/'runtime'); numpy_before=numpy_hashes()","start_ticks=ticks(); runtime_before=hashes(source/'runtime')")
worker=replace(worker,"assert len(runtime_before)==389",'assert len(runtime_before)==403')
worker=replace(worker,"assert numpy_before==config['numpy_pins']['sources']\n",'')
worker=replace(worker,'; numpy_after=None','')
worker=replace(worker,'exact ordered55 required','exact ordered59 required')
worker=replace(worker,"selected=config['selected_files']; assert len(selected)==6","selected=config['selected_files']; assert selected==['test_device_integration.py']")
worker=replace(worker,"runtime_after=hashes(source/'runtime'); numpy_after=numpy_hashes()","runtime_after=hashes(source/'runtime')")
worker=replace(worker,'assert runtime_after==runtime_before and numpy_after==numpy_before','assert runtime_after==runtime_before')
worker=replace(worker,'runtime_module_origins={},numpy_sources_before=numpy_before,numpy_sources_after=numpy_after,','runtime_module_origins={},')
worker=replace(worker,"assert os.environ.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD') == '1'","assert os.environ.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD') == '1'\nassert os.environ.get('FULL28_FOCUSED_SOURCE_MAP')==str(out/'sources-before.json')\nassert os.environ.get('FULL28_FOCUSED_SOURCE_MAP_SHA256')==config['source_manifest_sha256']")
(out/'worker_driver.py').write_text(worker)
worker_sha=sha(worker.encode());inspector_sha=sha((out/'inspect_children.py').read_bytes())
xml=[dict(classname=n.split('::')[0][:-3],name=n.split('::')[1])for n in selection['nodes']]
config=dict(schema='qwen-full28-device-focused-proof-v1',synthetic_only=True,actual_execution_authorized=False,
 source_root=str(source),proof_root=str(out),source_count=458,runtime_count=403,
 source_manifest_sha256=sha(manifest),source_archive_sha256=sha((out/'sources.tar.gz').read_bytes()),
 source_manifest_encoding='compact sorted JSON + newline',runtime_source_expectations=runtime,runtime_module_origins={},
 selected_files=selection['files'],selected_cases=selection['cases'],selected_nodes=selection['nodes'],selected_xml_cases=xml,
 selected_nodes_sha256=sha(pretty(selection['nodes'])),selected_xml_cases_sha256=sha(pretty(xml)),
 selection_sha256=sha((source/'selected_tests.json').read_bytes()),artifact_plan=plan,artifact_plan_sha256=sha((source/'ARTIFACT_SELECTION.json').read_bytes()),
 worker_forbidden_roots=sorted(['ckks_ctx','flax','jax','jax_plugins','jaxite_word','jaxlib','key_gen','libtpu','numpy','polynomial','safetensors','tensorflow','tokenizers','torch','torch_xla','transformers']),
 owned_tmpdir=str(out/'tmp'),pytest_basetemp=str(out/'pytest-owned'),cpu_affinity=[112,113,114,115],
 python='/home/jianming_gatech/miniconda3/envs/jaxite/bin/python',pytest_config='/dev/null',
 scope='59 fresh-child Python-only device/policy/owner/loader/stage/audit/process recordings; no actual device, native package, capacity, body or scientific admission',
 temporary_numeric_names='Only child-owned honest recording_jax and empty numpy definition name, removed before normal child receipt',
 expected_child_records=59,expected_regular_child_files=236,expected_owned_fixture_metadata_files=14,
 external_source_review=dict(path='/tmp/qwen-full28-device-focused-root-source-review-20261105/review.json',sha256=sha(Path('/tmp/qwen-full28-device-focused-root-source-review-20261105/review.json').read_bytes())))
rawconfig=pretty(config);config_sha=sha(rawconfig);(out/'run_config.json').write_bytes(rawconfig)
# Keep the admitted original collector Popen/wait/finally ownership block byte-identical.
driver=replace(driver,'Q5 427/55','full28 458/59')
driver=replace(driver,"'8b423a4ddbfda4ad4c50aa039f38fa1876f2ba74034e0e60300ac676417d917c'",repr(config_sha))
driver=replace(driver,"'57fd322b69eaf64b6a1ca58dcf5e9b23dac707b933b2285c15a5b191748ec04d'",repr(worker_sha))
driver=replace(driver,"'5beb2916d31d09bfb267859ae659412b1bd81c2928e1496e24b0f37001996fa2'",repr(inspector_sha))
driver=replace(driver,'qwen-toy5-role-focused-proof-v1','qwen-full28-device-focused-proof-v1')
driver=replace(driver,'{108,109,110,111}','{112,113,114,115}')
driver=replace(driver,'[108,109,110,111]','[112,113,114,115]')
driver=replace(driver,"'108-111'","'112-115'")
driver=replace(driver,'==55','==59');driver=replace(driver,'tests=55','tests=59');driver=replace(driver,'==427','==458');driver=replace(driver,'source_count=427,runtime_count=389','source_count=458,runtime_count=403')
start="    assert sha((execution/'numpy-pins.json').read_bytes())==config['numpy_pins_sha256']\n    numpy_before={p:sha(Path(p).read_bytes())for p in config['numpy_pins']['sources']}\n    assert numpy_before==config['numpy_pins']['sources']\n"
driver=replace(driver,start,'')
driver=replace(driver,"OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')","OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',\n             FULL28_FOCUSED_SOURCE_MAP=str(out/'sources-before.json'),FULL28_FOCUSED_SOURCE_MAP_SHA256=config['source_manifest_sha256'])")
driver=replace(driver,"archive_after=archived(); numpy_after={p:sha(Path(p).read_bytes())for p in config['numpy_pins']['sources']}",'archive_after=archived()')
driver=replace(driver,"'q5_receipt_inspector'","'full28_receipt_inspector'")
driver=replace(driver,' and numpy_after==numpy_before','')
driver=replace(driver,"        and audit['numpy_sources_before']==audit['numpy_sources_after']==numpy_before\n",'')
driver=replace(driver,"        numpy_four_sources_unchanged=numpy_after==numpy_before,numpy_imported_in_worker=False,\n        numpy_child_origin_scope=config['numpy_pins']['scope'],scope=config['scope'],","        numpy_imported_in_worker=False,scope=config['scope'],\n        temporary_numeric_names=config['temporary_numeric_names'],")
driver=replace(driver,'Q5 proof collector infrastructure','Full28 proof collector infrastructure')
(out/'executed_driver.py').write_text(driver)
for name in('worker_driver.py','executed_driver.py','inspect_children.py'):ast.parse((out/name).read_bytes(),filename=str(out/name))
# Exact original ownership region rather than a behavioral equivalence claim.
def ownership(text):return text.split("    with (out/'tests.log').open('xb')as log:\n",1)[1].split('    current,after=inventory(source),inventory(execution)',1)[0]
assert ownership(driver)==ownership((base/'executed_driver.py').read_text())
for name in('worker_driver.py','executed_driver.py'):
 original=(out/('before_'+name)).read_text();now=(out/name).read_text();(out/(name+'.patch')).write_text(''.join(difflib.unified_diff(original.splitlines(True),now.splitlines(True),fromfile='admitted-q5-55/'+name,tofile='unrun-full28-59/'+name)))
(out/'driver_provenance.json').write_bytes(pretty(dict(schema='full28-focused-harness-provenance-v1',prior_driver=str(base/'executed_driver.py'),prior_driver_sha256=sha((base/'executed_driver.py').read_bytes()),prior_worker=str(base/'worker_driver.py'),prior_worker_sha256=sha((base/'worker_driver.py').read_bytes()),owned_popen_wait_finally_text_identical=True,changes=['exact458/403/59 source/selection/config pins','CPU112–115 and interpreter explicit','pure worker forbids NumPy and genuine backends; no NumPy source probing','source-map environment binds all child test/runtime files','new bounded whitelist-only record inspector; fixtures lstat only','unchanged healthy no-deadline ownership path'],source_only_unrun=True)))
argv=['env','-i','PATH=/usr/bin:/bin','HOME=/home/jianming_gatech','LANG=C.UTF-8','TZ=UTC','PYTHONDONTWRITEBYTECODE=1','PYTEST_DISABLE_PLUGIN_AUTOLOAD=1','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','taskset','-c','112-115',config['python'],'-I','-B',str(out/'executed_driver.py')]
proposal=dict(schema='full28-focused-launch-proposal-v1',status='UNRUN_REQUIRES_ROOT_REVIEW_AND_EXPLICIT_LAUNCH_AUTHORIZATION',argv=argv,source_manifest_sha256=sha(manifest),source_archive_sha256=config['source_archive_sha256'],driver_sha256=sha(driver.encode()),worker_sha256=worker_sha,inspector_sha256=inspector_sha,config_sha256=config_sha,selected_count=59,healthy_deadline_seconds=None,actual_role_grant=False)
(out/'launch-proposal.json').write_bytes(pretty(proposal))
proof_pins={p.name:sha(p.read_bytes())for p in out.iterdir()if p.is_file()}
h=dict(schema='full28-focused-unrun-harness-handoff-v1',proof_root=str(out),source_root=str(source),execution_root=str(out/'execution'),source_count=458,runtime_count=403,static_cases=59,source_manifest_sha256=sha(manifest),source_archive_sha256=config['source_archive_sha256'],driver_sha256=proposal['driver_sha256'],worker_sha256=worker_sha,inspector_sha256=inspector_sha,config_sha256=config_sha,proposal_sha256=sha((out/'launch-proposal.json').read_bytes()),pins=proof_pins,actual_execution_authorized=False,imports_collection_runs=0)
(out/'handoff.json').write_bytes(pretty(h));(out/'prepare_harness.py').write_bytes(Path(__file__).read_bytes())
assert {p.relative_to(out/'execution').as_posix():sha(p.read_bytes())for p in(out/'execution').rglob('*')if p.is_file()}==expected
assert {p.relative_to(source).as_posix():sha(p.read_bytes())for p in source.rglob('*')if p.is_file()}==expected
print(json.dumps({k:h[k]for k in('source_count','static_cases','driver_sha256','worker_sha256','inspector_sha256','config_sha256','proposal_sha256')},indent=2));print('handoff',sha((out/'handoff.json').read_bytes()))
