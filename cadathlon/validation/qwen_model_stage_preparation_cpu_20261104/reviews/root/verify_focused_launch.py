"""Admit one exact38/48 synthetic run from public source bytes only."""
import hashlib,json,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
PROOF=Path('/tmp/qwen-model-stage-boundary-proof-20261104/preparation-v1')
SOURCE=Path('/tmp/qwen-model-stage-aggregate-tests-20261104')
PARENT=Path('/tmp/qwen-key-producer-boundary-proof-20261104/preparation-v1')
PEER=Path('/tmp/qwen-model-stage-transformers-review-20261104/test-source-review.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def canonical(v):return(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
pins={'executed_driver.py':'27003446b2b49976d4dd6f067fdfae63a42d229538ec4f189d5d42a148f732fc',
      'worker_driver.py':'f26cf16ec3ad1fca60ea93c7fb6e07059f597123e86662e23dc0a7f437cda91b',
      'run_config.json':'da8269a2002ebeb55cb8084b27e26c592d95aee53e1f2c78cd168242c64a4137',
      'sources-before.json':'30ab709ada68471e65767357808398990cbb36329bb29650d91fc0b1a75a3f1a',
      'sources.tar.gz':'b456d1148ee8d2abe7e12444e5f9e3cdf2a806df7809771cf8b90c2d62780286',
      'launch-proposal.json':'627e31456b21a4157a447e3d5b7f6ce03dbda2f97225cf23730f1fab9cf9bb26'}
for name,digest in pins.items():assert sha(PROOF/name)==digest
assert sha(PEER)=='fbbf1cc22b51bb56b16877eb53361807d0fe45e5311e73e625bbd4572b8fa054'
expected=read(PROOF/'sources-before.json');assert len(expected)==38
for root in (SOURCE,PROOF/'execution'):
    assert not any(p.is_symlink()for p in root.rglob('*'))
    assert {p.relative_to(root).as_posix():sha(p)for p in root.rglob('*')if p.is_file()}==expected
with tarfile.open(PROOF/'sources.tar.gz')as archive:
    members=archive.getmembers();assert len(members)==len({m.name for m in members})==38
    assert all(m.isfile()and not Path(m.name).is_absolute()and'..'not in Path(m.name).parts for m in members)
    assert {m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest()for m in members}==expected
config=read(PROOF/'run_config.json');proposal=read(PROOF/'launch-proposal.json');source_review=read(HERE/'test-source-review.json')
assert source_review['passed']is True and config['selected_nodes']==source_review['selected_nodes']
assert len(config['selected_nodes'])==48
assert config['selected_files']==['test_model_admission.py','test_held_identity.py','test_model_recording.py']
assert config['selected_xml_cases']==[dict(classname=n.split('::')[0][:-3],name=n.split('::')[1])for n in config['selected_nodes']]
for key in ('selected_nodes','selected_xml_cases'):assert hashlib.sha256(canonical(config[key])).hexdigest()==config[key+'_sha256']
assert config['source_count']==38 and config['synthetic_only']is True and config['actual_execution_authorized']is False
assert config['source_root']==str(SOURCE)and config['proof_root']==str(PROOF)
for name,key in [('sources-before.json','source_manifest_sha256'),('sources.tar.gz','source_archive_sha256')]:assert config[key]==proposal[key]==pins[name]
assert config['runtime_source_expectations']==source_review['runtime_source_pins']
assert config['runtime_module_origins']=={n:r['path']for n,r in source_review['expected_loaded_numpy_modules'].items()}
assert len(config['runtime_source_expectations'])==4 and len(config['runtime_module_origins'])==3
for path,digest in config['runtime_source_expectations'].items():assert sha(Path(path))==digest
# Reconstruct the admitted71 harness by changing only labels and exact pins.
driver=(PROOF/'executed_driver.py').read_text()
old_driver=(PARENT/'executed_driver.py').read_text();old_config=read(PARENT/'run_config.json')
restored=driver.replace('model-stage','producer/store',2).replace('qwen-model-stage-focused-proof-v1','qwen-producer-store-focused-proof-v1')
restored=restored.replace(pins['run_config.json'],sha(PARENT/'run_config.json')).replace(pins['worker_driver.py'],sha(PARENT/'worker_driver.py'))
assert restored==old_driver
worker=(PROOF/'worker_driver.py').read_text()
assert worker.replace('recording model-stage proof','recording producer/store proof')==(PARENT/'worker_driver.py').read_text()
assert proposal['argv'][:5]==['taskset','-c','108-111','env','-i']
assert proposal['argv'][5:-4]==[k+'='+v for k,v in [('PATH','/usr/bin:/bin'),('HOME','/home/jianming_gatech'),('LANG','C.UTF-8'),('TZ','UTC'),('PYTHONDONTWRITEBYTECODE','1'),('PYTEST_DISABLE_PLUGIN_AUTOLOAD','1'),('OPENBLAS_NUM_THREADS','4'),('OMP_NUM_THREADS','4'),('MKL_NUM_THREADS','4')]]
assert proposal['argv'][-4:]==['/tmp/cross-pretrained-validation-env/bin/python','-I','-B',str(PROOF/'executed_driver.py')]
assert all(not(PROOF/n).exists()for n in ('tests.log','tests.xml','report.json','worker-audit.json'))
report=dict(schema='qwen-model-stage-root-focused-launch-review-v1',passed=True,pins=pins,source_count=38,tests=48,
    source_review_sha256=sha(HERE/'test-source-review.json'),peer_review_sha256=sha(PEER),argv=proposal['argv'],
    unchanged_admitted_harness_semantics=True,affinity=[108,109,110,111],script_sha256=sha(Path(__file__)),
    synthetic_launch_authorized=True,actual_execution_authorized=False,
    scope='One exact48-case synthetic/owned-small-file/recording execution; collection checked before fixtures. No inherited suites, retained body reads, backend/context/model or actual role.')
(HERE/'focused-launch-review.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items()if k not in('pins','argv')},sort_keys=True))
