"""Root source-only admission for one exact focused synthetic test run."""
import ast,hashlib,json,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
PROOF=Path('/tmp/qwen-key-producer-boundary-proof-20261104/preparation-v1')
SOURCE=Path('/tmp/qwen-key-generation-producer-v2-20261104')
SECURITY=Path('/tmp/qwen-key-producer71-security-review-20261104/review.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def canonical(v):return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
pins={'executed_driver.py':'8423e904893925bbc4ef798fc331eddcc5f1b17c29c2a1b6acc8a00384bc48f1',
      'worker_driver.py':'6add929d793c524e20c79c642ab6a7019516036c7d7bb275803b879ff8e7549f',
      'run_config.json':'51b185a2ec124a7594aafa13d0d4016b54777d9fd2e31f628b72751fd169730a',
      'sources-before.json':'6b8a842e72694d691ea2f4a2d9831331b14e929ad521d701fe445e85f939d874',
      'sources.tar.gz':'536fe8696324dcc4d87b635a95b647e311daeef99b06b729955d0c1ffe9e0d97',
      'launch-proposal.json':'1f4ae755e85fccc75c7b3b9d8fe7cecb3e77aa5281da6c26a88db94c3441d7cf'}
for name,digest in pins.items():assert sha(PROOF/name)==digest
assert sha(SECURITY)=='a6b5f8e267949fd3fe7826df31cfd9be94e00a20ae2b4c51996edeb79319b167'
expected=read(PROOF/'sources-before.json');assert len(expected)==71
for root in (SOURCE,PROOF/'execution'):
    assert not any(p.is_symlink()for p in root.rglob('*'))
    assert {p.relative_to(root).as_posix():sha(p)for p in root.rglob('*')if p.is_file()}==expected
with tarfile.open(PROOF/'sources.tar.gz')as archive:
    members=archive.getmembers();assert len(members)==len({m.name for m in members})==71
    assert all(m.isfile()and not Path(m.name).is_absolute()and'..'not in Path(m.name).parts for m in members)
    assert {m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest()for m in members}==expected
config=read(PROOF/'run_config.json');proposal=read(PROOF/'launch-proposal.json')
source_review=read(HERE/'successor-source-review.json')
assert source_review['passed']is True and config['selected_nodes']==source_review['selected_nodes']
assert config['selected_files']==['test_key_store.py','test_key_producer.py']
assert config['selected_xml_cases']==[dict(classname=n.split('::')[0][:-3],name=n.split('::')[1])for n in config['selected_nodes']]
for key in ('selected_nodes','selected_xml_cases'):
    assert hashlib.sha256(canonical(config[key])).hexdigest()==config[key+'_sha256']
assert config['source_count']==71 and config['synthetic_only']is True and config['actual_execution_authorized']is False
assert config['source_root']==str(SOURCE)and config['proof_root']==str(PROOF)
for name,key in [('sources-before.json','source_manifest_sha256'),('sources.tar.gz','source_archive_sha256')]:
    assert config[key]==proposal[key]==pins[name]
for path,digest in config['runtime_source_expectations'].items():assert sha(Path(path))==digest
assert set(config['runtime_module_origins'].values())==set(config['runtime_source_expectations'])
assert proposal['argv'][:5]==['taskset','-c','108-111','env','-i']
assert proposal['argv'][5:-4]==[k+'='+v for k,v in [('PATH','/usr/bin:/bin'),('HOME','/home/jianming_gatech'),('LANG','C.UTF-8'),('TZ','UTC'),('PYTHONDONTWRITEBYTECODE','1'),('PYTEST_DISABLE_PLUGIN_AUTOLOAD','1'),('OPENBLAS_NUM_THREADS','4'),('OMP_NUM_THREADS','4'),('MKL_NUM_THREADS','4')]]
assert proposal['argv'][-4:]==['/tmp/cross-pretrained-validation-env/bin/python','-I','-B',str(PROOF/'executed_driver.py')]
assert all(not(PROOF/n).exists()for n in ('tests.log','tests.xml','report.json','worker-audit.json'))
report=dict(schema='qwen-key-producer-root-focused-launch-review-v1',passed=True,
    pins=pins,source_count=71,tests=30,source_review_sha256=sha(HERE/'successor-source-review.json'),
    security_review_sha256=sha(SECURITY),argv=proposal['argv'],affinity=[108,109,110,111],
    script_sha256=sha(Path(__file__)),synthetic_launch_authorized=True,actual_execution_authorized=False,
    scope='Exactly one selected30-case owned-small-file/recording test execution. Exact collection checked before fixtures; no inherited suite, backend/keygen/model or actual role.')
(HERE/'launch-review.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items()if k not in ('pins','argv')},sort_keys=True))
