"""Author only the unrun harness metadata successor; no candidate imports."""
import ast, difflib, hashlib, json, shutil, tarfile
from pathlib import Path
OLD=Path('/tmp/qwen-client-authority-boundary-proof-20261104/preparation-v1')
NEW=OLD.with_name('preparation-v2')
OUT=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
canonical=lambda x:(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def inventory(root):
    assert not any(p.is_symlink()for p in root.rglob('*'))
    return {p.relative_to(root).as_posix():sha(p.read_bytes())for p in sorted(root.rglob('*'))if p.is_file()}
before=inventory(OLD)
assert before['run_config.json']=='b54359259300c0e47daff411f93215eab25851160eedcc3a3591bebe4af85919'
assert before['executed_driver.py']=='263bd38d13275dc2cf81b7f512697a51bec6f14a06e781cba3426170e089dd20'
assert before['worker_driver.py']=='3dfb645c744143d1e1b7699791680dffd549ec179b16d708348e0e67c4b4ec97'
assert not any(n in before for n in ('tests.xml','tests.log','report.json','worker-audit.json','live-observation.json'))
assert not NEW.exists()
shutil.copytree(OLD,NEW)
config=json.loads((NEW/'run_config.json').read_bytes())
assert config['cpu_affinity']==[112,113,114,115]
assert config['source_manifest_encoding']=='canonical sorted compact JSON plus newline'
config['cpu_affinity']=[108,109,110,111]
config['source_manifest_encoding']='canonical sorted indent=2 JSON plus newline'
config['proof_root']=str(NEW)
(NEW/'run_config.json').write_bytes(canonical(config))
configsha=sha((NEW/'run_config.json').read_bytes())
driver=(NEW/'executed_driver.py').read_text()
assert driver.count(before['run_config.json'])==1
(NEW/'executed_driver.py').write_text(driver.replace(before['run_config.json'],configsha))
driversha=sha((NEW/'executed_driver.py').read_bytes())
proposal=json.loads((NEW/'launch-proposal.json').read_bytes())
assert proposal['argv'][-1]==str(OLD/'executed_driver.py')
proposal['argv'][-1]=str(NEW/'executed_driver.py')
proposal['config_sha256']=configsha;proposal['driver_sha256']=driversha
(NEW/'launch-proposal.json').write_bytes(canonical(proposal))
handoff=json.loads((NEW/'handoff.json').read_bytes())
handoff.update(proof_root=str(NEW),config_sha256=configsha,driver_sha256=driversha,
    launch_proposal_sha256=sha((NEW/'launch-proposal.json').read_bytes()))
(NEW/'handoff.json').write_bytes(canonical(handoff))
after=inventory(NEW)
changed=[n for n in before if before[n]!=after[n]]
assert set(before)==set(after)
assert set(changed)=={'run_config.json','executed_driver.py','launch-proposal.json','handoff.json'}
assert inventory(OLD)==before
oldconfig=json.loads((OLD/'run_config.json').read_bytes())
assert {k for k in config if config[k]!=oldconfig[k]}=={'cpu_affinity','source_manifest_encoding','proof_root'}
assert (OLD/'executed_driver.py').read_bytes().replace(before['run_config.json'].encode(),configsha.encode())==(NEW/'executed_driver.py').read_bytes()
expected=json.loads((NEW/'sources-before.json').read_bytes())
assert len(expected)==53
assert (NEW/'sources-before.json').read_bytes()==canonical(expected)
assert sha(canonical(expected))==config['source_manifest_sha256']
assert inventory(Path(config['source_root']))==inventory(NEW/'execution')==expected
assert sha((NEW/'sources.tar.gz').read_bytes())==config['source_archive_sha256']
with tarfile.open(NEW/'sources.tar.gz')as arc:
    members=arc.getmembers()
    assert len(members)==len({m.name for m in members})==53
    assert all(m.isfile()for m in members)
    assert {m.name:sha(arc.extractfile(m).read())for m in members}==expected
assert len(config['selected_nodes'])==len(set(config['selected_nodes']))==len(config['selected_xml_cases'])==95
assert sha(canonical(config['selected_nodes']))==config['selected_nodes_sha256']
assert sha(canonical(config['selected_xml_cases']))==config['selected_xml_cases_sha256']
constants={n.targets[0].id:ast.literal_eval(n.value)for n in ast.parse((NEW/'executed_driver.py').read_bytes()).body if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and n.targets[0].id in ('CONFIG_SHA256','WORKER_SHA256')}
assert constants==dict(CONFIG_SHA256=configsha,WORKER_SHA256=before['worker_driver.py'])
for name in ('executed_driver.py','worker_driver.py'):
    t=ast.parse((NEW/name).read_bytes())
    assert any(isinstance(n,ast.Set)and all(isinstance(v,ast.Constant)for v in n.elts)and {v.value for v in n.elts}==set(config['cpu_affinity'])for n in ast.walk(t))
assert proposal['argv'][proposal['argv'].index('-c')+1]=='108-111'
assert config['runtime_source_expectations']==config['runtime_module_origins']=={}
assert 'numpy'in config['forbidden_roots']
diff=''.join(''.join(difflib.unified_diff((OLD/n).read_text().splitlines(True),(NEW/n).read_text().splitlines(True),fromfile='preparation-v1/'+n,tofile='preparation-v2/'+n))for n in sorted(changed))
(OUT/'metadata-only.patch.txt').write_text(diff)
(OUT/'preparation-v1-files.json').write_bytes(canonical(before))
(OUT/'preparation-v2-files.json').write_bytes(canonical(after))
review=dict(schema='qwen-client-authority-harness-metadata-successor-v1',passed=True,
    scope='stdlib file/hash/AST preparation only; no candidate imports, collection, tests or execution',
    parent=str(OLD),successor=str(NEW),parent_preserved=True,source53_and_test95_unchanged=True,
    changes=changed,config_corrections=['cpu_affinity=[108,109,110,111] matching exact driver/worker/argv','source_manifest_encoding matches actual sorted indent=2 JSON plus newline'],
    derived_changes=['fresh proof root path','config digest in otherwise byte-identical driver','proposal paths/pins','handoff paths/pins'],
    unchanged_historical_patches='Copied executed_driver.py.patch and worker_driver.py.patch describe preparation-v1 versus pure39; metadata-only.patch.txt records the complete v1-to-v2 delta.',
    source_count=53,static_selected_count=95,collection_performed=False,execution_authorized=False,
    source_manifest_sha256=config['source_manifest_sha256'],source_archive_sha256=config['source_archive_sha256'],
    driver_sha256=driversha,worker_sha256=before['worker_driver.py'],config_sha256=configsha,
    proposal_sha256=sha((NEW/'launch-proposal.json').read_bytes()),handoff_sha256=sha((NEW/'handoff.json').read_bytes()),
    delta_sha256=sha((OUT/'metadata-only.patch.txt').read_bytes()),prepare_script_sha256=sha(Path(__file__).read_bytes()))
(OUT/'review.json').write_bytes(canonical(review))
print(json.dumps(review,sort_keys=True))
