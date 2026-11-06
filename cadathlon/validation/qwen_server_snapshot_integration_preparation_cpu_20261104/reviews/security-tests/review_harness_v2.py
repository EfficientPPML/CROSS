"""Narrow config-pin/path successor verification, without candidate execution."""
import hashlib,json,tarfile
from pathlib import Path
P=Path('/tmp/qwen-server-snapshot-integration-focused-proof-20261104/preparation-v2')
B=P.parent/'preparation-v1';H=lambda b:hashlib.sha256(b).hexdigest()
J=lambda p:json.loads(p.read_bytes())
h=J(P/'handoff.json');c=J(P/'run_config.json');old=J(B/'run_config.json')
assert H((P/'handoff.json').read_bytes())=='50ffb8a07c5d63d69ffd60e8e40a789f272075851427f7326eee5bf9b3af6555'
for name,digest in h['files'].items():assert H((P/name).read_bytes())==digest,name
changed=sorted(n for n in c if c[n]!=old[n])
assert set(c)==set(old)and changed==['owned_tmpdir','proof_root','pytest_basetemp','source_archive_sha256','source_count','source_manifest_sha256','source_root']
before=(B/'executed_driver.py').read_bytes();after=(P/'executed_driver.py').read_bytes()
oldpin=H((B/'run_config.json').read_bytes()).encode();newpin=H((P/'run_config.json').read_bytes()).encode()
assert before.count(oldpin)==1 and before.replace(oldpin,newpin)==after
assert (P/'worker_driver.py').read_bytes()==(B/'worker_driver.py').read_bytes()
assert (P/'before_worker_driver.py').read_bytes()==(B/'worker_driver.py').read_bytes()
assert (P/'before_executed_driver.py').read_bytes()==before
raw=(P/'sources-before.json').read_bytes();m=json.loads(raw)
assert H(raw)==c['source_manifest_sha256']and raw==(json.dumps(m,sort_keys=True,separators=(',',':'))+'\n').encode()
def scan(root):
    result={}
    for path in root.rglob('*'):
        assert not path.is_symlink()
        if path.is_file():result[path.relative_to(root).as_posix()]=H(path.read_bytes())
    return result
assert len(m)==405 and scan(Path(c['source_root']))==scan(P/'execution')==m
assert {n[8:]:v for n,v in m.items()if n.startswith('runtime/')}==c['runtime_source_expectations']and len(c['runtime_source_expectations'])==391
with tarfile.open(P/'sources.tar.gz')as archive:
    members=archive.getmembers();assert len(members)==len({x.name for x in members})==405 and all(x.isfile()for x in members)
    assert {x.name:H(archive.extractfile(x).read())for x in members}==m
assert c['selected_nodes']==J(P/'source-handoff.json')['static_nodeids']and len(c['selected_nodes'])==36
assert c['runtime_module_origins']=={}and c['cpu_affinity']==[112,113,114,115]
assert c['expected_regular_child_files']==144 and c['expected_owned_public_fixture_files']==6
provenance=J(P/'driver_provenance.json')
for name,digest in provenance['base_pins'].items():assert H((B/name).read_bytes())==digest
for name,digest in provenance['independent_failed_receipts'].items():assert H(Path(name).read_bytes())==digest
for name in('tests.log','tests.xml','report.json','worker-audit.json','live-observation.json','tmp','pytest-owned'):assert not(P/name).exists()
report=dict(schema='qwen-snapshot-integration-405-harness-delta-peer-review-v1',conclusion='No additional blocker in the narrow held unrun harness successor.',
    handoff_sha256=H((P/'handoff.json').read_bytes()),driver_sha256=H(after),worker_sha256=H((P/'worker_driver.py').read_bytes()),config_sha256=newpin.decode(),source_manifest_sha256=H(raw),source_archive_sha256=H((P/'sources.tar.gz').read_bytes()),
    exact_driver_delta='Only CONFIG_SHA256 literal',worker_exact=True,changed_config_fields=changed,source_count=405,runtime_count=391,static_cases=36,current_execution_archive_equal=True,
    inherited_checks='Original v1 reviewed Popen ownership/direct wait, alias walk,144 child joins+6 fixtures, prefixture36 selection/XML, empty backend/origin policy and CPU112-115 unchanged.',
    failed_v1_preserved=True,unrun_at_review=True,source_only=True,candidate_imports=False,collection=False,tests=False,probes=False,actual_authority=False)
out=Path(__file__).parent/'harness-v2-review.json';out.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps({'review_sha256':H(out.read_bytes()),'changed_config_fields':len(changed),'source_count':405},sort_keys=True))
