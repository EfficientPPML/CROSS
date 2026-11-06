"""Public source/proof verification only. No candidate imports or test execution."""
from pathlib import Path
import argparse,ast,hashlib,json,tarfile
import xml.etree.ElementTree as ET
P=Path(__file__).resolve().parent
sha=lambda raw:hashlib.sha256(raw).hexdigest()
canonical=lambda x:(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def read(n):return json.loads((P/n).read_bytes())
def digest(n):return sha((P/n).read_bytes())
def inventory(root):
    paths=list(root.rglob('*'));assert not any(p.is_symlink()for p in paths)
    return {p.relative_to(root).as_posix():sha(p.read_bytes())for p in paths if p.is_file()}
def archive(name,expected):
    with tarfile.open(P/name)as arc:
        members=arc.getmembers();assert len(members)==len({m.name for m in members})==len(expected)
        assert all(m.isfile()and not Path(m.name).is_absolute()and '..'not in Path(m.name).parts for m in members)
        blobs={m.name:arc.extractfile(m).read()for m in members}
    assert {n:sha(b)for n,b in blobs.items()}==expected
    assert all(Path(n).suffix in('.py','.json','.md','.txt')for n in blobs)
    return blobs
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--current',action='store_true');args=parser.parse_args()
    index=read('files.sha256.json');actual=inventory(P);assert actual.pop('files.sha256.json')and actual==index
    white=read('public-whitelist.json');assert set(white['files'])==set(index)|{'files.sha256.json'}
    assert white['private_or_synthetic_payload_bodies_included']is False
    origins=read('copy-origins.json');assert set(origins)|set(white['generated_files'])==set(white['files'])
    for n,row in origins.items():assert digest(n)==row['sha256']and len((P/n).read_bytes())==row['bytes']
    expected=read('measured/sources-before.json');assert len(expected)==53
    assert canonical(expected)==(P/'measured/sources-before.json').read_bytes()
    assert digest('measured/sources-before.json')=='597ca4c89e66a22608d28d2e67d9234bac37bfee40d84b04139ac727309dee18'
    assert digest('measured/sources.tar.gz')=='d568676329e12379f6b480980ab68d346928945de90832f151bb2eca76e329ea'
    assert expected==read('measured/sources-after.json')==read('measured/execution-after.json')
    blobs=archive('measured/sources.tar.gz',expected)
    for p in (P/'source').iterdir():assert p.read_bytes()==blobs[p.name]
    parent=read('history/source39/sources.json');assert len(parent)==39
    assert digest('history/source39/sources.json')=='da3b9014385331b2efe3b1633cf331f44e69a9853feab687c41183e5b09154d9'
    assert digest('history/source39/sources.tar.gz')=='e3ad62603cc944c19366891f53194415126c7f3e3699ef64bc917aacd20c0b6c'
    parentblobs=archive('history/source39/sources.tar.gz',parent)
    assert all(blobs[n]==b for n,b in parentblobs.items())
    p28=read('source/authority28_v2_sources.json');assert len(p28)==28 and all(parent[n]==h for n,h in p28.items())
    assert digest('source/authority28_v2_sources.json')=='f30d07f8070c778ec0193336e56bcfdf2403a7dd58d96a4e4b327f31f4adfb8d'
    p52=read('history/source52-unrun/sources.json');assert len(p52)==52
    assert digest('history/source52-unrun/sources.json')=='9ed6f0253faffabcfe1fb5fe71c436ceba897e363e93d6ccf326104f609498fc'
    reconstructed=dict(blobs);reconstructed.pop('PREEXECUTION_CORRECTIONS.json')
    correction=read('source/PREEXECUTION_CORRECTIONS.json');assert correction['parent52_map']==p52
    changed=correction['changed'];assert len(changed)==4
    for n,row in changed.items():
        old=(P/'history/source52-unrun/overlaid-files'/n).read_bytes()
        assert sha(old)==row['before']==p52[n]and sha(blobs[n])==row['after']
        reconstructed[n]=old
    assert {n:sha(b)for n,b in reconstructed.items()}==p52
    assert all(p52[n]==h for n,h in parent.items())
    historical=read('history/source52-unrun/handoff.json')
    for n,pin in historical['historical_fixture_pins'].items():assert sha(blobs[n])==pin
    assert sha(blobs['fixture_hooks.patch.txt'])==historical['fixture_patch_sha256']
    assert historical['tests_collected']==historical['tests_executed']==0
    currenthandoff=read('history/source53/handoff.json');assert currenthandoff['imports_collection_execution']is False
    assert currenthandoff['runtime39_unchanged']is True and currenthandoff['changed']==changed
    config=read('harness/run_config.json');oldconfig=read('history/harness-v1-unrun/run_config.json')
    assert {k for k in config if config[k]!=oldconfig[k]}=={'cpu_affinity','source_manifest_encoding','proof_root'}
    assert oldconfig['cpu_affinity']==[112,113,114,115]and config['cpu_affinity']==[108,109,110,111]
    assert oldconfig['source_manifest_encoding']=='canonical sorted compact JSON plus newline'
    assert config['source_manifest_encoding']=='canonical sorted indent=2 JSON plus newline'
    assert (P/'history/harness-v1-unrun/executed_driver.py').read_bytes().replace(digest('history/harness-v1-unrun/run_config.json').encode(),digest('harness/run_config.json').encode())==(P/'harness/executed_driver.py').read_bytes()
    constants={n.targets[0].id:ast.literal_eval(n.value)for n in ast.parse((P/'harness/executed_driver.py').read_bytes()).body if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and n.targets[0].id in('CONFIG_SHA256','WORKER_SHA256')}
    assert constants==dict(CONFIG_SHA256=digest('harness/run_config.json'),WORKER_SHA256=digest('harness/worker_driver.py'))
    proposal=read('harness/launch-proposal.json')
    assert proposal['config_sha256']==digest('harness/run_config.json')and proposal['driver_sha256']==digest('harness/executed_driver.py')and proposal['worker_sha256']==digest('harness/worker_driver.py')
    report=read('measured/report.json');audit=read('measured/worker-audit.json');live=read('measured/live-observation.json')
    assert digest('measured/report.json')=='2d475143212ff559a137670d6b8d0f87b0f60a178b3ceeed5579f4ff7ae3dd6d'
    assert report['passed']is True and report['sources_unchanged']is True and report['source_count']==53
    assert report['child_returncode']==0 and report['child_reaped']is True and report['child_group_absent']is True
    assert report['tests']==dict(tests=95,failures=0,errors=0,skipped=0)
    for n in ('actual_client_role_execution','actual_execution_authorized','actual_key_generation','backend_context_construction','encrypted_execution','model_execution','numpy_imported'):assert report[n]is False
    assert report['synthetic_only']is True
    for k,n in [('driver_sha256','executed_driver.py'),('worker_driver_sha256','worker_driver.py'),('config_sha256','run_config.json')]:assert report[k]==digest('harness/'+n)
    assert report['source_manifest_sha256']==config['source_manifest_sha256']==digest('measured/sources-before.json')
    assert report['source_archive_sha256']==config['source_archive_sha256']==digest('measured/sources.tar.gz')
    assert audit['denied']==audit['loaded_backend']==[]and audit['numpy_imported']is False
    assert audit['forbidden_roots']==config['forbidden_roots']and 'numpy'in audit['forbidden_roots']
    assert audit['runtime_sources_before']==audit['runtime_sources_after']==audit['runtime_module_origins']==config['runtime_source_expectations']==config['runtime_module_origins']=={}
    assert type(audit['pytest_exit_code'])is int and audit['pytest_exit_code']==0
    assert audit['affinity']==config['cpu_affinity']and (audit['pid'],audit['pgid'])==(live['worker_pid'],live['worker_pgid'])==(99602,99602)
    assert (live['outer_pid'],live['outer_pgid'])==(99601,99601)
    selection=read('source/selected_client_tests.json');nodes=config['selected_nodes']
    assert len(nodes)==len(set(nodes))==95
    assert nodes==selection['selected_nodes']==audit['collected_nodes']==report['selected_nodes']
    assert sha(canonical(nodes))==config['selected_nodes_sha256']
    xml=ET.parse(P/'measured/tests.xml').getroot();cases=[dict(classname=c.attrib['classname'],name=c.attrib['name'])for c in xml.iter('testcase')]
    assert len(cases)==95 and all(not list(xml.iter(n))for n in('failure','error','skipped'))
    assert cases==config['selected_xml_cases']==report['selected_xml_cases']==selection['selected_xml_cases']
    assert sha(canonical(cases))==config['selected_xml_cases_sha256']
    assert {n:sum(c['classname']==n[:-3]for c in cases)for n in config['selected_files']}==selection['per_file']=={'test_client_conditional.py':29,'test_client_phase.py':33,'test_client_server.py':33}
    root=read('reviews/root/client-authority53-terminal-review.json');grant=read('reviews/root/client-authority53-focused-run-grant.json')
    assert digest('reviews/root/client-authority53-terminal-review.json')=='e3e8efe8ba3f5151e14954848536c542f8ff3e1983785d4233235f58da7938e5'
    assert root['synthetic_metadata_admitted']is True and root['normal_exit_code']==0 and root['reaped']is True
    assert root['session']==27523 and root['terminal_tool']=='7c7d70'and root['launch_tool']=='07a6fa'
    assert root['original_processes_and_groups_absent']is True and root['tests']==report['tests']
    assert root['actual_role_execution_authorized']is False and root['actual_file_private_ordering_demonstrated']is False
    for field,path in [('report_sha256','measured/report.json'),('owner_sha256','measured/live-observation.json'),('XML_sha256','measured/tests.xml'),('worker_audit_sha256','measured/worker-audit.json'),('grant_sha256','reviews/root/client-authority53-focused-run-grant.json'),('peer_v2_sha256','reviews/security/focused/review-preparation-v2.json'),('source_review_sha256','reviews/root/client-authority53-preexecution-review.json')]:assert root[field]==digest(path)
    external=read('reviews/root/client-authority53-live-observation.json')
    assert external['owner_receipt_sha256']==digest('measured/live-observation.json')and external['session']==27523
    for role in ('outer','worker'):
        row=external['observed'][role];assert row['pid']==live[role+'_pid']and row['pgid']==live[role+'_pgid']and row['affinity']==[108,109,110,111]
    assert grant['one_pure_focused_run_authorized']is True and grant['actual_role_execution_authorized']is False
    assert grant['argv']==proposal['argv']and grant['proposal_sha256']==digest('harness/launch-proposal.json')
    assert grant['peer_review_sha256']==digest('reviews/security/focused/review-preparation-v2.json')
    peer=read('reviews/security/focused/review-preparation-v2.json')
    for n,h in peer['harness_pins'].items():assert digest('harness/'+n)==h
    assert peer['all39_unchanged']is True and peer['tests_executed']is False and peer['selected_response_root_successor']is False
    if args.current:
        assert inventory(Path(config['source_root']))==inventory(Path(config['proof_root'])/'execution')==expected
        assert inventory(Path('/tmp/qwen-client-role-authority-20261104'))==parent
        assert inventory(Path('/tmp/qwen-client-role-authority-tests-20261104'))==p52
        for row in origins.values():assert sha(Path(row['source']).read_bytes())==row['sha256']
    print(json.dumps(dict(passed=True,indexed_files=len(index),source_count=53,unchanged_parent39=True,unrun52_reconstructed=True,tests=95,current=args.current,actual_execution_authorized=False,scope='public source/proof checks only; no imports/tests/body readers'),sort_keys=True))
if __name__=='__main__':main()
