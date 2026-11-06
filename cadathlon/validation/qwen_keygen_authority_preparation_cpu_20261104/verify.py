"""Portable public-evidence verification; no candidate imports or body readers."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import tarfile
import xml.etree.ElementTree as ET

P=Path(__file__).resolve().parent
sha=lambda raw:hashlib.sha256(raw).hexdigest()
pretty=lambda v:(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
compact=lambda v:(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def read(name):return json.loads((P/name).read_bytes())
def digest(name):return sha((P/name).read_bytes())
def inventory(root):
    paths=list(root.rglob('*'));assert not any(p.is_symlink()for p in paths)
    return {p.relative_to(root).as_posix():sha(p.read_bytes())for p in paths if p.is_file()}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--current',action='store_true');args=parser.parse_args()
    index=read('files.sha256.json');actual=inventory(P);assert actual.pop('files.sha256.json')and actual==index
    white=read('public-whitelist.json');assert set(white['files'])==set(index)|{'files.sha256.json'}
    assert white['private_or_synthetic_payload_bodies_included']is False
    origins=read('copy-origins.json')
    for name,row in origins.items():
        raw=(P/name).read_bytes();assert sha(raw)==row['sha256']and len(raw)==row['bytes']
    assert set(origins)|set(white['generated_files'])==set(white['files'])
    expected=read('measured/sources-before.json');assert len(expected)==39
    assert compact(expected)==(P/'measured/sources-before.json').read_bytes()
    assert expected==read('measured/sources-after.json')==read('measured/execution-after.json')
    assert digest('measured/sources-before.json')=='3b2ddbcde655dc3485a36b160ed97f166ef6aad69512dd824bc409545a13db48'
    assert digest('measured/sources.tar.gz')=='532c9f51972f851f535f373115022876fa2553af12f21ed84617de83e0f877b2'
    with tarfile.open(P/'measured/sources.tar.gz')as archive:
        members=archive.getmembers();assert len(members)==len({m.name for m in members})==39
        assert all(m.isfile()and not Path(m.name).is_absolute()and '..'not in Path(m.name).parts for m in members)
        blobs={m.name:archive.extractfile(m).read()for m in members}
    assert {n:sha(raw)for n,raw in blobs.items()}==expected
    assert all(Path(n).suffix in('.py','.json','.md')for n in blobs)
    for file in (P/'source').iterdir():assert file.is_file()and file.read_bytes()==blobs[file.name]
    parent=read('history/parent28_sources.json');assert len(parent)==28
    assert digest('history/parent28_sources.json')=='d67a5df7aff7cd6560e88bd7731cf124ceefba18f4a2c83c443aa1ad1aa72ff8'
    assert all(expected[n]==h for n,h in parent.items())and len(set(expected)-set(parent))==11
    assert blobs['parent28_sources.json']==(P/'history/parent28_sources.json').read_bytes()
    source_handoff=read('history/source28-handoff.json');test_handoff=read('history/test39-handoff.json')
    assert source_handoff['source_map']==parent and test_handoff['source_map']==expected
    assert source_handoff['capture_script_sha256']==digest('history/source28-capture.py')
    assert test_handoff['capture_script_sha256']==digest('history/test39-capture.py')
    assert test_handoff['all_parent28_unchanged']is True and test_handoff['static_expected_count']==48
    tree=ast.parse(blobs['keygen_authority.py']);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='create_actual_keygen_process')
    assert len(fn.body)==1 and isinstance(fn.body[0],ast.Raise)and fn.body[0].exc.func.id=='PermissionError'
    sets={n.targets[0].id:set(ast.literal_eval(n.value))for n in tree.body if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and isinstance(n.value,ast.Set)}
    schema=json.loads(blobs['boundary.schema.json'])
    for label,key in {'actual_intent':'INTENT_FIELDS','expected_launch_pins':'LAUNCH_PIN_FIELDS','grant':'GRANT_FIELDS','expected_ready_pins':'READY_PIN_FIELDS','positive_report':'REPORT_FIELDS','outer':'OUTER_FIELDS','root_readiness':'ROOT_FIELDS'}.items():
        row=schema['$defs'][label];assert row['additionalProperties']is False and set(row['required'])==set(row['properties'])==sets[key]
    report=read('measured/report.json');config=read('harness/run_config.json');audit=read('measured/worker-audit.json')
    owner=read('measured/owner-terminal.json');live=read('measured/live-observation.json');handoff=read('measured/terminal-handoff.json')
    assert digest('measured/report.json')=='c022c5917868ce0666dcd2024024bedf7a16c2bdcf121587017d4f2b8acbf54f'
    assert digest('measured/owner-terminal.json')=='55faf18e48c75dda9c788d9d0fe5d66865d4189db0c578d7762413fa77e69708'
    assert digest('measured/terminal-handoff.json')=='53037bd05667d42c437fee628e64bef2d358ab75f0ca8cd2d1f0d60b83909592'
    assert report['passed']is True and report['sources_unchanged']is True and report['source_count']==39
    assert report['tests']==dict(tests=48,failures=0,errors=0,skipped=0)==owner['tests']
    assert report['child_returncode']==0 and report['child_reaped']is True and report['child_group_absent']is True
    for n in ('actual_execution_authorized','actual_key_generation','backend_context_construction','encrypted_execution','model_execution','numpy_imported'):assert report[n]is False
    assert report['synthetic_only']is True and config['synthetic_only']is True
    assert owner['owner_session']==handoff['owner_session']==22566 and owner['owner_terminal_chunk']==handoff['owner_terminal_chunk']=='c646b3'
    assert owner['owner_tool_exit_code']==handoff['exit_code']==0 and owner['owner_reaped']is True and handoff['reaped']is True
    assert all(row['pid_absent']and row['group_absent']for row in owner['process_states'].values())
    assert set(owner['process_states'])=={str(live['outer_pid']),str(live['worker_pid'])}
    owner_locations={name:'harness/'+name for name in ('executed_driver.py','worker_driver.py','run_config.json','launch-proposal.json')}
    owner_locations['source-review.json']='reviews/security/harness-source-review.json'
    for name,h in owner['files'].items():assert digest(owner_locations.get(name,'measured/'+name))==h
    assert handoff['owner_receipt_sha256']==digest('measured/owner-terminal.json')and handoff['report_sha256']==digest('measured/report.json')
    assert handoff['owner_script_sha256']==digest('measured/owner_terminal_review.py')
    assert report['source_manifest_sha256']==config['source_manifest_sha256']==handoff['source_map_sha256']==digest('measured/sources-before.json')
    assert report['source_archive_sha256']==config['source_archive_sha256']==handoff['source_archive_sha256']==digest('measured/sources.tar.gz')
    assert report['driver_sha256']==digest('harness/executed_driver.py')and report['worker_driver_sha256']==digest('harness/worker_driver.py')and report['config_sha256']==digest('harness/run_config.json')
    assert audit['denied']==audit['loaded_backend']==[]and audit['numpy_imported']is False
    assert 'numpy'in audit['forbidden_roots']and audit['forbidden_roots']==config['forbidden_roots']
    assert audit['runtime_sources_before']==audit['runtime_sources_after']==audit['runtime_module_origins']==config['runtime_source_expectations']==config['runtime_module_origins']=={}
    assert type(audit['pytest_exit_code'])is int and audit['pytest_exit_code']==0
    assert audit['affinity']==[112,113,114,115]and audit['pid']==live['worker_pid']and audit['pgid']==live['worker_pgid']
    nodes=config['selected_nodes'];assert len(nodes)==len(set(nodes))==48
    assert nodes==audit['collected_nodes']==report['selected_nodes']==test_handoff['ordered_nodeids']
    assert sha(pretty(nodes))==config['selected_nodes_sha256']
    xml=ET.parse(P/'measured/tests.xml').getroot();cases=[dict(classname=c.attrib['classname'],name=c.attrib['name'])for c in xml.iter('testcase')]
    assert len(cases)==48 and cases==config['selected_xml_cases']==report['selected_xml_cases']
    assert sha(pretty(cases))==config['selected_xml_cases_sha256']and all(not list(xml.iter(n))for n in ('failure','error','skipped'))
    assert {n:sum(c['classname']==n[:-3]for c in cases)for n in config['selected_files']}==test_handoff['group_counts']=={'test_keygen_launch.py':17,'test_keygen_readiness.py':25,'test_keygen_retention.py':6}
    race=read('measured/late-live-observation-race.json');assert race['proof_test_failure']is False and race['source_or_proof_mutated']is False
    root=read('reviews/root/keygen-terminal-review.json');assert digest('reviews/root/keygen-terminal-review.json')=='dd38d9c1d95842ab6e60d0cc134f744b8c54bc0f704f826c89dd971ee0d81563'
    assert root['passed']is True and root['source_count']==39 and root['parent_count']==28 and root['tests']==report['tests']
    assert root['independent_root_live_observation']is False and all(r['live']is False for r in read('reviews/root/keygen-live-observation.json')['observations'])
    for field,path in [('report_sha256','measured/report.json'),('owner_terminal_sha256','measured/owner-terminal.json'),('handoff_sha256','measured/terminal-handoff.json'),('audit_sha256','measured/worker-audit.json'),('xml_sha256','measured/tests.xml'),('script_sha256','reviews/root/verify_keygen_terminal.py'),('launch_review_sha256','reviews/root/keygen-launch-review.json')]:assert root[field]==digest(path)
    launch=read('reviews/root/keygen-launch-review.json');assert launch['passed']and launch['one_synthetic_test_run_authorized']and launch['actual_role_authorized']is False
    for field,name in [('config_sha256','run_config.json'),('driver_sha256','executed_driver.py'),('worker_sha256','worker_driver.py'),('proposal_sha256','launch-proposal.json')]:assert launch[field]==digest('harness/'+name)
    assert launch['source_review_sha256']==digest('reviews/root/source-review.json')and launch['peer_review_sha256']==digest('reviews/security/harness-source-review.json')
    peer=read('reviews/cnn/terminal-review.json');assert peer['passed']and peer['tests']==report['tests']and peer['review_script_sha256']==digest('reviews/cnn/terminal-review.py')
    assert peer['report_sha256']==digest('measured/report.json')and peer['owner_receipt_sha256']==digest('measured/owner-terminal.json')
    assert read('reviews/security/source28-review.json')['source_hashes']==parent
    if args.current:
        assert inventory(Path(config['source_root']))==inventory(Path(config['proof_root'])/'execution')==expected
        assert inventory(Path('/tmp/qwen-keygen-authority-boundary-20261104'))==parent
        for row in origins.values():assert sha(Path(row['source']).read_bytes())==row['sha256']
    print(json.dumps(dict(passed=True,indexed_files=len(index),source_count=39,unchanged_parent_sources=28,tests=48,current=args.current,actual_execution_authorized=False,scope='Public source/proof verification only; no test or candidate execution/body descriptor follows.'),sort_keys=True))
if __name__=='__main__':main()
