"""Portable public proof checks only; no runtime imports or original data reads."""
from pathlib import Path
import argparse,ast,hashlib,itertools,json,tarfile
import xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent
sha=lambda raw:hashlib.sha256(raw).hexdigest()
compact=lambda value:(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
pretty=lambda value:(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
load=lambda name:json.loads((HERE/name).read_bytes())

def inventory(root):
    paths=sorted(root.rglob('*'))
    assert all(not p.is_symlink() and (p.is_file() or p.is_dir()) for p in paths)
    return {p.relative_to(root).as_posix():sha(p.read_bytes()) for p in paths if p.is_file()}

def archive(prefix,count):
    raw=(HERE/prefix/'sources.json').read_bytes();expected=json.loads(raw)
    assert raw==compact(expected) and len(expected)==count
    with tarfile.open(HERE/prefix/'sources.tar.gz') as tar:
        members=tar.getmembers()
        assert len(members)==len({m.name for m in members})==count
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        content={m.name:tar.extractfile(m).read() for m in members}
    assert {name:sha(raw) for name,raw in content.items()}==expected
    return expected,content

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--current',action='store_true',help='Hash original frozen source trees only; no runtime imports or model/key data.')
    args=parser.parse_args()
    index=load('index.json');current=inventory(HERE)
    assert set(current)==set(index)|{'index.json'}
    assert all(current[n]==h for n,h in index.items())
    whitelist=load('public_whitelist.json')
    assert len(whitelist['copied_files'])==len({x['destination'] for x in whitelist['copied_files']})
    assert all(current[row['destination']]==row['sha256'] for row in whitelist['copied_files'])
    old,oldbytes=archive(Path('source85-v1'),85)
    new,newbytes=archive(Path('source85-v2'),85)
    measured,body=archive(Path('focused97'),97)
    changed=[n for n in old if old[n]!=new[n]]
    assert changed==['keygen_entry_status.json','keygen_static_policy.json']
    for n in changed:
        assert json.loads(oldbytes[n])==json.loads(newbytes[n])
        assert newbytes[n]==compact(json.loads(newbytes[n])) and oldbytes[n]!=newbytes[n]
        assert body['history/'+n]==oldbytes[n]
        assert (HERE/'source85-v1'/n).read_bytes()==oldbytes[n]
        assert (HERE/'source85-v2'/n).read_bytes()==newbytes[n]
    assert set(measured)==set('runtime/'+n for n in new)|set(n for n in measured if not n.startswith('runtime/'))
    assert {n.removeprefix('runtime/'):h for n,h in measured.items() if n.startswith('runtime/')}==new
    assert body['parent85_sources.json']==(HERE/'source85-v2/sources.json').read_bytes()
    assert len([n for n in measured if not n.startswith('runtime/')])==12
    copies=json.loads(newbytes['keygen_selected_copies.json'])
    assert len(copies)==74 and all(new[n]==old[n]==row['sha256'] for n,row in copies.items())
    assert sorted(json.loads(newbytes['keygen_source_inventory.json'])['files'])==sorted(new)
    for name in ('keygen_entry.py','keygen_output.py','keygen_dependencies.py','keygen_process.py'):
        assert oldbytes[name]==newbytes[name]==body['runtime/'+name]
    assert json.loads(newbytes['keygen_entry_status.json'])['actual_entry_ready'] is False
    c=load('focused97/run_config.json');r=load('focused97/report.json');a=load('focused97/worker-audit.json')
    terminal=load('root/keygen97-terminal-review.json');owner=load('root/keygen97-direct-owner-receipt.json')
    grant=load('root/keygen97-focused-run-grant.json');pre=load('root/keygen97-preexecution-review.json')
    assert c['source_manifest_sha256']==r['source_manifest_sha256']==sha((HERE/'focused97/sources.json').read_bytes())
    assert c['source_archive_sha256']==r['source_archive_sha256']==sha((HERE/'focused97/sources.tar.gz').read_bytes())
    assert (HERE/'focused97/sources-after.json').read_bytes()==(HERE/'focused97/execution-after.json').read_bytes()==compact(measured)
    assert c['source_manifest_encoding']=='canonical sorted compact JSON plus newline'
    assert c['selection_digest_encoding']=='sorted indent=2 JSON plus newline'
    nodes=[]
    for filename in c['selected_files']:
        tree=ast.parse(body[filename])
        for fn in tree.body:
            if not isinstance(fn,ast.FunctionDef) or not fn.name.startswith('test_'):continue
            axes=[]
            for d in reversed(fn.decorator_list):
                assert isinstance(d,ast.Call) and isinstance(d.func,ast.Attribute) and d.func.attr=='parametrize'
                assert len(d.args)==2 and not d.keywords
                axis=ast.literal_eval(d.args[1]);assert type(axis)is list;axes.append(axis)
            suffixes=[''] if not axes else ['['+'-'.join(map(str,row))+']' for row in itertools.product(*axes)]
            nodes.extend(filename+'::'+fn.name+s for s in suffixes)
    assert nodes==c['selected_nodes']==r['selected_nodes']==a['collected_nodes'] and len(nodes)==len(set(nodes))==61
    cases=[dict(classname=n.split('::')[0][:-3],name=n.split('::')[1]) for n in nodes]
    xml=ET.parse(HERE/'focused97/tests.xml').getroot()
    assert [dict(classname=n.attrib['classname'],name=n.attrib['name'])for n in xml.iter('testcase')]==cases==c['selected_xml_cases']==r['selected_xml_cases']
    counts={k:len(list(xml.iter(tag))) for k,tag in [('tests','testcase'),('failures','failure'),('errors','error'),('skipped','skipped')]}
    assert counts==r['tests']==dict(tests=61,failures=0,errors=0,skipped=0)
    assert sha(pretty(nodes))==c['selected_nodes_sha256'] and sha(pretty(cases))==c['selected_xml_cases_sha256']
    assert a['denied']==a['loaded_backend']==[] and a['numpy_imported'] is False
    assert a['runtime_sources_before']==a['runtime_sources_after']==c['runtime_source_expectations']=={}
    assert a['runtime_module_origins']==c['runtime_module_origins']=={}
    assert a['forbidden_roots']==c['forbidden_roots'] and 'numpy' in a['forbidden_roots']
    assert a['affinity']==c['cpu_affinity']==[112,113,114,115]
    assert type(a['pytest_exit_code'])is int and a['pytest_exit_code']==0
    assert r['passed'] is True and r['sources_unchanged'] is True and r['synthetic_only'] is True
    assert type(r['child_returncode'])is int and r['child_returncode']==0 and r['child_reaped'] is True and r['child_group_absent'] is True
    for n in ('actual_execution_authorized','actual_key_generation','backend_context_construction','model_execution','encrypted_execution','actual_client_role_execution','genuine_backend_imports_authorized','capacity_acquisition_authorized'):
        assert r[n] is False
    assert 'ModuleType' in r['scope'] and r['temporary_module_name_doubles']==c['temporary_module_name_doubles']
    live=load('focused97/live-observation.json')
    assert a['pid']==live['worker_pid']==live['worker_pgid']==a['pgid']
    assert terminal['passed'] is True and terminal['direct_exit_code']==0 and terminal['direct_reaped'] is True
    assert terminal['direct_tool_chunk']=='03b09d' and terminal['independent_live_observation'] is False
    assert terminal['process_absence']==dict(outer_group=True,outer_pid=True,worker_group=True,worker_pid=True)
    for name,digest in terminal['evidence_sha256'].items():
        local='sources.json' if name=='sources-before.json' else name
        assert sha((HERE/'focused97'/local).read_bytes())==digest
    assert sha((HERE/'root/root_keygen97_terminal_20261104.py').read_bytes())==terminal['review_script_sha256']
    assert sha((HERE/'root/root_keygen97_review_20261104.py').read_bytes())==pre['review_script_sha256']
    for name,key in [('executed_driver.py','driver_sha256'),('worker_driver.py','worker_driver_sha256'),('run_config.json','config_sha256')]:
        assert sha((HERE/'focused97'/name).read_bytes())==r[key]
    assert grant['authorized'] is True and grant['tests']==61 and grant['source_count']==97
    assert grant['root_review_sha256']==sha((HERE/'root/keygen97-preexecution-review.json').read_bytes())
    assert grant['source_peer_review_sha256']==sha((HERE/'peer/focused/review.json').read_bytes())
    assert grant['harness_peer_review_sha256']==sha((HERE/'peer/focused/harness-review.json').read_bytes())
    assert grant['launch_proposal_sha256']==sha((HERE/'focused97/launch-proposal.json').read_bytes())
    assert sha((HERE/'root/keygen97-direct-owner-receipt.json').read_bytes())=='46e46a2fb3d86d7bee0e7dbb83098145e9f12ee670a7aa3ef07aae45de56d085'
    assert owner['tool_exit_code']==0 and owner['direct_reaped'] is True and owner['session_id'] is None
    assert owner['launch_tool_chunk']=='03b09d' and owner['independent_live_observation'] is False
    assert owner['report_sha256']==sha((HERE/'focused97/report.json').read_bytes())
    assert owner['root_grant_sha256']==sha((HERE/'root/keygen97-focused-run-grant.json').read_bytes())
    assert owner['root_terminal_review_sha256']==sha((HERE/'root/keygen97-terminal-review.json').read_bytes())
    assert load('root/keygen85-peer-metadata-finding.json')['runtime_failure_observed'] is False
    assert load('root/keygen85-v2-encoding-review.json')['runtime4_and74_selected_copies_unchanged'] is True
    if args.current:
        assert inventory(Path('/tmp/qwen-keygen-actual-runtime-20261104'))==old
        assert inventory(Path('/tmp/qwen-keygen-actual-runtime-v2-20261104'))==new
        assert inventory(Path(c['source_root']))==measured
        assert inventory(Path(c['proof_root'])/'execution')==measured
    print(json.dumps(dict(passed=True,indexed_files=len(index),source85_v1=85,source85_v2=85,selected_copies=74,
        focused_sources=97,tests=counts,current_sources_checked=args.current,candidate_imports=False,
        actual_key_generation_authorized=False,independent_live_observation=False),sort_keys=True))

if __name__=='__main__':main()
