"""Portable stdlib evidence checks only. Never imports or runs measured code."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def raw(name):return (ROOT/name).read_bytes()
def obj(name):return json.loads(raw(name))
def closed_files():
    files=set()
    for base,dirs,names in os.walk(ROOT,followlinks=False):
        for name in dirs:assert not (Path(base)/name).is_symlink()
        for name in names:
            path=Path(base)/name
            assert path.is_file() and not path.is_symlink()
            files.add(path.relative_to(ROOT).as_posix())
    return files
def archive(prefix,count):
    mapping=obj(prefix+'/sources.json')
    assert type(mapping) is dict and len(mapping)==count
    with tarfile.open(ROOT/prefix/'sources.tar.gz',mode='r:gz') as tar:
        members=tar.getmembers()
        assert len(members)==len({m.name for m in members})==count
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        assert sum(m.size for m in members)<32*1024*1024
        data={m.name:tar.extractfile(m).read() for m in members}
    assert {n:sha(b) for n,b in data.items()}==mapping
    hand=obj(prefix+'/handoff.json');assert hand['source_count']==count
    map_pin=hand.get('source_manifest_sha256',hand.get('source_map_sha256'))
    tar_pin=hand.get('source_archive_sha256')
    if map_pin is None:map_pin=hand['source_map']['sha256']
    if tar_pin is None:tar_pin=hand['source_archive']['sha256']
    assert map_pin==sha(raw(prefix+'/sources.json')) and tar_pin==sha(raw(prefix+'/sources.tar.gz'))
    return mapping,data

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--current',action='store_true')
    args=parser.parse_args()
    index=obj('files.json');white=obj('whitelist.json');origins=obj('origins.json')
    assert set(index)==closed_files()-{'files.json'}
    assert set(white['files'])==set(index)|{'files.json'} and len(white['files'])==len(set(white['files']))
    assert sum((ROOT/n).stat().st_size for n in index)<64*1024*1024
    for n,d in index.items():assert sha(raw(n))==d,(n,'capsule digest differs')
    assert set(origins['copied'])|set(origins['authored'])==set(index)-{'origins.json','whitelist.json'}
    assert not (set(origins['copied'])&set(origins['authored']))
    for n,item in origins['copied'].items():
        assert item['sha256']==index[n]
        if args.current:
            p=Path(item['path']);assert p.is_file() and not p.is_symlink()
            assert sha(p.read_bytes())==item['sha256'],(n,'current original differs')
    d1,bd1=archive('history/device-v1',5);d2,bd2=archive('history/device-v2',6)
    t1,bt1=archive('history/tests-v1',49);t2,bt2=archive('history/tests-v2',50)
    assert {n for n in d1 if d1[n]!=d2[n]}=={'API.md','qwen_device_capacity.py'}
    assert set(d2)-set(d1)=={'successor_history.json'}
    assert {n for n in t1 if t1[n]!=t2[n]}=={'device_child.py','TEST_SCOPE.md'}
    assert set(t2)-set(t1)=={'successor_history.json'}
    runtime={n[8:]:d for n,d in t2.items() if n.startswith('runtime/')}
    assert len(runtime)==34 and all(t1['runtime/'+n]==d for n,d in runtime.items())
    for n in ('qwen_device_policy.py','qwen_device_capacity.py'):assert runtime[n]==d2[n]
    for n in ('API.md','provenance.json','source_refs.json','successor_history.json'):assert t2['device_metadata/'+n]==d2[n]
    assert all(raw('source/'+n)==b for n,b in bt2.items())
    assert bt1['selected_tests.json']==bt2['selected_tests.json']
    selected=json.loads(bt2['selected_tests.json']);nodes=selected['static_nodeids']
    assert len(nodes)==len(set(nodes))==30
    assert obj('source/runtime_sources.json')==runtime

    config=obj('measured/run_config.json');report=obj('measured/report.json');audit=obj('measured/worker-audit.json')
    for name in ('sources-before.json','sources-after.json','execution-after.json'):
        assert raw('measured/'+name)==raw('history/tests-v2/sources.json')
    assert raw('measured/sources.tar.gz')==raw('history/tests-v2/sources.tar.gz')
    assert config['source_count']==50 and config['runtime_count']==34 and config['selected_nodes']==nodes
    assert config['runtime_source_expectations']==runtime
    assert config['source_manifest_sha256']==sha(raw('measured/sources-before.json'))
    assert config['source_archive_sha256']==sha(raw('measured/sources.tar.gz'))
    assert report['passed'] is True and report['source_count']==50 and report['sources_unchanged'] is True
    assert report['tests']==dict(tests=30,failures=0,errors=0,skipped=0)
    assert type(report['child_returncode']) is int and report['child_returncode']==0
    assert report['child_reaped'] is True and report['child_group_absent'] is True
    assert report['early_parent_records_validated']==30 and report['device_component_recording_only'] is True
    assert report['temporary_recording_jax_children']==17 and report['retained_owned_files']==141
    for name in ('numpy_imported','actual_execution_authorized','model_execution','encrypted_execution','backend_context_construction','actual_key_generation','actual_client_role_execution','full_scientific_admission_tested','actual_proofs_present','actual_device_initialized','genuine_capacity_measured','observed_child_module_origin_table'):
        assert report[name] is False
    tree=ET.fromstring(raw('measured/tests.xml'))
    assert {n:len(list(tree.iter(t))) for n,t in [('tests','testcase'),('failures','failure'),('errors','error'),('skipped','skipped')]}==report['tests']
    xml_cases=[dict(classname=n.attrib['classname'],name=n.attrib['name']) for n in tree.iter('testcase')]
    assert xml_cases==config['selected_xml_cases']==report['selected_xml_cases']
    assert audit['collected_nodes']==nodes and audit['denied']==audit['loaded_backend']==[]
    assert audit['numpy_imported'] is False and audit['scientific_admission_tested'] is False
    assert audit['runtime_sources_before']==audit['runtime_sources_after']==runtime
    assert audit['runtime_module_origins']=={}
    assert audit['affinity']==[112,113,114,115] and audit['pytest_exit_code']==0
    for field,name in [('config_sha256','run_config.json'),('driver_sha256','executed_driver.py'),('worker_driver_sha256','worker_driver.py')]:assert report[field]==sha(raw('measured/'+name))
    metadata=obj('measured/owned-pytest-metadata.json')
    regular={n:v for n,v in metadata.items() if v['kind']=='regular'}
    aliases={n:v for n,v in metadata.items() if v['kind']=='owned_pytest_alias'}
    assert len(regular)==120 and len(aliases)==21 and len(regular)+len(aliases)==len(metadata)
    for n,v in regular.items():
        assert sha(raw('measured/owned/'+n))==v['sha256'] and len(raw('measured/owned/'+n))==v['bytes']
    for n,v in aliases.items():
        assert not (ROOT/'measured/owned'/n).exists()
        assert n.startswith('pytest-owned/test_device_component_') and n.endswith('current')
        assert not Path(v['resolved_target']).is_absolute() and '..' not in Path(v['resolved_target']).parts
        assert (ROOT/'measured/owned'/v['resolved_target']).is_dir()
    records=obj('measured/test-child-terminals.json')
    assert records==report['child_records'] and len(records)==30
    scenarios=[n.split('[',1)[1][:-1] for n in nodes]
    assert [r['terminal']['scenario'] for r in records]==scenarios
    temporary_count=0;joined=set()
    for r,scenario in zip(records,scenarios,strict=True):
        early=r['parent_terminal'];positive=r['terminal'];result=r['result']
        for label,value in [('parent_terminal_path',early),('terminal_path',positive),('result_path',result)]:
            assert obj('measured/owned/'+r[label])==value
            joined.add(r[label])
        joined.add(r['log_path'])
        pid=result['pid'];assert type(pid) is int and pid>0
        assert type(result['pgid']) is int and result['pgid']==pid
        assert early['schema']=='qwen-device-component-owned-child-v1' and early['child_result_read'] is False
        assert early['scenario']==positive['scenario']==result['scenario']==scenario
        for owner in (early,positive):
            assert type(owner['pid']) is int and owner['pid']==pid
            assert type(owner['exit_code']) is int and owner['exit_code']==0
            assert owner['reaped'] is True and owner['group_absent'] is True
            assert owner['observed_pgid'] is None or (type(owner['observed_pgid']) is int and owner['observed_pgid']==pid)
        assert type(early['owned_session_pgid']) is int and early['owned_session_pgid']==pid
        assert type(positive['child_reported_pgid']) is int and positive['child_reported_pgid']==pid
        assert early['observed_pgid']==positive['observed_pgid']
        for field in ('assertions_completed','original_phase_identities_restored','original_resource_identities_preserved','profile_observer_restored','recording_phase_validation','resource_validator_is_real','temporary_jax_removed'):assert result[field] is True
        for field in ('actual_policy_or_grant','device_sources_modified','genuine_backend_imported','genuine_capacity_measured','scientific_admission_tested'):assert result[field] is False
        assert result['backend_denials']==[]
        assert type(result['resource_calls']) is int and type(result['logical_calls']) is int
        assert result['resource_calls']==result['logical_calls']
        if scenario in ('document_role_order','document_noncanonical','document_source_phase_join'):assert result['resource_calls']==0
        else:assert result['resource_calls']>0
        expected_fake=not scenario.startswith('document_') and scenario not in ('lease_failure','binding_failure','reentry_caught','reentry_uncaught','initialization_failure')
        assert result['temporary_recording_jax_used'] is expected_fake
        temporary_count+=int(result['temporary_recording_jax_used'])
        assert sha(raw('measured/owned/'+r['log_path']))==r['log_sha256']
        assert sha(raw('measured/owned/'+r['parent_terminal_path']))==r['parent_terminal_sha256']
    assert temporary_count==17 and joined==set(regular)

    review=obj('root-review/terminal-review.json');owner=obj('root-review/direct-owner.json')
    assert review['passed'] is True and review['all_child_pids_and_groups_absent'] is True
    assert review['outer_worker_pids_and_groups_absent'] is True and review['independent_live_observation'] is False
    assert review['full_scientific_admission_tested'] is False and review['device_integrated_into_actual_server'] is False
    assert review['direct_owner_receipt_sha256']==sha(raw('root-review/direct-owner.json'))
    assert owner['session_id']==24307 and owner['launch_tool_chunk']=='ab0197' and owner['terminal_tool_chunk']=='6454a9'
    assert owner['direct_session_reaped'] is True and type(owner['exit_code']) is int and owner['exit_code']==0
    assert owner['independent_live_observation'] is False
    assert review['counts']==report['tests'] and review['elapsed_s']==report['elapsed_s']
    for name,d in review['evidence'].items():assert sha(raw('measured/'+name))==d
    assert review['root_preexecution_review_sha256']==sha(raw('root-review/preexecution-review.json'))
    assert review['root_run_grant_sha256']==sha(raw('root-review/run-grant.json'))
    assert review['script_sha256']==sha(raw('root-review/scripts/root_device_terminal_20261104.py'))
    peer=obj('peer-review/terminal/review.json')
    assert peer['passed'] is True and peer['tests']==report['tests']
    assert peer['source_count']==50 and peer['runtime_count']==34 and peer['unfollowed_aliases']==21
    assert peer['temporary_source_owned_jax_children']==17
    assert peer['reviewer_waited'] is False and peer['reviewer_live_observation'] is False
    for n,d in peer['child_hashes'].items():assert regular[n]['sha256']==d
    root_pins=obj('peer-review/terminal/root-receipt-pins.json')['files']
    origin_by_path={v['path']:v['sha256'] for v in origins['copied'].values()}
    assert all(origin_by_path[p]==d for p,d in root_pins.items())
    cnn=obj('peer-review/tests-v2/review.json')
    assert cnn['reviewer_history']['initial_source_check_exit']==1 and cnn['tests_executed'] is False
    assert cnn['reviewer_history']['initial_script_sha256']==sha(raw('peer-review/tests-v2/review_source_initial.py'))
    assert cnn['review_script_sha256']==sha(raw('peer-review/tests-v2/review_source.py'))
    print(json.dumps(dict(verified=True,mode='current public origins' if args.current else 'portable',indexed_files=len(index),
        tests_passed=30,source_files=50,runtime_files=34,regular_child_files=120,recorded_unfollowed_aliases=21,
        temporary_recording_jax_children=17,candidate_imported=False,tests_rerun=False,processes_probed=False,
        genuine_capacity_measured=False,scientific_admission_tested=False),sort_keys=True))

if __name__=='__main__':main()
