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
        for name in dirs:assert not(Path(base)/name).is_symlink()
        for name in names:
            path=Path(base)/name
            assert path.is_file()and not path.is_symlink()
            files.add(path.relative_to(ROOT).as_posix())
    return files
def archive(prefix,count):
    mapping=obj(prefix+'/sources.json')
    assert type(mapping)is dict and len(mapping)==count
    with tarfile.open(ROOT/prefix/'sources.tar.gz',mode='r:gz')as tar:
        members=tar.getmembers()
        assert len(members)==len({m.name for m in members})==count
        assert all(m.isfile()and not Path(m.name).is_absolute()and '..'not in Path(m.name).parts for m in members)
        assert sum(m.size for m in members)<32*1024*1024
        data={m.name:tar.extractfile(m).read()for m in members}
    assert{n:sha(b)for n,b in data.items()}==mapping
    hand=obj(prefix+'/handoff.json')
    assert hand['source_count']==count
    assert hand['source_archive_sha256']==sha(raw(prefix+'/sources.tar.gz'))
    map_pin=hand.get('source_manifest_sha256',hand.get('source_map_sha256'))
    assert map_pin==sha(raw(prefix+'/sources.json'))
    return mapping,data
def xml(name,failures):
    tree=ET.fromstring(raw(name))
    assert {n:len(list(tree.iter(t)))for n,t in [('tests','testcase'),('failures','failure'),('errors','error'),('skipped','skipped')]}==dict(tests=18,failures=failures,errors=0,skipped=0)
    return [dict(classname=n.attrib['classname'],name=n.attrib['name'])for n in tree.iter('testcase')]

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--current',action='store_true')
    args=parser.parse_args()
    index=obj('files.json');white=obj('whitelist.json');origins=obj('origins.json')
    assert set(index)==closed_files()-{'files.json'}
    assert set(white['files'])==set(index)|{'files.json'}and len(white['files'])==len(set(white['files']))
    assert sum((ROOT/n).stat().st_size for n in index)<64*1024*1024
    for n,d in index.items():assert sha(raw(n))==d,(n,'capsule digest differs')
    assert set(origins['copied'])|set(origins['authored'])==set(index)-{'origins.json','whitelist.json'}
    assert not(set(origins['copied'])&set(origins['authored']))
    for n,item in origins['copied'].items():
        assert item['sha256']==index[n]
        if args.current:
            p=Path(item['path']);assert p.is_file()and not p.is_symlink()
            assert sha(p.read_bytes())==item['sha256'],(n,'current original differs')
    g1,dg1=archive('history/guard-v1',4);g2,dg2=archive('history/guard-v2',5)
    t1,dt1=archive('history/tests-v1',36);t2,dt2=archive('history/tests-v2',37);t3,dt3=archive('history/tests-v3',38)
    assert dg1['qwen_admission_snapshot.py']!=dg2['qwen_admission_snapshot.py']
    assert t1['snapshot_child.py']==t2['snapshot_child.py']!=t3['snapshot_child.py']
    assert [n for n in t1 if t1[n]!=t2[n]]==['test_snapshot_guard.py']
    assert set(t2)-set(t1)=={'successor_history.json'}
    assert {n for n in t2 if t2[n]!=t3[n]}=={'snapshot_child.py','test_snapshot_guard.py','TEST_SCOPE.md'}
    assert set(t3)-set(t2)=={'successor_history_v3.json'}
    runtime={n:d for n,d in t3.items()if n.startswith('runtime/')}
    assert len(runtime)==28 and all(t1[n]==t2[n]==d for n,d in runtime.items())
    assert dt3['runtime/qwen_admission_snapshot.py']==dg2['qwen_admission_snapshot.py']
    assert all(raw('source/'+n)==b for n,b in dt3.items())
    assert dt1['selected_tests.json']==dt2['selected_tests.json']==dt3['selected_tests.json']
    selected=json.loads(dt3['selected_tests.json']);nodes=selected['static_nodeids']
    assert len(nodes)==18 and len(set(nodes))==18

    config=obj('measured/run_config.json');report=obj('measured/report.json');audit=obj('measured/worker-audit.json')
    for name in ('sources-before.json','sources-after.json','execution-after.json'):
        assert raw('measured/'+name)==raw('history/tests-v3/sources.json')
    assert raw('measured/sources.tar.gz')==raw('history/tests-v3/sources.tar.gz')
    assert config['source_count']==38 and config['runtime_count']==28 and config['selected_nodes']==nodes
    assert config['source_manifest_sha256']==sha(raw('measured/sources-before.json'))
    assert config['source_archive_sha256']==sha(raw('measured/sources.tar.gz'))
    assert report['passed']is True and report['source_count']==38 and report['sources_unchanged']is True
    assert report['tests']==dict(tests=18,failures=0,errors=0,skipped=0)
    assert report['child_returncode']==0 and type(report['child_returncode'])is int
    assert report['child_reaped']is True and report['child_group_absent']is True
    assert report['early_parent_records_validated']==18 and report['snapshot_guard_recording_only']is True
    for name in ('numpy_imported','actual_execution_authorized','model_execution','encrypted_execution','backend_context_construction','actual_key_generation','actual_client_role_execution','full_scientific_admission_tested','actual_proofs_present'):
        assert report[name]is False
    assert xml('measured/tests.xml',0)==config['selected_xml_cases']==report['selected_xml_cases']
    assert audit['collected_nodes']==nodes and audit['denied']==audit['loaded_backend']==[]
    assert audit['numpy_imported']is False and audit['scientific_admission_tested']is False
    assert audit['runtime_sources_before']==audit['runtime_sources_after']==audit['runtime_module_origins']=={}
    assert audit['affinity']==[112,113,114,115]and audit['pytest_exit_code']==0
    assert report['config_sha256']==sha(raw('measured/run_config.json'))
    assert report['driver_sha256']==sha(raw('measured/executed_driver.py'))
    assert report['worker_driver_sha256']==sha(raw('measured/worker_driver.py'))
    metadata=obj('measured/owned-pytest-metadata.json')
    regular={n:v for n,v in metadata.items()if v['kind']=='regular'}
    aliases={n:v for n,v in metadata.items()if v['kind']=='owned_pytest_alias'}
    assert len(regular)==72 and len(aliases)==17 and len(regular)+len(aliases)==len(metadata)
    for n,v in regular.items():
        assert sha(raw('measured/owned/'+n))==v['sha256']and len(raw('measured/owned/'+n))==v['bytes']
    for n,v in aliases.items():
        assert not(ROOT/'measured/owned'/n).exists()
        assert n.startswith('pytest-owned/test_snapshot_guard_')and n.endswith('current')
        assert '..'not in Path(v['resolved_target']).parts
        assert (ROOT/'measured/owned'/v['resolved_target']).is_dir()
    records=obj('measured/test-child-terminals.json')
    assert records==report['child_records']and len(records)==18
    scenarios=[n.split('[',1)[1][:-1]for n in nodes]
    assert [r['terminal']['scenario']for r in records]==scenarios
    for r,scenario in zip(records,scenarios,strict=True):
        early=r['parent_terminal'];positive=r['terminal'];result=r['result']
        for label,value in [('parent_terminal_path',early),('terminal_path',positive),('result_path',result)]:
            assert obj('measured/owned/'+r[label])==value
        pid=result['pid'];assert type(pid)is int and pid>0
        assert type(result['pgid'])is int and result['pgid']==pid
        assert early['schema']=='qwen-snapshot-owned-child-terminal-v1'and early['child_result_read']is False
        assert early['scenario']==positive['scenario']==result['scenario']==scenario
        for owner in (early,positive):
            assert type(owner['pid'])is int and owner['pid']==pid
            assert type(owner['exit_code'])is int and owner['exit_code']==0
            assert owner['reaped']is True and owner['group_absent']is True
            assert owner['observed_pgid']is None or(type(owner['observed_pgid'])is int and owner['observed_pgid']==pid)
        assert type(early['owned_session_pgid'])is int and early['owned_session_pgid']==pid
        assert type(positive['child_reported_pgid'])is int and positive['child_reported_pgid']==pid
        assert early['observed_pgid']==positive['observed_pgid']
        assert result['assertions_completed']is True and result['original_validator_identities_restored']is True
        assert result['recording_full_validation_seam']is True and result['scientific_admission_tested']is False
        assert result['actual_proof_instances']is False and result['backend_imported']is False and result['backend_denials']==[]
        assert type(result['graph_type_count'])is int and result['graph_type_count']==13
        assert sha(raw('measured/owned/'+r['log_path']))==r['log_sha256']
        assert sha(raw('measured/owned/'+r['parent_terminal_path']))==r['parent_terminal_sha256']
    retention=obj('failed/files.json');assert len(retention)==44
    for n,d in retention.items():assert sha(raw('failed/'+n))==d
    failed_root=obj('root-review/failed/failed-terminal-review.json')
    assert failed_root['test_outcome']=='FAIL'and failed_root['guard_exercised']is False
    assert failed_root['child_pid_pgid_receipts_available']is False and failed_root['child_group_absence_proved']is False
    assert failed_root['direct_owner']['session_id']==97824 and failed_root['direct_owner']['terminal_tool_chunk']=='9b71cb'
    assert failed_root['direct_owner']['direct_session_reaped']is True and failed_root['direct_owner']['exit_code']==1
    assert failed_root['retention_index_sha256']==sha(raw('failed/files.json'))
    assert xml('failed/evidence/tests.xml',18)==config['selected_xml_cases']
    for name in ('sources-before.json','sources-after.json','execution-after.json'):
        assert raw('failed/evidence/'+name)==raw('history/tests-v2/sources.json')
    assert raw('failed/evidence/sources.tar.gz')==raw('history/tests-v2/sources.tar.gz')
    assert raw('failed/evidence/worker_driver.py')==raw('measured/worker_driver.py')
    review=obj('root-review/passed/terminal-review.json');owner=obj('root-review/passed/direct-owner.json')
    assert review['passed']is True and review['all_child_pids_and_groups_absent']is True
    assert review['outer_worker_pids_and_groups_absent']is True and review['independent_live_observation']is False
    assert review['full_scientific_admission_tested']is False and review['guard_integrated_into_actual_server']is False
    assert review['direct_owner_receipt_sha256']==sha(raw('root-review/passed/direct-owner.json'))
    assert owner['session_id']==91616 and owner['launch_tool_chunk']=='e197dd'and owner['terminal_tool_chunk']=='70406f'
    assert owner['direct_session_reaped']is True and type(owner['exit_code'])is int and owner['exit_code']==0
    assert owner['independent_live_observation']is False
    assert review['counts']==report['tests']and review['elapsed_s']==report['elapsed_s']
    for name,d in review['evidence'].items():assert sha(raw('measured/'+name))==d
    assert review['root_preexecution_review_sha256']==sha(raw('root-review/passed/preexecution-review.json'))
    assert review['root_run_grant_sha256']==sha(raw('root-review/passed/run-grant.json'))
    print(json.dumps(dict(verified=True,mode='current public origins'if args.current else'portable',indexed_files=len(index),
        tests_passed=18,prior_fixture_setup_failures=18,regular_child_files=72,recorded_unfollowed_aliases=17,
        candidate_imported=False,tests_rerun=False,processes_probed=False,scientific_admission_tested=False),sort_keys=True))

if __name__=='__main__':main()
