"""Stdlib-only public capsule readback. Never imports or runs candidate code."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import tarfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
sha = lambda raw: hashlib.sha256(raw).hexdigest()
def read(name): return (ROOT/name).read_bytes()
def doc(name): return json.loads(read(name))
def mapping(raw):
    value=json.loads(raw)
    assert raw==(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    return value
def tar_bytes(name):
    with tarfile.open(ROOT/name,'r:gz') as t:
        members=t.getmembers()
        assert len(members)==len({m.name for m in members})
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        return {m.name:t.extractfile(m).read() for m in members}
def pins(files): return {name:sha(raw) for name,raw in files.items()}

def verify(current=False):
    wl=doc('WHITELIST.json')['files']
    assert len(wl)==len(set(wl)) and wl==sorted(wl)
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in wl)
    assert not any(p.is_symlink() for p in ROOT.rglob('*'))
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()}
    assert actual==set(wl)
    indexed={}
    for line in read('SHA256SUMS').decode().splitlines():
        digest,name=line.split('  ',1)
        assert name not in indexed and len(digest)==64 and sha(read(name))==digest,name
        indexed[name]=digest
    assert set(indexed)==set(wl)-{'SHA256SUMS'}
    origins=doc('copy_origins.json')['files']
    assert set(origins)<=set(wl)
    for name,item in origins.items():
        assert sha(read(name))==item['sha256'] and len(read(name))==item['bytes'],name
        assert ('source' in item) != ('source_archive' in item)
    important={
        'sources/404/sources.json':'9fa73f2f59e6bef71a2de475a09e0aaccb25d56375f7a06452cac7b75e4afc1c',
        'sources/404/sources.tar.gz':'bbc4c0bd3ce4bc6a1cd0efa10a7afc799e280887132869b846849fee10b2e713',
        'sources/405/sources.json':'7c4f95e68dabba24ee489bfe05ddd77b5de0ca5e7ba1a3e77139ef57a410bbd4',
        'sources/405/sources.tar.gz':'7814e934402820f57da5084c6cefc4d0a924cc5e68a06c219bdeacd47d03ee59',
        'sources/405-pre-receipt/sources.json':'701d3911a7c7542c60ec304e8674d34c1da216f8d924de82a9cc955befff4b3e',
        'sources/405-pre-receipt/sources.tar.gz':'8fb5450802f5051238cdb7fe1b16a0137b4fe14dcdf662427052bc88a2d720ec',
        'proof/v2/report.json':'98c2642ab0cdc5dd5e7871c3d55bd6b8ff7b1da3e7b76e6e3de29db06f56afbb',
        'reviews/root-tests/failed-terminal-v1.json':'5505f782e2850765d27940190332fb0c8596622207c406832abd16df75ff0c49',
        'reviews/root-tests/terminal-v2-review.json':'33ba5f07ea82aa1c4c406af25c96ff57c2a374eb98bac0f0a49fb7d48a40da1d',
        'reviews/security-terminal/review.json':'81c8494c1b76f933ce47ce008292935fb56e2a58dc72cb9ee6c8e601f854f26d',
    }
    for name,digest in important.items():assert sha(read(name))==digest,name
    archives={}
    for version,count in [('404',404),('405',405),('405-pre-receipt',405)]:
        m=mapping(read('sources/'+version+'/sources.json'))
        files=tar_bytes('sources/'+version+'/sources.tar.gz')
        assert len(m)==count and pins(files)==m
        archives[version]=files
    old,new,pre=archives['404'],archives['405'],archives['405-pre-receipt']
    assert set(new)-set(old)=={'fixture_successor_history.json'} and set(old)<=set(new)
    assert [n for n in old if old[n]!=new[n]]==['integration_fixture.py']
    assert set(new)==set(pre) and [n for n in new if new[n]!=pre[n]]==['fixture_successor_history.json']
    assert pre['fixture_successor_history.json']==read('sources/405-pre-receipt/fixture_successor_history.json')
    assert old['selected_tests.json']==new['selected_tests.json']
    for n in ['integration_cases.py','integration_checks.py','integration_child.py','test_snapshot_integration.py','conftest.py']:
        assert old[n]==new[n]==read('selected-tests/'+n)
    for name in origins:
        if name.startswith('selected-tests/'):
            assert read(name)==new[name[len('selected-tests/'):]]
    runtime={n[8:]:raw for n,raw in new.items() if n.startswith('runtime/')}
    assert len(runtime)==391 and runtime=={n[8:]:raw for n,raw in old.items() if n.startswith('runtime/')}
    summary=doc('summary.json')
    reconstructed={}
    for count,spec in summary['runtime_lineage'].items():
        expected=mapping(read(spec['source_map']))
        files={n:read(spec['overrides'][n]) if n in spec['overrides'] else runtime[n] for n in expected}
        assert len(files)==int(count) and pins(files)==expected
        assert spec['historical_archive_bundled'] is False and spec['historical_archive_hash_offline_verified'] is False
        reconstructed[count]=files
    assert reconstructed['391']==runtime
    assert sum(reconstructed['390'][n]==runtime[n] for n in reconstructed['390'])==386
    assert sum(reconstructed['387'][n]==runtime[n] for n in reconstructed['387'])==384
    assert sha(runtime['qwen_admission_snapshot.py'])==summary['literal_guard_sha256']=='8790711a1a402f4c21ef444ccdb76f773b4c62cd334f5be07a49d2cd4535b395'
    for n in ['qwen_server_resource.py','qwen_server_entry_status.json']:
        assert reconstructed['387'][n]==reconstructed['390'][n]==runtime[n]
    for name in origins:
        if name.startswith('selected-runtime/'):
            assert read(name)==runtime[name[len('selected-runtime/'):]]
    # Parse-only normalized delegate correction; never executes the fixture.
    before=ast.parse(old['integration_fixture.py']);after=ast.parse(new['integration_fixture.py'])
    def capture(tree):
        return next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='capture_runtime')
    method=capture(after)
    delegates=[n for n in method.body if isinstance(n,ast.FunctionDef)]
    assert [n.name for n in delegates]==['source_check','capacity_check']
    for node,target in zip(delegates,['full_boundary','capacity']):
        assert len(node.body)==1 and isinstance(node.body[0],ast.Return)
        call=node.body[0].value
        assert isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute)
        assert isinstance(call.func.value,ast.Name) and call.func.value.id=='self' and call.func.attr==target
        assert call.args==[] and call.keywords==[]
    method.body=[n for n in method.body if n not in delegates]
    class Undo(ast.NodeTransformer):
        def visit_Name(self,node):
            if isinstance(node.ctx,ast.Load) and node.id in ('source_check','capacity_check'):
                return ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr={'source_check':'full_boundary','capacity_check':'capacity'}[node.id],ctx=ast.Load())
            return node
    Undo().visit(method)
    assert ast.dump(before,include_attributes=False)==ast.dump(after,include_attributes=False)

    for version,src,failures,passes in [('v1','404',25,11),('v2','405',0,36)]:
        prefix='proof/'+version+'/'
        c=doc(prefix+'run_config.json');h=doc(prefix+'handoff.json')
        for name,digest in h['files'].items():
            location='sources/'+src+'/sources.tar.gz' if name=='sources.tar.gz' else prefix+name
            assert sha(read(location))==digest,location
        assert sha(read(prefix+'executed_driver.py'))==h['driver_sha256']
        assert sha(read(prefix+'worker_driver.py'))==h['worker_sha256']
        assert sha(read(prefix+'run_config.json'))==h['config_sha256']
        assert c['source_count']==int(src) and c['runtime_count']==391 and c['cpu_affinity']==[112,113,114,115]
        assert c['runtime_source_expectations']==pins(runtime) and c['runtime_module_origins']=={}
        nodes=c['selected_nodes'];assert len(nodes)==36 and nodes==json.loads(new['selected_tests.json'])['static_nodeids']
        for mp in ['sources-before.json','sources-after.json','execution-after.json']:
            assert read(prefix+mp)==read('sources/'+src+'/sources.json')
        tree=ET.fromstring(read(prefix+'tests.xml'))
        counts={k:len(list(tree.iter(t))) for k,t in [('tests','testcase'),('failures','failure'),('errors','error'),('skipped','skipped')]}
        assert counts==dict(tests=36,failures=failures,errors=0,skipped=0)
        assert [dict(classname=n.attrib['classname'],name=n.attrib['name']) for n in tree.iter('testcase')]==c['selected_xml_cases']
        audit=doc(prefix+'worker-audit.json')
        assert audit['collected_nodes']==nodes and audit['denied']==audit['loaded_backend']==[]
        assert audit['runtime_sources_before']==audit['runtime_sources_after']==pins(runtime)
        assert audit['runtime_module_origins']=={} and audit['numpy_imported'] is False
        assert audit['pytest_exit_code']==(1 if version=='v1' else 0)
        grant=doc('reviews/root-tests/'+('run-grant.json' if version=='v1' else 'run-v2-grant.json'))
        assert grant['authorized'] is True and grant['argv']==h['argv']
        for n in ['source_count','source_manifest_sha256','source_archive_sha256','config_sha256','driver_sha256','worker_sha256']:
            assert grant[n]==h[n]
        owner=doc('reviews/root-tests/'+('direct-owner-failed-v1.json' if version=='v1' else 'direct-owner-v2.json'))
        assert owner['root_owned_direct_completion_and_reap'] is True and owner['exit_code']==(1 if version=='v1' else 0)
        live=doc(prefix+'live-observation.json')
        assert live['worker_pid']==audit['pid']==audit['pgid']==owner['worker_pid']
        assert live['outer_pid']==owner['outer_pid']
        child_spec=summary['children'][version];evidence=doc(child_spec['pin_table'])
        regular=evidence['regular_files'];assert len(regular)==child_spec['regular_count']
        assert len(evidence['unfollowed_aliases'])==15
        for name,item in regular.items():
            location='children/'+version+'/'+name
            assert len(read(location))==item['bytes'] and sha(read(location))==item['sha256']
        assert sum(Path(n).name=='tiny-public-proof.json' for n in regular)==6
        parents=[n for n in regular if n.endswith('/parent-terminal.json')]
        results=[n for n in regular if n.endswith('/result.json')]
        assert len(parents)==36 and len(results)==passes
        expected_cases=[n.split('[',1)[1][:-1] for n in nodes]
        seen=[];codes=[]
        for name in parents:
            parent=doc('children/'+version+'/'+name);pid=parent['pid']
            assert type(pid)is int and pid>0 and parent['owned_session_pgid']==pid
            assert parent['observed_pgid'] is None or (type(parent['observed_pgid'])is int and parent['observed_pgid']==pid)
            assert parent['reaped'] is parent['group_absent'] is True and parent['child_result_read'] is False
            seen.append(parent['scenario']);codes.append(parent['exit_code'])
        assert set(seen)==set(expected_cases) and codes.count(0)==passes and codes.count(1)==failures
        for name in results:
            result=doc('children/'+version+'/'+name)
            assert type(result['pid'])is int and type(result['pgid'])is int and result['pid']==result['pgid']
            for flag in ['assertions_completed','original_main_restored','temporary_names_removed','profile_restored','declared_exports_restored','runtime_sources_unchanged','definition_only_entry']:
                assert result[flag] is True
            for flag in ['genuine_cli_bootstrap','genuine_backend_imported','scientific_admission','capacity_measurement','model_key_ciphertext_body_reads','actual_server_execution']:
                assert result[flag] is False
            assert result['backend_denials']==[]
            if 'replacement' in result['counters']:assert result['counters']['replacement']==0
        assert sum(Path(n).name=='child-terminal.json' for n in regular)==passes
        assert sum(Path(n).name=='child.log' for n in regular)==36
    assert 'proof/v1/report.json' not in wl
    report=doc('proof/v2/report.json')
    assert report['passed'] is True and report['tests']==dict(tests=36,failures=0,errors=0,skipped=0)
    assert report['sources_unchanged'] and report['child_reaped'] and report['child_group_absent']
    assert report['child_returncode']==0 and len(report['child_records'])==36
    root_terminal=doc('reviews/root-tests/terminal-v2-review.json')
    assert root_terminal['passed'] and root_terminal['report_sha256']==sha(read('proof/v2/report.json'))
    assert len(root_terminal['all38_PIDs_and_groups_absent'])==38
    assert root_terminal['direct_owner_sha256']==sha(read('reviews/root-tests/direct-owner-v2.json'))
    peer=doc('reviews/security-terminal/review.json')
    assert peer['verification_passed'] and peer['run_passed'] and peer['regular_child_artifacts']==144
    assert peer['tiny_owned_public_fixtures']==6 and peer['unfollowed_aliases']==15
    c1=doc('proof/v1/run_config.json');c2=doc('proof/v2/run_config.json')
    assert sorted(k for k in c1 if c1[k]!=c2[k])==doc('proof/v2/handoff.json')['changed_config_fields']
    assert read('proof/v1/worker_driver.py')==read('proof/v2/worker_driver.py')
    assert read('proof/v2/executed_driver.py').replace(sha(read('proof/v2/run_config.json')).encode(),sha(read('proof/v1/run_config.json')).encode())==read('proof/v1/executed_driver.py')
    current_checks=0
    if current:
        checked_archives={}
        for name,item in origins.items():
            if 'source' in item:
                p=Path(item['source']);assert p.is_file() and not p.is_symlink()
                assert sha(p.read_bytes())==item['sha256'],str(p)
            else:
                p=Path(item['source_archive'])
                if str(p) not in checked_archives:
                    assert not p.is_symlink() and sha(p.read_bytes())==item['source_archive_sha256']
                    with tarfile.open(p) as t:
                        checked_archives[str(p)]={m.name:sha(t.extractfile(m).read()) for m in t.getmembers() if m.isfile()}
                assert checked_archives[str(p)][item['member']]==item['sha256']
            current_checks+=1
        for version,source in [('404','/tmp/qwen-server-snapshot-integration-tests-20261104'),('405','/tmp/qwen-server-snapshot-integration-tests-v2-20261104')]:
            base=Path(source)
            for name,raw in archives[version].items():
                p=base/name;assert p.is_file() and not p.is_symlink() and sha(p.read_bytes())==sha(raw)
            current_checks+=len(archives[version])
    return dict(passed=True,files=len(wl),indexed=len(indexed),copied=len(origins),source_archives=3,
                reconstructed_runtime_maps=[387,390,391],historical_tar_bytes_offline=False,
                failed404=dict(tests=36,failures=25,passed=11),final405=dict(tests=36,failures=0,passed=36),
                child_artifacts=[94,144],tiny_synthetic_fixtures=[6,6],aliases_metadata_only=[15,15],
                current=current,current_public_checks=current_checks,candidate_imports=False,rerun=False,
                capacity_or_process_probe=False,borrowed_body_read=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--current',action='store_true')
    args=parser.parse_args();print(json.dumps(verify(args.current),sort_keys=True))
