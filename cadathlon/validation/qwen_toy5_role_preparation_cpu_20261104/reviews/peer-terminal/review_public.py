"""Stdlib terminal readback: public sources and exact392 receipts/logs only."""
from pathlib import Path
import hashlib,json,os,stat,tarfile,xml.etree.ElementTree as ET
P=Path('/tmp/qwen-toy5-role-focused-proof-20261104/preparation-v1')
S=Path('/tmp/qwen-toy5-role-focused-tests-20261104')
R=Path('/tmp/qwen-toy5-role-focused-root-harness-review-20261104')
O=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def j(p):return json.loads(p.read_bytes())
def inventory(p):
    assert not any(x.is_symlink()for x in p.rglob('*'))
    return {x.relative_to(p).as_posix():sha(x.read_bytes())for x in p.rglob('*')if x.is_file()}
def read_fixed(p):
    s=p.lstat();assert stat.S_ISREG(s.st_mode)and s.st_uid==os.geteuid()and 0<=s.st_size<=1048576
    assert not p.is_symlink();raw=p.read_bytes();assert len(raw)==s.st_size and p.lstat()==s
    return raw
source=j(P/'sources-before.json');assert len(source)==427
assert source==inventory(S)==inventory(P/'execution')==j(P/'sources-after.json')==j(P/'execution-after.json')
assert sha((P/'sources-before.json').read_bytes())=='43a7e90b04a2b99edfb953a7724c3a50580dcff844c2da987a32ec1da7a41df9'
assert sha((P/'sources.tar.gz').read_bytes())=='269a3661b3a84309bed36c7527c7ccedf06c086b1dd9e907e269b0a03b87380b'
with tarfile.open(P/'sources.tar.gz')as archive:
    members=archive.getmembers();assert len(members)==427 and all(m.isfile()for m in members)
    assert {m.name:sha(archive.extractfile(m).read())for m in members}==source
runtime=j(S/'runtime-map.json');assert len(runtime)==389
assert {n.removeprefix('runtime/'):v for n,v in source.items()if n.startswith('runtime/')}==runtime
config=j(P/'run_config.json');report=j(P/'report.json');audit=j(P/'worker-audit.json');art=j(P/'child-artifact-review.json')
root=j(R/'terminal-review.json');live=j(R/'owner-live.json')
assert sha((R/'terminal-review.json').read_bytes())=='72b3f644faab265021fbdb563522feb7321952b6dc3a009654cc5d67b45c9e80'
assert sha((P/'report.json').read_bytes())=='8cb363c0cb94424c7c2a814800f88ffc9104739aa1da89fd253e06205fedfc71'
assert sha((P/'child-artifact-review.json').read_bytes())==report['artifact_review_sha256']==root['artifact_review_sha256']
assert report['passed']is root['passed']is art['passed']is True
xml=ET.parse(P/'tests.xml').getroot();cases=list(xml.iter('testcase'))
assert len(cases)==55 and not any(list(xml.iter(n))for n in('failure','error','skipped'))
assert [dict(classname=x.attrib['classname'],name=x.attrib['name'])for x in cases]==config['selected_xml_cases']==report['selected_xml_cases']
assert audit['collected_nodes']==config['selected_nodes']==report['selected_nodes']
assert audit['denied']==audit['loaded_backend']==[]and audit['runtime_module_origins']=={}
assert audit['numpy_imported']is False and audit['pytest_exit_code']==0
assert audit['numpy_sources_before']==audit['numpy_sources_after']==config['numpy_pins']['sources']
assert len(config['numpy_pins']['sources'])==4
for path,digest in config['numpy_pins']['sources'].items():assert sha(Path(path).read_bytes())==digest
files=art['files'];assert len(files)==392 and len(art['records'])==56 and art['errors']==[]
assert art['fixture_bodies_read']is False and art['aliases_followed']is False
allowed=set(config['artifact_plan']['fixed_child_file_names']);assert len(allowed)==7
pins={};pids=set();numpy_children=0
expected={(x['nodeid'],x['variant']):x for x in config['artifact_plan']['children']};observed=set()
for record in art['records']:
    key=(record['nodeid'],record['variant']);assert key in expected and key not in observed;observed.add(key)
    parent=record['parent_terminal'];early=record['child_early'];terminal=record['child_terminal'];res=record['result']
    pid=parent['pid'];assert type(pid)is int and pid>0 and pid not in pids;pids.add(pid)
    assert type(early['pgid'])is int and early['pgid']==pid==terminal['pgid']
    assert type(early['start_ticks'])is int and early['start_ticks']>0 and early['start_ticks']==terminal['start_ticks']
    assert parent['start_ticks']is None or parent['start_ticks']==early['start_ticks']
    assert parent['observed_pgid']is None or parent['observed_pgid']==pid
    assert parent['direct_wait_completed']is True and parent['group_absent']is True
    assert parent['exit_code']==expected[key]['expected_exit']and parent['primary_type']is None and parent['cleanup_types']==[]
    assert res['passed']is terminal['passed']is True and terminal['forbidden_imports']==[]
    assert terminal['actual_authority']is False and terminal['recording_only']is True
    assert set(record['files'])==allowed
    for name,pin in record['files'].items():
        relative=pin['path'];path=P/relative
        assert relative.startswith('pytest-owned/')and path.name==name and '..'not in Path(relative).parts
        assert '/owned_io/'not in relative and '/owned_assets/'not in relative
        raw=read_fixed(path);assert sha(raw)==pin['sha256']==files[relative]['sha256']and len(raw)==pin['bytes']
        pins[relative]=dict(sha256=sha(raw),bytes=len(raw))
    for name,value in [('parent-terminal.json',parent),('child-early.json',early),('child-terminal.json',terminal),('result.json',res),('parent-live.json',record['parent_live'])]:
        assert j(P/record['files'][name]['path'])==value
    if terminal['numpy_source_pins_checked']:
        numpy_children+=1
        assert terminal['numpy_origins'].get('numpy')==config['numpy_pins']['origins']['numpy']
    else:assert terminal['numpy_origins']=={}
    for name,path in terminal['numpy_origins'].items():assert path==config['numpy_pins']['origins'][name]
    for name,path in terminal['helper_origins'].items():assert path==str(P/'execution'/'runtime'/(name+'.py'))
assert set(pins)==set(files)and len(pins)==392 and observed==set(expected)and numpy_children==10
worker=j(P/'worker-owner-terminal.json');early=j(P/'worker-early.json');wt=j(P/'worker-terminal.json');collector=j(P/'collector-terminal.json')
assert worker['direct_wait_completed']is worker['child_reaped']is worker['pid_absent']is worker['group_absent']is True
assert worker['exit_code']==0 and worker['primary_type']is None and worker['cleanup_types']==[]
assert early['pid']==early['pgid']==worker['worker_pid']==worker['owned_session_pgid']==audit['pid']==audit['pgid']==wt['pid']==wt['pgid']
assert early['start_ticks']==worker['start_ticks']==audit['start_ticks']==wt['start_ticks']
assert collector['normal_main_return']is True and collector['planned_exit_code']==0
assert root['session']==79231 and root['launch_tool']=='9220a4'and root['terminal_tool']=='ecf1c0'
assert root['actual_collector_exit_code']==0 and root['tool_reaped_collector']is True
assert root['collector_pid_pgid']==collector['pid']==collector['pgid']
for role,row in live['root_process_observations'].items():
    actual=collector if role=='collector'else early
    assert row['pid']==actual['pid']and row['pgid']==actual['pgid']and row['start_ticks']==actual['start_ticks']
    assert row['affinity']==[108,109,110,111]
pids.update((collector['pid'],worker['worker_pid']));assert len(pids)==58
absence=[]
for pid in sorted(pids):
    for group in(False,True):
        try:(os.killpg if group else os.kill)(pid,0)
        except ProcessLookupError:pass
        else:raise ValueError('original process/group still present')
    absence.append(dict(pid=pid,pgid=pid,pid_absent=True,pgid_absent=True))
for item in art['fixture_metadata']:assert item['body_read']is False
fifo=[x['path']for x in art['fixture_metadata']if x['kind']=='fifo_metadata_only'];assert len(fifo)==1
# No fixture path is stat'ed or opened by this readback; these records remain
# producer lstat evidence. No alias target is traversed.
(O/'child-files.json').write_text(json.dumps(pins,sort_keys=True,separators=(',',':'))+'\n')
(O/'process-absence.json').write_text(json.dumps(absence,sort_keys=True,separators=(',',':'))+'\n')
review=dict(schema='qwen-toy5-role-focused-terminal-peer-v1',passed=True,source427_exact=True,runtime389_exact=True,
    ordered55_XML_and_prefixture_collection=True,child_count=56,regular_receipt_log_files=392,numpy_children=10,
    numpy_four_sources_checked=True,forbidden_backend_audit_empty=True,root_direct_owner_session=79231,
    root_terminal_tool='ecf1c0',independent_peer_wait_or_live_claim=False,postexit58_pid_groups_absent=True,
    copied_fixture_bodies=False,read_or_statted_fixture_bodies=False,aliases_followed=False,
    original_FIFO_metadata_only=fifo,root_terminal_sha256=sha((R/'terminal-review.json').read_bytes()),
    report_sha256=sha((P/'report.json').read_bytes()),artifact_review_sha256=sha((P/'child-artifact-review.json').read_bytes()),
    source_map_sha256=sha((P/'sources-before.json').read_bytes()),source_archive_sha256=sha((P/'sources.tar.gz').read_bytes()),
    child_pin_map_sha256=sha((O/'child-files.json').read_bytes()),process_absence_sha256=sha((O/'process-absence.json').read_bytes()),
    scope=report['scope'],candidate_imports=False,tests_rerun=False,actual_role_or_HE_authority=False)
(O/'review.json').write_text(json.dumps(review,sort_keys=True,separators=(',',':'))+'\n')
print(json.dumps(dict(passed=True,review=str(O/'review.json'),sha256=sha((O/'review.json').read_bytes())),sort_keys=True))
