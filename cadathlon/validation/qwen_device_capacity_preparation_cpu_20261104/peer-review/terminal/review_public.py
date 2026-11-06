import ast,hashlib,json,os,pathlib,stat,tarfile,time,xml.etree.ElementTree as ET
P=pathlib.Path('/tmp/qwen-device-component-focused-proof-20261104/preparation-v1');O=pathlib.Path(__file__).parent
H=lambda b:hashlib.sha256(b).hexdigest(); J=lambda p:json.loads(p.read_bytes())
proof_hashes={}
def evidence(name):
    p=P/name;assert stat.S_ISREG(p.lstat().st_mode) and not p.is_symlink();raw=p.read_bytes();proof_hashes[name]=H(raw);return raw
h=json.loads(evidence('handoff.json'));c=json.loads(evidence('run_config.json'));r=json.loads(evidence('report.json'));a=json.loads(evidence('worker-audit.json'));live=json.loads(evidence('live-observation.json'))
assert proof_hashes['handoff.json']=='b46829a7876dc45f445d36dbb05271ca579548ec3c26fed2016d441f4aa350a9'
assert proof_hashes['report.json']=='a95a2b30377e88f6c27da2ceb7376bb052c970f1a1c6da6aa4f29faa2b8fd2a5'
for n,v in h['files'].items():assert H(evidence(n))==v
m=json.loads(evidence('sources-before.json'));assert len(m)==50
assert json.loads(evidence('sources-after.json'))==json.loads(evidence('execution-after.json'))==m
assert proof_hashes['sources-before.json']==c['source_manifest_sha256']==h['source_manifest_sha256']
def current(root):
    assert not any(p.is_symlink()for p in root.rglob('*'))
    return {p.relative_to(root).as_posix():H(p.read_bytes())for p in root.rglob('*')if p.is_file()}
assert current(pathlib.Path(c['source_root']))==current(P/'execution')==m
with tarfile.open(P/'sources.tar.gz')as t:
    members=t.getmembers();assert len(members)==50 and all(x.isfile()for x in members)
    assert {x.name:H(t.extractfile(x).read())for x in members}==m
runtime={k[8:]:v for k,v in m.items()if k.startswith('runtime/')};assert len(runtime)==34
assert runtime==c['runtime_source_expectations']==a['runtime_sources_before']==a['runtime_sources_after']
assert r['passed']is True and r['sources_unchanged']is True and r['child_returncode']==0 and type(r['child_returncode'])is int
assert r['child_reaped']is True and r['child_group_absent']is True
assert a['denied']==a['loaded_backend']==[] and a['numpy_imported']is False and a['runtime_module_origins']=={}
assert a['collected_nodes']==c['selected_nodes'] and a['affinity']==[112,113,114,115] and a['forbidden_roots']==c['forbidden_roots']
assert a['device_component_recording_only']is True and a['scientific_admission_tested']is False
assert type(a['pytest_exit_code'])is int and a['pytest_exit_code']==0
xml=ET.fromstring(evidence('tests.xml'));cases=list(xml.iter('testcase'))
counts={k:len(list(xml.iter(v)))for k,v in [('tests','testcase'),('failures','failure'),('errors','error'),('skipped','skipped')]}
assert counts==r['tests']==dict(tests=30,failures=0,errors=0,skipped=0)
assert [dict(classname=x.attrib['classname'],name=x.attrib['name'])for x in cases]==c['selected_xml_cases']==r['selected_xml_cases']
metadata=json.loads(evidence('owned-pytest-metadata.json'));records=json.loads(evidence('test-child-terminals.json'));assert records==r['child_records']
assert len(metadata)==141 and sum(v['kind']=='regular'for v in metadata.values())==120 and sum(v['kind']=='owned_pytest_alias'for v in metadata.values())==21
observed_names=set();alias_records={};child_hashes={}
# Enumerate only the two explicitly owned proof leaves; do not follow aliases.
for leaf in(P/'tmp',P/'pytest-owned'):
    assert stat.S_ISDIR(leaf.lstat().st_mode)
    def walk(directory):
        with os.scandir(directory)as items:
            for item in items:
                path=pathlib.Path(item.path);name=path.relative_to(P).as_posix();info=item.stat(follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode):
                    row=metadata[name];assert row['kind']=='owned_pytest_alias'and path.parent==P/'pytest-owned'
                    assert info.st_uid==os.geteuid() and path.name.startswith('test_device_component_')and path.name.endswith('current')
                    target=os.readlink(path);assert target==row['target']and len(target)<=4096
                    target_path=pathlib.Path(target);target_path=target_path if target_path.is_absolute()else path.parent/target_path
                    assert '..'not in target_path.parts and target_path==P/row['resolved_target']and target_path.parent==path.parent
                    target_info=target_path.lstat();assert stat.S_ISDIR(target_info.st_mode)and target_info.st_uid==os.geteuid()
                    assert not target_path.is_symlink()
                    after=path.lstat();assert(info.st_dev,info.st_ino,info.st_mode,info.st_uid)==(after.st_dev,after.st_ino,after.st_mode,after.st_uid)
                    assert os.readlink(path)==target
                    alias_records[name]=dict(target=target,followed=False);observed_names.add(name)
                elif stat.S_ISDIR(info.st_mode):walk(path)
                else:
                    assert stat.S_ISREG(info.st_mode)and info.st_uid==os.geteuid();row=metadata[name]
                    assert row['kind']=='regular'and info.st_size==row['bytes'];raw=path.read_bytes();assert H(raw)==row['sha256']
                    child_hashes[name]=H(raw);observed_names.add(name)
    walk(leaf)
assert observed_names==set(metadata)
expected=[x.split('[',1)[1][:-1]for x in c['selected_nodes']];assert [x['result']['scenario']for x in records]==expected
assert len(records)==30 and len({x['result']['pid']for x in records})==30
pids={live['outer_pid'],live['worker_pid']};pgids={live['outer_pgid'],live['worker_pgid']};allpaths=set();counters={};temporaries=0
for row in records:
    e=row['parent_terminal'];t=row['terminal'];v=row['result'];scenario=v['scenario']
    for key,obj in [('parent_terminal_path',e),('terminal_path',t),('result_path',v)]:
        assert row[key]in child_hashes and J(P/row[key])==obj;allpaths.add(row[key])
    assert row['log_path']in child_hashes and child_hashes[row['log_path']]==row['log_sha256'];allpaths.add(row['log_path'])
    assert child_hashes[row['parent_terminal_path']]==row['parent_terminal_sha256']
    assert e['child_result_read']is False and e['schema']=='qwen-device-component-owned-child-v1'
    assert e['scenario']==t['scenario']==scenario
    pid=v['pid'];assert type(pid)is int and pid>0 and type(v['pgid'])is int and v['pgid']==pid
    for owner in(e,t):
        assert type(owner['pid'])is int and owner['pid']==pid and type(owner['exit_code'])is int and owner['exit_code']==0
        assert owner['reaped']is True and owner['group_absent']is True
        assert owner['observed_pgid']is None or(type(owner['observed_pgid'])is int and owner['observed_pgid']==pid)
    assert e['observed_pgid']==t['observed_pgid']and e['owned_session_pgid']==t['child_reported_pgid']==pid
    assert t['runtime_scope']=='synthetic device policy/state only'
    for flag in('assertions_completed','recording_phase_validation','resource_validator_is_real','original_phase_identities_restored','original_resource_identities_preserved','profile_observer_restored','temporary_jax_removed'):assert v[flag]is True
    for flag in('actual_policy_or_grant','device_sources_modified','genuine_backend_imported','genuine_capacity_measured','scientific_admission_tested'):assert v[flag]is False
    assert v['backend_denials']==[] and type(v['temporary_recording_jax_used'])is bool
    temporaries+=v['temporary_recording_jax_used']
    for name in('resource_calls','logical_calls'):
        assert type(v[name])is int and v[name]>=0
        assert(v[name]>0)is(scenario not in('document_role_order','document_noncanonical','document_source_phase_join'))
    counters[scenario]={k:v[k]for k in('resource_calls','logical_calls','temporary_recording_jax_used')}
    pids.add(pid);pgids.add(pid)
assert allpaths==set(child_hashes)and len(allpaths)==120 and temporaries==r['temporary_recording_jax_children']==17
assert a['pid']==a['pgid']==live['worker_pid']==live['worker_pgid']
assert len(pids)==len(pgids)==32
absence=[]
for kind,values,fn in [('pid',pids,os.kill),('pgid',pgids,os.killpg)]:
    for value in sorted(values):
        try:fn(value,0)
        except ProcessLookupError:absence.append(dict(kind=kind,id=value,absent=True))
        else:raise AssertionError('named process/group currently present')
absence_record=dict(observation='reviewer independent post-exit signal0 absence; no live/wait/reap authority',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),checks=absence)
(O/'process-absence.json').write_text(json.dumps(absence_record,sort_keys=True,indent=2)+'\n')
report=dict(schema='qwen-device-terminal-security-review-v1',passed=True,proof_root=str(P),source_count=50,runtime_count=34,tests=counts,exact_ordered_collection_and_XML=True,source_current_execution_start_end_archive_equal=True,regular_child_files=120,unfollowed_aliases=21,early_parent_records=30,positive_child_records=30,temporary_source_owned_jax_children=17,restoration_and_resource_counter_joins=counters,worker_backend_audit_empty=True,proof_hashes=proof_hashes,child_hashes=child_hashes,aliases=alias_records,process_absence_sha256=H((O/'process-absence.json').read_bytes()),root_reported_owner=dict(launch_chunk='ab0197',session=24307,terminal_chunk='6454a9',exit_code=0,reaped=True,independent_root_live=False),reviewer_waited=False,reviewer_live_observation=False,scope='Completed30 synthetic policy/state cases using real pure validators and disclosed recording phase/process/JAX seams; no genuine backend/device/capacity/actual scientific authority.',origin_scope='Source-bound child assertions; no observed module-origin table.',rerun=False,candidate_imports=False,backend_or_capacity_probe=False,private_or_body_reads=False)
(O/'review.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n');print(H((O/'review.json').read_bytes()))
