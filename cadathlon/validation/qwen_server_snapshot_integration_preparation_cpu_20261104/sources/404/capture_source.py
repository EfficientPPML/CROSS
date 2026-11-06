"""Static source authoring/closure only: never import candidate/test modules."""
import ast
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

H=Path(__file__).resolve().parent
R=Path('/tmp/qwen-server-snapshot-integration-tests-20261104')
P=Path('/tmp/qwen-server-snapshot-runtime-v2-20261104')
PH=Path('/tmp/qwen-server-snapshot-runtime-handoff-v2-20261104')
sha=lambda b:hashlib.sha256(b).hexdigest()
def canonical(v):return(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def inventory(root):
    out={}
    for p in sorted(root.rglob('*')):
        assert not p.is_symlink(),p
        if p.is_file():out[p.relative_to(root).as_posix()]=p.read_bytes()
        else:assert p.is_dir(),p
    return out
def mapping(raw):return{n:sha(b)for n,b in raw.items()}
expected=json.loads((PH/'sources.json').read_bytes())
assert len(expected)==391 and sha((PH/'sources.json').read_bytes())=='fb5beaa2b7d86aa7a1fce40edb0052898a59fea25323ecc4ecc267609f77edaa'
assert expected==mapping(inventory(P))==mapping(inventory(R/'runtime'))
assert(R/'runtime_sources.json').read_bytes()==(PH/'sources.json').read_bytes()
with tarfile.open(PH/'sources.tar.gz','r:gz')as t:
    members=t.getmembers();assert len(members)==391 and all(m.isfile()for m in members)
    assert {m.name:sha(t.extractfile(m).read())for m in members}==expected
case_tree=ast.parse((R/'integration_cases.py').read_bytes())
cases=ast.literal_eval(next(n.value for n in case_tree.body if isinstance(n,ast.Assign)and n.targets[0].id=='CASES'))
assert len(cases)==len(set(cases))==36
nodes=['test_snapshot_integration.py::test_snapshot_integration['+n+']'for n in cases]
selection=dict(schema='qwen-snapshot-integration-static-selection-v1',static_count=36,static_nodeids=nodes,
    selected_files=['test_snapshot_integration.py'],child_entry='integration_child.py',runtime_files=391,
    definition_only_entry=True,genuine_cli_bootstrap=False,installed_backend_origins={},temporary_native_names=['numpy','jax'],
    registry_reset=False,scientific_admission=False,capacity_measurement=False,actual_authorization=False,collected=False,executed=False)
(R/'selected_tests.json').write_bytes(canonical(selection))
entry_raw=(R/'runtime/qwen_server_entry.py').read_bytes();entry_tree=ast.parse(entry_raw)
dispatch=ast.parse("if __name__=='__main__':\n    raise SystemExit(main())\n").body[0]
dump=lambda n:ast.dump(n,include_attributes=False)
assert [i for i,n in enumerate(entry_tree.body)if dump(n)==dump(dispatch)]==[len(entry_tree.body)-1]
defs=ast.Module(body=entry_tree.body[:-1],type_ignores=entry_tree.type_ignores)
entry_provenance=dict(schema='qwen-definition-only-entry-fixture-v1',entry_sha256=sha(entry_raw),
    excluded_dispatch_ast=dump(dispatch),excluded_dispatch_ast_sha256=sha(dump(dispatch).encode()),
    definition_ast_sha256=sha(dump(defs).encode()),only_one_top_level_node_excluded=True,
    actual_snapshot_origins_unchanged=True,main_spec=None,original_child_main_restored=True,
    call_main=False,call_dispatch=False,call_authenticate=False,genuine_bootstrap=False,
    runtime_internal_inventory_unchanged=True,fixture_root_is_separate=True)
(R/'entry_fixture_provenance.json').write_bytes(canonical(entry_provenance))
seams=[]
def seam(module,exports,groups,description):
    seams.append(dict(module=module,source_sha256=expected[module+'.py'],exports=exports,groups=groups,description=description))
seam('qwen_file_contract',['PublicContract constructor bypass'],['all_nonterminal'],'Exact type allocated with marker tuple; no full public validation claim. No public method export replaced.')
seam('qwen_server_authority_packet',['ServerPacket.check'],['all_nonterminal'],'Recording full validation counter on original typed graph.')
seam('qwen_admission_snapshot',['_FULL_PACKET_CHECK'],['all_nonterminal'],'Same recording function as ServerPacket.check; guard capture/check/retire bodies unchanged.')
seam('qwen_server_admission',['ServerPhaseAdmission.check','ExpectedServerPhasePins.check'],['all_nonterminal'],'Recorded validation/pin dictionary. expected_ids and input_digest remain original code, independently counted by profiler.')
seam('keygen_authority',['GeneratedKeyReadiness.expected_ids'],['all_nonterminal'],'Returns one genuine typed ExpectedIds fixture; no readiness or key generation authentication.')
seam('qwen_server_process',['exact_environment'],['all_nonterminal'],'Explicit recording environment binding; no host probe. Real source-file reader and entry.origins remain active.')
seam('qwen_key_readers',['map_array'],['backend_constructor'],'Records five original forwarding argument identities. ReadDirectory class remains original/unconstructed; no array/file reads.')
seam('qwen_backend_composition',['_runtime_dependencies'],['backend_constructor'],'Returns real Dependencies dataclass with fail-if-called numerical callbacks; genuine factory/materialize never selected.')
seam('qwen_server_process',['admit','check_host_capacity','hash_installed','imported_origins','AccessScope','Output','authenticate_public','bind_files','termination_signals','cleanup_signals'],['process'],'Explicit admission, host/dependency, audit, publication, key-binding and signal owner recordings. Actual run, source/public rereads, control, Finalizer and terminal_status remain selected.')
seam('qwen_server_process',['require_complete_capacity'],['positive_process_only'],'Recording positive capacity boundary only. Default-capacity-denial case preserves the actual unchanged denying function.')
seam('qwen_backend_identity',['install_core'],['process'],'Recording installer returns Core fixture; no core import/alias registration. Denial control asserts this hook was never reached.')
seam('qwen_server_backend',['ServerBackend'],['positive_process_only'],'Recorded process boundary only; actual ServerBackend constructor/checks are separately exercised by backend_constructor cases.')
seam('qwen_stage_composition',['_server_for_stage'],['positive_process_only'],'Recording stage and loaded-context retirement; no actual loader, model, arrays or ciphertext processing.')
table=dict(schema='qwen-snapshot-integration-explicit-seams-v1',export_and_allocation_seams=seams,
    injected_objects=[dict(name='Core',description='No real modules: four module-like namespace values and retained check leases; no sys.modules core aliases.'),
        dict(name='Audit',description='Original state/control sees recorded enable/check/close callbacks and flags; os.open remains original.'),
        dict(name='FilesAdmission',description='Real dataclass containing recorded keys/public/handle objects; its full validation is not executed.'),
        dict(name='temporary numpy/jax',description='Owned ModuleType names, specNone/no falsely stamped source. Empty for constructor; process adds recorded devices/config only. Removed before positive terminal.'),
        dict(name='__main__',description='Exact entry definitions without final dispatch; original source filename/specNone. Genuine bootstrap excluded.'),
        dict(name='profile observer',description='Original code-object counters for phase derivation and guard retirement; restored before result.')],
    pre_capture_export_patches_only=True,intentional_mutations_separate=True,private_retention_keywords_never_supplied=True,
    real_SourceFile_and_origin_checks=True,real_Finalizer=True,all_original_registries_never_reset=True,
    synthetic_observations_not_actual_proofs=True)
(R/'SEAMS.json').write_bytes(canonical(table))
parent_test=Path('/tmp/qwen-device-component-tests-v2-20261104/test_device_component.py')
parent_tree=ast.parse(parent_test.read_bytes());test_tree=ast.parse((R/'test_snapshot_integration.py').read_bytes())
def ownership(tree):
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef))
    with_node=next(n for n in function.body if isinstance(n,ast.With))
    assert isinstance(with_node.body[0],ast.Assign)and isinstance(with_node.body[1],ast.Try)
    return with_node.body[1]
assert dump(ownership(parent_tree))==dump(ownership(test_tree))
functions=[n for n in test_tree.body if isinstance(n,ast.FunctionDef)]
assert [n.name for n in functions]==['test_snapshot_integration']
assert ast.unparse(functions[0].decorator_list[0])=="pytest.mark.parametrize('scenario', CASES, ids=CASES)"
tests=['integration_cases.py','integration_fixture.py','integration_checks.py','integration_child.py','test_snapshot_integration.py','conftest.py']
parsed={n:ast.parse((R/n).read_bytes())for n in tests}
for name,tree in parsed.items():
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            assert not any(k.arg in('_dispatch','_require','_require_runtime','_check_runtime')for k in node.keywords),(name,ast.unparse(node))
        if isinstance(node,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
            targets=node.targets if isinstance(node,ast.Assign)else[node.target]
            for target in targets:
                assert not(isinstance(target,ast.Attribute)and target.attr in('_REGISTRY','_ISSUED','_CAPTURE_ATTEMPT','_RUNTIME_CONTROLS','_QUARANTINE')),(name,ast.unparse(node))
refs=[Path('/tmp/qwen-server-snapshot-test-architecture-20261104/PLAN.md'),
    Path('/tmp/qwen-server-snapshot-integration-test-plan-20261104/PLAN.md'),
    Path('/tmp/qwen-server-snapshot-root-source-review-v2-20261104/review.json'),
    Path('/tmp/qwen-server-snapshot-integration-security-review-20261104/source391/review.json'),
    Path('/tmp/qwen-server-admission-snapshot-tests-v3-20261104/snapshot_child.py'),parent_test]
origin=dict(schema='qwen-snapshot-integration-test-source-origin-v1',runtime_root=str(P),runtime_map_sha256=sha((PH/'sources.json').read_bytes()),
    runtime_archive_sha256=sha((PH/'sources.tar.gz').read_bytes()),runtime_count=391,all_runtime_exact=True,
    references={str(p):sha(p.read_bytes())for p in refs},parent_owned_child_try_AST_exact=True,
    static_selection_only=True,imports=False,collection=False,execution=False,actual_authority=False,
    test_scope='Named synthetic graph/owners and pure definition-only entry; no real native initialization, scientific evidence, memory reservation or body integrity.',
    default_capacity_denial_exact=True,positive_process_preenabled_gate_is_recording=True,
    finalizer_failure_scope='Close-only failure requires terminal70 and unretired quarantine, not invented guard failed-state.')
(R/'origin.json').write_bytes(canonical(origin))
raw=inventory(R);m=mapping(raw);assert len(m)==404
assert {n[8:]:v for n,v in m.items()if n.startswith('runtime/')}==expected
(H/'sources.json').write_bytes(canonical(m))
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w',format=tarfile.PAX_FORMAT)as t:
    for name,data in raw.items():
        info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;info.mtime=0;info.uid=info.gid=0;info.uname=info.gname='';t.addfile(info,io.BytesIO(data))
(H/'sources.tar.gz').write_bytes(gzip.compress(buf.getvalue(),mtime=0))
with tarfile.open(H/'sources.tar.gz','r:gz')as t:
    assert {member.name:sha(t.extractfile(member).read())for member in t.getmembers()}==m
assert mapping(inventory(P))==expected and mapping(inventory(R))==m
handoff=dict(schema='qwen-snapshot-integration-held-test-handoff-v1',root=str(R),source_count=len(m),runtime_count=391,
    source_manifest_sha256=sha((H/'sources.json').read_bytes()),source_archive_sha256=sha((H/'sources.tar.gz').read_bytes()),
    source_archive_bytes=(H/'sources.tar.gz').stat().st_size,static_count=36,static_nodeids=nodes,
    selected_files=['test_snapshot_integration.py'],test_hashes={n:m[n]for n in tests},
    seams_sha256=m['SEAMS.json'],entry_fixture_provenance_sha256=m['entry_fixture_provenance.json'],
    origin_sha256=m['origin.json'],parent391_exact=True,ownership_try_AST_exact=True,
    imports=False,collected=False,executed=False,capacity_probes=False,actual_authority=False,
    source_capture_script_sha256=sha(Path(__file__).read_bytes()))
(H/'handoff.json').write_bytes(canonical(handoff))
print(json.dumps(dict(passed=True,handoff_sha256=sha((H/'handoff.json').read_bytes()),**handoff),sort_keys=True))
