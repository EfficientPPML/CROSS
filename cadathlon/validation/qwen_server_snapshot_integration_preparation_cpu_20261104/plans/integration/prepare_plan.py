"""Pin the reviewed public source slices for a plan; no candidate imports."""
import ast,hashlib,json,pathlib,stat
HERE=pathlib.Path(__file__).resolve().parent
SERVER=pathlib.Path('/tmp/qwen-server-actual-runtime-20261104')
GUARD=pathlib.Path('/tmp/qwen-server-admission-snapshot-source-v2-20261104')
FIXTURE=pathlib.Path('/tmp/qwen-server-admission-snapshot-tests-v2-20261104/runtime')
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):
    p=pathlib.Path(p);a=p.lstat();assert stat.S_ISREG(a.st_mode) and not p.is_symlink()
    raw=p.read_bytes();z=p.lstat()
    assert(a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns)
    return raw
def dump(path,d):path.write_text(json.dumps(d,sort_keys=True,indent=2)+'\n')
pure=['qwen_server_authority_packet.py','qwen_server_admission.py','qwen_scientific_admission.py','qwen_file_contract.py','model_admission.py','keygen_authority.py','client_role_common.py','freeze_admission.py','freeze_issuer.py','qwen_key_contract.py','qwen_public_parse.py','public_source_pins.py','keygen_boundary_pins.py','scientific_source_pins.py']
other=['qwen_server_process.py','qwen_server_backend.py','qwen_server_entry.py','qwen_server_dependencies.py','qwen_server_resource.py','qwen_key_files_admission.py','qwen_key_adapter.py','qwen_key_loader.py','qwen_key_readers.py','qwen_backend_identity.py','qwen_finalizer.py']
paths=[SERVER/n for n in pure+other]+[GUARD/'qwen_admission_snapshot.py',FIXTURE/'qwen_admission_snapshot.py']+[FIXTURE/n for n in pure]+[
 pathlib.Path('/tmp/qwen-server-admission-snapshot-handoff-v2-20261104/handoff.json'),
 pathlib.Path('/tmp/qwen-server-admission-snapshot-tests-v2-20261104/TEST_SCOPE.md'),
 pathlib.Path('/tmp/qwen-server-admission-snapshot-focused-proof-20261104/preparation-v2/run_config.json'),
 pathlib.Path('/tmp/qwen-server-admission-snapshot-root-review-v2-20261104/terminal-review.json')]
refs={str(p):{'sha256':sha(read(p)),'bytes':len(read(p))}for p in paths}
guard_hash='8790711a1a402f4c21ef444ccdb76f773b4c62cd334f5be07a49d2cd4535b395'
assert refs[str(GUARD/'qwen_admission_snapshot.py')]['sha256']==refs[str(FIXTURE/'qwen_admission_snapshot.py')]['sha256']==guard_hash
comparison={n:{'server_sha256':refs[str(SERVER/n)]['sha256'],'tested_guard_dependency_sha256':refs[str(FIXTURE/n)]['sha256'],'exact':read(SERVER/n)==read(FIXTURE/n)}for n in pure}
assert all(v['exact']for v in comparison.values())
assert refs[str(SERVER/'qwen_server_process.py')]['sha256'].startswith('e8347')
assert refs[str(SERVER/'qwen_server_backend.py')]['sha256'].startswith('682e5')
tree=ast.parse(read(SERVER/'qwen_server_resource.py'));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='require_complete_capacity')
assert any(isinstance(n,ast.Raise)for n in fn.body)
terminal=json.loads(read(paths[-1]));assert terminal['passed'] is True and terminal['source_count']==38 and terminal['runtime_files']==28
assert terminal['full_scientific_admission_tested'] is False and terminal['guard_integrated_into_actual_server'] is False
dump(HERE/'source_refs.json',{'schema':'qwen-server-snapshot-plan-public-refs-v1','files':refs})
dump(HERE/'compatibility.json',{'schema':'qwen-server-snapshot-helper-compatibility-v1','helper_count':14,'guard_sha256':guard_hash,'helpers':comparison,'source_only':True,'candidate_imports':False,'full_current387_captured':False})
dump(HERE/'review_limits.json',{
 'schema':'qwen-server-snapshot-plan-scope-v1','status':'source-only plan awaiting parent review',
 'current_server_changed':False,'candidate_imports':False,'tests_or_collection':False,'body_reads':False,'backend_or_capacity_probes':False,'actual_grants':False,
 'reviewed_server_slices':other,'pure_helpers_compared_exact':pure,
 'snapshot_proof':{'source_count':38,'runtime_files':28,'cases':18,'root_terminal_sha256':refs[str(paths[-1])]['sha256'],'scope':'Thirteen exact admission types with non-proof markers, explicit PublicContract construction bypass and a recording full-validator seam; no real scientific admission or server integration.'},
 'source_count_387':'Parent-reported current development tree; only listed slices independently pinned by this plan.',
 'required_before_implementation':'Root acceptance; fresh exact full parent capture and explicit overlays. No current source edit authorized by this artifact.',
 'future_proof_limits':['No measured acceleration','No global proof cache','No implied body scope/capacity authority','No inherited18 integration PASS','Live FD/keyless/security/audit checks remain genuine selected paths'],
 'preparation_script_sha256':sha(read(pathlib.Path(__file__)))})
inventory={p.name:sha(read(p))for p in HERE.iterdir()if p.is_file()and p.name not in('sources.json',)}
dump(HERE/'sources.json',inventory)
handoff={'schema':'qwen-server-snapshot-integration-plan-handoff-v1','root':str(HERE),'source_count':len(inventory),'map_sha256':sha(read(HERE/'sources.json')),'plan_sha256':sha(read(HERE/'PLAN.md')),'callgraph_sha256':sha(read(HERE/'callgraph.json')),'source_refs_sha256':sha(read(HERE/'source_refs.json')),'helper14_exact':True,'guard_sha256':guard_hash,'implementation':False,'candidate_imports_or_tests':False,'actual_authority':False}
out=pathlib.Path('/tmp/qwen-server-snapshot-integration-plan-handoff-20261104.json');dump(out,handoff)
print(json.dumps({'passed':True,'handoff':str(out),'handoff_sha256':sha(read(out)),**handoff},sort_keys=True))
