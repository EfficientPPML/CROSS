"""Freeze public source/AST artifacts only; no candidate imports or probes."""
import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile

BASE=Path('/tmp/qwen-actual-client-role-runtime-20261104')
ROOT=Path('/tmp/qwen-actual-client-role-runtime-v2-20261104')
TEST=Path('/tmp/qwen-client-kernel-allowlist-tests-20261104')
OUT=Path(__file__).parent
def h(raw):return hashlib.sha256(raw).hexdigest()
def compact(value):return(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def pretty(value):return(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def snap(root):return{str(p.relative_to(root)):h(p.read_bytes())for p in sorted(root.rglob('*'))if p.is_file()}
def node(raw,name):
    nodes=ast.parse(raw).body
    for bit in name.split('.'):
        selected=[n for n in nodes if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name==bit]
        assert len(selected)==1,name
        found=selected[0];nodes=found.body
    return found
def archive(root,mapping,path):
    with tarfile.open(path,'w:gz')as tf:
        for name,digest in mapping.items():
            data=(root/name).read_bytes();assert h(data)==digest
            info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;info.mtime=0
            tf.addfile(info,io.BytesIO(data))

oldraw=(OUT/'parent464-sources.json').read_bytes();old=json.loads(oldraw)
assert h(oldraw)=='41d563410ec152d2ae390396284fc776da940276141552a867ae775a797f93d2'
assert snap(BASE)==old
current=snap(ROOT);assert len(current)==464 and set(current)==set(old)
assert {n for n in old if old[n]!=current[n]}=={'qwen_client_dependencies.py'}
raw=(ROOT/'qwen_client_dependencies.py').read_bytes()
security=Path('/tmp/qwen-keygen-actual-runtime-v3-20261104/keygen_dependencies.py').read_bytes()
assert h(security)=='9e6760172efb58342b4aee5d7a3ae1799930869d0ae479258d691ba9e7365ca4'
assert ast.dump(node(raw,'_literal_proc_self'))==ast.dump(node(security,'_literal_proc_self'))
def branch(data):
    return next(n for n in node(data,'AccessScope.access').body if isinstance(n,ast.If)and '_PROC_SELF_KERNEL'in ast.unparse(n.test))
assert ast.dump(branch(raw))==ast.dump(branch(security))
oldtree=ast.parse((BASE/'qwen_client_dependencies.py').read_bytes());newtree=ast.parse(raw)
def functions(tree):
    out={}
    for n in tree.body:
        if isinstance(n,ast.FunctionDef):out[n.name]=ast.dump(n)
        elif isinstance(n,ast.ClassDef):
            for child in n.body:
                if isinstance(child,ast.FunctionDef):out[n.name+'.'+child.name]=ast.dump(child)
    return out
a,b=functions(oldtree),functions(newtree)
assert {n for n in a if a[n]!=b[n]}=={'AccessScope.access','AccessScope.install'}
assert set(b)-set(a)=={'_literal_proc_self'}
assert json.loads((ROOT/'qwen_client_entry_status.json').read_bytes())['actual_entry_ready']is False
newraw=compact(current);(OUT/'sources.json').write_bytes(newraw)
archive(ROOT,current,OUT/'sources.tar.gz')

runtime=TEST/'runtime';assert not runtime.exists()
for name in current:
    target=runtime/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
(TEST/'runtime-sources.json').write_bytes(newraw)
(TEST/'parent464-sources.json').write_bytes(oldraw)
(TEST/'before_qwen_client_dependencies.py').write_bytes((BASE/'qwen_client_dependencies.py').read_bytes())
tests=ast.parse((TEST/'test_kernel_allowlist.py').read_bytes())
values={}
for n in tests.body:
    if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and n.targets[0].id in('SUCCESSOR_CASES','BASELINE_CASES'):
        values[n.targets[0].id]=ast.literal_eval(n.value)
assert len(values['SUCCESSOR_CASES'])==21 and len(values['BASELINE_CASES'])==3
nodes=[f'test_kernel_allowlist.py::test_successor_kernel_policy[{case}]'for case in values['SUCCESSOR_CASES']]
nodes += [f'test_kernel_allowlist.py::test_old464_alias_controls[{case}]'for case in values['BASELINE_CASES']]
assert len(nodes)==len(set(nodes))==24
selection={'schema':'qwen-client-kernel-focused-selection-v1','static_node_count':24,'nodes':nodes,
    'test_files':['test_kernel_allowlist.py'],'successor_source_manifest_sha256':h(newraw),
    'parent_source_manifest_sha256':h(oldraw),'cpu_affinity_proposed':[108,109,110,111],
    'collected':False,'executed':False,'actual_role_authorized':False,
    'metadata_probe_authorized':False,'scope':'Exact audit/kernel path repair only; no backend or actual admission/capacity.'}
(TEST/'selected_tests.json').write_bytes(pretty(selection))
for name in('kernel_probe.py','test_kernel_allowlist.py','conftest.py'):ast.parse((TEST/name).read_bytes(),filename=name)
testmap=snap(TEST);(OUT/'focused-sources.json').write_bytes(compact(testmap))
archive(TEST,testmap,OUT/'focused-sources.tar.gz')
correction={'schema':'qwen-client-kernel-path-correction-v1','parent_root':str(BASE),'parent_source_count':464,
    'parent_source_manifest_sha256':h(oldraw),'successor_root':str(ROOT),'successor_source_count':464,
    'successor_source_manifest_sha256':h(newraw),'unchanged_files':463,
    'only_changed_file':'qwen_client_dependencies.py','before_sha256':old['qwen_client_dependencies.py'],
    'after_sha256':current['qwen_client_dependencies.py'],'shared_keygen_dependency_sha256':h(security),
    'shared_literal_helper_ast_exact':True,'shared_special_branch_ast_exact':True,
    'actual_entry_ready':False,'source_only':True,'imports_tests_probes_capacity_body_reads':False,
    'finding_receipt':{'path':'/tmp/qwen-actual-client-source-peer-review-20261104/review.json',
        'sha256':'d80e5ef6f54596b38f8a64ea16ebe6e9eb96bd90d4dc4e86117f383957602263'},
    'limitations':['Only exact original os.open strings/no dir_fd or direct audit-event strings receive the read-only current-PID exception.',
        'Builtin audit events cannot recover a pre-normalized caller PathLike type.',
        'No broad proc PID/alias exception; unchanged actual role/status/dependency policy remains disabled.',
        'Focused24 static nodes have not been collected/executed; no new harness or run grant.']}
(OUT/'correction.json').write_bytes(pretty(correction))
handoff={'schema':'qwen-client-kernel-successor-source-handoff-v1','source_root':str(ROOT),
    'source_count':464,'source_manifest':str(OUT/'sources.json'),'source_manifest_sha256':h(newraw),
    'source_archive_sha256':h((OUT/'sources.tar.gz').read_bytes()),'changed_source_sha256':current['qwen_client_dependencies.py'],
    'focused_source_root':str(TEST),'focused_source_count':len(testmap),
    'focused_source_manifest_sha256':h((OUT/'focused-sources.json').read_bytes()),
    'focused_source_archive_sha256':h((OUT/'focused-sources.tar.gz').read_bytes()),
    'static_node_count':24,'selection_sha256':h((TEST/'selected_tests.json').read_bytes()),
    'probe_sha256':h((TEST/'kernel_probe.py').read_bytes()),'test_sha256':h((TEST/'test_kernel_allowlist.py').read_bytes()),
    'guard_sha256':h((TEST/'conftest.py').read_bytes()),'correction_sha256':h((OUT/'correction.json').read_bytes()),
    'source_only':True,'collected':False,'executed':False,'actual_roles':False,'probe_authorized':False}
(OUT/'handoff.json').write_bytes(pretty(handoff))
assert snap(BASE)==old and snap(ROOT)==current
print(json.dumps(handoff,sort_keys=True))
