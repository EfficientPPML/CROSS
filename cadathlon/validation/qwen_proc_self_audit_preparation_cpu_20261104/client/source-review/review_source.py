"""Public-source/AST readback only; never import or execute candidate modules."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path('/tmp/qwen-actual-client-role-runtime-20261104')
HANDOFF = Path('/tmp/qwen-actual-client-role-runtime-handoff-20261104')
OUT = Path(__file__).parent
MAP_SHA = '41d563410ec152d2ae390396284fc776da940276141552a867ae775a797f93d2'
TAR_SHA = '41f61e4ae593336f63565ebf3f7da24948db14ff361a4706ee3d78b02b426938'
def h(raw): return hashlib.sha256(raw).hexdigest()
def read(path): return Path(path).read_bytes()
def canonical(value): return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def tree(path): return ast.parse(read(path),filename=str(path))
def definition(path,name):
    nodes=tree(path).body
    for part in name.split('.'):
        found=[n for n in nodes if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==part]
        assert len(found)==1,(str(path),name)
        node=found[0];nodes=node.body
    return copy.deepcopy(node)
def dump(node): return ast.dump(node,include_attributes=False)
def literal(path,name):
    for node in tree(path).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(name)

raw_map=read(HANDOFF/'sources.json');assert h(raw_map)==MAP_SHA
expected=json.loads(raw_map);assert len(expected)==464
assert raw_map==(json.dumps(expected,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
current={str(p.relative_to(ROOT)):h(read(p)) for p in ROOT.rglob('*') if p.is_file()}
assert not any(p.is_symlink() for p in ROOT.rglob('*'))
assert current==expected
archive=read(HANDOFF/'sources.tar.gz');assert h(archive)==TAR_SHA
with tarfile.open(HANDOFF/'sources.tar.gz','r:gz') as tf:
    members=tf.getmembers();assert all(m.isfile() for m in members)
    assert len({m.name for m in members})==len(members)==464
    assert {m.name:h(tf.extractfile(m).read()) for m in members}==expected
base=json.loads(read(ROOT/'actual_client_selection_base.json'))['selected']
selection=json.loads(read(ROOT/'actual_client_selection.json'))
overlays=selection['declared_overlays'];assert len(base)==431 and len(overlays)==5
assert {n for n,row in base.items() if expected[n]!=row['sha256']}==set(overlays)
for name,row in base.items():
    assert h(read(row['path']))==row['sha256']
for name,row in overlays.items():
    assert row['parent']==base[name]
    assert expected[name]==row['selected_sha256']
assert sum(expected[n]==row['sha256'] for n,row in base.items())==426
fixed=literal(ROOT/'client_role_source_pins.py','FIXED_CLIENT_SOURCES')
assert len(fixed)==93 and all(expected[n]==value for n,value in fixed.items())
assert len(literal(ROOT/'client_role_source_pins.py','ACTUAL_CLIENT_MODULES'))==14
core=literal(ROOT/'qwen_backend_identity.py','CORE_SHA256')
assert len(core)==34 and all(expected['runtime/jaxite_word/'+n]==v for n,v in core.items())
assert expected['runtime/jaxite_word/ckks_ctx.py']=='bc978cb7412b55416f7ed634a689ae616a263b0f47fd9eef996ba244f2505db6'
assert expected['cleanup_origin.py']=='4543b26bf1168e29bc21375c2f80a9aa99648205a9fd9aea6d5a8059c6d18175'

# Admission factoring preserves the selected original loader body except its
# name and first admission call. No semantic helper execution occurs here.
old_loader=Path(base['qwen_key_loader.py']['path'])
old=definition(old_loader,'_load_for_recording');new=definition(ROOT/'qwen_key_loader.py','_load_selected')
new.name=old.name;new.body[0]=copy.deepcopy(old.body[0]);assert dump(old)==dump(new)
assert dump(definition(old_loader,'_parameters'))==dump(definition(ROOT/'qwen_key_loader.py','_parameters'))
assert 'p=56' in ast.unparse(definition(ROOT/'qwen_key_loader.py','_parameters'))
old_stage=Path(base['client_stage.py']['path'])
for old_name,new_name in (('publish_input_for_recording','_publish_input'),('consume_output_for_recording','_consume_output'),('_admit','_admit_recording')):
    a=definition(old_stage,old_name);b=definition(ROOT/'client_stage.py',new_name);b.name=a.name
    assert dump(a)==dump(b),(old_name,new_name)
a=definition(old_stage,'_consumer_scope');b=definition(ROOT/'client_stage.py','_consumer_scope')
for node in ast.walk(b):
    if isinstance(node,ast.Name) and node.id=='_load_for_stage':node.id='_load_for_recording'
assert dump(a)==dump(b)
old_lifecycle=Path(base['qwen_file_lifecycle.py']['path'])
assert dump(definition(old_lifecycle,'ResponseFiles.run_client'))==dump(definition(ROOT/'qwen_file_lifecycle.py','ResponseFiles.run_client'))
old_publisher=Path(base['qwen_publisher.py']['path'])
for name in ('_Descriptor','Publisher.write_ciphertext','Publisher.finish','Publisher.close'):
    assert dump(definition(old_publisher,name))==dump(definition(ROOT/'qwen_publisher.py',name)),name

# Source-order evidence for the one concrete blocker, without observing /proc
# or executing any candidate function or capacity operation.
dep=read(ROOT/'qwen_client_dependencies.py').decode()
access=definition(ROOT/'qwen_client_dependencies.py','AccessScope.access')
access_text=ast.unparse(access)
assert access_text.index("p.resolve() != p")<access_text.index('self.kernel')
assert all(p in dep for p in ('/proc/self/status','/proc/self/cgroup','/proc/self/mountinfo'))
resource=read(ROOT/'qwen_client_resource.py').decode()
assert "rss=kibibytes('/proc/self/status','VmRSS')" in resource
assert "os.listdir('/proc/self/fd')" in resource
process=read(ROOT/'qwen_client_process.py').decode()
assert process.index('audit.install()')<process.index('capacity_observations.append(check_capacity(')<process.index('from qwen_backend_identity import install_core')
assert "event=='os.listdir'" not in dep and "event in('os.listdir'" not in dep

for name in ('qwen_client_entry_status.json','qwen_client_role_policy.json'):
    assert json.loads(read(ROOT/name))['actual_entry_ready'] is False
assert json.loads(read(ROOT/'qwen_client_dependencies.json'))['configured'] is False

read_targets = [
 'ACTUAL_CLIENT_SOURCE.md','actual_client_selection.json','actual_client_selection_base.json',
 'qwen_client_entry.py','qwen_client_process.py','qwen_client_authority_packet.py',
 'qwen_client_dependencies.py','qwen_client_resource.py','qwen_backend_identity.py',
 'qwen_client_backend.py','qwen_client_input_files.py','qwen_client_orchestration.py',
 'model_output.py','client_stage.py','qwen_key_loader.py','qwen_key_readers.py',
 'qwen_file_lifecycle.py','qwen_publisher.py','client_role_source_pins.py',
 'qwen_backend_composition.py','qwen_client_entry_status.json','qwen_client_role_policy.json',
 'qwen_client_dependencies.json','runtime/jaxite_word/__init__.py','runtime/jaxite_word/ckks_ctx.py',
 'client_loader_factoring.patch.txt','client_stage_factoring.patch.txt',
 'publisher_actual_parent.patch.txt','client_output_shared85.patch.txt']
receipt = {
 'schema':'qwen-actual-client464-source-peer-review-v1',
 'reviewer':'cnn_performance','source_root':str(ROOT),'source_manifest_sha256':MAP_SHA,
 'source_archive_sha256':TAR_SHA,'source_count':464,'current_archive_map_equal':True,
 'selected_base':431,'unchanged_base':426,'declared_overlays':overlays,
 'fixed_prerequisites':93,'required_role_module_names':14,'fixed_core_sources':34,
 'source_only':True,'candidate_imports':False,'tests_or_collection':False,
 'backend_or_context_calls':False,'capacity_or_proc_observations':False,
 'borrowed_body_or_private_reads':False,'source_edits':False,'actual_execution_ready':False,
 'review_conclusion':'One concrete source-derived audit/resource integration blocker; no additional concrete issue found in the bounded reviewed paths.',
 'findings':[{
  'id':'client-kernel-self-alias','severity':'blocks future actual execution',
  'source_files':['qwen_client_dependencies.py','qwen_client_resource.py','qwen_client_process.py'],
  'mechanism':'AccessScope.kernel contains literal /proc/self/status,cgroup,mountinfo. access rejects p.resolve()!=p before its kernel allowlist. On the specified Linux process /proc/self resolves to this PID; check_capacity after audit.install reaches status and triggers sticky open alias denial.',
  'ordering':'First /proc/meminfo read may pass; /proc/self/status is next, before backend import or output creation. Earlier independent admission failures remain possible.',
  'listdir_distinction':'check_capacity later uses os.listdir(/proc/self/fd) for FD count. This audit does not intercept os.listdir; that call is not the reported open-alias blocker. No listdir/capacity call was executed by this reviewer.',
  'evidence_kind':'Exact source/AST ordering, no runtime reproduction',
  'repair_boundary':'Use a separately reviewed exact current-process kernel path identity policy; retain fail-closed aliases for ordinary public/key/input/output paths. Preserve held464.'}],
 'reviewed_paths':{
  'entry':'External canonical compact map/full tree and fresh main/file/spec checks precede local helpers; false source status denies before packet/backend access; source cap64MiB is separate from unchanged body limits.',
  'public_authority':'Externally pinned complete packet reauthenticates model/key conditional/phase and distinct server-source envelope; no outcome-derived pins or inherited actual PASS.',
  'aliases':'Fixed34 bytes, paired flat/package identity and retained core/class bindings; loader retains original p56/defaultbc978 and patches both keygen aliases before context.',
  'embedding':'Granted file state precedes hash/header/body; original parent+file FDs retained; immutable float64 snapshot and original positions feed unchanged layout; no model/token reader.',
  'response':'Complete38 original held/named effective-UID root and six-field payload checks precede private manifest/key admission; original consumer arrays retire inside nested key scope; actual-scale/full65536/approximationTrue decoder and immutable snapshots unchanged.',
  'outputs':'Original external parent, UID/modes/exclusive outputs/fsync/readback; captured callback checks and shared cleanup policy; no deletion or uncertain-close retry in selected new output path.',
  'terminal':'Postpublication source/sticky audit and original output/audit retirement are attempted; escaped BaseException maps70, reports alone do not establish successful independent normal-zero/reap/root admission.'},
 'remaining_obligations':['Repair audit/kernel path mismatch in separate successor','Focused current-source orchestration tests and installed import/origin proof','Externally reviewed client resource policy','Ready source successor and independent conditional/phase grants','Real key/body/backend behavior remains unexecuted','Independent normal-zero/reap/group/root admission for any eventual actual result'],
 'read_source_pins':{name:expected[name] for name in read_targets},
 'review_script_sha256':h(read(__file__)),
}
assert {str(p.relative_to(ROOT)):h(read(p)) for p in ROOT.rglob('*') if p.is_file()}==expected
(OUT/'review.json').write_bytes(canonical(receipt))
print(json.dumps({'source_readback_passed':True,'review_has_blocker':True,'receipt':str(OUT/'review.json'),'sha256':h(read(OUT/'review.json'))},sort_keys=True))
