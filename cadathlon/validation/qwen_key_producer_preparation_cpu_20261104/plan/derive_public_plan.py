"""Read pinned public metadata/source only; no producer import, bodies or tests."""
from pathlib import Path
import ast,hashlib,json,math
HERE=Path(__file__).resolve().parent
ROOT=Path('/tmp/qwen-organizer-file-engine-integration-v2-20261104')
OLD=Path('/tmp/bert-client-clean-launch-source-proposals-20261104/combined125/source')
TOY=Path('/tmp/bert-toy4-checkpoint-component-v4-20261104')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(path):return path.read_bytes()
def pin(path):return dict(path=str(path),sha256=sha(read(path)))
def write(name,v):(HERE/name).write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')
profile_path=ROOT/'key_adapter_source/public_inputs/profile_audit.json';plan_path=ROOT/'key_adapter_source/public_inputs/actual_key_plan.json'
profile=read(profile_path);plan=read(plan_path)
assert sha(profile)=='2b2457e50da73b9ccf87984c089afcb99586fde863a48e98b89a1b7ccf872a17'
assert sha(plan)=='bac55debfe95752a16a9a2e849e66488f4afbb576959403a2af4698358dd32ca'
p=next(x for x in json.loads(profile)['profiles']if x['name']=='depth49_p60_d8');keys=json.loads(plan)['combined_keys']
assert len(keys)==166 and keys==sorted(set(keys))and all(type(i)is int for i in keys)and 65535 in keys and 262143 in keys
q=p['q'];P=p['p'];n=p['degree'];assert(len(q),len(P),n,p['depth'],p['dnum'])==(100,14,131072,49,8)
alpha=(len(q)+7)//8;parts=[math.prod(q[i:i+alpha])for i in range(0,len(q),alpha)];assert len(parts)==8
qp=math.prod(q+P);assert qp.bit_length()<=3523 and math.prod(P)>=max(parts)
rows=[dict(name='public_key.npy',shape=[2,100,n],moduli='Q100',raw_bytes=2*100*n*4)]
rows +=[dict(name='relin_'+s+'.npy',shape=[8,114,n],moduli='Q100+P14',raw_bytes=8*114*n*4)for s in ('a','b')]
rows +=[dict(name=f'rot_{k}_{s}.npy',shape=[8,114,n],moduli='Q100+P14',raw_bytes=8*114*n*4)for k in keys for s in ('a','b')]
assert len(rows)==335
raw=sum(x['raw_bytes']for x in rows);private=100*n*4
public=dict(schema='qwen-key-generation-public-plan-v1',execution_authorized=False,scope='Static public descriptor derivation only; not generated keys, backend security execution or a launch grant',profile=pin(profile_path),rotation_plan=pin(plan_path),parameters=dict(degree=n,num_slots=65536,q_towers=q,p_towers=P,composite_degree=2,degree_layout=[256,512],precision=32,max_level=49,scaling_factor=p['canonical_scale'],dnum=8,batch=1,sigma=3.190000057220458984375,secret_distribution='uniform_ternary'),rotations=keys,public_array_inventory=rows,private_array=dict(name='secret.npy',shape=[100,n],dtype='uint32',moduli='Q100',raw_bytes=private),public_files=336,private_files=2,security=dict(target_bits=128,attack='classical',actual_log2_qp=sum(math.log2(x)for x in q+P),actual_qp_bit_length=qp.bit_length(),inclusive_ceiling=3523,p_product_bits=math.prod(P).bit_length(),partition_product_bits=[x.bit_length()for x in parts],p_coverage_exact=math.prod(P)>=max(parts),scope='Integer public-plan replay; real selected predicates remain a future pre-generation requirement'),resources=dict(public_raw_bytes=raw,private_raw_bytes=private,public_npy_upper=raw+335*4096,private_npy_upper=private+4096,public_required_bytes_existing_allowance=raw+335*4096+(1<<30),private_required_bytes_existing_allowance=private+4096+(1<<20),shared_filesystem_required_sum=raw+335*4096+(1<<30)+private+4096+(1<<20),held_output_descriptor_count=342,scope='Logical file inventory and inherited disk reserve only. No live capacity measurement, backend working-set, compiler/device/context buffers, peak memory or performance claim. Temporary verification descriptors require an implementation-specific extra allowance.'))
write('public_plan.json',public)
core={}
for path in sorted((ROOT/'runtime/jaxite_word').rglob('*')):
 if path.is_file():
  name=path.relative_to(ROOT).as_posix();other=OLD/name
  assert other.is_file()and read(path)==read(other)
  core[name]=pin(path)|dict(default125_path=str(other),selection='unchanged existing34 core closure; no imports performed')
assert len(core)==34
active=['qwen_key_contract.py','qwen_key_adapter.py','qwen_key_files_admission.py','qwen_file_contract.py','qwen_public_parse.py','public_source_pins.py','qwen_key_readers.py','key_descriptor_io.py','cleanup_origin.py','cleanup.py','qwen_finalizer.py']
assert sha(read(ROOT/'runtime/jaxite_word/key_gen.py'))=='742e6af9e1ce9025416fa2ef68ae037bca7e772d48fbbd6ca387cc79820119c8'
metadata={}
tree=ast.parse(read(ROOT/'public_source_pins.py'))
pins=next(ast.literal_eval(node.value)for node in tree.body if isinstance(node,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='PUBLIC_PINS'for t in node.targets))
for name,digest in pins.items():
 path=ROOT/'adapter238/parent208'/name;assert sha(read(path))==digest;metadata[name]=pin(path)
refs={
 'default125_client_keys':pin(OLD/'runtime/stage/client_keys.py'),
 'toy4v4_client_keys':pin(TOY/'stage/client_keys.py'),
 'publisher28_owner_reference':pin(ROOT/'qwen_publisher.py'),
 'admitted366_selection':pin(ROOT/'integration_selection.json'),
 'admitted366_manifest':pin(Path('/tmp/qwen-stage-integration-v2-sources-20261104.json')),
 'reader43_manifest':pin(Path('/tmp/qwen-key-reader-proof-20261104/development-v1/sources-before.json')),
 'default125_manifest':pin(OLD.parent/'source_manifest.json'),
 'terminal_status_reference':pin(Path('/tmp/qwen-expanded-candidate-checker-terminal-exit-successor-20261104/entry.py')),
}
funcs={}
oldast=ast.parse(read(OLD/'runtime/stage/client_keys.py'));toyast=ast.parse(read(TOY/'stage/client_keys.py'))
for name in ['_array','_switch_metadata','_security_check','inventory','resources']:
 a=next(x for x in oldast.body if isinstance(x,ast.FunctionDef)and x.name==name)
 b=next(x for x in toyast.body if isinstance(x,ast.FunctionDef)and x.name==name)
 sa=ast.dump(a,include_attributes=False);sb=ast.dump(b,include_attributes=False);assert sa==sb
 funcs[name]=dict(ast_sha256=sha(sa.encode()),equal_default125_toy4v4=True,reuse_scope='exact function candidate; module globals/entry/storage/schema are not selected wholesale')
write('source_selection.json',dict(schema='qwen-key-generation-producer-plan-selection-v1',status='PROPOSED_SOURCE_SELECTION_ONLY',actual_execution_authorized=False,core=core,unchanged_component_candidates={n:pin(ROOT/n)for n in active},public_metadata=metadata,reference_only=refs,exact_function_candidates=funcs,planned_new_files=['qwen_key_producer.py','qwen_key_store.py','test_qwen_key_producer.py','test_qwen_key_store.py'],excluded=['default125/Toy4 generate wrapper and BERT bindings/schema','old fdopen-owning/dup key-generation IO','Toy4 _authenticate_array header parser without pre-read cap','BERT130/199 rotation ledgers','key_contexts/context constructor/encode/decode/encrypt/decrypt and model/input stages','any actual entry/worker/grant until separately reviewed'],resolution='new root; selected runtime for package jaxite_word; exact existing dual-spelling package aliases pinned to these same34 sources; no ancestor/provenance dirs or installed fallback'))
print(json.dumps(dict(scope='stdlib source/public-metadata-only derivation',core_sources=34,component_candidates=len(active),public_metadata=len(metadata),keys=len(keys),public_arrays=len(rows),public_raw_bytes=raw,private_raw_bytes=private,generated_keys=False)))
