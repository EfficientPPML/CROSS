"""Public source/shape audit only. No NumPy, JAX, keys, contexts or ciphertexts."""
from pathlib import Path
import ast,hashlib,io,json,tarfile
ROOT=Path('/tmp/bert-server-compute-stage4-successor-v2-20261103')
HERE=Path(__file__).resolve().parent
FILES=['runtime/jaxite_word/herot.py','runtime/jaxite_word/he_ops.py','runtime/jaxite_word/he_params.py','runtime/jaxite_word/finite_field.py','runtime/jaxite_word/bconv.py']
FILES += ['joined/checkout/cadathlon/common/'+n+'.py' for n in ('encrypted_rotation','encrypted_compilation','encrypted_polynomial','encrypted_packed_qwen','encrypted_shallow_score_routing','encrypted_factored_score_routing')]
FILES += ['runtime/stage/server_inputs/p60_profile_and_refresh.json']

def digest(b):return hashlib.sha256(b).hexdigest()
def method(raw,cls,name):
 tree=ast.parse(raw);c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
 return next(n for n in c.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
def calls(node):
 return sorted((n.lineno,n.func.attr) for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute))
def main():
 raw={n:(ROOT/n).read_bytes() for n in FILES};raw['audit.py']=Path(__file__).read_bytes();before={n:digest(v) for n,v in raw.items()}
 h=raw['runtime/jaxite_word/herot.py'];ordinary=method(h,'_HERotKernel','_rotate_array');decomp=method(h,'_HERotKernel','_decompose_array');part=method(h,'_HERotKernel','_decompose_part')
 tail=[name for line,name in calls(ordinary) if name in ('_key_switch_array','_mod_down_array','_add_first_component_array','_automorphism_array')]
 assert tail==['_key_switch_array','_mod_down_array','_add_first_component_array','_automorphism_array']
 assert not ({'eval_a','eval_b','coef_map','rot_index'} & {n.id for n in ast.walk(part) if isinstance(n,ast.Name)})
 assert not ({'eval_a','eval_b','coef_map','rot_index'} & {n.id for n in ast.walk(decomp) if isinstance(n,ast.Name)})
 s=raw['joined/checkout/cadathlon/common/encrypted_shallow_score_routing.py'];stage=method(s,'ShallowScoreRouting','_stage')
 rotations=[n for n in ast.walk(stage) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='rotate']
 assert len(rotations)==1 and isinstance(rotations[0].args[0],ast.Name) and rotations[0].args[0].id=='value'
 p=json.loads(raw[FILES[-1]])['profile'];q=p['q'];aux=p['p'];N=p['degree'];d=p['dnum'];alpha=(len(q)+d-1)//d
 assert (N,p['depth'],len(q),len(aux),d,alpha)==(131072,49,100,14,8,13)
 assert max(q+aux)<2**31
 levels=[]
 for level in (49,48,45,25,24,23,21,15,9,2):
  L=2*(level+1);parts=(L+alpha-1)//alpha;digit=parts*N*(L+len(aux))*4;ct=2*N*L*4
  levels.append(dict(level=level,q_count=L,p_count=len(aux),effective_parts=parts,full_digits_uint32_bytes=digit,
   c1_intt_uint32_bytes=N*L*4,source_ciphertext_bytes=ct,one_formatted_key_pair_uint32_bytes=2*digit,
   two_key_accumulators_uint64_bytes=2*N*(L+len(aux))*8,
   retained_nonzero_outputs_head11_bytes=11*ct,retained_nonzero_outputs_key15_bytes=15*ct,
   all15_key_pairs_if_stacked_bytes=30*digit))
 public_counts=dict(origin='Parent/CNN reported fullK128 model counts; not recomputed by this source-only audit',
  eligible_nonzero_rotations=638976,total_model_rotations=906829,nonzero_rotations_per_map=26,
  maps=638976//26,decomposition_groups=2*(638976//26),decomposition_calls_saved_if_one_per_group=638976-2*(638976//26))
 assert public_counts['maps']*26==public_counts['eligible_nonzero_rotations']
 report=dict(passed=True,scope='Public AST and integer storage arithmetic only; no numerical/HE/JAX execution or performance/peak-memory claim.',
  source_hashes_before=before,checks=dict(ordinary_generic_order=tail,prepared_digits_independent_of_key_map=True,
   routing_stage_rotates_same_source=True,sub31_bit_barrett_envelope=True),profile=dict(degree=N,depth=p['depth'],Q=len(q),P=len(aux),dnum=d,alpha=alpha),
  public_counts=public_counts,storage=levels,
  feasibility='Separate same-input decomposition from each original per-key completion; keep per-key ModDown before c0-add and coefficient map.',
  not_a_drop_in='_hoisted_rotate_array changes completion order by adding/automorphing inQP; do not substitute it.',
  exactness_status='Plausible deterministic factorization. No new bit-exact or secure parity execution was performed; fused Barrett versus composed generic tail must be checked.',
  initial_scope='Explicit opt-in serial single-device CD2 standard Barrett, orderedQ/P and immutable input, full-ring supplied keys, bounded group lifetime; ordinary rotate fallback elsewhere.',
  source_hashes_after={n:digest((ROOT/n).read_bytes()) if n!='audit.py' else digest(Path(__file__).read_bytes()) for n in raw})
 assert report['source_hashes_after']==before
 with tarfile.open(HERE/'sources.tar.gz','w:gz') as tar:
  for n,v in raw.items():m=tarfile.TarInfo(n);m.size=len(v);tar.addfile(m,io.BytesIO(v))
 with tarfile.open(HERE/'sources.tar.gz') as tar:
  ms=tar.getmembers();assert len(ms)==len(before)==len({m.name for m in ms});assert {m.name:digest(tar.extractfile(m).read()) for m in ms}==before
 report['source_archive_sha256']=digest((HERE/'sources.tar.gz').read_bytes());report['source_archive_verified']=True
 (HERE/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(dict(passed=True,source_count=len(before),source_archive_sha256=report['source_archive_sha256'],storage_levels=len(levels))))
if __name__=='__main__':main()
