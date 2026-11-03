from collections import Counter
from pathlib import Path
import hashlib,json,shutil
base=Path('/tmp/qwen2-stack-sampling-20261103')
metadata=json.loads(base.with_suffix('.json').read_text())
raw=base.with_suffix('.raw').read_text()
assert hashlib.sha256(raw.encode()).hexdigest()==metadata['raw_sha256']
owners=Counter();leaves=Counter();stages=Counter();threads=Counter();total=compilation=0
for line in raw.splitlines():
 stack,count=line.rsplit(' ',1);n=int(count);frames=stack.split(';');total+=n
 leaves[frames[-1]]+=n;threads[frames[0]]+=n;compilation+=n*('backend_compile_and_load (' in stack)
 for frame in frames:
  if frame.startswith('_qwen ('):stages[frame]+=n
 if any(marker in stack for marker in ('get_shared_barrett_ntt_context (','basis_aligned_transformation (','ntt_coefficients_precompute (')):owner='NTT table construction including nested JAX work'
 elif any(marker in stack for marker in ('control_gen (','_build_level_params (','get_rot_key (')):owner='Other public control/key preparation including nested JAX work'
 elif '_encode_public (' in stack:owner='Plaintext encoding excluding controls'
 elif '_clear_working_caches (' in stack:owner='Cache release'
 else:owner='Ciphertext arithmetic/dispatch/compilation outside preparation'
 owners[owner]+=n
report=dict(metadata,sample_count=total,sampling_errors=4,threads=dict(threads),mutually_exclusive_stack_ownership={name:dict(count=count,percent=100*count/total) for name,count in owners.items()},inclusive_backend_compile_samples=compilation,inclusive_backend_compile_percent=100*compilation/total,leaf_frames=dict(leaves.most_common()),graph_frames=dict(stages),interpretation='This window spans the end of V projection and start of Q projection. Control/table preparation owns 147/195 sampled stacks (75.38%). Backend compilation appears in 99/195 (50.77%) and overlaps those ownership categories. These are sample fractions from one 20-second phase window, not CPU utilization, whole-model fractions or speedup predictions.')
out=Path('/home/jianming_gatech/CROSS/cadathlon/validation');prefix='qwen2_tpu_stack_window_20261103'
for suffix in ('raw','log','json'):shutil.copyfile(base.with_suffix('.'+suffix),out/(prefix+'_collection.'+suffix))
shutil.copyfile(Path('/tmp/profile_qwen2_stack.py'),out/(prefix+'_collector.py'))
shutil.copyfile(Path(__file__),out/(prefix+'_analysis.py'))
(out/(prefix+'.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({key:report[key] for key in ('sample_count','sampling_errors','mutually_exclusive_stack_ownership','inclusive_backend_compile_percent','elapsed_s','observer_child_user_cpu_s','observer_child_system_cpu_s')},indent=2))
