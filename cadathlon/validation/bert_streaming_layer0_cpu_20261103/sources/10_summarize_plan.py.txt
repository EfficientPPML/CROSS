"""Read-only row-packed full-width BERT planning; no keygen or HE execution."""
from pathlib import Path
import datetime, hashlib, json, math, os
HERE=Path(__file__).resolve().parent
proof_path=HERE/'native-result3/report.json'
inventory_path=Path('/tmp/streaming-bert-review/inventory.json')
proof=json.loads(proof_path.read_text());inventory=json.loads(inventory_path.read_text())
assert proof['passed'] and proof['source_files_unchanged']
slots=65536; degree=131072; q=100; p=24; partitions=4; rows=16; tiles=8
ct=2*degree*q*4; pt=degree*q*4
key=lambda n:2*partitions*degree*(n+p)*4
storage={name:{'available_bytes':os.statvfs(name).f_bavail*os.statvfs(name).f_frsize,
                'total_bytes':os.statvfs(name).f_blocks*os.statvfs(name).f_frsize} for name in ('/tmp','/dev/shm')}
maxerr=max(max(c['maximum_absolute_errors'].values()) for c in proof['cases'])
matrices={}
for name,m in inventory['dense_linear_matrices'].items():
 r,c,n1=m['rows'],m['columns'],m['baby_step']
 active=[d%slots for d in range(-(r-1),c)]
 babies={d%n1 for d in active}; giants={d//n1 for d in active}
 # Current preprocess stages one giant key even for the j=0 contribution.
 bound=(len(babies)-1)*key(q)+len(giants)*key(q-2)
 matrices[name]={
  'shape':[r,c],'active_diagonals_upper_bound':len(active),'baby_vectors':len(babies),
  'giant_groups':len(giants),'baby_vector_payload_at_level49_bytes':len(babies)*ct,
  'one_max_giant_plaintext_group_bytes':n1*pt,
  'staged_rotation_key_uint32_payload_bytes':bound,
  'scope':'Current BSGS dense-support upper bound at level49, excludes copies, coefficient maps, controls and kernel workspaces; giant keys use level48. memory_bounded streams diagonals, but still stages all baby/giant keys and baby ciphertexts.'}
report={
 'scope':'Plaintext pinned layer0 layout proof plus public payload/key/depth planning; no encrypted block, full model, MRPC, runtime or HBM claim.',
 'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'proof':{'path':str(proof_path),'sha256':hashlib.sha256(proof_path.read_bytes()).hexdigest(),
          'cases':len(proof['cases']),'passed':proof['passed'],'max_abs_block_or_attention_error':maxerr,
          'elapsed_s':proof['elapsed_s'],'peak_rss_kib_plaintext_oracle_process':proof['peak_rss_kib'],
          'retains_actual_padded_query_rows':True},
 'layout':{'row_width':4096,'head_count':12,'head_width':64,'active_hidden':768,'mlp_width':3072,
           'slots':slots,'rows_per_tile':rows,'sequence':128,'tiles_per_example':tiles,
           'ciphertext_period':slots,'alias_count':1,
           'streaming':'Query tile outer loop, key tile middle loop, 16 cyclic whole-tile row offsets; all keys meet each query exactly once.',
           'batching':'Stream examples independently; no cross-example ciphertext batching assumed.',
           'public_masks':'Key-validity and head-anchor masks only. Logical padded queries remain computed; tail rows beyond sequence length are discarded by public position.'},
 'attention_counts_per_example_layer':proof['cases'][0]['layout_plan']['actual_simulated_counts'],
 'attention_rotation_keys':proof['cases'][0]['layout_plan']['attention_rotation_indices'],
 'depth':{
  'opaque_exponential_depth':'E','opaque_reciprocal_depth':'R',
  'schedule':[{'stage':'Q/K/V public linear projection','cumulative_depth':'1'},
              {'stage':'Q*K','cumulative_depth':'2'},
              {'stage':'head sum anchor mask and public score/mean scale','cumulative_depth':'3'},
              {'stage':'mean subtraction then key-validity mask before exp','cumulative_depth':'4'},
              {'stage':'opaque exp','cumulative_depth':'4+E'},
              {'stage':'zero invalid exp outputs','cumulative_depth':'5+E'},
              {'stage':'denominator reciprocal; neutral inactive denominators are public ones','cumulative_depth':'5+E+R'},
              {'stage':'reciprocal head-anchor mask','cumulative_depth':'6+E+R'},
              {'stage':'numerator times broadcast reciprocal','cumulative_depth':'7+E+R'},
              {'stage':'attention output public projection','cumulative_depth':'8+E+R'}],
  'parallel_numerator_depth':'6+E before the final product',
  'excluded':'LayerNorm, GELU, residual alignment, MLP, CKKS refresh and all realized nonlinear circuits. No numeric total depth or refresh count is claimed.'},
 'payload_plan':{
  'ciphertext_max_level_bytes':ct,'plaintext_max_level_bytes':pt,
  'eight_tile_activation_set_bytes':8*ct,'resident_x_q_k_v_32_tile_payload_bytes':32*ct,
  'proposed_host_or_disk_spill':'X/K/V plus completed output: at most32 tile payloads per example,3.125GiB at maxlevel. Q is produced one tile at a time. Disk spill and asynchronous lifetime implementation are not present in this NumPy prototype.',
  'proposed_attention_explicit_schedule':{
    'permanent_second_pass_CT_buffers':['query','mean','numerator','denominator','current_key_tile','current_value_tile'],
    'max_additional_CT_buffers':3,'ordinary_arithmetic_slot_buffer_count':9,
    'ordinary_arithmetic_max_level_payload_bytes':9*ct,
    'assumptions':['Old immutable arithmetic temporaries are released as soon as their sole consumer has synchronized.',
      'A rotate/add butterfly has old, rotated and new accumulator payloads: three including its current work buffer.',
      'The exp/reciprocal call is atomic only in this accounting; its additional working set must be added separately.',
      'Public plaintext masks, key bindings, NTT tables, decomposition scratch, compiler buffers, cached inputs and output queues are excluded. This is not an HBM peak bound.',
      'The current NumPy simulator does not implement ciphertext disk/host transfers or this explicit release schedule.']},
  'bsqs_current_implementation_upper_bound':matrices,
  'full_slot_refresh':'No sparse alias averaging; the period128 numerical proof does not cover period65536. Full-slot refresh numerical accuracy and memory remain unvalidated.'},
 'key_inventory':{
  'independent_inventory_path':str(inventory_path),'independent_inventory_sha256':hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
  'actual_qp_profile':inventory['profile'],'raw_pair_bytes':key(q),
  'direct_graph_rotations':inventory['alternatives']['direct_token']['graph_rotation_count'],
  'fullslot_bootstrap_rotations':inventory['alternatives']['direct_token']['bootstrap_rotation_count'],
  'union_rotations':inventory['alternatives']['direct_token']['combined_rotation_count'],
  'union_raw_pair_bytes':inventory['alternatives']['direct_token']['combined_raw_public_key_bytes'],
  'includes':'All row graph layernorm/dense/classifier keys plus planned full-slot refresh, not just the demonstrated attention routine.',
  'excludes':'Relinearization/public encryption keys, maps, manifests, file-format overhead, copied/device-staged arrays and compiler memory.'},
 'live_filesystem_snapshot':storage,
 'domain_observations':{c['case']:c['observed_ranges'] for c in proof['cases']},
 'next_minimal_prototype':[
  'Use a separate scalar-score layout per query tile:16 queries ×128 keys ×16 padded heads=32768 slots, versus repeated4096 features.',
  'Construct and independently verify a public linear compaction/scatter map between head anchors and scalar scores. Its rotations and mask levels are additional and not covered by this proof.',
  'Apply the previously planned normalized repeated-squaring softmax to that scalar layout, with new public domains covering padded queries or explicitly retain public valid-query compaction.',
  'Scatter each fixed key offset probability to head anchors and stream weighted V accumulation; the probability route avoids storing large raw exponentials.',
  'Prove full-slot refresh and actual row-linear controls/workspaces before proposing a full encrypted layer or checkpoint.'],
 'limitations':['Observed domains on seven cases are not certified bounds or a replacement calibration.',
  'The exact raw-exponential route reaches4.496e6 on a public boundary mask; it cannot inherit bound16 refresh assumptions.',
  'Current fully staged BSGS key controls can exceed an individual TPU chip even with memory_bounded=True; lower-level scheduling or streaming key bindings needs separate implementation/proof.',
  'The union raw public-key store exceeds current /tmp free space. /dev/shm capacity is a live snapshot shared with other jobs; fit is not reserved or measured performance.',
  'No fitted logits/labels, clipping, private-value branch, HE inference, full12layer model or MRPC accuracy result.']}
source_paths=[Path(__file__),HERE/'row_attention.py',HERE/'check_pretrained_block.py',
 Path('/home/jianming_gatech/CROSS/jaxite_word/bsgs.py'),Path('/home/jianming_gatech/CROSS/jaxite_word/he_params.py')]
report['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
(HERE/'planning.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'proof':report['proof'],'keys':report['key_inventory'],'linear_working_sets':matrices,'storage':storage},indent=2))
