"""Independent BERT5 public security/device/source evidence review; no runtime imports."""
import ast,hashlib,json,math,os,tarfile
from datetime import datetime,timezone
from pathlib import Path
BASE=Path('/tmp/cross-optimized-twelve-20261103');SRC=BASE/'checkout';E=BASE/'evidence';CASE=E/'bert/test5';OUT=Path(__file__).parent
sha=lambda b:hashlib.sha256(b).hexdigest()
PUBLIC=('process.json','security_device.json','resources.json','resource_samples.jsonl','stdout.json','stderr.log')
blobs={n:(CASE/n).read_bytes()for n in PUBLIC};before={n:sha(b)for n,b in blobs.items()}
p=json.loads(blobs['process.json']);s=json.loads(blobs['security_device.json']);res=json.loads(blobs['resources.json']);result=json.loads(blobs['stdout.json'])
assert p['passed']is False and p['returncode']==1 and type(p['returncode'])is int and p['result']==result
assert result=={'case':'test5.in','passed':False,'error':'int too large to convert to float','error_type':'OverflowError','notes':[]}
assert p['timeout_s']is None
for flag in ('source_files_unchanged','process_reaped','public_key_store_removed','owned_directory_handle_closed'):assert p[flag]is True
assert not any(k in p for k in ('process_cleanup_error','key_cleanup_error','directory_handle_cleanup_error'))
assert s['passed']is False and s['error']=='SystemExit: 1'
assert len(s['events'])==2 and [(e['index'],e['boundary'])for e in s['events']]==[(0,'input'),(1,'output')]
manifest_raw=(E/'snapshot_manifest.json').read_bytes();m=json.loads(manifest_raw);expected=m['source_hashes']
assert sha(manifest_raw)=='1fbca19a43f0830f4beb9ad00659c5b79e01ea10d89aa6466801340373f210bc'
assert len(expected)==199 and (SRC/'validation_snapshot.json').read_bytes()==manifest_raw
current={n:sha((SRC/n).read_bytes())for n in expected};assert current==expected
folders=('jaxite_word','demos','cadathlon/common','cadathlon/agent_pset/p2_bert','cadathlon/agent_pset/p4_qwen')
suffixes={'.py','.json','.in','.out','.md','.golden','.sh','.toml','.txt'}
paths={p.relative_to(SRC).as_posix()for folder in folders for p in (SRC/folder).rglob('*') if p.is_file()and '__pycache__'not in p.parts and (p.suffix in suffixes or p.name=='Makefile')}
paths|={'cadathlon/validation/check_transformer_suite.py','validation_sweep.py'}
assert paths==set(expected) and all(not(SRC/n).is_symlink()for n in expected)
archive=E/'sources.tar.gz';assert sha(archive.read_bytes())=='bc0ad1b2a3796fe4ecb5944d68f3ca65561ad5740357470a1e634cc243ec387b'
with tarfile.open(archive,'r:gz')as tf:
 ms=tf.getmembers();assert len(ms)==199 and all(x.isfile()for x in ms)and len({x.name for x in ms})==199
 assert {x.name:sha(tf.extractfile(x).read())for x in ms}==expected
# Read the exact tabulated ceiling from the pinned AST, without importing HE/JAX.
tree=ast.parse((SRC/'jaxite_word/he_params.py').read_bytes())
table=next(n.value for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='_OPENFHE_CEILINGS_RAW'for t in n.targets))
ceiling=None
for key,value in zip(table.keys,table.values):
 if isinstance(key,ast.Tuple)and isinstance(key.elts[0],ast.Attribute)and key.elts[0].attr=='CLASSICAL'and ast.literal_eval(key.elts[1])==128:
  ceiling=ast.literal_eval(value)[131072]
assert ceiling==3523
summaries=[]
for e in s['events']:
 rings=e['actual_rings'];assert len(rings)==2 and [x['dnum']for x in rings]==[None,4]
 assert {k:v for k,v in rings[0].items()if k!='dnum'}=={k:v for k,v in rings[1].items()if k!='dnum'}
 for a in rings:
  for k,v in (('degree',131072),('num_slots',65536),('num_q',100),('num_p',24),('max_level',49),('security_bits',128)):
   assert type(a[k])is int and a[k]==v
  assert a['security_model']=='classical'
  q=a['q_towers'];aux=a['p_towers'];assert len(q)==100 and len(aux)==24
  assert all(type(x)is int and 1<x<2**31 and (x-1)%(2*131072)==0 for x in q+aux)
  assert len(set(q+aux))==124
  qp=math.prod(q+aux);P=math.prod(aux);parts=[math.prod(q[i:i+25])for i in range(0,100,25)]
  assert qp.bit_length()==3491 and qp.bit_length()<=ceiling and P>=max(parts)
  assert math.log2(math.prod(q))==a['log2_Q'] and math.log2(qp)==a['log2_QP']
 assert rings==s['events'][0]['actual_rings']
 assert len(e['ciphertexts'])==1;ct=e['ciphertexts'][0]
 towers=100 if e['boundary']=='input'else40
 assert ct['shape']==[1,2,256,512,towers] and ct['degree']==131072 and ct['moduli_count']==towers
 assert ct['payload_bytes']==math.prod(ct['shape'])*4
 assert type(ct['scale'])is float and math.isfinite(ct['scale'])and ct['scale']>0
 assert ct['nsd']==(None if e['boundary']=='input'else1)
 assert len(ct['devices'])==1;device=ct['devices'][0]
 assert device['platform']=='tpu'and device['id']==0 and device['device_kind']=='TPU v6 lite'
 assert device['memory_stats']['bytes_in_use']<=device['memory_stats']['peak_bytes_in_use']<=device['memory_stats']['bytes_limit']
 summaries.append(dict(boundary=e['boundary'],shape=ct['shape'],physical_q_towers=towers,payload_bytes=ct['payload_bytes'],scale=ct['scale'],scale_hex=ct['scale'].hex(),nsd=ct['nsd'],devices=ct['devices']))
input_scale=summaries[0]['scale'];output_scale=summaries[1]['scale']
assert output_scale-input_scale==math.ulp(input_scale)==8.0
samples=[json.loads(line)for line in blobs['resource_samples.jsonl'].splitlines()]
assert len(samples)==res['samples']==18957 and res==p['resources']
assert all(x['leader']==p['pid']for x in samples)
assert all(x['aggregate_rss_bytes']==sum(y['rss_bytes']for y in x['processes'].values())for x in samples)
assert max(x['aggregate_rss_bytes']for x in samples)==res['sampled_peak_process_tree_rss_bytes']==151212969984
assert all(samples[i]['monotonic_s']<=samples[i+1]['monotonic_s']for i in range(len(samples)-1))
assert p['pid']==2610682 and not Path('/proc',str(p['pid'])).exists()
try:os.killpg(p['pid'],0)
except ProcessLookupError:group_absent=True
else:group_absent=False
assert group_absent
try:os.lstat(p['owned_public_key_directory'])
except FileNotFoundError:key_directory_absent=True
else:key_directory_absent=False
assert key_directory_absent
selected=('validation_sweep.py','cadathlon/validation/check_transformer_suite.py','cadathlon/common/support.py','cadathlon/common/transformer_runner.py','cadathlon/agent_pset/p2_bert/src/runner.py','cadathlon/agent_pset/p2_bert/src/bert_optimized_reference.py','jaxite_word/he_params.py','jaxite_word/util.py')
assert {n:sha((SRC/n).read_bytes())for n in expected}==expected
assert {n:sha((CASE/n).read_bytes())for n in PUBLIC}==before
receipt=dict(schema='bert-toy5-security-device-public-review-v1',created_utc=datetime.now(timezone.utc).isoformat(),
 scope='Public failed-case/source/resource evidence only. No TPU/backend invocation, ciphertext/key/private/model body read, decryption, numerical rerun, or source change.',
 case_root=str(CASE),case_public_sha256=before,case_public_bytes_unchanged=True,
 outcome='Preserved FAILURE: runner OverflowError after both recorded security/device boundaries. No decoded numerical metrics or prediction result published.',
 original_runner_returncode=1,returncode_basis='Pinned process.json recorded direct-child wait; reviewer observes present absence, not a second wait.',
 process_pid=p['pid'],process_group_absent_at_review=True,key_directory_absent_at_review=True,recorded_key_cleanup=True,recorded_handle_close=True,
 source_count=199,source_manifest_sha256=sha(manifest_raw),source_archive_sha256=sha(archive.read_bytes()),current_manifest_archive_equal=True,reviewed_source_sha256={n:expected[n]for n in selected},
 actual_ring=dict(degree=131072,slots=65536,num_q=100,num_p=24,max_level=49,context_dnum_values=[None,4],q_log2=math.log2(math.prod(q)),p_log2=math.log2(P),qp_log2=math.log2(qp),qp_integer_bit_length=3491,classical128_ceiling=ceiling,integer_security_gate=True,p_covers_four_consecutive25_tower_partitions=True,p_coverage_margin_log2=math.log2(P)-max(math.log2(v)for v in parts),moduli_distinct_below2pow31_and_ntt_congruent=True),
 event_summaries=summaries,output_scale_change=dict(delta=8.0,ulp=8.0,scope='Observed metadata only, not an identified overflow cause or numerical gate.'),
 resources=dict(samples=18957,sampled_process_tree_rss_peak_bytes=151212969984,recorded_output_device_peak_bytes=29580395008,process_elapsed_s=p['process_elapsed_s'],scope=res['scope']),
 interpretation=[
 'validation_sweep.gate calls the unchanged require_secure_execution before appending either event; both recorded events therefore attest that original live-context/ciphertext security gate returned.',
 'The gate checks actual context Q/P security, sigma/default threshold, secret distribution and ciphertext ring membership. Sigma/distribution are not independently serialized here; only the gate-return evidence supports those checks.',
 'The dnum-null context and dnum4 context expose identical actual Q/P; the null field is retained, not converted into a second decomposition claim.',
 'benchmark_session synchronizes evaluator result then performs output security gate before session.decrypt, metrics and check_output. Output event is not decoded correctness evidence.',
 'security_device.passedFalse/errorSystemExit1 is the instrumented wrapper preserving runner failure, not an observed ring-security rejection.',
 'TPU placement and device memory are checkpoint observations, not proof of every internal operation or continuous peak. Process-tree RSS is sampled, may double-count shared pages, and is not device HBM.',
 'No benchmark sample or decoded error/prediction metric exists in the failed runner result. The outer process elapsed time is not a validated model benchmark.',
 'Overflow location is left to the separate source diagnosis; public evidence has no traceback or ciphertext values to prove a precise arithmetic cause.',
 'The twelve-case sweep remains independently owned and active; this failed-case review does not establish an overall sweep pass.'
 ])
pout=OUT/'review.json';pout.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps({'receipt':str(pout),'sha256':sha(pout.read_bytes()),'source_count':199,'qp_bit_length':3491,'ceiling':ceiling,'preserved_failure':True}))
