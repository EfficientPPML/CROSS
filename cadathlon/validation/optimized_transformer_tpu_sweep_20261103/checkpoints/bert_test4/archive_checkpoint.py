"""Read-only failed checkpoint copy; never polls/reaps the live parent sweep."""
from pathlib import Path
import datetime,hashlib,json,math,os,shutil,tarfile
LAB=Path('/tmp/cross-optimized-twelve-20261103');EV=LAB/'evidence';SRC=EV/'bert/test4'
DEST=Path('/home/jianming_gatech/CROSS/cadathlon/validation/optimized_transformer_tpu_sweep_20261103/checkpoints/bert_test4')
SHA=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ARCHIVE='bc0ad1b2a3796fe4ecb5944d68f3ca65561ad5740357470a1e634cc243ec387b'
names={'process.json','stdout.json','stderr.log','resource_samples.jsonl','resources.json','security_device.json'}
assert {f.name for f in SRC.iterdir()}==names
before={n:SHA(SRC/n) for n in sorted(names)}
manifest=json.loads((EV/'snapshot_manifest.json').read_bytes());hashes=manifest['source_hashes'];assert len(hashes)==199
assert {n:SHA(LAB/'checkout'/n) for n in hashes}==hashes
with tarfile.open(EV/'sources.tar.gz')as t:
 ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==199 and all(m.isfile() for m in ms)
 assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in ms}==hashes
assert SHA(EV/'sources.tar.gz')==ARCHIVE
process=json.loads((SRC/'process.json').read_bytes());numeric=json.loads((SRC/'stdout.json').read_bytes());security=json.loads((SRC/'security_device.json').read_bytes())
assert process['passed']is False and process['returncode']==1 and process['result']==numeric and numeric['passed']is False
assert numeric['error_type']=='ApproximationErrorTooHigh' and 'stddev=inf' in numeric['error'] and '3.125e-02' in numeric['error']
assert all(process[n]is True for n in ('process_reaped','public_key_store_removed','owned_directory_handle_closed','source_files_unchanged'))
assert not Path('/proc',str(process['pid'])).exists() and not Path(process['owned_public_key_directory']).exists()
try:os.killpg(process['pid'],0)
except ProcessLookupError:pass
else:raise AssertionError('completed failed child group remains')
assert Path('/proc/944985').exists()
assert security['passed']is False and security['error']=='SystemExit: 1'
assert [e['boundary'] for e in security['events']]==['input','output']
for e in security['events']:
 for r in e['actual_rings']:
  assert r['degree']==131072 and r['security_bits']==128 and r['security_model']=='classical' and len(r['q_towers'])==100 and len(r['p_towers'])==24
  qp=sum(math.log2(q) for q in r['q_towers']+r['p_towers']);assert abs(qp-r['log2_QP'])<1e-8 and qp<=3523
 for c in e['ciphertexts']:assert c['devices'] and all(d['platform']=='tpu' for d in c['devices'])
resources=json.loads((SRC/'resources.json').read_bytes());assert resources==process['resources']
samples=[json.loads(line) for line in (SRC/'resource_samples.jsonl').read_text().splitlines()]
assert len(samples)==resources['samples'] and max(r['aggregate_rss_bytes'] for r in samples)==resources['sampled_peak_process_tree_rss_bytes']
parent_bytes=(EV/'report.json').read_bytes();parent=json.loads(parent_bytes);cases=[r for m in parent['models'].values() for r in m['cases'].values()]
assert len(cases)==8 and sum(r['passed']is True for r in cases)==7 and parent['passed']is False
assert parent['models']['bert']['cases']['test4.in']==process
assert before=={n:SHA(SRC/n) for n in names}
DEST.mkdir(exist_ok=False)
for n in names:shutil.copy2(SRC/n,DEST/n)
(DEST/'parent_report_at_checkpoint.json').write_bytes(parent_bytes)
summary=dict(scope='Eighth completed optimized toy sweep case; first failure, BERT test4 only. Parent twelve-case sweep remains live and root-owned.',
 verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),case_passed=False,case_returncode=1,case_pid=process['pid'],case_pid_and_group_absent=True,
 parent_pid=944985,parent_present=True,parent_tool_session=12657,parent_tool_reaping_owner='root',parent_tool_exit_code=None,
 completed_cases=8,passed_cases=7,failed_cases=1,required_total_cases=12,overall_sweep_passed=False,
 source_count=199,source_archive_and_current_snapshot_verified=True,source_archive_sha256=ARCHIVE,source_file_sha256=hashes,
 failed_case_files_before=before,failed_case_files_after={n:SHA(SRC/n) for n in names},
 numerical_result=numeric,logit_metrics_available=False,runner_timing_metrics_available=False,
 external_case_elapsed_s=process['process_elapsed_s'],resources=resources,
 input_output_security_gate_return_observed=True,security_report_passed=False,security_report_error=security['error'],
 output_ciphertext=security['events'][1]['ciphertexts'][0],
 cleanup={n:process[n] for n in ('process_reaped','public_key_store_removed','owned_directory_handle_closed')},
 limitations=['Toy dimensions hidden24/MLP48/heads4/layers2,batch2/sequence8; not a full pretrained BERT/MRPC run.',
 'Decoder error estimate overflowed; no decoded output or max-error/nRMSE/top1 values are available.',
 'Two metadata/device boundaries and five completed refresh log pairs do not localize first numerical corruption.',
 'Conditional polynomial/public-amplitude contracts exclude accumulated CKKS/refresh errors; no gate/profile/threshold was changed.',
 'Sampled RSS may double count shared pages and miss peaks; output device stats are checkpoint observations.',
 'No new TPU execution, retry, source mutation, parent reap or calibration was performed by this archive action.'])
(DEST/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
shutil.copyfile(Path(__file__),DEST/'archive_checkpoint.py')
print(json.dumps(dict(destination=str(DEST),source_count=199,case_passed=False,completed_cases=8,passed_cases=7,failed_cases=1)))
