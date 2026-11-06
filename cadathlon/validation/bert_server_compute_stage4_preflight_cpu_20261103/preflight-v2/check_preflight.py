from pathlib import Path
import hashlib,importlib.util,json,os,subprocess,sys,time
ROOT=Path('/tmp/bert-server-compute-stage4-successor-v2-20261103');sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('stage4_preflight_collector',ROOT/'collect.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
OUT=ROOT/'source-bound-preflight-v2';OUT.mkdir(exist_ok=False)
EXTERNAL={'external/preflight.py':Path(__file__),'external/public_gates.py':Path('/tmp/check_stage4_actual_gates_public_20261103.py')}
def raw_sources():return c.sources()|{n:p.read_bytes() for n,p in EXTERNAL.items()}
raw=raw_sources();before=c.hashes(raw);c.archive(OUT/'sources.tar.gz',raw)
start=time.monotonic();records=[]
commands=[('tests',[sys.executable,'-m','pytest','-q',*[str(ROOT/n) for n in ('test_worker.py','test_collector.py','test_boundaries.py','test_server_readback.py','test_stage3_gate.py')],f'--junitxml={OUT}/tests.xml']),('public_gates',[sys.executable,str(EXTERNAL['external/public_gates.py'])])]
for name,command in commands:
 with (OUT/(name+'.log')).open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
 records.append(dict(name=name,command=command,returncode=result.returncode))
 if result.returncode:break
if len(records)==2 and records[-1]['returncode']==0:
 public_report=Path('/tmp/stage4_actual_gates_public_20261103.json').read_bytes();(OUT/'public_gates.json').write_bytes(public_report)
else:public_report=None
after=c.hashes(raw_sources());origin=json.loads((ROOT/'source_origin.json').read_text());original=origin['assembly_source_hashes']
changed={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in original if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=original[n]}
expected_overlay=origin['reviewed_runtime_overlays'];archive_ok=c.check_archive(OUT/'sources.tar.gz',before)
report=dict(passed=len(records)==2 and all(r['returncode']==0 for r in records) and before==after and archive_ok and changed==expected_overlay,
 scope='Mocked stage4 transactions/terminal/public metadata contracts and authentication of completed public stage1/2/3 proofs. No ciphertext/key/model payloads, context construction or server execution.',
 source_count=len(before),candidate_source_count=len(c.sources()),source_hashes_before=before,source_hashes_after=after,source_files_unchanged=before==after,
 source_archive_sha256=hashlib.sha256((OUT/'sources.tar.gz').read_bytes()).hexdigest(),source_archive_verified=archive_ok,commands=records,
 original_198_changed_only_adopted_overlays=changed==expected_overlay,adopted_runtime_overlays=changed,
 public_gate_report_sha256=hashlib.sha256(public_report).hexdigest() if public_report else None,
 cpu_affinity=sorted(os.sched_getaffinity(0)),elapsed_s=time.monotonic()-start,actual_server_execution=False)
(OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in report.items() if not isinstance(v,(dict,list))},indent=2));raise SystemExit(0 if report['passed'] else 1)
