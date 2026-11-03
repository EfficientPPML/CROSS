import hashlib,importlib.metadata,json,os,resource,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('/home/jianming_gatech/CROSS');OUT=Path('/tmp/cross-cpu-checkpoint-20261102')
OUT.mkdir(exist_ok=False)
scopes=['cadathlon/tests','jaxite_word/ckks_ctx_test.py','jaxite_word/codec_preparation_test.py',
        'jaxite_word/mapping_compile_test.py','jaxite_word/bsgs_test.py','jaxite_word/packing_test.py',
        'jaxite_word/composite_prime_gen_test.py','jaxite_word/bootstrapping_test.py']
command=['taskset','-c','148-151',sys.executable,'-m','pytest',*scopes,'-q','--junitxml='+str(OUT/'junit.xml')]
environment={'JAX_PLATFORMS':'cpu','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
paths=set()
for folder in ('jaxite_word','demos','cadathlon/common','cadathlon/tests'):
 paths.update((ROOT/folder).rglob('*.py'))
for folder in (ROOT/'cadathlon/non_agent_pset').glob('*/src'):
 paths.update(folder.rglob('*.py'))
for folder in (ROOT/'cadathlon/agent_pset').glob('*/src'):
 paths.update(folder.rglob('*.py'))
paths={p for p in paths if '__pycache__' not in p.parts and '/suite/run/' not in str(p) and not p.is_symlink()}
def snapshot():return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
before=snapshot();(OUT/'source-before.json').write_text(json.dumps(before,indent=2)+'\n')
report={'scope':'CPU regression checkpoint; opt-in encrypted integration tests remain disabled and separately reported.',
        'command':command,'environment':environment,'source_scope':['all Python files under jaxite_word, demos, common, tests and problem src directories; exclude caches/run/symlinks'],
        'source_file_count':len(before),'review_base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'versions':{name:importlib.metadata.version(name) for name in ('jax','jaxlib','numpy','torch','libtpu','pytest')},
        'python':sys.version,'started_utc':datetime.now(timezone.utc).isoformat(),
        'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'opt_in_test_environment':{k:v for k,v in os.environ.items() if k.startswith('CROSS_') and ('TEST' in k or 'REFRESH' in k or 'SECURE' in k)}}
start=time.monotonic()
with (OUT/'pytest.log').open('wb') as stream:
 process=subprocess.run(['/usr/bin/time','-v','-o',str(OUT/'time.txt'),*command],cwd=ROOT,env=dict(os.environ,**environment),stdout=stream,stderr=subprocess.STDOUT)
report['returncode']=process.returncode;report['parent_elapsed_s']=time.monotonic()-start
report['completed_utc']=datetime.now(timezone.utc).isoformat()
after=snapshot();(OUT/'source-after.json').write_text(json.dumps(after,indent=2)+'\n')
report['source_unchanged']=before==after;report['changed_sources']=[p for p in before if before[p]!=after.get(p)]
report['passed']=process.returncode==0
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
raise SystemExit(process.returncode)
