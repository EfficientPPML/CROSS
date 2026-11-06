from pathlib import Path
import argparse,hashlib,json,shutil,tarfile,xml.etree.ElementTree as ET
LAB=Path('/tmp/bert-ranked-genuine-seven-bridge-20261103')
DEST=Path('/home/jianming_gatech/CROSS/cadathlon/validation/bert_ranked_genuine_seven_bridge_preflight_cpu_20261103')
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_bytes())
def write(p,v):p.write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')
def proofcheck(base,*,passed,current=False):
 r=read(base/'report.json');assert r['passed']is passed
 x=r['source_hashes_start'];assert len(x)==309 and r['source_hashes_end']==x and r['source_files_unchanged']is True and r['source_archive_verified']is True
 assert sha((base/'sources.tar.gz').read_bytes())==r['source_archive_sha256']
 with tarfile.open(base/'sources.tar.gz') as t:
  rows=t.getmembers();assert len(rows)==len(x) and len({z.name for z in rows})==len(x) and all(z.isfile() for z in rows)
  assert {z.name:sha(t.extractfile(z).read()) for z in rows}==x
 if current:assert {n:sha((LAB/n).read_bytes()) for n in x}==x
 proc=r['test_process'];assert proc['process_reaped'] and proc['public_key_store_removed'] and proc['owned_directory_handle_closed']
 if passed:
  assert r['tests']==dict(tests=478,failures=0,errors=0,skipped=0)
  assert proc['returncode']==0 and r['terminal_scope_completed'] and r['termination_scope_completed'] and r['cleanup_completed']
 return r

def copy_public(src,dst):
 dst.mkdir(parents=True,exist_ok=True)
 for p in src.rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.log','.xml','.gz','.md'):
   out=dst/p.relative_to(src);out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(p.read_bytes())

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--observed-exit-code',type=int,required=True);parser.add_argument('--session-id',type=int,required=True);args=parser.parse_args();assert args.observed_exit_code==0 and args.session_id==59964
 final=proofcheck(LAB/'source-bound-v2',passed=True,current=True);proofcheck(LAB/'source-bound-v1',passed=False)
 for pid in (final['pid'],final['test_process']['pid']):assert not Path(f'/proc/{pid}').exists()
 DEST.mkdir(exist_ok=False)
 for version in ('v1','v2'):
  source=LAB/f'source-bound-{version}';target=DEST/('measured' if version=='v2' else 'history/pytest-discovery-failure-v1');target.mkdir(parents=True)
  for name in ('report.json','source_manifest.json','sources.tar.gz','tests.log','tests.xml','session_terminal.json'):
   if (source/name).exists():(target/name).write_bytes((source/name).read_bytes())
  for n in ('process.json','stdout.log','stderr.log'):
   p=source/'test-process'/n
   if p.exists():(target/f'test-process-{n}').write_bytes(p.read_bytes())
  (target/'collector.log').write_bytes((LAB/f'source-bound-{version}.log').read_bytes())
 histories={
  'development-contract-fixture':LAB/'development-negative-v1',
  'development-private-umask-fixture':LAB/'development-negative-phases-v1',
  'original-synthetic-exit-negative':Path('/tmp/bert-seven-preflight-exit-review-20261103/negative-v1'),
  'worker-envelope-negative':Path('/tmp/bert-genuine-bridge-review-20261103/negative-worker-envelope-v1'),
  'cleanup-negative':Path('/tmp/bert-genuine-bridge-review-20261103/negative-lifecycle-v1'),
  'cleanup-independent-successor':Path('/tmp/bert-genuine-bridge-review-20261103/successor-lifecycle-v1'),
  'launch-gate-development-v1':Path('/tmp/bert-ranked-launch-authorization-review-20261103/development-v1'),
  'launch-gate-development-v2':Path('/tmp/bert-ranked-launch-authorization-review-20261103/development-v2'),
 }
 for name,source in histories.items():
  if source.exists():copy_public(source,DEST/'history'/name)
 (DEST/'BRIDGE_DESIGN.md').write_bytes((LAB/'BRIDGE_DESIGN.md').read_bytes())
 (DEST/'archive.py').write_bytes(Path(__file__).read_bytes())
 write(DEST/'measured/session_terminal.json',dict(session_id=args.session_id,observed_exit_code=0,reaped=True,parent_pid=final['pid'],child_pid=final['test_process']['pid'],parent_child_absent=True))
 write(DEST/'readback.json',dict(passed=True,current_before_after_tar_sources=309,parent_sources_unchanged=272,original_harness_files_unchanged=11,tests=final['tests'],report_sha256=sha((LAB/'source-bound-v2/report.json').read_bytes()),source_archive_sha256=final['source_archive_sha256'],actual_ranked_launch=False,actual_key_generation=False,actual_model_or_he=False))
 print(json.dumps(dict(archive=str(DEST),report_sha256=sha((LAB/'source-bound-v2/report.json').read_bytes()),source_archive_sha256=final['source_archive_sha256'])))
if __name__=='__main__':main()
