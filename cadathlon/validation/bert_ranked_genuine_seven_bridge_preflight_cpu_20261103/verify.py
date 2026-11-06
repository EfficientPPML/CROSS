from pathlib import Path
import argparse,hashlib,json,tarfile,xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent

def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(p):return json.loads(p.read_bytes())
def verify(current=None):
 artifact=read(HERE/'artifact_sha256.json')
 assert {p.relative_to(HERE).as_posix() for p in HERE.rglob('*') if p.is_file() and p.name!='artifact_sha256.json'}==set(artifact)
 assert {n:sha((HERE/n).read_bytes()) for n in artifact}==artifact
 result=read(HERE/'measured/report.json');expected=result['source_hashes_start']
 assert len(expected)==309 and result['source_hashes_end']==expected
 assert all(result.get(n)is True for n in ('passed','source_files_unchanged','source_archive_verified','all_processes_reaped','cleanup_completed','terminal_scope_completed','termination_scope_completed'))
 assert result['tests']==dict(tests=478,failures=0,errors=0,skipped=0)
 assert result['actual_ranked_launch']is False and result['real_key_generation']is False and result['real_model_forward']is False and result['real_ciphertext_work']is False
 assert sha((HERE/'measured/sources.tar.gz').read_bytes())==result['source_archive_sha256']
 with tarfile.open(HERE/'measured/sources.tar.gz') as t:
  members=t.getmembers();assert len(members)==len(expected)==len({m.name for m in members}) and all(m.isfile() for m in members)
  assert {m.name:sha(t.extractfile(m).read()) for m in members}==expected
  parent=json.loads(t.extractfile('bridge_parent_manifest.json').read());assert len(parent)==272
  assert {n:sha(t.extractfile(n).read()) for n in parent}==parent
  harness=json.loads(t.extractfile('upstream/manifest.json').read());assert len(harness)==11
  assert {n:sha(t.extractfile('upstream/'+n).read()) for n in harness}==harness
  assert all(Path(n).suffix in ('.py','.json','.md','.gz') and not Path(n).is_absolute() and '..' not in Path(n).parts for n in expected)
 suites=list(ET.parse(HERE/'measured/tests.xml').getroot().iter('testsuite'))
 assert {n:sum(int(s.get(n,0)) for s in suites) for n in ('tests','failures','errors','skipped')}==result['tests']
 process=result['test_process'];assert process['returncode']==0 and all(process[n] for n in ('process_reaped','public_key_store_removed','owned_directory_handle_closed'))
 terminal=read(HERE/'measured/session_terminal.json');assert terminal['session_id']==59964 and terminal['observed_exit_code']==0 and terminal['reaped']
 if current is not None:assert {n:sha((Path(current)/n).read_bytes()) for n in expected}==expected
 failed=read(HERE/'history/pytest-discovery-failure-v1/report.json');assert failed['passed']is False and failed['source_files_unchanged']is True and failed['test_process']['returncode']==1
 return dict(passed=True,sources=len(expected),inherited_parent_sources=len(parent),original_harness_sources=len(harness),tests=result['tests'],current_checked=current is not None,actual_ranked_launch=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--current',type=Path);args=p.parse_args();print(json.dumps(verify(args.current),sort_keys=True))
