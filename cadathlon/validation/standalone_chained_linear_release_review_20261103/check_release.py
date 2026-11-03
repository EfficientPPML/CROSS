"""Read-only release/source identity checks; no encrypted evaluation or backend."""
from pathlib import Path
import hashlib,importlib.util,json,os,subprocess,sys,tarfile,tempfile
ROOT=Path(os.environ.get('CROSS_REPO','/home/jianming_gatech/CROSS')).resolve()
HERE=Path(__file__).resolve().parent

def sha(data):return hashlib.sha256(data).hexdigest()
def main():
 proof_path=ROOT/'cadathlon/validation/standalone_chained_linear_secure_cpu_20261103/report.json'
 proof=json.loads(proof_path.read_text());before=proof['source_hashes_before']
 release=ROOT/'cadathlon/contestant-release.tar.gz';release_before=sha(release.read_bytes())
 checked={};names=[]
 with tarfile.open(release) as archive:
  names=[m.name for m in archive.getmembers() if m.isfile()]
  assert len(names)==len(set(names))==238
  for filename in ('encrypted_chained_streaming_linear.py','encrypted_streaming_linear.py','encrypted_packed_transformer.py','package.py'):
   path='cadathlon/common/'+filename;member='common/'+filename;data=archive.extractfile(member).read();expected=before['checkout/'+path]
   actual=sha((ROOT/path).read_bytes());assert sha(data)==actual==expected
   checked[filename]={'sha256':actual,'proof_main_release_identical':True}
  with tempfile.TemporaryDirectory(prefix='cross-independent-chained-release-') as temporary:
   common=Path(temporary)/'common';common.mkdir()
   for member in archive.getmembers():
    if member.isfile() and member.name.startswith('common/'):
     rel=Path(member.name).relative_to('common')
     assert not rel.is_absolute() and '..' not in rel.parts
     path=common/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(archive.extractfile(member).read())
   code="""import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'common'))
import package
from encrypted_chained_streaming_linear import ChainedStreamingLinear
from encrypted_streaming_linear import StreamingLinear
assert 'encrypted_chained_streaming_linear.py' in package.COMMON_RUNTIME_FILES
for name in ('__init__','_validate_ciphertext','iter_evaluate','clear'):
 assert getattr(ChainedStreamingLinear,name) is getattr(StreamingLinear,name)
assert 'jax' not in sys.modules and 'jaxite_word' not in sys.modules
assert Path(sys.modules[ChainedStreamingLinear.__module__].__file__).is_relative_to(Path.cwd())
print('extracted standalone class/package/inheritance imported with no backend')
"""
   process=subprocess.run([sys.executable,'-c',code],cwd=temporary,
    env=dict(os.environ,PYTHONPATH='',JAX_PLATFORMS='independent_review_no_backend',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),
    text=True,capture_output=True)
   assert process.returncode==0,process.stdout+process.stderr
 manifest=Path('/tmp/cross-optimized-twelve-20261103/checkout/validation_snapshot.json')
 frozen=json.loads(manifest.read_text())['source_hashes'];assert len(frozen)==199
 assert all(sha((manifest.parent/name).read_bytes())==digest for name,digest in frozen.items())
 calibrations={}
 for label,file in [('p1','p1_student_post_promotion_20261103/source_inventory.json'),('p3','p3_compact_post_promotion_20261103/source-start.json')]:
  inventory=json.loads((ROOT/'cadathlon/validation'/file).read_text())
  for name,value in inventory.items():
   if label=='p1':
    path=(ROOT/name[6:] if name.startswith('CROSS/') else ROOT/'cadathlon'/name if name.startswith('common/') else ROOT/'cadathlon/non_agent_pset/p1_benchmark/src'/name[8:])
    digest=value['sha256']
   else:path=ROOT/name;digest=value
   assert sha(path.read_bytes())==digest,str(path)
  calibrations[label]=len(inventory)
 assert sha(release.read_bytes())==release_before
 report=dict(scope=__doc__,passed=True,release_file=str(release),release_sha256=release_before,release_files=len(names),
  proof_report_sha256=sha(proof_path.read_bytes()),checked_helpers=checked,
  extracted_backend_free_import_returncode=process.returncode,extracted_import_stdout=process.stdout,
  untouched_active_sweep_files=len(frozen),active_sweep_manifest_sha256=sha(manifest.read_bytes()),
  unchanged_calibration_files=calibrations,release_unchanged_during_review=True,
  new_cryptographic_execution=False,review_script_sha256=sha(Path(__file__).read_bytes()))
 (HERE/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
