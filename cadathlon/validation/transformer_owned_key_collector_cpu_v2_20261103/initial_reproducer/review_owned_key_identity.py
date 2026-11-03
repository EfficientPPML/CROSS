from pathlib import Path
import hashlib,importlib.util,json,os,sys,tempfile
SOURCE=Path('/tmp/transformer-collector-ownedkeys-20261103/cadathlon/validation/check_transformer_suite.py')
spec=importlib.util.spec_from_file_location('collector_review',SOURCE);suite=importlib.util.module_from_spec(spec);spec.loader.exec_module(suite)
suite.child_environment_overrides=lambda:{'JAX_PLATFORMS':'cpu'}
with tempfile.TemporaryDirectory(prefix='collector-identity-review-') as directory:
 root=Path(directory);shared=root/'shared';shared.mkdir();foreign=shared/'unrelated-directory';foreign.mkdir();(foreign/'unrelated.txt').write_text('must survive')
 output=root/'result';output.mkdir();old=os.environ.get('CROSS_PUBLIC_KEY_DIRECTORY');os.environ['CROSS_PUBLIC_KEY_DIRECTORY']=str(shared)
 try:
  code="import os; from pathlib import Path; p=Path(os.environ['CROSS_PUBLIC_KEY_DIRECTORY']); p.rmdir(); (p.parent/'unrelated-directory').rename(p); print('{}')"
  record=suite.run_case([sys.executable,'-c',code],output,timeout=None)
 finally:
  if old is None:os.environ.pop('CROSS_PUBLIC_KEY_DIRECTORY',None)
  else:os.environ['CROSS_PUBLIC_KEY_DIRECTORY']=old
 owned=Path(record['owned_public_key_directory'])
 report={'candidate_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'unrelated_directory_survived':foreign.exists(),'replacement_directory_survived':owned.exists(),'record':record,'scope':'Real CPU subprocess replacement test; all paths are isolated temporary test artifacts.'}
Path('/tmp/review_owned_key_identity.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
