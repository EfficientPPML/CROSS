from pathlib import Path
import difflib,hashlib,json
HERE=Path(__file__).resolve().parent;ROOT=HERE/'checkout';LIVE=Path('/home/jianming_gatech/CROSS')
base=json.loads((HERE/'base_source_sha256.json').read_text())
names=['jaxite_word/key_gen.py','jaxite_word/he_params.py','jaxite_word/he_ops.py','jaxite_word/herot.py',
 'cadathlon/common/encrypted_reference.py','cadathlon/common/encrypted_ntt.py','cadathlon/common/encrypted_packed_transformer.py','cadathlon/common/package.py',
 'cadathlon/tests/test_package.py','cadathlon/tests/test_scoped_ntt.py','cadathlon/tests/test_linear_queue.py']
parts=[];manifest={}
for name in names:
 old=LIVE/name;new=ROOT/name
 if old.exists():
  before=old.read_bytes();assert hashlib.sha256(before).hexdigest()==base[name]
 else:before=b''
 after=new.read_bytes();assert before!=after
 parts.extend(difflib.unified_diff(before.decode().splitlines(keepends=True),after.decode().splitlines(keepends=True),
  fromfile='a/'+name if before else '/dev/null',tofile='b/'+name))
 manifest[name]={'base_sha256':hashlib.sha256(before).hexdigest() if before else None,'candidate_sha256':hashlib.sha256(after).hexdigest()}
(HERE/'combined_runtime.patch').write_text(''.join(parts))
(HERE/'patch_sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('combined patch files',len(names),'bytes',(HERE/'combined_runtime.patch').stat().st_size)
