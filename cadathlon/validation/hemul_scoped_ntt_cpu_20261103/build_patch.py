"""Build incremental HEMul opt-in patch against the reviewed combined candidate."""
from pathlib import Path
import argparse,difflib,hashlib,json
parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True);parser.add_argument('--candidate',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
files=['jaxite_word/hemul.py','jaxite_word/he_ops.py','jaxite_word/he_params.py','cadathlon/common/encrypted_ntt.py','cadathlon/tests/test_scoped_mul_ntt.py']
patch=[];manifest={}
for name in files:
 old_path=args.base/name;new_path=args.candidate/name
 old=old_path.read_bytes() if old_path.exists() else b'';new=new_path.read_bytes()
 assert old!=new,name
 patch.extend(difflib.unified_diff(old.decode().splitlines(keepends=True),new.decode().splitlines(keepends=True),fromfile='a/'+name if old_path.exists() else '/dev/null',tofile='b/'+name))
 manifest[name]={'combined_base_sha256':hashlib.sha256(old).hexdigest() if old_path.exists() else None,'candidate_sha256':hashlib.sha256(new).hexdigest()}
(args.output/'hemul_scoped_ntt_followup.patch').write_text(''.join(patch))
(args.output/'patch_sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
