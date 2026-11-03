"""Archive a facade-only rescale follow-up without applying it to live code."""
from pathlib import Path
import ast,difflib,hashlib,io,json,shutil,subprocess,tarfile,xml.etree.ElementTree as ET
LAB=Path(__file__).resolve().parent;before=LAB/'before';after=LAB/'checkout';OUT=LAB/'evidence';OUT.mkdir(exist_ok=False)
patch=[];changes={}
for rel in sorted({p.relative_to(root) for root in (before,after) for p in root.rglob('*.py') if '__pycache__' not in p.parts}):
 a=before/rel;b=after/rel;old=a.read_text() if a.exists() else '';new=b.read_text() if b.exists() else ''
 if old==new:continue
 changes[str(rel)]={'before_sha256':hashlib.sha256(old.encode()).hexdigest() if a.exists() else None,'after_sha256':hashlib.sha256(new.encode()).hexdigest()}
 patch.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+str(rel) if a.exists() else '/dev/null',tofile='b/'+str(rel)))
path=OUT/'rescale_scoped_ntt_followup.patch';path.write_text(''.join(patch));subprocess.run(['git','apply','--check',str(path)],cwd=before,check=True)
assert set(changes)=={'jaxite_word/rescale.py','jaxite_word/he_ops.py','jaxite_word/he_params.py','cadathlon/common/encrypted_ntt.py','cadathlon/tests/test_scoped_rescale_ntt.py'}
for name in ('_rescale_array','_rescale_array_unchecked','rescale'):
 def body(root):
  cls=next(n for n in ast.parse((root/'jaxite_word/rescale.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='_HERescaleKernel')
  return next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
 assert ast.dump(body(before))==ast.dump(body(after))
assert (before/'jaxite_word/hemul.py').read_bytes()==(after/'jaxite_word/hemul.py').read_bytes()
assert (before/'jaxite_word/bootstrapping.py').read_bytes()==(after/'jaxite_word/bootstrapping.py').read_bytes()
reports={name:json.loads((LAB/name).read_text()) for name in ('tests.json','fullring_controls.json')}
sources={}
for report in reports.values():
 assert report['passed'] and report['source_hashes_start']==report['source_hashes_end']
 for name,expected in report['source_hashes_start'].items():
  data=Path(name).read_bytes();assert hashlib.sha256(data).hexdigest()==expected
  if name in sources:assert sources[name]==data
  sources[name]=data
with tarfile.open(OUT/'sources.tar.gz','w:gz') as archive:
 for name,data in sorted(sources.items()):
  source=Path(name);member='repository/'+str(source.relative_to(after)) if source.is_relative_to(after) else 'validation/'+source.name
  info=tarfile.TarInfo(member);info.size=len(data);info.mode=0o644;archive.addfile(info,io.BytesIO(data))
for name in ('tests.json','tests.log','tests.xml','fullring_controls.json','fullring_controls.log','run_tests.py','check_fullring_controls.py','base_source_hashes.json'):
 shutil.copyfile(LAB/name,OUT/name)
shutil.copytree(LAB/'interim_failed',OUT/'interim_failed')
for name in ('initial_tests.log','tests_v2.log'):shutil.copyfile(LAB/name,OUT/'interim_failed'/name)
# Reconstruct only the two edited validation files and verify every byte
# against the saved interim hashes; never rewrite the failed report itself.
failed=json.loads((LAB/'interim_failed/tests.json').read_text());assert failed['source_files_unchanged']
with tarfile.open(OUT/'interim_failed/sources.tar.gz','w:gz') as archive:
 for name,expected in failed['source_hashes_start'].items():
  source=Path(name);data=source.read_bytes()
  if name.endswith('/test_scoped_rescale_ntt.py'):
   data=data.decode().replace('getattr(v,"_ckks_nsd",None)','v._ckks_nsd').encode()
  if name==str(LAB/'run_tests.py'):
   data=data.decode().replace("paths=[*ROOT.rglob('*.py'),ROOT/'pytest.ini',Path(__file__)]","paths=[*ROOT.rglob('*.py'),Path(__file__)]").encode()
  assert hashlib.sha256(data).hexdigest()==expected,name
  member='repository/'+str(source.relative_to(after)) if source.is_relative_to(after) else 'validation/'+source.name
  info=tarfile.TarInfo(member);info.size=len(data);info.mode=0o644;archive.addfile(info,io.BytesIO(data))
xml=ET.parse(LAB/'tests.xml').getroot();suite=xml.find('testsuite');skips=[{'name':c.attrib['name'],'reason':c.find('skipped').attrib.get('message')} for c in xml.findall('.//testcase') if c.find('skipped') is not None]
summary={'scope':'Optional rescale-facade table construction only; no full refresh, model, TPU or calibration claim.','passed':True,'tests_passed':97,'subtests_passed':3,'skips':skips,'pytest_seconds':float(suite.attrib['time']),'source_count_union':len(sources),'source_files_unchanged':True,'source_archive_sha256':hashlib.sha256((OUT/'sources.tar.gz').read_bytes()).hexdigest(),'patch_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'patch_after':'Five-candidate integration patch91ccdbb67c457fd4f3a2636e92316d2385a88e3036f5e5963fafa5b465ab1c61','changed_files':changes,'rescale_arithmetic_unchanged':True,'hemul_nested_rescales_unchanged':True,'bootstrap_unchanged':True,'failed_fixture_scope':'Initial failure used a normalization-only level-lowering helper on nsd2; corrected9-case test passed. The later source-hashed interim failure assumed fresh ciphertexts carry an explicit nsd attribute instead of the permitted implicit1. Both were validation-fixture failures, not observed payload/parity failures; final checks preserve attribute presence. Failed logs and verified interim source bytes are retained.','review':'Root and security agent independent read-only reviews found no blocker.'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='changed_files'},indent=2))
