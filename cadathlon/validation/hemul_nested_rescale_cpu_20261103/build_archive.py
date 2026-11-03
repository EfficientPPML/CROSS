"""Archive only the isolated nested-rescale follow-up and its exact evidence."""
from pathlib import Path
import ast,difflib,hashlib,io,json,shutil,subprocess,tarfile,xml.etree.ElementTree as ET
LAB=Path(__file__).resolve().parent;before=LAB/'before';after=LAB/'checkout';OUT=LAB/'evidence';OUT.mkdir(exist_ok=False)
changes={};patch=[]
for rel in sorted({p.relative_to(root) for root in (before,after) for p in root.rglob('*.py') if '__pycache__' not in p.parts}):
 a,b=before/rel,after/rel;old=a.read_bytes() if a.exists() else b'';new=b.read_bytes() if b.exists() else b''
 if old==new:continue
 changes[str(rel)]={'before_sha256':hashlib.sha256(old).hexdigest() if a.exists() else None,'after_sha256':hashlib.sha256(new).hexdigest()}
 patch.extend(difflib.unified_diff(old.decode().splitlines(True),new.decode().splitlines(True),fromfile='a/'+str(rel) if a.exists() else '/dev/null',tofile='b/'+str(rel)))
assert set(changes)=={'jaxite_word/hemul.py','jaxite_word/he_ops.py','cadathlon/common/encrypted_ntt.py','cadathlon/tests/test_scoped_nested_rescale.py','cadathlon/tests/test_scoped_mul_ntt.py'}
path=OUT/'hemul_nested_rescale_followup.patch';path.write_text(''.join(patch));subprocess.run(['git','apply','--check',str(path)],cwd=before,check=True)
# Only the three private and one public rescale-constructor keyword arguments
# may differ in core ASTs. No arithmetic, path enabling or existing code edits.
for rel in ('jaxite_word/hemul.py','jaxite_word/he_ops.py'):
 oldtree,newtree=(ast.parse((root/rel).read_text()) for root in (before,after))
 oldcalls=[n for n in ast.walk(oldtree) if isinstance(n,ast.Call)]
 newcalls=[n for n in ast.walk(newtree) if isinstance(n,ast.Call)]
 assert len(oldcalls)==len(newcalls)
 removed=0
 for oldcall,newcall in zip(oldcalls,newcalls):
  oldkeys={k.arg for k in oldcall.keywords}
  additions=[k for k in newcall.keywords if k.arg not in oldkeys]
  for kw in additions:
   assert kw.arg=='polynomial_factory'
   name=newcall.func.id if isinstance(newcall.func,ast.Name) else getattr(newcall.func,'attr',None)
   assert name=='_HERescaleKernel';newcall.keywords.remove(kw);removed+=1
 assert removed==(3 if rel.endswith('hemul.py') else 1)
 assert ast.dump(oldtree)==ast.dump(newtree)
assert (before/'jaxite_word/bootstrapping.py').read_bytes()==(after/'jaxite_word/bootstrapping.py').read_bytes()
assert (before/'jaxite_word/he_params.py').read_bytes()==(after/'jaxite_word/he_params.py').read_bytes()
tests=json.loads((LAB/'tests.json').read_text());diagnostic=json.loads((LAB/'cd1_precision.json').read_text())
assert tests['passed'] and diagnostic['diagnostic_completed'];sources={}
for report in (tests,diagnostic):
 assert report['source_hashes_start']==report['source_hashes_end']
 for name,expected in report['source_hashes_start'].items():
  data=Path(name).read_bytes();assert hashlib.sha256(data).hexdigest()==expected
  if name in sources:assert sources[name]==data
  sources[name]=data
with tarfile.open(OUT/'sources.tar.gz','w:gz') as stream:
 def add(name,data):
  info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;stream.addfile(info,io.BytesIO(data))
 for name,data in sorted(sources.items()):
  source=Path(name);member='repository/'+str(source.relative_to(after)) if source.is_relative_to(after) else 'validation/'+source.name
  add(member,data)
 for rel in changes:
  if (before/rel).exists():add('base_changed_files/'+rel,(before/rel).read_bytes())
for name in ('tests.json','tests.log','tests.xml','run_tests.py','cd1_precision.json','cd1_precision.log','check_cd1_precision.py','base_source_hashes.json'):
 shutil.copyfile(LAB/name,OUT/name)
interim=OUT/'interim';interim.mkdir()
for name in ('initial_tests.log','tests_v2.log','tests_v3.log','tests_v4.log','test_scoped_nested_rescale_v3.py'):
 shutil.copyfile(LAB/name,interim/name)
if (LAB/'interim_formal_failed').is_dir():shutil.copytree(LAB/'interim_formal_failed',interim/'formal_failed')
prereq=OUT/'prerequisites';prereq.mkdir();repo=Path('/home/jianming_gatech/CROSS/cadathlon/validation')
for source in (repo/'all_candidates_cpu_20261103/combined.patch',repo/'rescale_scoped_ntt_cpu_20261103/rescale_scoped_ntt_followup.patch'):
 shutil.copyfile(source,prereq/source.name)
xml=ET.parse(LAB/'tests.xml').getroot();suites=list(xml.iter('testsuite'));assert len(suites)==1;suite=suites[0]
skips=[{'name':c.attrib['name'],'reason':c.find('skipped').attrib.get('message')} for c in xml.findall('.//testcase') if c.find('skipped') is not None]
summary={'scope':'Optional nested HEMul rescale control constructors only; no full refresh/model/TPU or calibration claim.','passed':True,'junit_suite':suite.attrib,'skips':skips,'source_count_union':len(sources),'source_files_unchanged':True,'source_archive_sha256':hashlib.sha256((OUT/'sources.tar.gz').read_bytes()).hexdigest(),'patch_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'patch_after':'Facade rescale patch290c6330af2af9b27ae138fa48a19df0b5136f046d3686d2160f01ff2b8e22fe after combined91ccdbb67c457fd4f3a2636e92316d2385a88e3036f5e5963fafa5b465ab1c61','changed_files':changes,'core_arithmetic_ast_unchanged_after_removing_four_constructor_keywords':True,'bootstrap_and_he_params_unchanged':True,'split_cd2_path_executed':False,'split_cd2_literal_disabled_unchanged':True,'precision_diagnostic':diagnostic['trials'],'production_scoped_adapter_cd2_only':True,'review':'Security agent independent read-only review found no blocker; root authorized the isolated four-constructor scope.'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='changed_files'},indent=2))
