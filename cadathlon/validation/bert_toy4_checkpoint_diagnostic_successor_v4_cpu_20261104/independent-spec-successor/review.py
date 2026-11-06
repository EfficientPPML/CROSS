from pathlib import Path
import hashlib,json,io,os,shutil,subprocess,tarfile,xml.etree.ElementTree as ET
LAB=Path('/tmp/bert-toy4-checkpoint-component-v3-20261104');OUT=Path('/tmp/bert-toy4-spec-alias-review-20261104/successor-v3');OUT.mkdir(parents=True,exist_ok=False);SNAP=OUT/'snapshot';SNAP.mkdir()
h=lambda b:hashlib.sha256(b).hexdigest();names=set(json.loads((LAB/'source_inventory.json').read_bytes()))|{p.name for p in LAB.glob('*.py')};blobs={n:(LAB/n).read_bytes()for n in sorted(names)};before={n:h(b)for n,b in blobs.items()}
for n,b in blobs.items():p=SNAP/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
with tarfile.open(OUT/'sources.tar.gz','w:gz')as t:
 for n,b in blobs.items():m=tarfile.TarInfo(n);m.size=len(b);t.addfile(m,io.BytesIO(b))
cmd=['/home/jianming_gatech/miniconda3/envs/jaxite/bin/python','-I','-B','-m','pytest','--rootdir='+str(SNAP),'--confcutdir='+str(SNAP),'-q','--tb=short','-p','no:cacheprovider',str(SNAP/'test_spec_alias.py'),'--junitxml='+str(OUT/'tests.xml')]
with (OUT/'tests.log').open('wb')as log:
 child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={'PATH':'/usr/bin:/bin','HOME':'/home/jianming_gatech','PYTHONDONTWRITEBYTECODE':'1','PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'});code=child.wait()
try:os.killpg(child.pid,0)
except ProcessLookupError:gone=True
else:gone=False
after={n:h((SNAP/n).read_bytes())for n in names};assert after==before
cases=list(ET.parse(OUT/'tests.xml').iter('testcase'));passed=code==0 and len(cases)==28 and not any(c.findall(n)for c in cases for n in('error','failure','skipped'))and gone
r=dict(passed=passed,tests=len(cases),direct_returncode=code,child_pid=child.pid,direct_wait_completed=True,pid_and_group_absent=gone and not Path('/proc/'+str(child.pid)).exists(),source_count=len(names),source_hashes_before=before,source_hashes_after=after,source_archive_sha256=h((OUT/'sources.tar.gz').read_bytes()),tested_guard_sha256=before['source_guard.py'],test_sha256=before['test_spec_alias.py'],scope='Independent copied-source actual collector/worker admission matrix,13 rejection variants each plus2 clean boundary controls; source inventory rewritten only in synthetic nested copies. No helper/backend/data/HE execution. Other v3 owners may continue edits outside snapshot; no final whole-candidate freeze claimed.')
(OUT/'report.json').write_text(json.dumps(r,sort_keys=True,indent=2)+'\n');shutil.copyfile(__file__,OUT/'review.py');print(json.dumps({k:r[k]for k in('passed','tests','direct_returncode','source_count','tested_guard_sha256','test_sha256','source_archive_sha256')}));raise SystemExit(0 if passed else 1)
