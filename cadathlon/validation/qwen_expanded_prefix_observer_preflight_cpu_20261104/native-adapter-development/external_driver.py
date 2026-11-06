from pathlib import Path
import hashlib,io,json,os,shutil,subprocess,tarfile,xml.etree.ElementTree as ET
LAB=Path('/tmp/qwen-expanded-prefix-observer-runtime-20261104');OUT=Path('/tmp/qwen-expanded-native-observer-review-20261104/source-bound-v1');OUT.mkdir(parents=True,exist_ok=False);SNAP=OUT/'snapshot';SNAP.mkdir()
paths={n:LAB/n for n in('native_observer.py','ranges.py','test_observers.py','cleanup_origin.py','observer_provenance.json','observed_reference.py')}
paths.update({'origins/calibrate.py':Path('/tmp/qwen-native-calibration-20261103/calibrate.py'),'origins/native_capture.py':Path('/tmp/qwen-candidate3-calibration-20261103/native_capture.py'),'origins/observed_reference.py':Path('/tmp/qwen-native-calibration-20261103/observed_reference.py')})
h=lambda b:hashlib.sha256(b).hexdigest();raw={n:p.read_bytes()for n,p in paths.items()};before={n:h(b)for n,b in raw.items()}
assert raw['observed_reference.py']==raw['origins/observed_reference.py']
for n,b in raw.items():p=SNAP/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
with tarfile.open(OUT/'sources.tar.gz','w:gz')as t:
 for n,b in raw.items():m=tarfile.TarInfo(n);m.size=len(b);t.addfile(m,io.BytesIO(b))
cmd=['/tmp/cross-pretrained-validation-env/bin/python','-B','-m','pytest','--rootdir='+str(SNAP),'--confcutdir='+str(SNAP),'-q','--tb=short','-p','no:cacheprovider',str(SNAP/'test_observers.py'),'--junitxml='+str(OUT/'tests.xml')]
with (OUT/'tests.log').open('wb')as log:
 child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={'PATH':'/usr/bin:/bin','HOME':'/home/jianming_gatech','PYTHONDONTWRITEBYTECODE':'1','PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'});code=child.wait()
try:os.killpg(child.pid,0)
except ProcessLookupError:gone=True
else:gone=False
after={n:h(p.read_bytes())for n,p in paths.items()};snapafter={n:h((SNAP/n).read_bytes())for n in paths};cases=list(ET.parse(OUT/'tests.xml').iter('testcase'))
passed=code==0 and len(cases)==62 and not any(c.findall(n)for c in cases for n in('failure','error','skipped'))and before==after==snapafter and gone
r=dict(passed=passed,tests=len(cases),source_count=len(paths),source_hashes_before=before,source_hashes_after=after,execution_hashes_after=snapafter,source_archive_sha256=h((OUT/'sources.tar.gz').read_bytes()),direct_returncode=code,child_pid=child.pid,direct_wait_completed=True,pid_and_group_absent=gone and not Path('/proc/'+str(child.pid)).exists(),scope='Synthetic observer input telemetry/lifecycle only. Origins are public source captures, not executed model/calibration. No tensor/token/asset/body read, model/Torch/HE launch or whole-runtime freeze.')
(OUT/'report.json').write_text(json.dumps(r,sort_keys=True,indent=2)+'\n');shutil.copyfile(__file__,OUT/'review.py');print(json.dumps({k:r[k]for k in('passed','tests','source_count','source_archive_sha256','child_pid')}));raise SystemExit(0 if passed else 1)
