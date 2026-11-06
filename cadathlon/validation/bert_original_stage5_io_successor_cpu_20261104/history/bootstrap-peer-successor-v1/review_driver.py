from pathlib import Path
import ast,json,hashlib,subprocess,os,tarfile,io
source=Path('/tmp/bert-original-stage5-io-successor-20261104');out=Path('/tmp/bert-original-client-bootstrap-extra-review-20261104/successor-v1');out.mkdir(parents=True,exist_ok=False)
tree=ast.parse((source/'client_bootstrap.py').read_bytes());names=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='EXPECTED_NAMES'for t in n.targets)))
raw={n:(source/n).read_bytes()for n in names};sha=lambda b:hashlib.sha256(b).hexdigest();expected={n:sha(b)for n,b in raw.items()};rows=[]
for index,(entry,kind)in enumerate((e,k)for e in ['original_client_worker.py','original_client_collect.py']for k in ['clean','extra-platform','alias']):
 case=out/f'case{index}';case.mkdir();root=case/'execution';root.mkdir();marker=case/'new-canary'
 for n,b in raw.items():p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
 manifest=case/'expected.json';manifest.write_text(json.dumps(expected,sort_keys=True)+'\n')
 if kind=='extra-platform':(root/'platform.py').write_text('from pathlib import Path\nPath('+repr(str(marker))+').write_text("unexpected")\nraise RuntimeError("extra shadow ran")\n')
 args=[str(root/entry),'--source-manifest',str(manifest),'--source-manifest-sha256',sha(manifest.read_bytes()),'--help']
 command=['/tmp/cross-pretrained-validation-env/bin/python','-I','-B',*args]
 if kind=='alias':
  code="import sys,types,runpy; m=types.ModuleType('innocent_alias'); m.__file__=%r; sys.modules['innocent_alias']=m; sys.argv=%r; runpy.run_path(sys.argv[0],run_name='__main__')"%(str(root/'key_gate.py'),args)
  command=['/tmp/cross-pretrained-validation-env/bin/python','-I','-B','-c',code]
 child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
 stdout,stderr=child.communicate();(case/'stdout.log').write_text(stdout);(case/'stderr.log').write_text(stderr)
 assert child.returncode==(0 if kind=='clean'else 1),(entry,kind,child.returncode,stderr)
 assert not marker.exists()
 if kind=='extra-platform':assert 'undeclared'in stderr or 'uncaptured'in stderr,stderr
 if kind=='alias':assert 'alias'in stderr or 'already loaded'in stderr,stderr
 for fn in [os.kill,os.killpg]:
  try:fn(child.pid,0)
  except ProcessLookupError:pass
  else:raise AssertionError('child/group remains')
 assert all((root/n).read_bytes()==b for n,b in raw.items())
 rows.append(dict(entry=entry,case=kind,pid=child.pid,exit_code=child.returncode,direct_child_reaped=True,group_absent=True,canary_created=False,expected_admission_outcome=True))
assert all((source/n).read_bytes()==b for n,b in raw.items())
files={**raw,'external_expected.json':(json.dumps(expected,sort_keys=True)+'\n').encode(),'review_driver.py':Path(__file__).read_bytes()}
with tarfile.open(out/'sources.tar.gz','w:gz')as z:
 for n,b in files.items():m=tarfile.TarInfo(n);m.size=len(b);z.addfile(m,io.BytesIO(b))
report=dict(passed=True,source_count=len(raw),source_hashes_before=expected,source_hashes_after=expected,cases=rows,actual_source_unchanged=True,no_private_payload_or_services=True,source_archive_sha256=sha((out/'sources.tar.gz').read_bytes()),scope='Both actual isolated CLI entries with copied public sources, help-only and independent extra-platform/alias negatives; no actual phase work')
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed=True,sources=len(raw),cases=len(rows),archive=report['source_archive_sha256'],pids=[r['pid']for r in rows])))
