from pathlib import Path
import hashlib,io,json,os,shutil,subprocess,tarfile
LAB=Path('/tmp/qwen-expanded-prefix-token-freeze-20261104');MAP=Path('/tmp/qwen-expanded-prefix-token-freeze-sources-v1-20261104.json');OUT=Path('/tmp/qwen-token-entry-alias-review-20261104/negative-v1');OUT.mkdir(parents=True,exist_ok=False)
h=lambda b:hashlib.sha256(b).hexdigest();raw=MAP.read_bytes();assert h(raw)=='91650139ae02a6efd4b549bf9f73c99a310113545f367f262d794cc423abad72';expected=json.loads(raw);blobs={n:(LAB/n).read_bytes()for n in expected};assert {n:h(b)for n,b in blobs.items()}==expected
(OUT/'source_manifest.json').write_bytes(raw)
with tarfile.open(OUT/'sources.tar.gz','w:gz')as t:
 for n,b in sorted(blobs.items()):
  info=tarfile.TarInfo(n);info.size=len(b);t.addfile(info,io.BytesIO(b))
rows=[]
for kind in ('exact_origin_control','external_symlink_file','external_symlink_spec'):
 root=OUT/kind;root.mkdir();src=root/'source';src.mkdir()
 for n,b in blobs.items():
  p=src/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
 link=root/'external_source_alias';link.symlink_to(src,target_is_directory=True)
 origin=str((src if kind=='exact_origin_control'else link)/'input_plan.py')
 attr='m.__spec__=types.SimpleNamespace(origin='+repr(origin)+')'if kind.endswith('_spec')else'm.__file__='+repr(origin)
 script="import runpy,sys,types; m=types.ModuleType('review_unknown_alias'); "+attr+"; sys.modules['review_unknown_alias']=m; sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
 cmd=['/tmp/cross-pretrained-validation-env/bin/python','-I','-B','-c',script,str(src/'entry.py'),'--manifest',str(OUT/'source_manifest.json'),'--manifest-sha256',h(raw),'--mode','freeze-worker','--output',str(root/'public'),'--private-output',str(root/'private')]
 child=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True,env={'PATH':'/usr/bin:/bin','HOME':'/home/jianming_gatech','PYTHONDONTWRITEBYTECODE':'1'});stdout,stderr=child.communicate(timeout=60)
 (root/'stdout.log').write_bytes(stdout);(root/'stderr.log').write_bytes(stderr)
 try:os.killpg(child.pid,0)
 except ProcessLookupError:gone=True
 else:gone=False
 rejected='preloaded local helper or unknown local alias'in stderr.decode();advanced='separate reviewed tokenizer-only launch grant required'in stderr.decode()
 assert child.returncode==1 and gone and not Path('/proc/'+str(child.pid)).exists()and not (root/'public').exists()and not (root/'private').exists()
 assert {n:h((src/n).read_bytes())for n in expected}==expected
 rows.append(dict(case=kind,expected_alias_rejection=True,alias_rejected=rejected,advanced_to_grant_check=advanced,requirement_passed=rejected,pid=child.pid,exit_code=child.returncode,direct_child_waited=True,pid_absent=True,process_group_absent=gone,outputs_absent=True))
after={n:h((LAB/n).read_bytes())for n in expected};assert after==expected
report=dict(passed=all(r['requirement_passed']for r in rows),expected_failures=sum(not r['requirement_passed']for r in rows),passing_controls=sum(r['requirement_passed']for r in rows),scope='Harmless synthetic preloaded aliases through external symlink; no original token/corpus/tokenizer/model reads and no source edits',source_count=32,source_hashes_before=expected,source_hashes_after=after,source_archive_sha256=h((OUT/'sources.tar.gz').read_bytes()),cases=rows)
(OUT/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n');shutil.copyfile(__file__,OUT/'review.py');print(json.dumps({k:report[k]for k in('passed','expected_failures','passing_controls','source_archive_sha256')}));raise SystemExit(0 if report['passed']else 1)
