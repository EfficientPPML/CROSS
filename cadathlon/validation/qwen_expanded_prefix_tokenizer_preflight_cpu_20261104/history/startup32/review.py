from pathlib import Path
import hashlib,io,json,os,shutil,subprocess,tarfile,time
LAB=Path('/tmp/qwen-expanded-prefix-token-freeze-20261104')
MAP=Path('/tmp/qwen-expanded-prefix-token-freeze-sources-v1-20261104.json')
OUT=Path('/tmp/qwen-token-entry-review-20261104');OUT.mkdir(exist_ok=False)
h=lambda b:hashlib.sha256(b).hexdigest()
raw=MAP.read_bytes();assert h(raw)=='91650139ae02a6efd4b549bf9f73c99a310113545f367f262d794cc423abad72';expected=json.loads(raw)
blobs={n:(LAB/n).read_bytes()for n in expected};assert {n:h(b)for n,b in blobs.items()}==expected
(OUT/'source_manifest.json').write_bytes(raw)
with tarfile.open(OUT/'sources.tar.gz','w:gz')as t:
 for n,b in sorted(blobs.items()):
  info=tarfile.TarInfo(n);info.size=len(b);t.addfile(info,io.BytesIO(b))
rows=[]
for kind in ('missing_grant','changed_helper','extra_shadow','bad_manifest_pin'):
 root=OUT/kind;root.mkdir();src=root/'source';src.mkdir()
 for n,b in blobs.items():
  dest=src/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
 marker=root/'marker'
 if kind=='changed_helper':
  p=src/'freeze_process.py';p.write_bytes(('from pathlib import Path\nPath('+repr(str(marker))+').touch()\n').encode()+p.read_bytes())
 elif kind=='extra_shadow':(src/'platform.py').write_text('from pathlib import Path\nPath('+repr(str(marker))+').touch()\n')
 pin='0'*64 if kind=='bad_manifest_pin'else h(raw)
 cmd=['/tmp/cross-pretrained-validation-env/bin/python','-I','-B',str(src/'entry.py'),'--manifest',str(OUT/'source_manifest.json'),'--manifest-sha256',pin,'--mode','freeze-worker','--output',str(root/'public'),'--private-output',str(root/'private')]
 start=time.monotonic();child=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env={'PATH':'/usr/bin:/bin','HOME':'/home/jianming_gatech','PYTHONDONTWRITEBYTECODE':'1'},start_new_session=True)
 stdout,stderr=child.communicate(timeout=60);elapsed=time.monotonic()-start
 (root/'stdout.log').write_bytes(stdout);(root/'stderr.log').write_bytes(stderr)
 try:os.killpg(child.pid,0)
 except ProcessLookupError:gone=True
 else:gone=False
 reason={'missing_grant':'separate reviewed tokenizer-only launch grant required','changed_helper':'source closure SHA256 differs','extra_shadow':'unexpected or missing source closure member','bad_manifest_pin':'external source manifest SHA256 differs'}[kind]
 passed=child.returncode==1 and reason in stderr.decode()and not marker.exists()and not (root/'public').exists()and not (root/'private').exists()and gone and not Path('/proc/'+str(child.pid)).exists()
 rows.append(dict(case=kind,passed=passed,pid=child.pid,returncode=child.returncode,direct_child_waited=True,pid_absent=not Path('/proc/'+str(child.pid)).exists(),process_group_absent=gone,marker_absent=not marker.exists(),outputs_absent=not (root/'public').exists()and not (root/'private').exists(),expected_reason=reason,elapsed_s=elapsed))
after={n:h((LAB/n).read_bytes())for n in expected};assert after==expected
report=dict(passed=all(r['passed']for r in rows),scope='Independent source/public-metadata startup review only; no original token/corpus/tokenizer/model bodies opened, no model/HE/key work, no cleanup review duplication',source_manifest_sha256=h(raw),source_hashes_before=expected,source_hashes_after=after,source_count=len(expected),source_archive_sha256=h((OUT/'sources.tar.gz').read_bytes()),cases=rows,review_findings=[],source_review=dict(exact_manifest_child_handoff=True,isolated_no_bytecode_worker=True,verified_bytes_local_imports=True,child_exit_report_source_cleanup_signal_and_group_gates=True,canonical_disjoint_outputs_before_creation=True))
(OUT/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n');shutil.copyfile(__file__,OUT/'review.py');print(json.dumps({k:report[k]for k in('passed','source_count','source_archive_sha256')}));assert report['passed']
