from pathlib import Path
import hashlib,io,json,os,shutil,subprocess,tarfile
OUT=Path('/tmp/bert-client-clean-launch-source-proposals-20261104');OUT.mkdir(mode=0o700,exist_ok=False)
h=lambda b:hashlib.sha256(b).hexdigest()
profiles=[('original122','/tmp/bert-original-stage5-io-preflight-v1-20261104','/tmp/bert-original-stage5-io-successor-20261104','original_client_worker.py','original_client_collect.py',122,217),('combined125','/tmp/bert-prepared-bsgs-client-v3-preflight-v1-20261104','/tmp/bert-prepared-bsgs-client-component-v3-20261104','separate_client_worker.py','separate_client_collect.py',125,286)]
rows=[]
for name,evidence,lab,worker,collector,count,producer in profiles:
 evidence=Path(evidence);lab=Path(lab);root=OUT/name;root.mkdir(mode=0o700);sources=root/'source';sources.mkdir(mode=0o700)
 raw=(evidence/'source_manifest.json').read_bytes();expected=json.loads(raw);report=json.loads((evidence/'report.json').read_bytes());assert len(expected)==count and report['passed']and expected==report['source_hashes_start']==report['source_hashes_end']
 before={n:h((lab/n).read_bytes())for n in expected};assert before==expected
 with tarfile.open(evidence/'sources.tar.gz')as t:
  members=t.getmembers();assert len(members)==len({m.name for m in members})==count
  for m in members:
   p=Path(m.name);assert m.isfile()and not p.is_absolute()and '..'not in p.parts;data=t.extractfile(m).read();assert len(data)==m.size and h(data)==expected[m.name]
   dst=sources/m.name;dst.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
   fd=os.open(dst,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
   try:
    view=memoryview(data)
    while view:
     used=os.write(fd,view);assert used>0;view=view[used:]
   finally:os.close(fd)
 manifest=root/'source_manifest.json';manifest.write_bytes(raw)
 def tree():
  paths={p.relative_to(sources).as_posix():p for p in sources.rglob('*')if p.is_file()};assert set(paths)==set(expected);assert not any(p.is_symlink()for p in sources.rglob('*'));assert {n:h(p.read_bytes())for n,p in paths.items()}==expected;assert not any(p.suffix in('.pyc','.pyo','.so')for p in paths.values())
 tree();cases=[]
 for entry in(worker,collector):
  command=['/tmp/cross-pretrained-validation-env/bin/python','-I','-B',str(sources/entry),'--source-manifest',str(manifest),'--source-manifest-sha256',h(raw),'--help']
  child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True,env={'PATH':'/usr/bin:/bin','HOME':'/home/jianming_gatech','PYTHONDONTWRITEBYTECODE':'1','JAX_PLATFORMS':'cpu','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','NUMEXPR_NUM_THREADS':'4'})
  stdout,stderr=child.communicate(timeout=60);(root/(entry+'.stdout.log')).write_bytes(stdout);(root/(entry+'.stderr.log')).write_bytes(stderr)
  try:os.killpg(child.pid,0)
  except ProcessLookupError:gone=True
  else:gone=False
  assert child.returncode==0 and b'--source-manifest-sha256'in stdout and gone and not Path('/proc/'+str(child.pid)).exists()
  cases.append(dict(entrypoint=entry,command=command,returncode=child.returncode,pid=child.pid,direct_child_reaped=True,pid_absent=True,process_group_absent=True))
 tree();after={n:h((lab/n).read_bytes())for n in expected};assert after==before
 rows.append(dict(name=name,source_count=count,source_root=str(sources),source_manifest=str(manifest),source_manifest_sha256=h(raw),parent_report_sha256=h((evidence/'report.json').read_bytes()),parent_source_archive_sha256=h((evidence/'sources.tar.gz').read_bytes()),source_hashes_before=before,source_hashes_after=after,source_tree_exact_before_after=True,no_bytecode_or_unlisted_files=True,original_lab_caches_untouched=True,producer_source_count_unchanged=producer,actual_client=False,execution_authorized=False,help_cases=cases))
result=dict(passed=True,scope='Fresh exact-manifest source-copy and isolated CLI help preparation only; no source edits/deletions, private/payload/key/model reads, contexts, decode or actual client launch.',profiles=rows,remaining_requirements=['Corresponding217 or286 producer must complete and be independently pinned.','A distinct root launch grant and concrete complete command are required.','Rehash exact source tree and external manifest immediately before any authorized launch.','Use separate externally owned evidence/results paths as the unchanged profile requires.'])
(OUT/'report.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');shutil.copyfile(__file__,OUT/'prepare.py');print(json.dumps(dict(passed=True,profiles=[dict(name=x['name'],sources=x['source_count'],manifest_sha256=x['source_manifest_sha256'])for x in rows],help_checks=4)))
