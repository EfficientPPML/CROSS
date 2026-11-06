import gzip,hashlib,json,os,stat,tarfile
from pathlib import Path,PurePosixPath
CAP=Path('/tmp/qwen_server_device_integration_preparation_cpu_20261105')
assert not CAP.exists();CAP.mkdir()
sha=lambda raw:hashlib.sha256(raw).hexdigest()
canon=lambda x:(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
origins={}
def read(path,limit=64<<20):
 path=Path(path);s=path.lstat();assert stat.S_ISREG(s.st_mode)and 0<=s.st_size<=limit
 state=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mode,x.st_uid,x.st_mtime_ns,x.st_ctime_ns)
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC)
 try:
  assert state(os.fstat(fd))==state(s);data=[];size=0
  while size<=limit:
   b=os.read(fd,min(65536,limit+1-size))
   if not b:break
   data.append(b);size+=len(b)
  assert size==s.st_size and state(os.fstat(fd))==state(s)==state(path.lstat());return b''.join(data)
 finally:os.close(fd)
def put(name,raw):
 rel=PurePosixPath(name);assert str(rel)==name and not rel.is_absolute()and '..'not in rel.parts
 p=CAP/name;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw)
def add(source,name,pin=None,limit=64<<20):
 source=Path(source);raw=read(source,limit);h=sha(raw)
 if pin is not None:assert h==pin,(source,h,pin)
 put(name,raw)
 row=dict(source=str(source),sha256=h,bytes=len(raw))
 if name in origins:assert origins[name]==row
 else:origins[name]=row
 return raw
def copy_names(root,dest,names):
 for name in names.split():add(Path(root)/name,dest+'/'+name)
final=Path('/tmp/qwen-full28-device-focused-tests-v3-20261105')
source_stages={
 '458':dict(root='/tmp/qwen-full28-device-focused-tests-20261104',map='/tmp/qwen-full28-device-focused-test-sources-20261104.json',handoff='/tmp/qwen-full28-device-focused-test-handoff-20261104'),
 '459':dict(root='/tmp/qwen-full28-device-focused-tests-v2-20261105',map='/tmp/qwen-full28-device-focused-test-sources-v2-20261105.json',handoff='/tmp/qwen-full28-device-focused-test-handoff-v2-20261105'),
 '460':dict(root=str(final),map='/tmp/qwen-full28-device-focused-test-sources-v3-20261105.json',handoff='/tmp/qwen-full28-device-focused-test-handoff-v3-20261105')}
lineage={};maps={}
for stage,row in source_stages.items():
 raw=add(row['map'],'source/maps/'+stage+'.json');maps[stage]=json.loads(raw);assert len(maps[stage])==int(stage)
 h=json.loads(read(Path(row['handoff'])/'handoff.json'))
 row['archive']=str(Path(row['handoff'])/'sources.tar.gz');row['archive_sha256']=h['source_archive_sha256']
 row['manifest_sha256']=sha(raw)
add(source_stages['460']['archive'],'source/final460.tar.gz',source_stages['460']['archive_sha256'])
with tarfile.open(source_stages['460']['archive'])as a:
 base={m.name:a.extractfile(m).read()for m in a.getmembers()if m.isfile()}
 assert {n:sha(b)for n,b in base.items()}==maps['460']
for stage,row in source_stages.items():
 wanted=maps[stage];overrides={};remove=sorted(set(base)-set(wanted))
 for name,pin in wanted.items():
  if name not in base or sha(base[name])!=pin:
   dest='source/history_blobs/'+stage+'/'+pin+'-'+Path(name).name
   add(Path(row['root'])/name,dest,pin);overrides[name]=dest
 reconstructed={n:sha(b)for n,b in base.items()if n not in remove}
 for name,dest in overrides.items():reconstructed[name]=sha((CAP/dest).read_bytes())
 assert reconstructed==wanted
 lineage[stage]=dict(map='source/maps/'+stage+'.json',remove=remove,overrides=overrides,current_source_root=row['root'],
  historical_archive_path=row['archive'],historical_archive_sha256=row['archive_sha256'],historical_archive_bundled=(stage=='460'),manifest_sha256=row['manifest_sha256'])
for name,pin in maps['460'].items():
 if '/'not in name and Path(name).suffix in('.py','.md','.json'):add(final/name,'source/readable/'+name,pin)
# Source-stage provenance, preserving original source-only statuses.
copy_names(source_stages['458']['handoff'],'source/handoffs/458','capture_source.py handoff.json static-review.json')
copy_names(source_stages['459']['handoff'],'source/handoffs/459','before_owned_child.py.txt failed-peer-review.json handoff.json prepare_source.py source.patch')
copy_names(source_stages['460']['handoff'],'source/handoffs/460','LE_handoff.json LE_review.json SEAMS.json.patch TEST_SCOPE.md.patch cases_loader_engine.py.patch cases_process.py.patch failed_terminal_peer.json failed_terminal_root.json fixture_base.py.patch handoff.json index.json owned_child.py.patch parent_handoff.json root-source-review.json source_peer.json source_refs.json')
rootbase=Path('/tmp/qwen-full28-device-focused-proof-20261104')
peers={
 '458':Path('/tmp/qwen-full28-device-failed-security-peer-20261105'),
 '459':Path('/tmp/qwen-full28-device459-terminal-cnn-review-20261105'),
 '460':Path('/tmp/qwen-full28-device460-terminal-cnn-review-20261105')}
runs=[]
for version,stage,counts in ((2,'458',dict(tests=59,failures=59,errors=0,skipped=0)),(3,'459',dict(tests=59,failures=16,errors=0,skipped=0)),(4,'460',dict(tests=59,failures=0,errors=0,skipped=0))):
 proof=rootbase/('preparation-v'+str(version));dest='runs/'+stage
 hraw=add(proof/'handoff.json',dest+'/handoff.json');h=json.loads(hraw)
 for name,pin in h['pins'].items():
  if name=='sources.tar.gz':continue
  add(proof/name,dest+'/'+name,pin)
 peer=peers[stage]
 pubs=json.loads(read(peer/'public-pins.json'))
 for name,row in pubs.items():
  if name=='sources.tar.gz':continue
  add(row['path'],dest+'/'+name,row['sha256'],2<<20)
 recordpins=json.loads(read(peer/'owned-record-pins.json'))
 for name,row in recordpins.items():add(proof/name,dest+'/'+name,row['sha256'],2<<20)
 add(peer/'owned-record-pins.json',dest+'/owned-record-pins.json')
 add(peer/'owned-records.json',dest+'/owned-records.json')
 add(peer/'aliases.json',dest+'/aliases.json')
 if stage!='458':add(peer/'fixture-metadata.json',dest+'/fixture-metadata.json')
 else:put(dest+'/fixture-metadata.json',canon({}))
 runs.append(dict(stage=stage,proof_root=str(proof),destination=dest,counts=counts,child_records=len(recordpins),
  fixture_metadata_count=0 if stage=='458'else len(json.loads(read(peer/'fixture-metadata.json'))),
  peer_review='reviews/terminal/cnn'+stage+'/review.json',source_map=lineage[stage]['map']))
# Original v1 was unrun; its pinned source-only harness and original template delta.
proof=rootbase/'preparation-v1';h=json.loads(add(proof/'handoff.json','unrun/harness_v1/handoff.json'))
for name,pin in h['pins'].items():
 if name=='sources.tar.gz':continue
 add(proof/name,'unrun/harness_v1/'+name,pin)
# Fixed reviewer/source-author directories only; no recursive /tmp copying.
copy_names(peers['458'],'reviews/terminal/cnn458','review.json review_public.py public-pins.json owned-record-pins.json owned-records.json aliases.json process-absence.json review_initial_relative_alias.py reviewer-history.json')
copy_names(peers['459'],'reviews/terminal/cnn459','review.json review_public.py public-pins.json owned-record-pins.json owned-records.json failure-locations.json fixture-metadata.json aliases.json process-absence.json review_initial_directory_assumption.py.txt reviewer-history.json')
copy_names(peers['460'],'reviews/terminal/cnn460','review.json review_public.py public-pins.json owned-record-pins.json owned-records.json failure-locations.json fixture-metadata.json aliases.json process-absence.json')
rootreview=Path('/tmp/qwen-full28-device-focused-root-harness-review-20261105')
copy_names(rootreview,'reviews/root','v1-review.json v2-review-and-launch.json v3-review-and-launch.json v4-review-and-launch.json failed-terminal-review.json failed459-terminal-review.json success460-terminal-review.json tool-terminal459.json tool-terminal460.json live459.json live460.json live460-progress-0118.json review_failed_terminal.py review_failed459_terminal.py review_success459_terminal.py review_success460_terminal.py review_success460_terminal_v2.py success460-reader-preparation.json success460-reader-v2-preparation.json')
copy_names('/tmp/qwen-full28-device-focused-root-source-review-20261105','reviews/source/root','review.json correction459-review.json')
copy_names('/tmp/qwen-full28-device-shared-security-review-20261105','reviews/source/security','review.json review_source.py')
copy_names('/tmp/qwen-full28-device459-successor-source-cnn-review-20261105','reviews/source/cnn_correction','review.json review_source.py source-pins.json reviewer-history.json observed8539-reconstructed-exact.py.txt')
copy_names('/tmp/qwen-full28-device-focused-root-plan-review-20261104','reviews/plan/root','review.json')
copy_names('/tmp/qwen-full28-native403-security-review-20261104','reviews/runtime403/security','review.json review_source.py')
copy_names('/tmp/qwen-full28-device-packet-cases-root-handoff-20261104','authoring/PF17','document-byte-shape.patch handoff.json original-draft.py.txt')
copy_names('/tmp/qwen-full28-device-process-cases-root-handoff-20261104','authoring/R7','handoff.json initial-draft.py.txt profile-marker.patch')
copy_names('/tmp/qwen-full28-device-loader-engine-source-handoff-20261104','authoring/LE8','README.md SEAMS.json capture_source.py files.json handoff.json selected_nodes.json source_refs.json')
copy_names('/tmp/qwen-full28-device-loader-engine-correction-handoff-20261105','authoring/LE8_correction','author_source.py files.json handoff.json review.json source.patch source_refs.json sources.json')
put('source/lineage.json',canon(dict(base_archive='source/final460.tar.gz',stages=lineage,
 offline_history_scope='Exact458/459/460 source file maps reconstruct from final460 bytes plus declared sparse overrides. Historical458/459 tar-byte hashes are provenance only offline; archives not bundled.')))
put('RUNS.json',canon(runs))
put('SUMMARY.json',canon(dict(schema='full28-device-integration-preparation-capsule-v1',completed_recording_proof=True,
 final=dict(source_count=460,runtime_count=403,tests=59,passed=59,failed=0,regular_child_records=236,fixture_metadata_files=14,aliases_metadata_only=15,retired_pid_pgid_pairs=61),
 predecessors=[dict(source_count=458,passed=0,failed=59,regular_child_records=177,fixture_metadata_files=0),dict(source_count=459,passed=43,failed=16,regular_child_records=220,fixture_metadata_files=6)],
 runs_not_aggregated=True,actual_ready=False,actual_model=False,actual_HE=False,actual_capacity=False,native_backend=False,genuine_scientific_admission=False,genuine_cli_bootstrap=False,fixture_bodies_included=False,
 final_root_terminal='reviews/root/tool-terminal460.json',final_root_review='reviews/root/success460-terminal-review.json',final_peer_review='reviews/terminal/cnn460/review.json')))
put('COPY_ORIGINS.json',canon(origins))
put('BUILD_PLAN.json',canon(dict(copied_files=len(origins),copied_origin_paths=[v['source']for k,v in sorted(origins.items())],
 allowlist='Only fixed source maps/public source archives, declared harness handoff pins, explicitly named reviewer files and peer-selected bounded child records. No directory symlinks followed; fixture bodies excluded.')))
print(json.dumps(dict(capsule=str(CAP),copied=len(origins),bytes=sum(v['bytes']for v in origins.values()),runs=runs),indent=2))
