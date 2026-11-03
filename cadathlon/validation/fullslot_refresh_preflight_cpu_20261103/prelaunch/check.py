from pathlib import Path
import hashlib,io,json,os,shutil,subprocess,sys,tarfile,time
HERE=Path(__file__).resolve().parent;LAB=HERE.parent;sys.path.insert(0,str(LAB/'prototype'))
import collect
snapshot=collect.source_bytes();snapshot[str(Path(__file__))]=Path(__file__).read_bytes();before=collect.hashes(snapshot)
with tarfile.open(HERE/'sources.tar.gz','w:gz') as archive:
 for name,data in snapshot.items():
  member=tarfile.TarInfo(str(Path(name).relative_to(LAB)));member.size=len(data);archive.addfile(member,io.BytesIO(data))
count=collect.verify_archive(HERE/'sources.tar.gz',snapshot)
report={'passed':False,'scope':'Public secure-profile, source/archive, synthetic metadata/lifecycle/codec guards only; no keygen, encryption, refresh or TPU.','source_hashes_start':before,'verified_archive_members':count,'cpu_affinity':sorted(os.sched_getaffinity(0)),'software':collect.process.software_versions(),'environment':{k:os.environ.get(k) for k in ('JAX_PLATFORMS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}}
(HERE/'report.json').write_text(json.dumps(report,indent=2)+'\n');start=time.perf_counter()
with (HERE/'pytest.log').open('w') as stream:
 child=subprocess.run([sys.executable,'-m','pytest','-q',str(LAB/'prototype/test_preflight.py'),'--junitxml='+str(HERE/'pytest.xml')],stdout=stream,stderr=subprocess.STDOUT,check=False)
after_snapshot=collect.source_bytes();after_snapshot[str(Path(__file__))]=Path(__file__).read_bytes();after=collect.hashes(after_snapshot)
report.update(returncode=child.returncode,elapsed_s=time.perf_counter()-start,source_hashes_end=after,source_files_unchanged=before==after,passed=child.returncode==0 and before==after,source_archive_sha256=collect.process.sha256(HERE/'sources.tar.gz'))
(HERE/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('passed','returncode','elapsed_s','source_files_unchanged','verified_archive_members','source_archive_sha256')}));raise SystemExit(0 if report['passed'] else 1)
