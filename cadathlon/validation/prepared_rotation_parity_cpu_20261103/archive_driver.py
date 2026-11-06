"""Terminal-only public evidence archiver; never reads retained payload workspaces."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,math,re,shutil,stat,tarfile
LAB=Path('/tmp/bert-prepared-rotation-parity-20261103');RUN=LAB/'actual-v1'
def require(value,message):
    if not value:raise ValueError(message)
def read(path,limit=32<<20):
    info=path.lstat();require(stat.S_ISREG(info.st_mode) and info.st_size<=limit,'bounded regular public artifact required')
    raw=path.read_bytes();require(len(raw)==info.st_size,'public artifact changed during archive read');return raw
def sha(path):return hashlib.sha256(read(path)).hexdigest()
def aggregate(value):
    forbidden={'input_ids','token_ids','token_type_ids','labels','logits','embeddings','secret_key','private_key','sk','decrypted'}
    if isinstance(value,dict):
        require(not set(value)&forbidden,'private-valued report field forbidden')
        for v in value.values():aggregate(v)
    elif isinstance(value,list):
        for v in value:aggregate(v)
    elif isinstance(value,float):require(math.isfinite(value),'nonfinite public report')
def main(destination,observed_exit_code):
    report=json.loads(read(RUN/'report.json'));worker=json.loads(read(RUN/'worker/worker_report.json'));process=json.loads(read(RUN/'worker/process.json'))
    for value in (report,worker,process):aggregate(value)
    pids=[1836931,1836933,1837017]
    require(all(not Path('/proc',str(p)).exists() for p in pids),'owned process still exists; not terminal')
    require(process.get('process_reaped')is True,'worker has not been reaped')
    require(report.get('all_processes_reaped')is True,'collector reap evidence incomplete')
    expected=json.loads(read(RUN/'source_manifest.json'));require(len(expected)==231,'fixed source closure differs')
    with tarfile.open(RUN/'sources.tar.gz') as archive:
        members=archive.getmembers()
        require(len(members)==len(expected)==len({m.name for m in members}),'source archive inventory differs')
        require(all(m.isfile() and not PurePosixPath(m.name).is_absolute() and '..' not in PurePosixPath(m.name).parts for m in members),'unsafe source member')
        require({m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in members}==expected,'exact source archive bytes differ')
    current={n:sha(LAB/n) for n in expected};copied={n:sha(RUN/'execution_source'/n) for n in expected}
    passed=report.get('passed')is True
    require((observed_exit_code==0)==passed,'numeric exit and final report differ')
    if passed:
        require(all(report.get(n)is True for n in ('execute','actual_rotation_parity','actual_execution_attempted','operation_complete','cleanup_completed','terminal_scope_completed','io_workspace_retained','external_workspaces_never_removed','source_files_unchanged','source_archive_verified')),'collector positive proof incomplete')
        require(expected==current==copied==report['source_hashes_end']==report['execution_source_hashes_end']==worker['source_hashes_start']==worker['source_hashes_end'],'frozen current/execution/worker source mismatch')
        require(all(worker.get(n)is True for n in ('passed','all_rotations_completed','compiled_owner_cleanup_completed','loader_cleanup_completed','upload_payloads_unchanged','keygen_codecs_native_private_model_forbidden','terminal_scope_completed')),'worker positive proof incomplete')
        require(not worker['forbidden_attempts'] and not report.get('cleanup_errors'),'forbidden operation or cleanup failure')
        require(process['returncode']==0 and process['public_key_store_removed'] and process['owned_directory_handle_closed'],'process terminal cleanup differs')
        require(process['result']['worker_report_sha256']==sha(RUN/'worker/worker_report.json'),'worker digest differs')
        arithmetic=worker['arithmetic'];expected_rows=[(L,k,i) for L in (49,25) for i,k in enumerate((1,1,64,64,1,64))]
        rows=arithmetic['comparisons'];require([(r['level'],r['index'],r['sequence']) for r in rows]==expected_rows,'fixed comparison schedule differs')
        require(arithmetic['passed'] and arithmetic['actual_rotation_calls']==24 and arithmetic['input_unchanged'] and arithmetic['level_reduction']['passed'],'arithmetic component proof incomplete')
        for row in rows:
            require(row['passed'] and row['metadata_equal'] and row['mismatched_words']==0 and row['compared_words']==2*131072*((row['level']+1)*2) and row['left']==row['right'],'all-word metadata parity failure')
        for item,bytes_ in zip(arithmetic['levels'],(1913651200,554696704)):
            require(item['retained_entries']==2 and item['retained_bytes']==bytes_ and item['stats_delta']['hits']==4 and item['stats_delta']['published']==2,'prepared cache count/bytes differ')
    require(report['cpu_affinity']==[140,141,142,143],'fixed actual affinity differs')
    timing=read(LAB/'actual-v1.time').decode();match=re.search(r'Exit status: (\d+)\s*$',timing)
    require(match is not None and int(match[1])==observed_exit_code,'external exit differs')
    files=['report.json','sources.tar.gz','source_manifest.json','worker/process.json','worker/stdout.json','worker/stderr.log','worker/worker_report.json']
    destination.mkdir(exist_ok=False)
    for name in files:
        source=RUN/name;target=destination/'measured'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(read(source))
    for name in ('actual-v1.log','actual-v1.time'): (destination/name).write_bytes(read(LAB/name))
    observation=dict(passed=passed,observed_exit_code=observed_exit_code,owned_pids_absent=pids,source_count=231,
        current_sources_equal=current==expected,execution_sources_equal=copied==expected,source_archive_sha256=sha(RUN/'sources.tar.gz'),
        report_sha256=sha(RUN/'report.json'),worker_report_sha256=sha(RUN/'worker/worker_report.json'),
        scope='Public archive only; no retained key/model/upload/download/private payload files were read or copied by this archiver.')
    (destination/'terminal_readback.json').write_text(json.dumps(observation,indent=2,allow_nan=False)+'\n')
    (destination/'archive_driver.py').write_bytes(Path(__file__).read_bytes())
    (destination/'artifact_hashes.json').write_text(json.dumps({p.relative_to(destination).as_posix():sha(p) for p in sorted(destination.rglob('*')) if p.is_file()},indent=2)+'\n')
    print(json.dumps(observation))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('destination',type=Path);parser.add_argument('--observed-exit-code',type=int,required=True);main(**vars(parser.parse_args()))
