"""Preserve this one completed proof, including a failure, without re-snapshotting code."""
from pathlib import Path
import hashlib, json, os, shutil, tarfile
from datetime import datetime, timezone
ROOT=Path('/home/jianming_gatech/CROSS')
SOURCE=Path('/tmp/cross-bounded-qwen2-tpu-20261103')
DESTINATION=ROOT/'cadathlon/validation'
PREFIX='qwen_bounded_test2_tpu'
report=json.loads((SOURCE/'report.json').read_text())
if not isinstance(report.get('elapsed_s'),(int,float)) or 'source_hashes_end' not in report:
    raise RuntimeError('proof report is not terminal')
for pid in (report.get('wrapper_pid'), report.get('case',{}).get('pid')):
    if type(pid) is not int or pid<=1:
        raise RuntimeError('missing owned process identity')
    try: os.kill(pid,0)
    except ProcessLookupError: pass
    else: raise RuntimeError(f'owned process {pid} is still present')
archive=SOURCE/'sources.tar.gz'
if hashlib.sha256(archive.read_bytes()).hexdigest()!=report['source_archive_sha256']:
    raise RuntimeError('source archive changed')
with tarfile.open(archive,'r:gz') as stream:
    names={item.name for item in stream.getmembers()}
    if names!=set(report['archive_members'].values()):
        raise RuntimeError('source archive member set differs from evidence')
    for source,name in report['archive_members'].items():
        payload=stream.extractfile(name).read()
        if hashlib.sha256(payload).hexdigest()!=report['source_hashes_start'][source]:
            raise RuntimeError(f'archived source mismatch: {source}')
files={
    'evidence.json':SOURCE/'report.json',
    'raw.json':SOURCE/'test2/stdout.json',
    'log':SOURCE/'test2/stderr.log',
    'process.json':SOURCE/'test2/process.json',
    'wrapper.log':Path(str(SOURCE)+'.log'),
    'sources.tar.gz':archive,
}
for suffix in files:
    if (DESTINATION/f'{PREFIX}_{suffix}').exists():
        raise RuntimeError('preserved proof destination already exists')
for suffix, source in files.items():
    if not source.is_file(): raise RuntimeError(f'missing terminal artifact: {source}')
for suffix,source in files.items():
    shutil.copyfile(source,DESTINATION/f'{PREFIX}_{suffix}')
# These nonblocking stack-only observations contain no frame locals. Their
# sampling overhead was not calibrated; they are phase diagnostics only.
monitoring=[]
for source in sorted(Path('/tmp').glob('qwen2-stack-20261103-*.txt')):
    target=DESTINATION/(PREFIX+'_monitor_'+source.name)
    if target.exists(): raise RuntimeError('monitor artifact destination already exists')
    shutil.copyfile(source,target)
    monitoring.append({'artifact':target.name,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'observed_file_mtime_utc':datetime.fromtimestamp(source.stat().st_mtime,timezone.utc).isoformat()})
summary={
    'scope':report['scope'], 'passed':report['passed'],
    'stack_observations':monitoring,
    'stack_scope':'Nonblocking stack-only diagnostics; no frame locals or calibrated overhead/speed claim.',
    'all_six_passed':False, 'full_model_evaluated':False, 'scoring_calibrated':False,
    'elapsed_s':report['elapsed_s'], 'warmups':0, 'repeats':1,
    'source_count':len(report['source_hashes_start']),
    'source_files_unchanged':report['source_files_unchanged'],
    'source_archive_sha256':report['source_archive_sha256'],
    'process_reaped':report['process_reaped'],
    'public_key_store_removed':report['public_key_store_removed'],
    'refresh_progress':report.get('refresh_progress'),
    'observed_refresh_progress':{'started':files['log'].read_text().count('bounded homomorphic refresh starting'),'completed':files['log'].read_text().count('bounded homomorphic refresh complete')},
    'error':report.get('error'),
    'runner_result':report.get('case',{}).get('result'),
    'artifacts':{suffix:f'{PREFIX}_{suffix}' for suffix in files},
}
(DESTINATION/f'{PREFIX}.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps({key:summary[key] for key in ('passed','source_count','source_files_unchanged','process_reaped','public_key_store_removed')},indent=2))
