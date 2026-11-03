"""Validation-only, source-sealed public transformer sweep; never calibration."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import sys
import tarfile
import threading
import time

MODELS = {'bert': 'p2_bert', 'qwen': 'p4_qwen'}
INTERLEAVED = [('qwen',1),('bert',1),('bert',2),('qwen',2),('bert',3),
               ('qwen',3),('qwen',4),('bert',4),('bert',5),('qwen',5),('qwen',6),('bert',6)]
SOURCE_DIRS = ('jaxite_word','demos','cadathlon/common',
               'cadathlon/agent_pset/p2_bert','cadathlon/agent_pset/p4_qwen')
COLLECTOR = 'cadathlon/validation/check_transformer_suite.py'
PINNED_WRAPPER = 'validation_sweep.py'
MANIFEST = 'validation_snapshot.json'
KEY_STORE_PREFLIGHT_BYTES = 117610135552  # Exact twelve-case public plan, largest BERT5 store + reserve.

def sha(data):
    return hashlib.sha256(data).hexdigest()

def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)

def load_suite(root):
    spec=importlib.util.spec_from_file_location('snapshot_transformer_suite', root/COLLECTOR)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.ROOT == root
    return module

def source_paths(root):
    paths=[]
    for folder in SOURCE_DIRS:
        for path in (root/folder).rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            # Runtime Python, public data/config/oracles, and text contracts.
            # No keys, model weights, compiled outputs, caches or private data.
            if path.suffix in {'.py','.json','.in','.out','.md','.golden','.sh','.toml','.txt'} or path.name=='Makefile':
                paths.append(path)
    paths.extend([root/COLLECTOR,root/PINNED_WRAPPER])
    return sorted(set(paths))

def read_sources(root):
    result={}
    for path in source_paths(root):
        if path.is_symlink():
            raise ValueError(f'source symlinks are not accepted: {path}')
        result[str(path.relative_to(root))]=path.read_bytes()
    return result

def prepare(args):
    source=args.source_root.resolve(strict=True);snapshot=args.snapshot.resolve()
    if snapshot==source or source in snapshot.parents:
        raise ValueError('snapshot must be outside the live checkout')
    snapshot.mkdir(parents=True,exist_ok=False)
    paths=[]
    for folder in SOURCE_DIRS:
        for path in (source/folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and (
                path.suffix in {'.py','.json','.in','.out','.md','.golden','.sh','.toml','.txt'} or path.name=='Makefile'):
                paths.append(path)
    paths.append(source/COLLECTOR)
    before={str(path.relative_to(source)):sha(path.read_bytes()) for path in paths}
    for path in paths:
        if path.is_symlink():raise ValueError(f'source symlink is not accepted: {path}')
        target=snapshot/path.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
    shutil.copy2(Path(__file__),snapshot/PINNED_WRAPPER)
    after={str(path.relative_to(source)):sha(path.read_bytes()) for path in paths}
    if before!=after:raise RuntimeError('source changed while snapshot was copied')
    raw=read_sources(snapshot);hashes={name:sha(data) for name,data in raw.items()}
    if any(hashes[name]!=value for name,value in before.items()):raise RuntimeError('snapshot copy mismatch')
    manifest={'scope':__doc__,'source_root':str(source),'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'source_hashes':hashes,'instrumentation_sha256':hashes[PINNED_WRAPPER],
        'copy_verified':True,'runtime_modified':False,
        'closure':'core/demos/common/P2/P4 Python and public data/text plus collector and pinned wrapper; installed packages recorded at run time'}
    write_json(snapshot/MANIFEST,manifest)
    # This protects against accidental writes, not a malicious same-user process.
    for path in snapshot.rglob('*'):
        if path.is_file():path.chmod(0o444)
    for path in sorted((p for p in snapshot.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):path.chmod(0o555)
    snapshot.chmod(0o555)
    print(json.dumps({'prepared':True,'snapshot':str(snapshot),'files':len(hashes),
        'manifest_sha256':sha((snapshot/MANIFEST).read_bytes()),'TPU_launched':False},indent=2))
    return 0

def verified_sources(root):
    manifest=json.loads((root/MANIFEST).read_text());raw=read_sources(root)
    hashes={name:sha(data) for name,data in raw.items()}
    if hashes!=manifest['source_hashes']:raise ValueError('sealed snapshot source inventory/hash mismatch')
    return manifest,raw

def archive_sources(root,output,manifest,raw):
    with tarfile.open(output/'sources.tar.gz','w:gz') as archive:
        for name,data in sorted(raw.items()):
            info=tarfile.TarInfo(name);info.size=len(data);archive.addfile(info,io.BytesIO(data))
    write_json(output/'snapshot_manifest.json',manifest)
    return sha((output/'sources.tar.gz').read_bytes())

def machine_record(suite,key_parent):
    cpu_models=sorted({line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                       if line.startswith('model name')})
    return {'uname':platform.uname()._asdict(),'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'cpu_models':cpu_models,'meminfo':Path('/proc/meminfo').read_text(),
        'software':suite.software_versions(),'key_filesystem':dict(zip(('total','used','free'),shutil.disk_usage(key_parent))),
        'environment':{key:os.environ.get(key) for key in (*suite.ENVIRONMENT,'MKL_NUM_THREADS','PYTHONDONTWRITEBYTECODE')}}

@contextmanager
def execution_environment(root,key_parent):
    settings={'CROSS_ROOT':str(root),'PYTHONPATH':str(root)+':'+str(root/'demos'),
        'CROSS_PUBLIC_KEY_DIRECTORY':str(key_parent),'JAX_PLATFORMS':'tpu','PYTHONDONTWRITEBYTECODE':'1',
        'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
        'CROSS_LEVEL_PARAMS_MAX':'1','CROSS_BARRETT_NTT_CONTEXT_CACHE_CAPACITY':'2',
        'XLA_FLAGS':None,'CROSS_SKIP_TOPLEVEL_ROTKEYS':None}
    previous={key:os.environ.get(key) for key in settings}
    try:
        for key,value in settings.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
        yield
    finally:
        for key,value in previous.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value

def process_table():
    result={};page_size=os.sysconf('SC_PAGE_SIZE')
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            fields=(entry/'stat').read_text().rsplit(')',1)[1].split()
            result[int(entry.name)]={'ppid':int(fields[1]),'pgid':int(fields[2]),
                'start_ticks':int(fields[19]),'rss_bytes':max(0,int(fields[21]))*page_size,
                'cpu_ticks':int(fields[11])+int(fields[12])}
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    return result

def owned_processes(table,leader):
    selected={pid for pid,row in table.items() if pid==leader or row['pgid']==leader}
    while True:
        children={pid for pid,row in table.items() if row['ppid'] in selected}
        expanded=selected|children
        if expanded==selected:return {pid:table[pid] for pid in sorted(selected)}
        selected=expanded

class ResourceSampler:
    def __init__(self,directory,interval):
        self.directory=directory;self.interval=interval;self.stop=threading.Event()
        self.error=None;self.rows=0;self.peak=0;self.thread=threading.Thread(target=self.sample,daemon=True)
    def sample(self):
        try:
            with (self.directory/'resource_samples.jsonl').open('w') as output:
                while not self.stop.is_set():
                    try:record=json.loads((self.directory/'process.json').read_text())
                    except FileNotFoundError:record={}
                    leader=record.get('pid')
                    if leader is not None:
                        members=owned_processes(process_table(),leader)
                        rss=sum(row['rss_bytes'] for row in members.values());self.peak=max(self.peak,rss)
                        row={'monotonic_s':time.monotonic(),'leader':leader,'processes':members,'aggregate_rss_bytes':rss}
                        output.write(json.dumps(row)+'\n');output.flush();self.rows+=1
                    self.stop.wait(self.interval)
        except BaseException as error:self.error=error
    def finish(self):
        self.stop.set();self.thread.join(timeout=5)
        if self.thread.is_alive():raise RuntimeError('resource sampler did not stop')
        if self.error is not None:raise self.error
        report={'samples':self.rows,'sample_interval_s':self.interval,'sampled_peak_process_tree_rss_bytes':self.peak,
            'scope':'Sampled sum of RSS for observed owned process-group/descendant tree; shared pages can be counted twice, transient peaks/escaped descendants can be missed. Not an OS sandbox or a continuous peak guarantee.'}
        write_json(self.directory/'resources.json',report)
        return report

def json_value(value):
    if value is None or isinstance(value,(str,bool,int,float)):return value
    if isinstance(value,(list,tuple)):return [json_value(v) for v in value]
    if isinstance(value,dict):return {str(k):json_value(v) for k,v in value.items()}
    if hasattr(value,'item'):return json_value(value.item())
    return str(value)

def ciphertext_records(bundle):
    from jaxite_word import Polynomial
    records=[]
    def visit(value):
        if isinstance(value,Polynomial):
            devices=[]
            for device in sorted(value.polynomial.devices(),key=lambda d:(str(d.platform),int(d.id))):
                entry={name:json_value(getattr(device,name,None)) for name in
                       ('platform','id','device_kind','process_index','local_hardware_id','coords','slice_index')}
                entry['platform_version']=str(getattr(device.client,'platform_version','unknown'))
                entry['memory_stats']=json_value(device.memory_stats())
                devices.append(entry)
            if not devices or any(d['platform']!='tpu' for d in devices):
                raise ValueError('actual ciphertext payload must reside on TPU')
            records.append({'shape':list(value.shape),'degree':value.degree,'moduli_count':len(value.moduli),
                'payload_bytes':value.polynomial.nbytes,'scale':getattr(value,'_ckks_scale',None),
                'nsd':getattr(value,'_ckks_nsd',None),'devices':devices})
        elif isinstance(value,(tuple,list)):
            for item in value:visit(item)
        else:raise TypeError('unexpected ciphertext container in validation probe')
    visit(bundle)
    return records

def instrumented_child(args):
    root=args.root.resolve(strict=True);verified_sources(root)
    if Path(__file__).resolve()!=root/PINNED_WRAPPER:raise ValueError('child must use pinned snapshot instrumentation')
    suite=load_suite(root)
    sys.path[:0]=[str(root),str(root/'cadathlon/common')]
    import transformer_runner
    original=transformer_runner.require_secure_execution
    report={'passed':False,'events':[],'scope':'Original runner security gate return and ciphertext metadata; no payload/plaintext/key serialization.',
        'timing_scope':'Probe overhead is included in request end-to-end timing. No time is subtracted; ciphertext computation is unchanged.',
        'device_scope':'Payload placement and memory_stats at gate checkpoints; not proof of every internal operation or a continuous device-memory peak.'}
    caught=None
    def gate(owner,bundle):
        manifests=original(owner,bundle)  # The existing security gate always runs first.
        started=time.perf_counter()
        event={'index':len(report['events']),'boundary':'input' if len(report['events'])%2==0 else 'output',
               'actual_rings':manifests,'ciphertexts':ciphertext_records(bundle)}
        report['events'].append(event)
        event['metadata_capture_s']=time.perf_counter()-started
        write_json(args.security_output,report)
        event['capture_and_publication_s']=time.perf_counter()-started
        return manifests
    transformer_runner.require_secure_execution=gate
    try:
        try:suite.child_runner(args.runner,args.runner_args)
        except SystemExit as error:
            if error.code not in (None,0):raise
        if len(report['events'])!=2*args.requests:raise ValueError('missing validated request boundaries')
        report['passed']=True
    except BaseException as error:
        caught=error;report['error']=f'{type(error).__name__}: {error}';raise
    finally:
        transformer_runner.require_secure_execution=original
        try:write_json(args.security_output,report)
        except BaseException as error:
            if caught is None:raise
            caught.add_note(f'Final live-ring/device evidence write failed: {type(error).__name__}: {error}')
    return 0

def request_schedule(order):
    if order=='interleaved':return INTERLEAVED
    return [(model,number) for model in ('qwen','bert') for number in range(1,7)]

def collect_case(suite,command,directory,*,timeout,grace,interval,state):
    sampler=ResourceSampler(directory,interval);sampler.thread.start();caught=None
    try:
        state['collector_entered']=True
        record=suite.run_case(command,directory,timeout=timeout,cleanup_grace_s=grace)
    except BaseException as error:
        caught=error;raise
    finally:
        try:resources=sampler.finish()
        except BaseException as error:
            if caught is None:raise
            caught.add_note(f'Resource sampler cleanup/evidence failed: {type(error).__name__}: {error}')
    record['resources']=resources
    return record

def run(args):
    root=args.root.resolve(strict=True)
    if Path(__file__).resolve()!=root/PINNED_WRAPPER:raise ValueError('run must use pinned snapshot instrumentation')
    suite=load_suite(root);output=args.output.resolve();key_parent=args.key_parent.resolve()
    output.mkdir(parents=True,exist_ok=False)
    report={'passed':False,'scope':'All twelve public toy encrypted correctness cases only; no full pretrained model or calibration.',
        'models':{model:{'cases':{}} for model in MODELS},'schedule':request_schedule(args.order),
        'warmups':args.warmups,'repeats':args.repeats,'scoring_calibrated':False,'full_model_evaluated':False,
        'source_files_unchanged':False,'phase':'preparing','errors':[]}
    started=time.monotonic();caught=None;owned_fd=identity=None;created_keys=False;all_cleanup=True
    manifest=raw=None;attempts={}
    try:
        write_json(output/'report.json',report)
        manifest,raw=verified_sources(root)
        report['source_hashes_start']=manifest['source_hashes']
        report['source_archive_sha256']=archive_sources(root,output,manifest,raw)
        key_parent.mkdir(parents=True,exist_ok=False);created_keys=True
        owned_fd,identity=suite.open_owned_directory(key_parent)
        report['owned_public_key_parent']=str(key_parent);report['key_parent_identity']=identity
        with execution_environment(root,key_parent):
            report['machine']=machine_record(suite,key_parent)
            # Planning bound for this unchanged profile set; per-session store
            # additionally checks its exact requested allocation before keys.
            required=KEY_STORE_PREFLIGHT_BYTES
            report['key_space_preflight_bytes']=required
            if shutil.disk_usage(key_parent).free<required:raise ValueError('insufficient space for largest public key store')
            for model,number in report['schedule']:
                directory=output/model/f'test{number}';directory.mkdir(parents=True)
                problem=root/'cadathlon/agent_pset'/MODELS[model]
                implementation=problem/f'src/{model}_optimized_reference.py'
                command=[sys.executable,str(root/PINNED_WRAPPER),'child','--root',str(root),
                    '--security-output',str(directory/'security_device.json'),'--requests',str(args.warmups+args.repeats),
                    '--runner',str(problem/'src/runner.py'),'--','--case',str(problem/f'test/test{number}.in'),
                    '--implementation',str(implementation),'--require-tpu','--warmups',str(args.warmups),'--repeats',str(args.repeats)]
                report['phase']=f'{model}/test{number}';write_json(output/'report.json',report)
                print(f'Starting {report["phase"]}; evidence {directory}',flush=True)
                state=attempts[str(directory)]={'collector_entered':False}
                record=collect_case(suite,command,directory,timeout=args.case_timeout_s,
                                    grace=args.cleanup_grace_s,interval=args.sample_interval_s,state=state)
                clean=all(record.get(name) is True for name in
                          ('process_reaped','public_key_store_removed','owned_directory_handle_closed'))
                all_cleanup=all_cleanup and clean
                try:
                    verified_sources(root)
                    record['source_files_unchanged']=True
                    if record.get('error'):raise ValueError(record['error'])
                    if not clean:raise ValueError('incomplete owned process/key/handle cleanup')
                    if record['resources']['samples']<1 or record['resources']['sampled_peak_process_tree_rss_bytes']<=0:
                        raise ValueError('missing observed process resource samples')
                    suite.validate_result(record.get('result'),model=model,case=f'test{number}.in',
                        implementation=implementation,warmups=args.warmups,repeats=args.repeats)
                    security=json.loads((directory/'security_device.json').read_text())
                    if security.get('passed') is not True or len(security.get('events',[]))!=2*(args.warmups+args.repeats):
                        raise ValueError('missing actual-ring/device request evidence')
                    record['security_device_sha256']=sha((directory/'security_device.json').read_bytes())
                    record['passed']=True
                except Exception as error:
                    record['passed']=False;record['validation_error']=f'{type(error).__name__}: {error}'
                report['models'][model]['cases'][f'test{number}.in']=record
                write_json(directory/'process.json',record);write_json(output/'report.json',report)
                if not clean or record.get('source_files_unchanged') is not True:break
    except BaseException as error:
        caught=error;report['errors'].append(f'{type(error).__name__}: {error}');raise
    finally:
        cleanup_errors=[];cleanup_cancellation=None
        # Unknown active child state means the outer parent must be retained.
        for directory,state in attempts.items():
            if not state['collector_entered']:continue
            process_file=Path(directory)/'process.json'
            try:
                state=json.loads(process_file.read_text())
                all_cleanup=all_cleanup and all(state.get(name) is True for name in
                    ('process_reaped','public_key_store_removed','owned_directory_handle_closed'))
            except BaseException as error:
                all_cleanup=False;cleanup_errors.append(f'child cleanup evidence: {type(error).__name__}: {error}')
                if not isinstance(error,Exception):cleanup_cancellation=error
        removed=not created_keys
        if created_keys:
            try:
                if not all_cleanup:raise RuntimeError('inner cleanup unknown/incomplete; key parent retained')
                removed=suite.remove_owned_directory(key_parent,owned_fd,identity)
            except BaseException as error:
                cleanup_errors.append(f'key parent: {type(error).__name__}: {error}')
                if not isinstance(error,Exception) and cleanup_cancellation is None:cleanup_cancellation=error
        handle_closed=owned_fd is None
        if owned_fd is not None:
            try:os.close(owned_fd);handle_closed=True
            except BaseException as error:
                cleanup_errors.append(f'key handle: {type(error).__name__}: {error}')
                if not isinstance(error,Exception) and cleanup_cancellation is None:cleanup_cancellation=error
        try:
            final_manifest,_=verified_sources(root)
            report['source_hashes_end']=final_manifest['source_hashes']
            report['source_files_unchanged']=manifest is not None and final_manifest==manifest
        except BaseException as error:
            report['source_files_unchanged']=False;cleanup_errors.append(f'final source inspection: {type(error).__name__}: {error}')
            if not isinstance(error,Exception) and cleanup_cancellation is None:cleanup_cancellation=error
        for model,entry in report['models'].items():
            entry.update(suite.summarize(entry['cases'],suite.CASES,warmups=args.warmups,repeats=args.repeats))
            entry['passed']=entry['all_six_passed'] and report['source_files_unchanged'] and not cleanup_errors and caught is None
        report.update(phase='finished',elapsed_s=time.monotonic()-started,cleanup_errors=cleanup_errors,
            owned_key_parent_removed=removed,owned_key_parent_handle_closed=handle_closed,inner_cleanup_complete=all_cleanup)
        report['passed']=all(entry['passed'] for entry in report['models'].values()) and removed and handle_closed and all_cleanup
        try:write_json(output/'report.json',report)
        except BaseException as error:
            primary=caught if caught is not None else cleanup_cancellation
            if primary is None:raise
            primary.add_note(f'Final sweep report publication failed: {type(error).__name__}: {error}')
        if caught is None and cleanup_cancellation is not None:raise cleanup_cancellation
    return 0 if report['passed'] else 1

def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='mode',required=True)
    prep=commands.add_parser('prepare');prep.add_argument('--source-root',type=Path,required=True);prep.add_argument('--snapshot',type=Path,required=True)
    child=commands.add_parser('child');child.add_argument('--root',type=Path,required=True);child.add_argument('--runner',type=Path,required=True)
    child.add_argument('--security-output',type=Path,required=True);child.add_argument('--requests',type=int,required=True);child.add_argument('runner_args',nargs=argparse.REMAINDER)
    execute=commands.add_parser('run');execute.add_argument('--root',type=Path,required=True);execute.add_argument('--output',type=Path,required=True)
    execute.add_argument('--key-parent',type=Path,required=True,help='new owned directory on sufficiently large filesystem')
    execute.add_argument('--order',choices=['interleaved','qwen-first'],default='interleaved')
    execute.add_argument('--warmups',type=int,default=0);execute.add_argument('--repeats',type=int,default=1)
    execute.add_argument('--case-timeout-s',type=float);execute.add_argument('--cleanup-grace-s',type=float,default=30.)
    execute.add_argument('--sample-interval-s',type=float,default=1.)
    args=parser.parse_args()
    if args.mode=='prepare':return prepare(args)
    if args.mode=='child':
        if args.runner_args[:1]==['--']:args.runner_args=args.runner_args[1:]
        return instrumented_child(args)
    if args.warmups<0 or args.repeats<1:parser.error('invalid request counts')
    for field in ('cleanup_grace_s','sample_interval_s','case_timeout_s'):
        value=getattr(args,field)
        if value is not None and (not math.isfinite(value) or value<=0):parser.error(f'invalid {field}')
    suite=load_suite(args.root.resolve(strict=True))
    with suite.termination_signals():return run(args)

if __name__=='__main__':raise SystemExit(main())
