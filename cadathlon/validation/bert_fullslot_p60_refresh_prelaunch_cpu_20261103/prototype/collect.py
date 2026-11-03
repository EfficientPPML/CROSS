"""Isolated full-slot refresh collector with owned tmpfs key cleanup."""
from pathlib import Path
import argparse,hashlib,io,json,os,sys,tarfile,time
LAB=Path(__file__).resolve().parents[1];ROOT=LAB/'checkout';HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'cadathlon/validation'))
import check_transformer_suite as process


def source_bytes():
    paths=[*ROOT.rglob('*.py'),*HERE.glob('*.py'),LAB/'public_plan.json',LAB/'profile_audit.json',LAB/'base_provenance.json']
    return {str(path):path.read_bytes() for path in sorted(paths)}


def hashes(mapping):return {name:hashlib.sha256(value).hexdigest() for name,value in mapping.items()}


def verify_archive(path, expected):
    wanted={str(Path(name).relative_to(LAB)):value for name,value in expected.items()}
    with tarfile.open(path,'r:gz') as archive:
        members=archive.getmembers()
        if len(members)!=len(wanted) or {m.name for m in members}!=set(wanted) or any(not m.isfile() for m in members):
            raise ValueError('source archive membership differs')
        for member in members:
            if archive.extractfile(member).read()!=wanted[member.name]:
                raise ValueError('source archive member bytes differ')
    return len(wanted)


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    report={'passed':False,'scope':'Two CPU p60 full-independent-slot refresh and encrypted square probes; not model inference/calibration.',
        'wrapper_pid':os.getpid(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'public_key_parent':'/dev/shm',
        'full_model_evaluated':False,'scoring_calibrated':False,'launch_was_explicit':True,
        'software':None,'environment':{name:os.environ.get(name) for name in ('JAX_PLATFORMS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BOOTSTRAP_DISABLE_HOISTED_LT')}}
    caught=None;before=None
    try:
        process.write_json(output/'report.json',report)
        report['software']=process.software_versions()
        if sorted(os.sched_getaffinity(0))!=[128,129,130,131]:raise ValueError('collector requires CPU128-131')
        snapshot=source_bytes();before=hashes(snapshot);report['source_hashes_start']=before
        with tarfile.open(output/'sources.tar.gz','w:gz') as archive:
            for name,data in snapshot.items():
                path=Path(name);member=str(path.relative_to(LAB));info=tarfile.TarInfo(member);info.size=len(data);info.mode=0o644;archive.addfile(info,io.BytesIO(data))
        report['verified_source_archive_members']=verify_archive(output/'sources.tar.gz',snapshot)
        report['source_archive_sha256']=process.sha256(output/'sources.tar.gz')
        report['tmpfs']=dict(zip(('total','used','free'),__import__('shutil').disk_usage('/dev/shm')))
        process.write_json(output/'report.json',report)
        os.environ['CROSS_PUBLIC_KEY_DIRECTORY']='/dev/shm'
        process.child_environment_overrides=lambda:{'CROSS_ROOT':str(ROOT),'JAX_PLATFORMS':'cpu','PYTHONUNBUFFERED':'1',
            'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','CROSS_LEVEL_PARAMS_MAX':'1',
            'CROSS_BARRETT_NTT_CONTEXT_CACHE_CAPACITY':'2','BOOTSTRAP_DISABLE_HOISTED_LT':'0'}
        directory=output/'process';directory.mkdir()
        command=[sys.executable,str(HERE/'proof.py'),'--report',str(output/'numerical.json')]
        record=process.run_case(command,directory,timeout=None,cleanup_grace_s=60.)
        report['process']=record
        if not isinstance(record,dict):raise TypeError('missing process evidence')
        raw=json.loads((output/'numerical.json').read_text())
        report['numerical']=raw
        required=('process_reaped','public_key_store_removed','owned_directory_handle_closed')
        report['all_gates_passed']=(record.get('returncode')==0 and not record.get('error')
            and all(record.get(key) is True for key in required) and raw.get('passed') is True
            and raw.get('full_slots_checked')==65536 and raw.get('actual_refresh_count')==2
            and raw.get('all_four_outputs_completed_before_first_client_decode') is True
            and set(raw.get('probes',{}))=={'B16','B128'}
            and all(row.get('passed') is True and row.get('output_level')==25 and row.get('square_level')==24
                and row.get('full_slots_checked')==65536 for row in raw.get('probes',{}).values())
            and raw.get('server_completed_before_client_decode') is True
            and raw.get('source_hashes_start')==before and raw.get('source_hashes_end')==before
            and raw.get('input_payload_and_metadata_unchanged') is True
            and raw.get('actual_security_and_P_coverage_checked_before_keygen') is True)
    except BaseException as error:
        caught=error;report['error']=type(error).__name__+': '+str(error)
    finally:
        try:
            report['source_hashes_end']=hashes(source_bytes());report['source_files_unchanged']=before is not None and before==report['source_hashes_end']
        except BaseException as error:
            report['source_files_unchanged']=False
            if caught is None:caught=error
            else:caught.add_note('Source verification also failed: '+repr(error))
        report['elapsed_s']=time.perf_counter()-start
        report['passed']=caught is None and report.get('all_gates_passed') is True and report.get('source_files_unchanged') is True
        try:process.write_json(output/'report.json',report)
        except BaseException as error:
            if caught is None:raise
            caught.add_note('Final collector report also failed: '+repr(error))
    print(json.dumps({k:report.get(k) for k in ('passed','error','elapsed_s','source_files_unchanged')}),flush=True)
    if caught is not None and not isinstance(caught,Exception):raise caught
    return 0 if report['passed'] else 1

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    with process.termination_signals():raise SystemExit(run(args.output))
