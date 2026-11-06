"""Source-bound complete layer0 ideal graph checks; no encryption or refresh proof."""
from pathlib import Path
import argparse,hashlib,importlib.metadata,io,json,os,shutil,signal,subprocess,sys,tarfile,time

HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def inventory():
    paths=list((HERE/'checkout').rglob('*.py'))+list(HERE.glob('*.py'))
    paths += [HERE/name for name in ('candidate_manifest.json','actual_profile_audit.json',
        'p60_profile_and_refresh.json','source_origin.json','fixture_preparation.json',
        'verified_model_manifest.json','matrix_inventory.json','README.md','JOINED_README.md',
        'joined_source_origin.json','native_block_preparation.json')]
    return {str(path.relative_to(HERE)):path for path in sorted(paths)}
def read_archive(path):
    with tarfile.open(path) as archive:
        members=archive.getmembers()
        if len(members)!=len({m.name for m in members}) or not all(m.isfile() for m in members):
            raise ValueError('invalid source archive inventory')
        return {m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in members}
def stop(process):
    if process is None:return
    if process.poll() is None:
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:process.wait(timeout=30)
        except subprocess.TimeoutExpired:pass
    # A reaped leader can still have children. The owned new process group is
    # cleaned after its graceful interval as well as after normal completion.
    try:os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    process.wait()

def main(output):
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();process=None;primary=None;paths={};before={};handlers={}
    report=dict(passed=False,scope=__doc__,encrypted_execution=False,actual_refresh=False,
        model_arrays_archived=False,cpu_affinity=sorted(os.sched_getaffinity(0)))
    def interrupted(signum,frame):raise KeyboardInterrupt('received signal '+str(signum))
    try:
        save(output/'report.json',report)
        for signum in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT):
            handlers[signum]=signal.signal(signum,interrupted)
        paths=inventory();before={name:sha(path) for name,path in paths.items()}
        fixture=Path('/tmp/bert-first-attention-owner-20261103/client_fixtures.npz');fixture_hash=sha(fixture)
        native_fixture=HERE/'native_block_fixtures.npz';native_hash=sha(native_fixture)
        if native_hash!=json.loads((HERE/'native_block_preparation.json').read_text())['output_fixture_sha256']:
            raise ValueError('native full-block fixture identity changed')
        if fixture_hash!=json.loads((HERE/'fixture_preparation.json').read_text())['fixture_npz_sha256']:
            raise ValueError('client-only fixture identity changed')
        snapshot=output/'snapshot';snapshot.mkdir()
        for name,path in paths.items():
            target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        if {name:sha(snapshot/name) for name in paths}!=before:raise ValueError('snapshot changed while copying')
        with tarfile.open(output/'sources.tar.gz','w:gz') as archive:
            for name in paths:
                data=(snapshot/name).read_bytes();member=tarfile.TarInfo(name);member.size=len(data)
                archive.addfile(member,io.BytesIO(data))
        if read_archive(output/'sources.tar.gz')!=before:raise ValueError('source archive bytes differ')
        tests=['test_first_attention_contract.py','test_first_attention_guards.py','test_joined_layer0.py']
        command=[sys.executable,'-m','pytest','-q','--tb=short',*tests,'--junitxml='+str(output/'tests.xml')]
        env=os.environ.copy();env['BERT_FIRST_ATTENTION_FIXTURE']=str(fixture);env['BERT_NATIVE_BLOCK_FIXTURE']=str(native_fixture)
        report.update(source_hashes_before=before,source_archive_sha256=sha(output/'sources.tar.gz'),
            source_archive_verified=True,fixture_npz_sha256=fixture_hash,native_block_fixture_sha256=native_hash,tests=tests,command=command,
            versions={name:importlib.metadata.version(name) for name in ('numpy','pytest','scipy')},python=sys.version)
        save(output/'report.json',report)
        with (output/'tests.log').open('w') as log:
            process=subprocess.Popen(command,cwd=snapshot,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            report['child_pid']=process.pid;save(output/'report.json',report)
            report['returncode']=process.wait()
        cases={}
        for K in (15,128):
            name=f'layer0-K{K}.json'
            if (snapshot/name).exists():
                shutil.copy2(snapshot/name,output/name);cases[str(K)]=json.loads((output/name).read_text())
        report['cases']=cases
        report['passed']=report['returncode']==0 and set(cases)=={'15','128'} and all(c['passed'] for c in cases.values())
        if not report['passed']:raise RuntimeError('complete layer0 ideal checks failed')
    except BaseException as error:
        primary=error;report['passed']=False;report['error']=dict(type=type(error).__name__,message=str(error))
    finally:
        for signum in handlers:signal.signal(signum,signal.SIG_IGN)
        try:
            stop(process);report['child_reaped']=process is None or process.poll() is not None
        except BaseException as error:
            report['passed']=False;report['child_reaped']=False
            if primary is None:primary=error
            else:primary.add_note('Child cleanup: '+repr(error))
        try:
            after={name:sha(path) for name,path in paths.items()}
            report.update(source_hashes_after=after,source_files_unchanged=bool(before) and before==after)
            if before:
                report['executed_snapshot_unchanged']={name:sha(output/'snapshot'/name) for name in paths}==before
                report['source_archive_verified']=read_archive(output/'sources.tar.gz')==before
                report['fixture_unchanged']=sha(Path('/tmp/bert-first-attention-owner-20261103/client_fixtures.npz'))==report.get('fixture_npz_sha256') and sha(HERE/'native_block_fixtures.npz')==report.get('native_block_fixture_sha256')
            else:report['executed_snapshot_unchanged']=False
            report['passed']=report['passed'] and report['source_files_unchanged'] and report['executed_snapshot_unchanged'] and report.get('source_archive_verified',False) and report.get('fixture_unchanged',False)
        except BaseException as error:
            report['passed']=False
            if primary is None:primary=error
            else:primary.add_note('Source readback: '+repr(error))
        report['elapsed_s']=time.monotonic()-started
        try:save(output/'report.json',report)
        except BaseException as error:
            if primary is None:raise
            primary.add_note('Final publication: '+repr(error))
        finally:
            for signum,handler in handlers.items():signal.signal(signum,handler)
    if primary is not None:raise primary
    return 0 if report['passed'] else 1

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();raise SystemExit(main(args.output.resolve()))
