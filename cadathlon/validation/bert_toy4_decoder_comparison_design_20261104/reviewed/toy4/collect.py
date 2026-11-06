"""One explicit diagnostic role or disabled public preflight; no healthy deadline.

Setup and server completion never authorize private validation automatically.
Validation requires a separately supplied, externally hashed root review.
"""
from pathlib import Path
import argparse,json,os,secrets,sys,tarfile,time
import xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent
# Before the first local import: authenticate and load the stdlib-only guard.
_BOOTSTRAP=None
if __name__=='__main__':
    import hashlib,importlib.util,json,stat
    parser0=argparse.ArgumentParser(add_help=False)
    parser0.add_argument('--source-manifest',type=Path,required=True)
    parser0.add_argument('--source-manifest-sha256',required=True)
    args0,_=parser0.parse_known_args()
    def _initial_read(path,limit):
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK);primary=None
        try:
            info=os.fstat(fd)
            if not stat.S_ISREG(info.st_mode)or not 0<info.st_size<=limit:raise ValueError('bounded bootstrap source required')
            data=b''
            while len(data)<=limit:
                part=os.read(fd,min(1<<20,limit+1-len(data)))
                if not part:break
                data+=part
            if len(data)!=info.st_size or len(data)>limit:raise ValueError('bootstrap source size differs')
            return data
        except BaseException as error:primary=error;raise
        finally:
            try:os.close(fd)
            except BaseException as error:
                if error is not primary and getattr(error,'_cross_toy4_body_origin',False)is not True:error._cross_toy4_cleanup_origin=True
                if primary is None:raise
                primary.add_note('bootstrap descriptor close: '+repr(error))
    manifest0=_initial_read(args0.source_manifest,1<<20)
    if hashlib.sha256(manifest0).hexdigest()!=args0.source_manifest_sha256:raise ValueError('external manifest digest differs before helper imports')
    expected0=json.loads(manifest0)
    guard_path=HERE/'source_guard.py';guard_raw=_initial_read(guard_path,8<<20)
    if type(expected0)is not dict or expected0.get('source_guard.py')!=hashlib.sha256(guard_raw).hexdigest():raise ValueError('source guard differs before import')
    if 'source_guard'in sys.modules:raise ValueError('source guard preloaded')
    spec0=importlib.util.spec_from_file_location('source_guard',guard_path)
    guard0=importlib.util.module_from_spec(spec0);sys.modules['source_guard']=guard0
    # Execute the already authenticated bytes; do not reopen a replaced path.
    exec(compile(guard_raw,str(guard_path),'exec'),guard0.__dict__)
    _BOOTSTRAP=guard0.bootstrap(args0.source_manifest,args0.source_manifest_sha256,root=HERE,entry=__file__)

sys.path[:0]=[str(HERE),str(HERE/'stage'),str(HERE/'runtime/cadathlon/common'),str(HERE/'runtime'),str(HERE/'runtime/jaxite_word')]
import component_protocol as p
from component_contract import ROOT,canonical
from component_processes import OwnedChild
from cleanup import cleanup_signals
from terminal import terminal_scope
from finish import finish
from publication import OwnedOutput,separate
from cleanup_origin import protect_body
EXPECTED_TESTS=371

def snapshot(out,publication):
    publication.check();raw=p.sources();pins=p.hashes(raw);p.check_sources(pins)
    if _BOOTSTRAP is not None and pins!=_BOOTSTRAP[0]:raise ValueError("source bytes changed after fresh admission")
    execution=out/'execution';execution.mkdir(mode=0o700)
    for name,body in raw.items():
        dest=execution/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(body)
    with tarfile.open(out/'sources.tar.gz','x:gz')as tar:
        for name in pins:tar.add(execution/name,arcname=name,recursive=False)
    archive=p.read(out/'sources.tar.gz');p.verify_archive(archive,pins)
    publication.check();publication.write(out/'source_hashes.json',pins)
    return execution,pins,p.sha(archive)

def execution_hashes(execution,pins):return{name:p.sha(p.read(execution/name,8<<20))for name in pins}

def environment(actual):
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',JAX_PLATFORMS='tpu'if actual else'cpu')
    env.pop('CROSS_SKIP_TOPLEVEL_ROTKEYS',None)
    return env

def stage(out,execution,pins,config,*,publication):
    report=dict(stage=config['stage'],complete=False,passed=False,source_verified=False,worker_verified=False,cleanup_completed=False,child_launch_attempted=False,
        workspace=config['workspace'],workspace_identity=config['workspace_identity'],source_hashes=pins)
    path=out/'report.json';primary=None;active=None;start=time.monotonic()
    publication.write(out/'config.json',config);publication.write(out/'source_hashes.json',pins)
    try:
        publication.write(path,report)
        command=[sys.executable,'-I','-B',str(execution/'component_worker.py'),'--execute','--config',str(out/'config.json'),
            '--config-sha256',p.sha(canonical(config)),'--source-manifest',str(out/'source_hashes.json'),'--source-manifest-sha256',p.sha(canonical(pins)),'--output',str(out/'worker')]
        publication.check()
        with(out/'worker.log').open('xb')as log:
            report['child_launch_attempted']=True;publication.write(path,report)
            active=OwnedChild(command,cwd=execution,stdout=log,env=environment(True));report['child']=active.record;publication.write(path,report)
            code=active.wait()
        report['child']=dict(active.record);active=None
        publication.check()
        worker_raw=p.read(out/'worker/report.json');worker=p.parse(worker_raw);report['worker_report_sha256']=p.sha(worker_raw)
        archive=p.read(out/'worker/sources.tar.gz');p.verify_archive(archive,pins)
        report['worker_verified']=bool(worker.get('source_archive_sha256')==p.sha(archive)and worker.get('source_hashes_start')==pins and worker.get('source_hashes_end')==pins)
        fresh=worker.get('fresh_source_guard',{})
        if fresh.get('passed')is not True or fresh.get('fresh')is not True or type(fresh.get('source_count'))is not int or fresh['source_count']!=len(pins)or fresh.get('manifest_sha256')!=p.sha(canonical(pins)):raise ValueError('fresh authenticated worker admission required')
        report['complete']=True
        report['passed']=bool(code==0 and report['worker_verified']and all(worker.get(k)is True for k in('passed','complete','source_verified','cleanup_completed','terminal_scope_completed','termination_scope_completed')))
    except BaseException as error:primary=protect_body(error);report['error']=dict(type=type(error).__name__,message=str(error))
    finally:
        with terminal_scope(report,path,publication.write,cleanup_signals,lambda:(primary,None)):
            cleanup=p.Finalizer(primary)
            cleanup.attempt('output ownership',publication.check)
            if active is not None:
                cleanup.attempt('owned child retirement',lambda:active.close(primary=primary));report['child']=dict(active.record)
            cleanup.attempt('workspace identity',lambda:p.check_identity(config['workspace'],config['workspace_identity']))
            current=cleanup.attempt('original source readback',lambda:p.hashes(p.sources()));copied=cleanup.attempt('execution source readback',lambda:execution_hashes(execution,pins))
            report.update(source_verified=current==pins and copied==pins,source_hashes_end=current,execution_hashes_end=copied,elapsed_s=time.monotonic()-start)
            child=report.get('child',{})
            report['cleanup_completed']=bool(not cleanup.errors and(not report['child_launch_attempted']or all(child.get(k)is True for k in('direct_child_reaped','group_empty_verified','cleanup_completed'))))
            report['cleanup_scope']='Owned OS child/group retirement only. Worker wrapper/map quarantine is separate; retained workspace is not deleted.'
            report['passed']=bool(report['passed']and report['source_verified']and report['cleanup_completed'])
            report['cleanup_errors']=[label+': '+repr(error)for label,error in cleanup.errors]
            cleanup.attempt('terminal publication',lambda:publication.write(path,report));cleanup.raise_failure()
    return report

def collect(out,execute,workspace_parent,*,publication,role='setup',config=None):
    report=dict(schema='toy4-checkpoint-diagnostic-collector-v2',execute=execute,stage=role if execute else 'preflight',
        fresh_source_guard=None if _BOOTSTRAP is None else _BOOTSTRAP[1],passed=False,complete=False,cleanup_completed=False,
        source_verified=False,roles=[],role_attempts=[],scope='One fixed toyBERT4 diagnostic phase, not a full-model benchmark')
    primary=None;active=None;path=out/'report.json';pins={};execution=None;start=time.monotonic()
    try:
        publication.write(path,report);execution,pins,archive_sha=snapshot(out,publication)
        from resource_plan import report as resource_report
        report.update(source_hashes_start=pins,source_archive_sha256=archive_sha,execution_root=str(execution),resource_plan=resource_report(workspace_parent));publication.write(path,report)
        if not execute:
            command=[sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider','--rootdir=.',
                '--junitxml',str(out/'tests.xml'),*sorted(name for name in pins if name.startswith('test_')and name.endswith('.py'))]
            publication.check()
            with(out/'tests.log').open('xb')as log:
                env=environment(False)|dict(TOY4_PREFLIGHT_AUDIT_PATH=str(out/'audit.json'))
                active=OwnedChild(command,cwd=execution,stdout=log,env=env);report['test_child']=active.record;publication.write(path,report);code=active.wait()
            report['test_child']=dict(active.record);active=None;publication.check()
            audit=p.parse(p.read(out/'audit.json'));report['preflight_audit']=audit
            tree=ET.fromstring(p.read(out/'tests.xml'));cases=tree.findall('.//testcase')
            xml_ok=len(cases)==EXPECTED_TESTS and not tree.findall('.//failure')and not tree.findall('.//error')and not tree.findall('.//skipped')
            report.update(tests_returncode=code,test_cases=len(cases),expected_test_cases=EXPECTED_TESTS,all_tests_executed_without_skip=bool(xml_ok))
            report['passed']=bool(code==0 and xml_ok and audit.get('zero_forbidden_attempts')is True and audit.get('backend_or_private_attempts')==[])
        else:
            p.require_ready()
            from component_worker import config_check,priors
            if role=='setup':
                if config is not None:raise ValueError('setup creates a fresh workspace; supplied prior config forbidden')
                if report['resource_plan']['capacity_passed']is not True:raise OSError('insufficient owned-workspace filesystem capacity')
                nonce=secrets.token_hex(16);workspace=workspace_parent/('toy4-diagnostic-'+nonce);workspace.mkdir(mode=0o700,exist_ok=False)
                config=dict(stage='setup',workspace=str(workspace),workspace_identity=p.identity(workspace),nonce=nonce,
                    setup_receipt=None,setup_sha256=None,server_receipt=None,server_sha256=None,server_root_review=None,server_root_review_sha256=None)
            else:
                if config is None or config.get('stage')!=role:raise ValueError('explicit exact one-role prior configuration required')
                config_check(config);priors(config,pins)
            config_check(config);report.update(workspace=config['workspace'],workspace_identity=config['workspace_identity'],workspace_retained=True)
            folder=out/role;owner=OwnedOutput(folder).acquire()
            report['role_attempts'].append(role);publication.write(path,report)
            receipt=finish(folder/'report.json',lambda:stage(folder,execution,pins,config,publication=owner),owner.write,owner.read)
            report['roles'].append(dict(stage=role,report_sha256=p.sha(owner.read(folder/'report.json')),passed=receipt['passed']));publication.write(path,report)
            if not receipt['passed']:raise RuntimeError('fixed diagnostic role failed: '+role)
            report['passed']=True
            report['next_phase_not_automatically_launched']=True
        report['complete']=True
    except BaseException as error:primary=protect_body(error);report['error']=dict(type=type(error).__name__,message=str(error))
    finally:
        with terminal_scope(report,path,publication.write,cleanup_signals,lambda:(primary,None)):
            cleanup=p.Finalizer(primary);cleanup.attempt('output ownership',publication.check)
            if active is not None:cleanup.attempt('test child retirement',lambda:active.close(primary=primary));report['test_child']=dict(active.record)
            if execution is not None:
                current=cleanup.attempt('original source readback',lambda:p.hashes(p.sources()));copied=cleanup.attempt('execution source readback',lambda:execution_hashes(execution,pins))
                cleanup.attempt('source archive readback',lambda:p.verify_archive(p.read(out/'sources.tar.gz'),pins))
                report.update(source_hashes_end=current,execution_hashes_end=copied,source_verified=current==pins and copied==pins)
            child=report.get('test_child',{});role_retirement={}
            for name in report['role_attempts']:
                receipt=cleanup.attempt('role retirement readback '+name,lambda name=name:p.parse(p.read(out/name/'report.json')))
                record={}if not isinstance(receipt,dict)else receipt
                role_retirement[name]=bool(record.get('cleanup_completed')is True and(record.get('child_launch_attempted')is False or all(record.get('child',{}).get(k)is True for k in('direct_child_reaped','group_empty_verified','cleanup_completed'))))
            report['role_retirement']=role_retirement
            retired=all(role_retirement.values())if execute else all(child.get(k)is True for k in('direct_child_reaped','group_empty_verified','cleanup_completed'))
            report['cleanup_completed']=bool(not cleanup.errors and retired)
            report['cleanup_scope']='Owned OS child/group retirement only; key/cipher/public-weight workspace retained. Failed worker arrays/maps remain separately quarantined until process exit.'
            report['passed']=bool(report['passed']and report['source_verified']and report['cleanup_completed']);report['elapsed_s']=time.monotonic()-start
            report['cleanup_errors']=[label+': '+repr(error)for label,error in cleanup.errors]
            cleanup.attempt('terminal publication',lambda:publication.write(path,report));cleanup.raise_failure()
    return report

def main():
    if _BOOTSTRAP is None:raise ValueError('fresh authenticated isolated script entry required')
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--source-manifest',type=Path,required=True);parser.add_argument('--source-manifest-sha256',required=True)
    parser.add_argument('--stage',choices=p.STAGES,default='setup');parser.add_argument('--config',type=Path);parser.add_argument('--config-sha256')
    parser.add_argument('--workspace-parent',type=Path,default=Path('/dev/shm'));args=parser.parse_args()
    if p.sha(p.read(args.source_manifest))!=args.source_manifest_sha256:raise ValueError('external manifest changed after bootstrap')
    args.output=args.output.absolute();args.workspace_parent=args.workspace_parent.absolute();config=None
    protected=[HERE,args.source_manifest,args.config]
    if args.execute:
        p.require_ready()
        if args.stage=='setup':
            if args.config is not None or args.config_sha256 is not None:raise ValueError('setup cannot accept prior config')
            # The parent itself is not protected: separate output may share /tmp.
        else:
            if args.config is None or type(args.config_sha256)is not str:raise ValueError('external role configuration required')
            raw=p.read(args.config)
            if p.sha(raw)!=args.config_sha256:raise ValueError('external role configuration hash differs')
            config=p.parse(raw)
            from component_worker import config_check,priors
            config_check(config)
            if config['stage']!=args.stage:raise ValueError('one exact role required')
            priors(config,_BOOTSTRAP[0])
            protected.extend(config[k]for k in ('workspace','setup_receipt','server_receipt','server_root_review'))
    elif args.config is not None or args.config_sha256 is not None or args.stage!='setup':raise ValueError('preflight has no prior role inputs')
    separate(args.output,protected);publication=OwnedOutput(args.output).acquire()
    report=finish(args.output/'report.json',lambda:collect(args.output,args.execute,args.workspace_parent,publication=publication,role=args.stage,config=config),publication.write,publication.read)
    return 0 if report['passed']else 1

if __name__=='__main__':raise SystemExit(main())
