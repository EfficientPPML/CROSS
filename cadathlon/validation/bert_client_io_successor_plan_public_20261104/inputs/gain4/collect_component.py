"""Snapshot/reap one proof phase or its no-HE preflight; no healthy deadline."""
from pathlib import Path
import argparse
import io
import json
import os
import resource
import secrets
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'runtime/stage'),str(HERE/'common'),str(HERE/'runtime')]
import component_protocol as protocol
import component_contract as contract
import component_key_gate as key_gate
from cleanup import cleanup_signals,termination_signals
from terminal import terminal_scope
from finish import finish
from component_processes import OwnedChild


class OwnedOutput:
    """Permit report republication only after this call created the directory.

    Exact directory identity is rechecked around access; this is not an atomic
    guarantee against an adversarial same-user filesystem replacement.
    """
    def __init__(self):self.path=None;self.identity=None
    def claim(self,path):
        if self.path is not None:raise ValueError('output ownership already acquired')
        path=Path(path).absolute();identity=protocol.identity(path)
        self.path=path;self.identity=identity
    def check(self,path):
        if self.path is None or Path(path).absolute()!=self.path/'report.json':raise ValueError('report path is not owned by this invocation')
        protocol.check_identity(self.path,self.identity)
    def write(self,path,value):
        self.check(path);protocol.write(path,value);self.check(path)
    def read(self,path):
        self.check(path);raw=key_gate.read(path);self.check(path);return raw


def test_counts(path):
    cases=ET.parse(path).findall('.//testcase')
    return dict(tests=len(cases),failures=sum(c.find('failure')is not None for c in cases),
        errors=sum(c.find('error')is not None for c in cases),skipped=sum(c.find('skipped')is not None for c in cases))


def source_snapshot(out,raw):
    snapshot=out/'execution_source';snapshot.mkdir()
    with tarfile.open(out/'sources.tar.gz','w:gz') as z:
        for name,body in sorted(raw.items()):
            protocol.relative(name)
            path=snapshot/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(body)
            entry=tarfile.TarInfo(name);entry.size=len(body);z.addfile(entry,io.BytesIO(body))
    expected=protocol.hashes(raw);protocol.write(out/'source_manifest.json',expected,replace=False)
    key_gate.verify_source_archive((out/'sources.tar.gz').read_bytes(),expected)
    return snapshot,expected


def preceding_config(stage,fixture_evidence,fixture_sha,server_evidence,server_sha,expected):
    if stage=='fixture':
        if any(v is not None for v in (fixture_evidence,fixture_sha,server_evidence,server_sha)):
            raise ValueError('new fixture phase cannot inherit previous evidence')
        request=contract.make_contract(contract.CONTEXT_ID,secrets.token_hex(16))
        workspace=Path('/tmp')/('cross-normalized-gain4-'+request['nonce'])
        return request,workspace,None
    if fixture_evidence is None or fixture_sha is None:raise ValueError('externally pinned completed fixture phase required')
    raw=key_gate.read(Path(fixture_evidence)/'report.json')
    if protocol.sha(raw)!=key_gate.sha_field(fixture_sha):raise ValueError('fixture parent pin differs')
    value=key_gate.parse(raw);request=contract.validate_contract(value.get('request_contract'))
    workspace=Path(value.get('workspace',''));identity=value.get('workspace_identity')
    if str(workspace)!=str(Path('/tmp')/('cross-normalized-gain4-'+request['nonce'])):
        raise ValueError('existing proof IO path differs')
    protocol.terminal_gate(fixture_evidence,fixture_sha,stage='fixture',expected_sources=expected,
        workspace=workspace,workspace_identity=identity,request=request)
    if stage=='validate':
        if server_evidence is None or server_sha is None:raise ValueError('externally pinned completed server phase required')
        server=protocol.terminal_gate(server_evidence,server_sha,stage='server',expected_sources=expected,
            workspace=workspace,workspace_identity=identity,request=request)
        if server['readback']['input_digest']!=value['readback']['input_digest']:raise ValueError('server/fixture identity differs')
    elif server_evidence is not None or server_sha is not None:raise ValueError('unused server evidence must be absent')
    return request,workspace,identity


def collect(*,output,key_evidence,key_workspace,stage='fixture',execute=False,fixture_evidence=None,
        fixture_report_sha256=None,server_evidence=None,server_report_sha256=None,_publication_owner=None):
    out=Path(output).absolute();key_evidence=Path(key_evidence).absolute();key_workspace=Path(key_workspace).absolute()
    resolved=out.resolve()
    for protected in (key_evidence,key_workspace):
        target=protected.resolve()
        if resolved==target or resolved.is_relative_to(target)or target.is_relative_to(resolved):raise ValueError('output must be separate from retained key/evidence directories before any write')
    publication=OwnedOutput()if _publication_owner is None else _publication_owner
    out.mkdir(mode=0o700,exist_ok=False);publication.claim(out)
    start=time.monotonic();primary=None;cleanup_error=None
    active=None;raw={};expected={};snapshot=None;workspace=None;workspace_fd=None;report_path=out/'report.json'
    report=dict(schema=protocol.TERMINAL_SCHEMA if execute else 'cross-normalized-gain4-preflight-v1',passed=False,
        stage=stage,execute=execute,termination_scope_completed=False,worker_reaped=False,worker_group_empty_verified=False,source_files_unchanged=False,source_archive_verified=False,
        children=[],cleanup_errors=[],cpu_affinity=sorted(os.sched_getaffinity(0)),
        scope='Separate component preparation or one genuine phase. No full model/ranked/calibrated timing claim.')
    def publish():publication.write(report_path,report)
    def child(command,name):
        nonlocal active
        with (out/(name+'.log')).open('wb') as log:
            active=OwnedChild(command,cwd=snapshot,stdout=log)
            active.record['name']=name;report['children'].append(active.record);publish()
            code=active.wait()
            if not active.record['group_empty_verified']:raise RuntimeError('owned process-group absence required')
            active=None;publish();return code
    try:
        publish()
        if os.environ.get('JAX_PLATFORMS')!='cpu':raise ValueError('explicit CPU parent required')
        raw=protocol.sources();snapshot,expected=source_snapshot(out,raw)
        report.update(source_hashes_start=expected,source_archive_sha256=protocol.sha((out/'sources.tar.gz').read_bytes()),source_archive_verified=True)
        profile,plan=protocol.check_sources(expected)
        keys=protocol.authenticate_keys(key_evidence,key_workspace,profile=profile,plan=plan)
        report['public_key_prerequisite']={n:keys[n] for n in ('context_id','key_report_sha256','key_source_archive_sha256')}
        if execute:
            protocol.require_ready()
            request,workspace,owner=preceding_config(stage,fixture_evidence,fixture_report_sha256,server_evidence,server_report_sha256,expected)
            # This is a storage reserve, never a claim about actual process RAM.
            storage=contract.numerical_plan()['io'];need=(storage['input_payload_bytes']+storage['output_payload_bytes']+104857600+(1<<30))
            if shutil.disk_usage(workspace.parent).free<need:raise ValueError('proof IO storage reserve unavailable')
            report['storage_reserve_bytes']=need
            if stage=='fixture':workspace.mkdir(mode=0o700,exist_ok=False);owner=protocol.identity(workspace)
            from owned_directory import open_owned_directory
            workspace_fd,held=open_owned_directory(workspace)
            if [held['device'],held['inode']]!=owner:raise ValueError('proof workspace changed before worker launch')
            protocol.check_identity(workspace,owner)
            report.update(workspace=str(workspace),workspace_identity=owner,request_contract=request)
            config=dict(stage=stage,workspace=str(workspace),workspace_identity=owner,request_contract=request,
                key_evidence=str(Path(key_evidence)),key_workspace=str(Path(key_workspace)),
                fixture_evidence=None if fixture_evidence is None else str(Path(fixture_evidence)),fixture_report_sha256=fixture_report_sha256,
                server_evidence=None if server_evidence is None else str(Path(server_evidence)),server_report_sha256=server_report_sha256)
            protocol.write(out/'worker_config.json',config,replace=False);(out/'worker').mkdir(mode=0o700)
            command=[sys.executable,str(snapshot/'component_worker.py'),'--config',str(out/'worker_config.json'),
                '--config-sha256',protocol.sha((out/'worker_config.json').read_bytes()),'--source-manifest',str(out/'source_manifest.json'),
                '--output',str(out/'worker/worker_report.json'),'--execute']
            code=child(command,'worker');report.update(worker_exit_code=code,worker_reaped=True,worker_group_empty_verified=True)
            if code!=0:raise RuntimeError('component worker failed; outputs retained for inspection')
            worker_raw=key_gate.read(out/'worker/worker_report.json');worker=key_gate.parse(worker_raw)
            if (any(worker.get(k)is not True for k in ('passed','phase_complete','loader_cleanup_completed','source_files_unchanged','terminal_scope_completed','termination_scope_completed'))
                    or worker.get('error') or worker.get('forbidden_attempts')!=[] or worker.get('source_hashes_start')!=expected
                    or worker.get('source_hashes_end')!=expected or worker.get('request_contract')!=request
                    or worker.get('workspace_identity')!=owner):raise ValueError('strict completed phase worker report required')
            protocol.check_identity(workspace,owner)
            report.update(worker_report_sha256=protocol.sha(worker_raw),readback=worker['readback'],workspace_retained=True)
            if stage=='validate':report['numeric_summary']=worker['numeric_summary']
        else:
            settings=key_gate.parse(key_gate.read(HERE/'preflight_expectations.json'))
            for name,files,count in (('mock_tests',settings['mock_test_files'],settings['mock_tests']),
                    ('installed_denials',[settings['installed_test_file']],settings['installed_tests'])):
                code=child([sys.executable,'-m','pytest','-q',*files,'--rootdir=.',f'--junitxml={out}/{name}.xml'],name)
                counts=test_counts(out/(name+'.xml'));report[name]=dict(exit_code=code,**counts)
                if code!=0 or counts!=dict(tests=count,failures=0,errors=0,skipped=0):raise ValueError('complete declared preparation suite required')
            report.update(no_private_or_ciphertext_payload_read=True,no_encrypted_execution=True)
        report['passed']=True
    except BaseException as error:
        primary=error;report.update(passed=False,error=dict(type=type(error).__name__,message=str(error)))
    finally:
        with terminal_scope(report,report_path,publication.write,cleanup_signals,lambda:(primary,cleanup_error)):
            if active is not None:
                try:
                    active.close(primary=primary)
                except BaseException as error:cleanup_error=error;report['cleanup_errors'].append(repr(error))
            if workspace_fd is not None:
                try:protocol.close(workspace_fd)
                except BaseException as error:
                    if cleanup_error is None or not isinstance(error,Exception) and isinstance(cleanup_error,Exception):
                        if cleanup_error is not None:error.add_note('earlier collector cleanup: '+repr(cleanup_error))
                        cleanup_error=error
                    else:cleanup_error.add_note('additional collector cleanup: '+repr(error))
                    report['cleanup_errors'].append(repr(error))
            try:
                report['cleanup_errors'].extend('child '+row['name']+': '+error for row in report['children'] for error in row['cleanup_errors'])
                after=protocol.hashes(protocol.sources());report.update(source_hashes_end=after,source_files_unchanged=bool(expected) and after==expected,
                    elapsed_s=time.monotonic()-start,all_direct_children_reaped=all(r['direct_child_reaped'] for r in report['children']),
                    all_owned_groups_empty=all(r['group_empty_verified'] for r in report['children']),
                    process_cleanup_completed=all(r['cleanup_completed'] for r in report['children']),
                    child_maxrss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                    memory_measurement_scope='OS maximum child RSS; not aggregate concurrent RSS or a capacity guarantee')
                if snapshot is not None and {n:protocol.sha((snapshot/n).read_bytes()) for n in expected}!=expected:
                    raise ValueError('execution snapshot changed')
                report['passed']=bool(report['passed'] and report['source_files_unchanged'] and report['all_direct_children_reaped']
                    and report['all_owned_groups_empty'] and report['process_cleanup_completed'] and not report['cleanup_errors'] and cleanup_error is None)
                publish()
            except BaseException as error:
                if primary is None:raise
                primary.add_note('component parent final evidence: '+repr(error))
            if primary is not None:
                if cleanup_error is not None:primary.add_note('component parent cleanup: '+repr(cleanup_error))
                raise primary
            if cleanup_error is not None:raise cleanup_error
    print(json.dumps(dict(passed=report['passed'],execute=execute,stage=stage,report_sha256=protocol.sha(report_path.read_bytes()),elapsed_s=report['elapsed_s'])))
    return 0 if report['passed'] else 1


def main():
    parser=argparse.ArgumentParser()
    for name in ('output','key-evidence','key-workspace'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('fixture-evidence','server-evidence'):parser.add_argument('--'+name,type=Path)
    for name in ('fixture-report-sha256','server-report-sha256'):parser.add_argument('--'+name)
    parser.add_argument('--stage',choices=protocol.STAGES,default='fixture');parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    publication=OwnedOutput()
    def operation():
        collect(**vars(args),_publication_owner=publication)
        return key_gate.parse(publication.read(args.output/'report.json'))
    report=finish(args.output/'report.json',operation,publication.write,publication.read,termination=termination_signals)
    return 0 if report['passed']else 1



if __name__=='__main__':raise SystemExit(main())
