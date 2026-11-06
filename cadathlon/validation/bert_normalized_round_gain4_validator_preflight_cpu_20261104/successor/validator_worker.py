"""Dedicated validator only: consume old163 evidence, execute new IO cleanup."""
from pathlib import Path
import argparse
import os
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'runtime/stage'),str(HERE/'common'),str(HERE/'runtime')]
import validator_common as common
import component_protocol as protocol
import component_key_gate as key_gate
import component_worker as original
from cleanup import cleanup_signals,termination_signals
from terminal import terminal_scope
from finish import finish


class Publication:
    def __init__(self):self.path=None;self.identity=None
    def begin(self,path,report):
        path=Path(path).absolute()
        if self.path is not None:raise ValueError('report ownership already acquired')
        identity=protocol.identity(path.parent)
        protocol.write(path,report,replace=False)
        protocol.check_identity(path.parent,identity)
        self.path=path;self.identity=identity
    def check(self,path):
        if self.path is None or Path(path).absolute()!=self.path:raise ValueError('report not created by this invocation')
        protocol.check_identity(self.path.parent,self.identity)
    def write(self,path,report):self.check(path);protocol.write(path,report);self.check(path)
    def read(self,path):self.check(path);value=key_gate.read(path);self.check(path);return value


def run(config,expected,output,*,publication=None):
    # All bootstrap restrictions precede the first publication, including a failure receipt.
    common.validate_config(config);common.separate(output,common.protected(config));common.require_ready()
    if os.environ.get('JAX_PLATFORMS')!='cpu':raise ValueError('explicit CPU validator required')
    publication=Publication()if publication is None else publication
    report=dict(schema='gain4-validator-successor-worker-v1',stage='validate',passed=False,phase_complete=False,
        pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)),loader_entered=False,loader_cleanup_completed=False,
        source_files_unchanged=False,termination_scope_completed=False,forbidden_attempts=[],
        workspace_identity=config['workspace_identity'],request_contract=config['request_contract'],
        producer_source_map_sha256=common.producer_gate.FIXED['producer_map_sha256'],
        external_stores_retained=True,scope='Only original54 fixed gain4 gates; no new fixture/server/full-model execution.')
    publication.begin(output,report);start=time.monotonic();primary=None
    def publish():publication.write(output,report)
    try:
        profile,plan,checked=common.inspect_sources(expected)
        report['source_hashes_start']=expected
        receipts=common.prior_gates(config)
        keys=protocol.authenticate_keys(config['key_evidence'],config['key_workspace'],profile=profile,plan=plan)
        report.update(key_report_sha256=protocol.KEY_REPORT,context_id=keys['context_id'])
        from component_boundaries import role_scope
        with role_scope('validate',calls=report['forbidden_attempts']):
            report['loaded_local_modules_start']=protocol.imported_sources(expected)
            # This body retains all original payload-before-private and fixed numeric checks.
            original.validate(config,keys,profile,plan,report,publish,receipts)
        if report['forbidden_attempts']:raise ValueError('forbidden validator operation attempted')
        protocol.check_identity(config['workspace'],config['workspace_identity'])
        if common.prior_gates(config)!=receipts:raise ValueError('producer terminal evidence changed')
        again=protocol.authenticate_keys(config['key_evidence'],config['key_workspace'],profile=profile,plan=plan)
        if again!=keys:raise ValueError('external key evidence changed')
        common.require_ready()
        report.update(phase_complete=True,passed=True)
    except BaseException as error:
        primary=error;report.update(passed=False,error=dict(type=type(error).__name__,message=str(error)))
    finally:
        with terminal_scope(report,output,publication.write,cleanup_signals,lambda:(primary,None)):
            try:
                after=protocol.hashes(protocol.sources())
                report.update(source_hashes_end=after,source_files_unchanged=after==expected,
                    loaded_local_modules_end=protocol.imported_sources(expected),elapsed_s=time.monotonic()-start)
                report['passed']=bool(report['passed']and report['source_files_unchanged']and report['loader_cleanup_completed'])
                publish()
            except BaseException as error:
                if primary is None:raise
                primary.add_note('validator terminal evidence: '+repr(error))
            if primary is not None:raise primary
    return report


def main():
    parser=argparse.ArgumentParser()
    for name in ('config','source-manifest','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--config-sha256',required=True);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute:raise ValueError('explicit validator execution required')
    common.require_ready()
    raw=key_gate.read(args.config)
    if protocol.sha(raw)!=key_gate.sha_field(args.config_sha256):raise ValueError('externally pinned worker configuration differs')
    config=key_gate.parse(raw);common.validate_config(config);common.separate(args.output,common.protected(config))
    if args.output.exists()or args.output.is_symlink():raise FileExistsError('existing worker report remains untouched')
    expected=key_gate.parse(key_gate.read(args.source_manifest));publication=Publication()
    report=finish(args.output,lambda:run(config,expected,args.output,publication=publication),publication.write,publication.read,termination=termination_signals)
    return 0 if report['passed']else 1


if __name__=='__main__':raise SystemExit(main())
