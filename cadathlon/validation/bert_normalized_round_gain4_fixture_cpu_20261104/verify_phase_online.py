"""Independent public terminal and full CT transport readback; no private keys."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
class NoBackend:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in ('jax','jaxlib','jaxite_word','torch','transformers'):
            raise RuntimeError('terminal readback forbids backend/model import: '+fullname)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('fixture','server','validate'));parser.add_argument('report_sha256');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    lab=Path('/tmp/bert-normalized-round-gain4-ready-20261104');evidence=lab/('actual-'+args.stage+'-v1');start=time.monotonic()
    if args.output.exists():raise FileExistsError('fresh independent readback report required')
    sys.meta_path.insert(0,NoBackend());sys.path[:0]=[str(lab),str(lab/'runtime/stage'),str(lab/'common'),str(lab/'runtime')]
    attempts=[]
    def audit(event,values):
        if event=='open'and values and isinstance(values[0],(str,bytes,os.PathLike)):
            path=os.fsdecode(values[0])
            if 'client_private'in Path(path).parts or path.endswith('.safetensors')or ('cross-full-bert-keys-'in path and path.endswith('.npy')):
                attempts.append(path);raise PermissionError('terminal readback forbids private/key/model payloads')
    sys.addaudithook(audit)
    import component_protocol as protocol
    import component_contract as contract
    sha=lambda b:hashlib.sha256(b).hexdigest()
    preflight_raw=(lab/'source-bound-preflight-v1/report.json').read_bytes()
    assert sha(preflight_raw)=='6fbf898fbcb8a884e2f2c7f1e2308dfd04171be990bfde82a20200d20f00e19c'
    expected=json.loads(preflight_raw)['source_hashes_start'];assert len(expected)==163
    before=protocol.hashes(protocol.sources());assert before==expected
    raw=(evidence/'report.json').read_bytes();assert sha(raw)==args.report_sha256;r=json.loads(raw)
    request=r['request_contract'];workspace=Path(r['workspace']);identity=r['workspace_identity']
    protocol.terminal_gate(evidence,args.report_sha256,stage=args.stage,expected_sources=expected,workspace=workspace,workspace_identity=identity,request=request)
    assert {n:sha((evidence/'execution_source'/n).read_bytes())for n in expected}==expected
    profile,plan=protocol.check_sources(expected)
    keys=protocol.authenticate_keys('/tmp/bert-full-key-generation-component-20261103/actual-v1','/dev/shm/cross-full-bert-keys-143b3f5c1573427d841760a3c9d5d33d',profile=profile,plan=plan)
    inputs=protocol.readback(workspace,request,role='request')
    assert inputs['tile_count']==2
    if args.stage=='fixture':assert inputs==r['readback']
    outputs=None
    if args.stage!='fixture':
        outputs=protocol.readback(workspace,request,role='response',input_digest=inputs['input_digest']);assert outputs['tile_count']==54
        if args.stage=='server':assert outputs==r['readback']
    protocol.check_identity(workspace,identity)
    assert protocol.hashes(protocol.sources())==before
    producer_pids=[]
    for row in r['children']:
        assert row['direct_child_reaped']is True and row['group_empty_verified']is True
        assert not Path('/proc',str(row['pid'])).exists();producer_pids.append(row['pid'])
        try:os.killpg(row['process_group'],0)
        except ProcessLookupError:pass
        else:raise RuntimeError('owned group still present')
    assert not attempts
    result=dict(passed=True,stage=args.stage,report_sha256=args.report_sha256,worker_report_sha256=r['worker_report_sha256'],source_archive_sha256=r['source_archive_sha256'],sources=163,current_execution_archive_equal=True,worker_pids_absent=producer_pids,owned_groups_absent=True,workspace=str(workspace),workspace_identity=identity,input_readback=inputs,output_readback=outputs,public_key_prerequisite={n:keys[n]for n in ('context_id','key_report_sha256','key_source_archive_sha256')},forbidden_attempts=attempts,backend_modules=[n for n in ('jax','jaxlib','jaxite_word','torch','transformers')if n in sys.modules],elapsed_s=time.monotonic()-start,scope='Independent full transport headers/residues/digest readback and public prerequisite/source proof; no raw arrays archived and no private/key/model payload read. Outer tool session reap is separately attested by the owner.')
    assert result['backend_modules']==[]
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:result[k]for k in ('passed','stage','sources','elapsed_s')}))
if __name__=='__main__':main()
