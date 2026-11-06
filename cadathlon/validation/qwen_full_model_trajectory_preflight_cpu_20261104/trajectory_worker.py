"""Dedicated-process, two fixed fixtures; run only after explicit review grant.

Actual immutable packed owner loops use NumPy payloads and identity refresh.
Independent dense and unchanged frozen references start the same original
embedding rows; neither supplies any owner intermediate. No HE/model forward.
"""
from pathlib import Path
import argparse,gc,hashlib,importlib.util,json,os,sys,time
import numpy as np
P=Path(__file__).resolve().parent
from trajectory_assets import requirement,runtime_gate,calibration_gate,asset_gate,selected_ids,PublicParameters,FrozenWeights,sha
from trajectory_observer import Observer,metric,TrajectoryFailure
from validation_cleanup import cleanup_signals,termination_signals
from validation_terminal import terminal_scope
from finish import finish
from full_reference import forward
class NoBackend:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('jax','jaxlib','torch','transformers','safetensors'):
            raise RuntimeError('wholegraph ideal proof forbids actual backend/model import: '+fullname)
sys.meta_path.insert(0,NoBackend())

def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def sources():
    return json.loads((P/'trajectory_inventory.json').read_bytes())['sources']
def source_hashes():return {n:sha(P/n) for n in sources()}
def error_row(error):
    row=dict(type=type(error).__name__,message=str(error),notes=getattr(error,'__notes__',[]))
    if hasattr(error,'label'):row['label']=error.label
    if hasattr(error,'record'):row['record']=error.record
    return row

def execute(index,out):
    if type(index)is not int or index not in (0,1):raise ValueError('exact fixed fixture index required')
    # main acquired this fresh output before entering republication scope.
    start=time.monotonic();primary=None;cleanup_error=None;parameters=None;observer=None;patch=None
    owner=result=reference=engine=original=inputs=None
    report=dict(passed=False,complete=False,cleanup_completed=False,termination_scope_completed=False,fixture_index=index,
        scope=__doc__,HE_executed=False,model_forward=False,coefficient_changes=False,parameter_records={})
    def status(row):
        print(json.dumps(dict(row,elapsed_s=time.monotonic()-start),allow_nan=False),flush=True)
    try:
        write(out/'report.json',report);report['source_hashes_before']=source_hashes();req=requirement();runtime_gate(req)
        report['calibration_prerequisite']=calibration_gate(req)
        report['asset_sha256_before']=asset_gate(req)
        ids,provenance=selected_ids(req);report['input_provenance']=provenance
        report['fixture']=req['fixtures'][index];positions=tuple(req['fixtures'][index]['positions'])
        parameters=PublicParameters(req)
        embeddings=parameters.matrix('model.embed_tokens.weight')[np.asarray(ids[index],dtype=np.int64)].copy()
        report['embedding_sha256']=hashlib.sha256(embeddings.tobytes()).hexdigest();report['embedding_shape']=list(embeddings.shape)
        report['token_vector_sha256']=hashlib.sha256(np.asarray(ids[index],dtype='<i8').tobytes()).hexdigest()
        manifest_bytes=(P/'inputs/candidate_manifest.json').read_bytes();config_bytes=(P/'inputs/config.json').read_bytes()
        manifest=json.loads(manifest_bytes);config=json.loads(config_bytes)
        status(dict(phase='authenticated_fixture',name=req['fixtures'][index]['name']))
        reference=forward(embeddings,positions,manifest_bytes=manifest_bytes,config_bytes=config_bytes,matrix=parameters.matrix,gamma=parameters.gamma)
        report['independent_reference_domains']=reference.domains;status(dict(phase='independent_dense_complete',operators=len(reference.operators),blocks=len(reference.blocks)))
        spec=importlib.util.spec_from_file_location('wholegraph_frozen_arithmetic',P/'inputs/frozen_arithmetic.py');frozen=importlib.util.module_from_spec(spec);sys.modules[spec.name]=frozen;spec.loader.exec_module(frozen)
        engine=frozen.QwenArithmetic(manifest)
        frozen_logits,engine,blocks,operators=frozen.approximate_forward(config,FrozenWeights(parameters),embeddings[None],manifest,engine=engine)
        comparison=dict(logits=metric(frozen_logits.data,reference.logits),blocks={str(i):metric(v,reference.blocks[i]) for i,v in enumerate(blocks)},
                        operators={n:metric(v,reference.operators[n]) for n,v in operators.items()})
        report['frozen_reference_comparison']=comparison;report['frozen_reference_domains']=engine.domains
        if len(blocks)!=28 or set(operators)!=set(reference.operators) or len(operators)!=421 or not all(r['passed'] for r in [comparison['logits'],*comparison['blocks'].values(),*comparison['operators'].values()]):raise TrajectoryFailure('independent dense versus frozen arithmetic gate failed')
        status(dict(phase='frozen_reference_complete',operators=421,blocks=28))
        # Import only explicit ideal doubles; real owner files remain unchanged.
        import pytest
        from tail_ideal import Arithmetic,Linear,OuterIdentityRefresh,Value,DELTA,install_matrix
        from plan_full import contract
        from encrypted_qwen_full_model import QwenParameterSource,QwenFullModel
        from full_metrics import check_logits
        patch=pytest.MonkeyPatch();install_matrix(patch);arithmetic=Arithmetic();linear=Linear(arithmetic);refresh=OuterIdentityRefresh(arithmetic)
        c=contract(positions)
        for key in c.required_keys:
            arithmetic.ctx._param_cache.raw_rotation_keys.setdefault(key,{'a':object(),'b':object()});arithmetic.ctx._param_cache.coef_maps.setdefault(key,object())
        provider=QwenParameterSource(c,matrix_loader=parameters.matrix,gamma_loader=parameters.gamma)
        observer=Observer(reference,manifest,positions,publish=status)
        owner=QwenFullModel(arithmetic,linear,c,provider,refresher=refresh,progress=observer)
        payload=np.zeros((16,4096));payload[:len(embeddings),:1024]=embeddings
        inputs=(Value(payload.ravel(),25,DELTA),);original=(inputs[0].polynomial,inputs[0].polynomial.copy())
        result=owner.evaluate(inputs)
        if inputs[0].polynomial is not original[0] or not np.array_equal(inputs[0].polynomial,original[1]):raise TrajectoryFailure('original embedding ciphertext stand-in changed')
        report['coverage']=observer.complete();report['owner_observations']=observer.summary()
        report['logit_metrics']=check_logits(tuple(v.polynomial for v in result.logit_blocks),reference.logits[0])
        if not report['logit_metrics']['passed']:raise TrajectoryFailure('all-slot full-vocabulary gate failed')
        expected_workload=next(r for r in json.loads((P/'public_plan.json').read_bytes())['workloads'] if r['sequence']==len(positions))
        if len(refresh.calls)!=expected_workload['refresh_counts']['total']:raise TrajectoryFailure('literal complete refresh count differs')
        report.update(primitive_counts=dict(arithmetic.calls),refresh_count=len(refresh.calls),expected_refresh_count=expected_workload['refresh_counts']['total'],
            actual_logit_metadata=[dict(row) for row in result.metadata],complete=True)
        report['backend_modules']=[n for n in ('jax','jaxlib','torch','transformers','safetensors') if n in sys.modules]
        if report['backend_modules']:raise RuntimeError('actual backend/model imported')
    except BaseException as error:
        primary=error;report['error']=error_row(error);report['failure_detail']=error_row(error)
        if engine is not None:report['frozen_reference_domains']=engine.domains
        if observer is not None:report['owner_observations']=observer.summary()
    finally:
        with terminal_scope(report,out/'report.json',write,cleanup_signals,lambda:(primary,cleanup_error)):
            def attempt(callback):
                nonlocal cleanup_error
                try:callback()
                except BaseException as error:
                    if cleanup_error is None:cleanup_error=error
                    elif isinstance(cleanup_error,Exception) and not isinstance(error,Exception):error.add_note(repr(cleanup_error));cleanup_error=error
                    else:cleanup_error.add_note(repr(error))
            if parameters is not None:
                report['parameter_records']=dict(parameters.records);attempt(parameters.readback)
            # All ideal arrays are synchronous. Production quarantine lists are
            # intentionally not reset on failure; dedicated process exit owns them.
            if patch is not None:attempt(patch.undo)
            if parameters is not None:attempt(parameters.close)
            if 'asset_sha256_before' in report:attempt(lambda:report.update(asset_sha256_after=asset_gate(requirement())))
            attempt(lambda:report.update(source_hashes_after=source_hashes()))
            report['source_files_unchanged']=report.get('source_hashes_after')==report.get('source_hashes_before') and 'source_hashes_before' in report
            report['assets_unchanged']=report.get('asset_sha256_after')==report.get('asset_sha256_before') and 'asset_sha256_after'in report
            report['cleanup_completed']=cleanup_error is None
            if cleanup_error is not None:report['cleanup_error']=error_row(cleanup_error)
            report['elapsed_s']=time.monotonic()-start
            report['passed']=primary is None and cleanup_error is None and report['complete'] and report['source_files_unchanged'] and report['assets_unchanged']
            write(out/'report.json',report)
    if primary is not None:
        if cleanup_error is not None:primary.add_note(repr(cleanup_error))
        raise primary
    if cleanup_error is not None:raise cleanup_error
    return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true',required=True);parser.add_argument('--fixture',type=int,choices=(0,1),required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(exist_ok=False)
    report=finish(args.output/'report.json',lambda:execute(args.fixture,args.output),write,lambda p:p.read_bytes(),termination=termination_signals,shield=cleanup_signals)
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
