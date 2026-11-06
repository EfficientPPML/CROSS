"""Synthetic producer proofs only; no real worker, payload or private-file access."""
from pathlib import Path
import copy,hashlib,io,json,math,tarfile
import pytest
import ranked_case_gate as cases
import ranked_stage_gate as gate
import transport_contract as tc

HERE=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()


@pytest.fixture(scope='module')
def schedule():
    return cases.authenticate(HERE/'public_evidence/fixture',cases.FIXTURE_REPORT_SHA,
        HERE/'public_evidence/reference',cases.REFERENCE_REPORT_SHA)


def summary(stage,case,previous=None):
    p=tc.parameters();n=case.count;scale=p['scaling_factor'];batch='b'*64
    if stage==3:
        shape=[1,2,256,512,100];per=math.prod(shape)*4
        rows=[dict(ordinal=i,request_id=f'{i:032x}',input_digest=sha(str(i).encode()),tiles=[dict(tile=t,payload_bytes=per,
            shape=shape,scale=scale,manifest_sha256=sha(f'{i}/{t}'.encode())) for t in range(8)]) for i in range(n)]
        return dict(sample_count=n,tile_count=8*n,public_batch_sha256=batch,ciphertext_payload_bytes=8*n*per,requests=rows)
    if stage==4:
        shape=[1,2,256,512,6];per=math.prod(shape)*4
        rows=[dict(ordinal=i,request_id=r['request_id'],input_digest=r['input_digest'],level=2,scale=scale,shape=shape,
            payload_bytes=per,response_manifest_sha256=sha(('response'+str(i)).encode())) for i,r in enumerate(previous['readback']['requests'])]
        return dict(sample_count=n,response_count=n,public_batch_sha256=batch,response_batch_sha256='c'*64,
            ciphertext_payload_bytes=n*per,responses=rows,full_model_compute_and_transport_seconds=1.)
    return dict(sample_count=n,public_batch_sha256=batch,private_receipt_validated=True,result_permissions_0600=True,
        decoded_result_sha256='d'*64,completed_receipt_sha256='e'*64,metrics=dict(sample_count=n,
            he_correct=case.official_correct,official_correct=case.official_correct,all_ordered_finite_logits_validated=True,
            postprocessed_predictions_validated=True,ranked_accuracy_floor_passed=True,ranked_accuracy_drop_passed=True,final_ranked_acceptance=True))


def proof(tmp_path,case,stage,previous=None,mutate=None):
    p=tmp_path/f'stage{stage}';(p/'worker').mkdir(parents=True)
    code=b'synthetic_fixture_only = True\n';expected={'tiny.py':sha(code)}
    with tarfile.open(p/'sources.tar.gz','w:gz') as t:
        m=tarfile.TarInfo('tiny.py');m.size=len(code);t.addfile(m,io.BytesIO(code))
    bindings=dict(case_binding=gate.case_binding(case),context_id='1'*64,model_id='2'*64,key_report_sha256='3'*64,
        model_report_sha256='4'*64,previous_stage_report_sha256=None if previous is None else previous['report_sha256'])
    readback=summary(stage,case,previous)
    worker=dict(passed=True,actual_stage_started=True,actual_stage_completed=True,readback_complete=True,
        source_files_unchanged=True,cleanup_completed=True,terminal_scope_completed=True,source_hashes_start=expected,
        source_hashes_end=expected,phases=dict(encryption=stage==3,decryption=stage==5,server_compute=stage==4,model_forward=False,key_generation=False),
        all_ordered_input_bindings_validated=True,full_model_graph_completed=True,all_ordered_response_bindings_validated=True,
        all_ordered_private_outputs_validated=True,bindings=bindings,readback=readback)
    report=dict(schema='cross-bert-ranked-stage-proof-v1',stage=stage,passed=True,execute=True,actual_execution_attempted=True,
        operation_complete=True,all_processes_reaped=True,source_files_unchanged=True,source_archive_verified=True,
        cleanup_completed=True,terminal_scope_completed=True,owned_artifacts_retained=True,source_hashes_start=expected,
        source_hashes_end=expected,execution_source_hashes_end=expected,source_archive_sha256=sha((p/'sources.tar.gz').read_bytes()),
        worker_process=dict(returncode=0,process_reaped=True,result=dict(passed=True)),bindings=bindings,readback=readback)
    if mutate:mutate(report,worker)
    worker_raw=tc.canonical(worker);(p/'worker/worker_report.json').write_bytes(worker_raw)
    report['worker_process']['result']['worker_report_sha256']=sha(worker_raw)
    raw=tc.canonical(report);(p/'report.json').write_bytes(raw)
    return gate.authenticate(p,sha(raw),stage=stage,case=case,expected_sources=expected,context_id='1'*64,
        model_id='2'*64,key_report_sha256='3'*64,model_report_sha256='4'*64,previous=previous)


@pytest.mark.parametrize('size',[1,2,3])
def test_complete_ordered_producer_chain(tmp_path,schedule,size):
    case=schedule.case(size,42);previous=None
    for stage in (3,4,5):previous=proof(tmp_path,case,stage,previous)
    assert previous['readback']['metrics']['final_ranked_acceptance']


@pytest.mark.parametrize('stage',[3,4,5])
@pytest.mark.parametrize('fault',['outer_terminal','worker_terminal','bool_exit','phase','bindings','source'])
def test_public_terminal_source_phase_failures(tmp_path,schedule,stage,fault):
    case=schedule.case(1,43);previous=None
    for earlier in range(3,stage):previous=proof(tmp_path,case,earlier,previous)
    def mutate(r,w):
        if fault=='outer_terminal':r['terminal_scope_completed']=False
        elif fault=='worker_terminal':w['terminal_scope_completed']=False
        elif fault=='bool_exit':r['worker_process']['returncode']=False
        elif fault=='phase':w['phases']['model_forward']=True
        elif fault=='bindings':w['bindings']=copy.deepcopy(w['bindings']);w['bindings']['case_binding']['seed']=44
        else:w['source_hashes_end']={}
    with pytest.raises(ValueError):proof(tmp_path,case,stage,previous,mutate)


@pytest.mark.parametrize('fault',['last_scale','swap','missing','payload','last_input','last_level'])
def test_bad_last_or_reordered_response_proof(tmp_path,schedule,fault):
    case=schedule.case(3,44);old=proof(tmp_path,case,3)
    def mutate(r,w):
        rows=w['readback']['responses']
        if fault=='last_scale':rows[-1]['scale']=1.
        elif fault=='swap':rows[0],rows[-1]=rows[-1],rows[0]
        elif fault=='missing':rows.pop()
        elif fault=='payload':rows[-1]['payload_bytes']=1
        elif fault=='last_input':rows[-1]['input_digest']='0'*64
        else:rows[-1]['level']=True
    with pytest.raises(ValueError):proof(tmp_path,case,4,old,mutate)


def test_decryption_without_postprocessing_cannot_pass_ranked_gate(tmp_path,schedule):
    case=schedule.case(1,42);old=proof(tmp_path,case,3);old=proof(tmp_path,case,4,old)
    with pytest.raises(ValueError):proof(tmp_path,case,5,old,lambda r,w:w['readback']['metrics'].update(postprocessed_predictions_validated=False))
