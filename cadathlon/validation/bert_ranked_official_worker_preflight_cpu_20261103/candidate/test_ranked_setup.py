"""Fresh official setup paths with actual handlers and synthetic key/model bytes."""
from pathlib import Path
from dataclasses import replace
from contextlib import nullcontext
from types import SimpleNamespace as NS
import hashlib,io,json,tarfile,copy
import pytest
import ranked_worker as worker,ranked_setup as setup,ranked_execution as ex,ranked_case_gate as cases
import file_stages as stages,client_keys,server_public_model as model
HERE=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()


@pytest.fixture(params=[1,2,3])
def prepared(tmp_path,monkeypatch,request):
    cs=cases.authenticate(HERE/'public_evidence/fixture',cases.FIXTURE_REPORT_SHA,HERE/'public_evidence/reference',cases.REFERENCE_REPORT_SHA)
    case=cs.case(request.param,42);root=tmp_path/'staged';attempt=ex.initialize_attempt(root,case);paths=ex.paths(root,case)
    request={k:None for k in worker.REQUEST_FIELDS};request.update(stage=1,size=case.size,seed=42,staged_root=str(root),attempt_sha256=attempt,
        fixture_evidence=str(HERE/'public_evidence/fixture'),fixture_report_sha256=cases.FIXTURE_REPORT_SHA,
        reference_evidence=str(HERE/'public_evidence/reference'),reference_report_sha256=cases.REFERENCE_REPORT_SHA,predecessors=[])
    raw={'runtime/jaxite_word/key_gen.py':b'# synthetic key generator\n','fixture.py':b'# synthetic test only\n'};expected={n:sha(b) for n,b in raw.items()}
    inputs,_=stages._inputs();p,r=client_keys.public_contract(**inputs);specs=client_keys.inventory(p,r)
    arrays={n:dict(shape=list(shape),dtype='<u4',file_sha256='1'*64) for n,(shape,q) in specs.items()}
    body=client_keys._body(p,r,arrays,expected['runtime/jaxite_word/key_gen.py']);public=body|{'context_id':client_keys.digest(body)}
    events=[]
    monkeypatch.setattr(setup,'setup_only',lambda *a,**kw:nullcontext())
    monkeypatch.setattr(client_keys,'preflight',lambda *a,**kw:(p,r,{'synthetic_preflight':True}))
    def generate(public_path,private_path,**kw):
        assert public_path==root/'io'/ex.NAMES[case.size]/'public_keys' and private_path==root/'io'/ex.NAMES[case.size]/'client_private'
        events.append('fresh keygen');public_path.mkdir(mode=0o700);private_path.mkdir(mode=0o700)
        (public_path/'public_manifest.json').write_bytes(client_keys.canonical(public))
        private=dict(context_id=public['context_id'],public_manifest_sha256=sha(client_keys.canonical(public)))
        for n,b in [('private_manifest.json',client_keys.canonical(private)),('secret.npy',b'synthetic secret sentinel')]:
            (private_path/n).write_bytes(b);(private_path/n).chmod(0o600)
        return dict(public_manifest=public,preflight={'synthetic':True})
    monkeypatch.setattr(client_keys,'generate',generate);monkeypatch.setattr(client_keys,'authenticate_public',lambda *a,**kw:public)
    manifest={}
    def preprocess(snapshot,output,*,context_id,**kw):
        assert output==root/'io/server_data'/public['context_id'];events.append('fresh model');output.mkdir(mode=0o700)
        records={n:dict(file=f'tensor_{i:03d}.npy',shape=list(shape),dtype='<f4',tensor_sha256=model.TENSOR_SHA256[n],file_sha256='2'*64) for i,(n,shape) in enumerate(model.tensor_shapes().items())}
        body=model._base(context_id)|{'tensors':records};manifest.update(body|{'model_id':model.digest(body)})
        (output/'model_manifest.json').write_bytes(model.canonical(manifest));return dict(manifest)
    monkeypatch.setattr(model,'preprocess',preprocess);monkeypatch.setattr(stages,'_snapshot',lambda:tmp_path/'never-read')
    arrays={n:NS(dtype=NS(str='<f8'),flags=NS(writeable=False),shape=shape,ndim=len(shape)) for n,shape in model.tensor_shapes().items()}
    monkeypatch.setattr(model,'load',lambda *a,**kw:(arrays,dict(manifest)))
    return NS(request=request,case=case,cs=cs,root=root,paths=paths,public=public,raw=raw,expected=expected,events=events,manifest=manifest)


def proof(p,tmp_path,stage,readback,key_sha=None,mutate=None):
    out=tmp_path/f'evidence{stage}';(out/'worker').mkdir(parents=True)
    request=p.request|{'stage':stage,'key_report_sha256':key_sha};bindings=setup.binding(request,p.case)
    with tarfile.open(out/'sources.tar.gz','w:gz') as t:
        for n,b in p.raw.items():m=tarfile.TarInfo(n);m.size=len(b);t.addfile(m,io.BytesIO(b))
    w=dict(passed=True,stage=stage,actual_stage_started=True,actual_stage_completed=True,readback_complete=True,source_files_unchanged=True,
        cleanup_completed=True,terminal_scope_completed=True,termination_scope_completed=True,phases=setup.phases(stage),bindings=bindings,readback=readback,
        source_hashes_start=p.expected,source_hashes_end=p.expected)
    report=dict(schema='cross-bert-ranked-setup-proof-v1',stage=stage,passed=True,execute=True,actual_execution_attempted=True,operation_complete=True,
        all_processes_reaped=True,source_files_unchanged=True,source_archive_verified=True,cleanup_completed=True,terminal_scope_completed=True,termination_scope_completed=True,
        owned_artifacts_retained=True,source_hashes_start=p.expected,source_hashes_end=p.expected,execution_source_hashes_end=p.expected,
        source_archive_sha256=sha((out/'sources.tar.gz').read_bytes()),bindings=bindings,readback=readback)
    if mutate:mutate(report,w)
    raw=client_keys.canonical(w);(out/'worker/worker_report.json').write_bytes(raw)
    report['worker_process']=dict(returncode=0,process_reaped=True,public_key_store_removed=True,owned_directory_handle_closed=True,result=dict(passed=True,worker_report_sha256=sha(raw)))
    raw=client_keys.canonical(report);(out/'report.json').write_bytes(raw);return out,sha(raw),request


def execute1(p):
    report={};worker.execute({'stage':1},p.cs,p.case,p.paths,None,None,None,report,[]);return report['readback']


def authenticate1(p,evidence,h,request):
    inputs,_=stages._inputs()
    return setup.authenticate_keys(evidence,h,request=request,case=p.case,paths=p.paths,expected_sources=p.expected,**inputs)


def test_setup_actual_handlers_use_fresh_official_layout_and_proof_chain(prepared,tmp_path):
    p=prepared;r1=execute1(p);e1,h1,request=proof(p,tmp_path,1,r1);keys=authenticate1(p,e1,h1,request)
    report={};worker.execute({'stage':2},p.cs,p.case,p.paths,keys,None,None,report,[])
    e2,h2,request=proof(p,tmp_path,2,report['readback'],h1)
    actual=setup.authenticate_model(e2,h2,request=request,case=p.case,paths=p.paths,keys=keys,expected_sources=p.expected)
    assert actual['server_data']==p.root/'io/server_data' and actual['context_id']==keys['context_id']
    assert p.events==['fresh keygen','fresh model'] and report['readback']['matrix_count']==74
    assert (p.paths['client_private']/'secret.npy').read_bytes()==b'synthetic secret sentinel'


@pytest.mark.parametrize('fault',['phase','source','outer_terminal','worker_terminal','attempt','private_attestation'])
def test_setup_public_proof_failures_precede_payload_reads(prepared,tmp_path,fault):
    p=prepared;r=execute1(p)
    def mutate(outer,w):
        if fault=='phase':w['phases']['decryption']=True
        elif fault=='source':w['source_hashes_end']={}
        elif fault=='outer_terminal':outer['terminal_scope_completed']=False
        elif fault=='worker_terminal':w['terminal_scope_completed']=False
        elif fault=='attempt':w['bindings']=dict(w['bindings'],attempt_sha256='0'*64)
        else:r['private_binding_verified']=False
    e,h,request=proof(p,tmp_path,1,r,mutate=mutate)
    with pytest.raises(ValueError):authenticate1(p,e,h,request)


def test_later_private_ledger_does_not_retroactively_require_count2(prepared,tmp_path):
    p=prepared;r=execute1(p);e,h,request=proof(p,tmp_path,1,r)
    (p.paths['client_private']/'requests.json').write_text('synthetic later ledger')
    assert authenticate1(p,e,h,request)['context_id']==p.public['context_id']


@pytest.mark.parametrize('stage',[1,2])
def test_no_retained_key_or_model_fallback(prepared,stage):
    p=prepared
    if stage==1:p.paths['public_keys'].mkdir(parents=True)
    else:p.paths['server_data'].mkdir(parents=True)
    with pytest.raises(FileExistsError):worker.execute({'stage':stage},p.cs,p.case,p.paths,None,None,None,{},[])


def test_bad_setup_case_gate_never_generates(prepared,monkeypatch):
    p=prepared;p.request['key_evidence']='retained-old';p.request['key_report_sha256']='1'*64
    with pytest.raises(ValueError,match='fallback'):worker.authenticate(p.request,p.expected)
    assert not p.events
