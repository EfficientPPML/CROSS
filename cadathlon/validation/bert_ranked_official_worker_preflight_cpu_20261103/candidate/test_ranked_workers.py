"""Synthetic preparation only: no retained private files or real HE are read."""
from pathlib import Path
from dataclasses import replace
from contextlib import nullcontext
from types import SimpleNamespace as NS
import hashlib,json,os
import pytest
import ranked_worker as worker,ranked_execution as execution,ranked_case_gate as cases
import test_ranked_handlers as handlers
import test_ranked_stage_gate as stage_fixtures
import ranked_private_reference,ranked_readback
HERE=Path(__file__).resolve().parent


@pytest.fixture(scope='module')
def case_set():
    return cases.authenticate(HERE/'public_evidence/fixture',cases.FIXTURE_REPORT_SHA,HERE/'public_evidence/reference',cases.REFERENCE_REPORT_SHA)


@pytest.mark.parametrize('size',[1,2,3])
def test_actual_all_record_handler_worker_chain(tmp_path,monkeypatch,case_set,size):
    paths,state,refs=handlers.flow(tmp_path,monkeypatch,size=size)
    paths['predictions']=tmp_path/'predictions';paths['test_input']=tmp_path/'test_input'
    raw=paths['preprocessed_input'].read_bytes();paths['test_input'].write_bytes(raw);paths['test_input'].chmod(0o600)
    handlers.stages.client_key_generation(paths);handlers.stages.server_preprocess_model({k:v for k,v in paths.items() if k in execution.SERVER_FIELDS});paths['preprocessed_input'].unlink()
    old=case_set.case(size,42);name=execution.NAMES[size];digest=hashlib.sha256(raw).hexdigest()
    files=dict(old.input_files);files[f'datasets/{name}/intermediate/test_input.txt']=digest;files[f'io/{name}/intermediate/client_preprocessed_input']=digest
    case=replace(old,input_sha256=digest,input_files=tuple(files.items()),official_correct=old.count)
    events=[]
    def preprocess(p):events.append('preprocess');p['preprocessed_input'].write_bytes(raw);p['preprocessed_input'].chmod(0o600)
    def postprocess(p):
        events.append('postprocess');rows=[json.loads(s) for s in p['decrypted_results'].read_text().splitlines()]
        p['predictions'].write_text(''.join(str(ranked_readback.label(r['he_logits']))+'\n' for r in rows));p['predictions'].chmod(0o600)
    monkeypatch.setattr(execution,'adapter',lambda:NS(preprocess=preprocess,postprocess=postprocess))
    monkeypatch.setattr(worker,'guard',lambda stage,calls,paths:nullcontext())
    monkeypatch.setattr(ranked_private_reference,'load',lambda *a,**kw:NS(records=lambda:refs))
    # Readback's private-directory contract requires the new synthetic parent
    #to have client-only mode, exactly as the real worker's umask077 does.
    tmp_path.chmod(0o700);previous=None
    for stage in (3,4,5):
        report={};request={'stage':stage}
        result=worker.execute(request,case_set,case,paths,{'context_id':'a'*64},{'model_id':'b'*64},previous,report,[])
        assert report['actual_stage_completed'] and report['readback_complete']
        assert report['official_harness_timing']is False
        previous={'stage':stage,'readback':result}
    assert state.encrypts==8*case.count and state.computes==state.decrypts==case.count
    assert events==['preprocess','postprocess']
    assert result['metrics']['he_correct']==case.count and result['metrics']['final_ranked_acceptance']
    assert (paths['client_private']/'secret.npy').exists()


def test_upstream_preprocess_body_really_tokenizes_fresh_records(tmp_path,monkeypatch):
    import sys
    calls=[]
    class Tokenizer:
        @classmethod
        def from_pretrained(cls,*a,**kw):calls.append(('load',a,kw));return cls()
        def __call__(self,a,b,**kw):calls.append(('tokenize',a,b,kw));return dict(input_ids=[101]*128,token_type_ids=[0]*128,attention_mask=[1]*128)
    monkeypatch.setitem(sys.modules,'transformers',NS(AutoTokenizer=Tokenizer))
    p=dict(test_input=tmp_path/'fresh.jsonl',preprocessed_input=tmp_path/'new'/'tokens',batch_size=100)
    p['test_input'].write_text(''.join(json.dumps(dict(target_idx=i,sentence1='synthetic a',sentence2='synthetic b'))+'\n' for i in range(100)))
    execution.adapter().preprocess(p)
    assert len(calls)==101 and len(p['preprocessed_input'].read_text().splitlines())==100
    assert calls[0][2]['revision']==cases.REVISION


@pytest.mark.parametrize('fault',['existing','old_fixture','old_reference','alias'])
def test_fresh_attempt_cannot_claim_retained_or_existing_root(tmp_path,case_set,fault):
    case=case_set.case(1,42);root=tmp_path/'attempt'
    if fault=='existing':root.mkdir()
    elif fault in ('old_fixture','old_reference'):
        private=tmp_path/'keep';private.mkdir();(private/'sentinel').write_text('keep')
        w=cases.Workspace(private,private.stat().st_dev,private.stat().st_ino)
        case=replace(case,**{('fixture_workspace' if fault=='old_fixture' else 'reference_workspace'):w});root=private/'attempt'
    else:
        private=tmp_path/'keep';private.mkdir();alias=tmp_path/'alias';alias.symlink_to(private,target_is_directory=True)
        case=replace(case,fixture_workspace=cases.Workspace(private,private.stat().st_dev,private.stat().st_ino));root=alias/'attempt'
    with pytest.raises((ValueError,FileExistsError)):execution.initialize_attempt(root,case)
    if fault!='existing':assert not root.exists()


def test_attempt_case_and_identity_bound_before_setup(tmp_path,case_set):
    case=case_set.case(1,42);root=tmp_path/'attempt';sha=execution.initialize_attempt(root,case)
    body=execution.authenticate_attempt(root,sha,case);assert body['retained_key_mode']is False
    with pytest.raises(ValueError):execution.authenticate_attempt(root,sha,case_set.case(1,43))
    root.rename(tmp_path/'old');root.mkdir(mode=0o700);(root/'.ranked_attempt.json').write_bytes((tmp_path/'old/.ranked_attempt.json').read_bytes())
    with pytest.raises(ValueError):execution.authenticate_attempt(root,sha,case)


@pytest.mark.parametrize('field',['requests.json','completed.json'])
def test_fresh_encode_rejects_existing_ledger(tmp_path,case_set,field):
    root=tmp_path/'attempt';case=case_set.case(1,42);execution.initialize_attempt(root,case);p=execution.paths(root,case)
    p['client_private'].mkdir(parents=True);(p['client_private']/field).write_text('keep')
    with pytest.raises(FileExistsError):execution.require_fresh(p,3)
    assert (p['client_private']/field).read_text()=='keep'


def test_directory_constructor_failure_keeps_primary_and_quarantines_close(tmp_path,monkeypatch):
    p=tmp_path/'held';p.mkdir();primary=ValueError('identity');close=OSError('close')
    monkeypatch.setattr(execution.HeldDirectory,'check',lambda self:(_ for _ in ()).throw(primary))
    real=execution.os.close
    def broken(fd):real(fd);raise close
    monkeypatch.setattr(execution.os,'close',broken)
    with pytest.raises(ValueError) as seen:execution.HeldDirectory(p)
    assert seen.value is primary and execution.QUARANTINE[-1].uncertain is not None
    owner=execution.QUARANTINE[-1];owner.close() # no retry of uncertain integer


@pytest.mark.parametrize('stage',[3,4,5])
def test_bad_public_prerequisite_never_reads_payload_or_calls_setup(monkeypatch,stage):
    request={n:None for n in worker.REQUEST_FIELDS};request.update(stage=stage)
    failure=ValueError('public prerequisite')
    monkeypatch.setattr(worker.cases,'authenticate',lambda *a:(_ for _ in ()).throw(failure))
    import ranked_setup
    monkeypatch.setattr(ranked_setup,'authenticate_keys',lambda *a,**kw:pytest.fail('setup touched before case gate'))
    monkeypatch.setattr(worker,'private_bytes',lambda *a:pytest.fail('private input touched'))
    with pytest.raises(ValueError) as got:worker.authenticate(request,{'x.py':'a'*64})
    assert got.value is failure


def test_existing_parent_241_source_bytes_unchanged():
    expected=json.loads((HERE/'parent_source_manifest.json').read_text())
    assert len(expected)==241
    assert {n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in expected}==expected


def test_all_seven_entrypoints_are_literal_original_bodies():
    import ranked_entrypoints as entry,file_stages
    assert len(entry.NAMES)==7
    assert entry.resolve('client_preprocess_input')is execution.adapter().preprocess
    assert entry.resolve('client_postprocess')is execution.adapter().postprocess
    for name in set(entry.NAMES)-{'client_preprocess_input','client_postprocess'}:
        assert entry.resolve(name)is getattr(file_stages,name)


def test_server_entrypoint_path_filter_includes_no_private_alias(monkeypatch):
    import ranked_entrypoints as entry
    seen=[];monkeypatch.setattr(entry,'resolve',lambda name:lambda paths:seen.append(paths))
    paths={n:Path('/synthetic')/n for n in execution.SERVER_FIELDS|{'client_private','test_input','preprocessed_input','decrypted_results','predictions'}}
    entry.invoke('server_encrypted_compute',paths)
    assert set(seen[0])==execution.SERVER_FIELDS
