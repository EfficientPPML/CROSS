"""Mocked server phase/order tests; no key, model or ciphertext payload reads."""
from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace as NS
import importlib.util,json,sys
import pytest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('group_test_worker',HERE/'group_worker.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)

@pytest.fixture
def case(tmp_path,monkeypatch):
    monkeypatch.setenv('JAX_PLATFORMS','cpu');events=[];failure={}
    paths={n:tmp_path/n for n in ('upload','download','server_report')};paths['upload'].mkdir()
    keyspace=tmp_path/'keys';keyspace.mkdir();modelspace=tmp_path/'model';modelspace.mkdir()
    keys=dict(context_id='c'*64,public_keys=keyspace/'public_keys',public_manifest={},key_report_sha256='1'*64,key_source_archive_sha256='2'*64)
    model=dict(model_id='m'*64,context_id='c'*64,server_data=modelspace,model_report_sha256='3'*64,model_manifest_sha256='4'*64)
    inputs=dict(upload=paths['upload'],stage3_report_sha256='5'*64,public_batch_sha256='b'*64,input_digest='d'*64,tiles=[dict(tile=i) for i in range(8)])
    request=dict(sample_count=1,tile_count=8,public_batch_sha256='b'*64,input_digest='d'*64,tiles=inputs['tiles'],batch={})
    response=dict(sample_count=1,response_count=1,level=2,public_batch_sha256='b'*64,input_digest='d'*64)
    expected={'worker.py':'a'*64};source=tmp_path/'source.json';source.write_text(json.dumps(expected))
    monkeypatch.setattr(w.protocol,'hashes',lambda x:expected)
    monkeypatch.setattr(w.protocol,'sources',lambda:{})
    monkeypatch.setattr(w.protocol,'environment',lambda:{})
    monkeypatch.setattr(w.protocol,'plan',lambda *a:{})
    monkeypatch.setattr(w.protocol,'imported_sources',lambda x:{})
    monkeypatch.setattr(w.protocol,'check_sources',lambda:events.append('dependencies') or {'ready':True})
    def gate(label,result):
        def fn(*a,**kw):
            events.append(label)
            if failure.get(label):raise failure[label]
            return result
        return fn
    monkeypatch.setattr(w.key_gate,'authenticate',gate('keys',keys));monkeypatch.setattr(w.stage4_worker.model_gate,'authenticate',gate('model',model))
    monkeypatch.setattr(w.stage4_worker.stage3_gate,'authenticate',gate('stage3',inputs))
    def handler(p):
        events.append('handler');assert not set(p)&{'client_private','preprocessed_input','test_input','decrypted_results'}
        if failure.get('handler'):raise failure['handler']
        paths['download'].mkdir();paths['server_report'].write_text('{}')
        return dict(sample_count=1,public_batch_sha256='b'*64)
    fake=NS(MODEL_ID='fixed',REVISION='pinned',server_encrypted_compute=handler)
    monkeypatch.setitem(sys.modules,'file_stages',fake)
    monkeypatch.setitem(sys.modules,'key_contexts',NS());monkeypatch.setitem(sys.modules,'full_model_engine',NS())
    @contextmanager
    def guarded(calls,**kw):
        events.append('guard-enter');assert kw['private_paths']==(keyspace/'client_private',)
        try:yield
        finally:events.append('guard-exit')
    monkeypatch.setattr(w.parity_boundaries,'rotation_only',guarded)
    monkeypatch.setattr(w.server_readback,'requests',gate('request',request))
    def actual(request,p,profile,plan,report,output,expected):
        report['actual_rotations_started']=True;handler(p)
        report.update(all_rotations_completed=True,loader_cleanup_completed=True,compiled_owner_cleanup_completed=True)
    monkeypatch.setattr(w,'actual',actual)
    args=dict(key_evidence=tmp_path/'ke',key_report_sha256='1'*64,key_workspace=keyspace,model_evidence=tmp_path/'me',
        model_report_sha256='3'*64,model_workspace=modelspace,stage3_evidence=tmp_path/'se',stage3_report_sha256='5'*64,
        io_workspace=tmp_path,output=tmp_path/'worker_report.json',source_manifest=source)
    return NS(args=args,events=events,failure=failure,keys=keys,model=model,inputs=inputs,request=request,response=response,paths=paths)

def read(c):return json.loads(c.args['output'].read_text())

def test_exact_public_phase_order_no_private_paths(case):
    assert w.run(**case.args)==0
    assert case.events==['dependencies','keys','model','stage3','guard-enter','request','handler','request','guard-exit','keys','model','stage3']
    r=read(case);assert r['passed'] and r['all_rotations_completed'] and r['upload_payloads_unchanged']
    assert case.paths['download'].is_dir()

@pytest.mark.parametrize('gate',['keys','model','stage3'])
def test_prerequisite_failure_precedes_upload_and_handler(case,gate):
    case.failure[gate]=ValueError('prerequisite')
    with pytest.raises(ValueError,match='prerequisite'):w.run(**case.args)
    assert 'guard-enter' not in case.events and 'request' not in case.events and 'handler' not in case.events
    assert not read(case)['actual_rotations_started']

def test_pending_overlay_precedes_all_prerequisites(case,monkeypatch):
    monkeypatch.setattr(w.protocol,'check_sources',lambda:(_ for _ in ()).throw(RuntimeError('pending review')))
    with pytest.raises(RuntimeError,match='pending review'):w.run(**case.args)
    assert not case.events and not read(case)['passed']

def test_request_failure_never_enters_server(case):
    case.failure['request']=ValueError('bad eighth ciphertext')
    with pytest.raises(ValueError,match='eighth'):w.run(**case.args)
    assert 'handler' not in case.events and case.events[-1]=='guard-exit' and not read(case)['actual_rotations_started']

def test_handler_cancellation_preserves_upload_and_restores_guard(case):
    error=KeyboardInterrupt('cancel');case.failure['handler']=error
    with pytest.raises(KeyboardInterrupt) as caught:w.run(**case.args)
    assert caught.value is error and case.events[-1]=='guard-exit' and case.paths['upload'].is_dir()
    assert read(case)['actual_rotations_started'] and not read(case)['passed']

def test_existing_sibling_paths_do_not_block_borrowed_input(case):
    case.paths['server_report'].write_text('other job')
    assert w.run(**case.args)==0
    assert case.paths['upload'].is_dir()

def test_wrong_source_rejected_before_dependencies(case,monkeypatch):
    monkeypatch.setattr(w.protocol,'hashes',lambda x:{})
    with pytest.raises(ValueError,match='source'):w.run(**case.args)
    assert not case.events and not read(case)['source_files_unchanged']

def test_terminal_write_preserves_primary(case,monkeypatch):
    error=KeyboardInterrupt('primary');case.failure['handler']=error;original=w.write
    def write(path,record):
        if 'source_hashes_end' in record:raise OSError('final report')
        original(path,record)
    monkeypatch.setattr(w,'write',write)
    with pytest.raises(KeyboardInterrupt) as caught:w.run(**case.args)
    assert caught.value is error and any('evidence' in n for n in error.__notes__)

def test_initial_write_failure_retries_failed_report(case,monkeypatch):
    original=w.write;calls=[]
    def write(path,record):
        calls.append(1)
        if len(calls)==1:raise OSError('first report')
        original(path,record)
    monkeypatch.setattr(w,'write',write)
    with pytest.raises(OSError,match='first report'):w.run(**case.args)
    assert not case.events and not read(case)['passed'] and len(calls)>=2

@pytest.mark.parametrize('failure_type',[OSError,KeyboardInterrupt,SystemExit])
def test_cleanup_scope_exit_republishes_failed_terminal_record(case,monkeypatch,failure_type):
    error=failure_type('shield exit')
    @contextmanager
    def shield():
        yield
        raise error
    monkeypatch.setattr(w,'cleanup_signals',shield)
    with pytest.raises(failure_type) as caught:w.run(**case.args)
    assert caught.value is error
    r=read(case);assert not r['passed'] and not r['terminal_scope_completed'] and r['terminal_failure_republished']

def test_primary_body_cancellation_survives_scope_exit_failure(case,monkeypatch):
    primary=KeyboardInterrupt('body');case.failure['handler']=primary
    @contextmanager
    def shield():
        try:yield
        finally:raise OSError('scope exit')
    monkeypatch.setattr(w,'cleanup_signals',shield)
    with pytest.raises(KeyboardInterrupt) as caught:w.run(**case.args)
    assert caught.value is primary and not read(case)['passed'] and not read(case)['terminal_scope_completed']
