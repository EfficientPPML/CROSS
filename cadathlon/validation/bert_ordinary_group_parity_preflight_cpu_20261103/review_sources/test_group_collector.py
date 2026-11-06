"""Mocked subprocess/retention transactions, no encrypted work."""
from pathlib import Path
from types import SimpleNamespace as NS
import importlib.util,json,sys
import pytest
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('group_test_collect',HERE/'group_collect.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def public_plan():
    import group_protocol as p
    rows=[]
    for level in p.LEVELS:
        q=2*(level+1);parts=(q+12)//13
        for name,indices in p.GROUPS:
            rows.append(dict(level=level,group=name,requests=len(indices),indices=list(indices),
                digit_bytes=parts*131072*(q+14)*4,result_bytes=sum(i%65536!=0 for i in indices)*2*131072*q*4))
    return dict(rows=rows)
def arithmetic():
    comparisons=[];groups=[]
    for planned in public_plan()['rows']:
        level=planned['level'];name=planned['group']
        for sequence,index in enumerate(planned['indices']):
            comparisons.append(dict(level=level,group=name,index=index,normalized_index=index%65536,sequence=sequence,
                zero_alias=index%65536==0,passed=True,metadata_equal=True,mismatched_words=0,
                compared_words=2*131072*(level+1)*2,left={'hash':'same'},right={'hash':'same'}))
        groups.append(dict(level=level,group=name,stats=dict(mode='digits',requests=planned['requests'],decompositions=1,
            digit_bytes=planned['digit_bytes'],result_bytes=planned['result_bytes'])))
    return dict(passed=True,actual_rotation_calls=116,nonzero_pair_comparisons=58,zero_alias_comparisons=8,shared_decompositions=6,
        input_unchanged=True,level_reduction={'passed':True},comparisons=comparisons,groups=groups)

@pytest.fixture
def case(tmp_path,monkeypatch):
    monkeypatch.setenv('JAX_PLATFORMS','cpu');events=[];fail={};owners=[]
    dirs={n:tmp_path/n for n in ('keys','model','io','ke','me','se')}
    for p in dirs.values():p.mkdir()
    (dirs['io']/'upload').mkdir();(dirs['keys']/'private-sentinel').write_bytes(b'untouched')
    inputs=dict(upload=dirs['io']/'upload',stage3_report_sha256='5'*64,public_batch_sha256='b'*64,input_digest='d'*64,tiles=[dict(tile=i) for i in range(8)])
    auth=(dict(context_id='c'*64,public_keys=dirs['keys']/'public_keys',key_report_sha256='1'*64,key_source_archive_sha256='2'*64),
        dict(model_id='m'*64,model_report_sha256='3'*64,model_manifest_sha256='4'*64,server_data=dirs['model']),inputs)
    raw={'worker.py':b'# mock\n','marker.py':b'# unchanged\n','runtime/stage/server_inputs/p60_profile_and_refresh.json':b'{}',
         'runtime/stage/server_inputs/twelve_layer_plan.json':b'{}'}
    monkeypatch.setattr(c,'sources',lambda:dict(raw));monkeypatch.setattr(c.group_protocol,'check_sources',lambda **kw:{'ready':True})
    monkeypatch.setattr(c.group_protocol,'environment',lambda:{})
    monkeypatch.setattr(c.group_protocol,'plan',lambda *a:public_plan())
    monkeypatch.setattr(c.process,'software_versions',lambda:{'mock':True})
    monkeypatch.setattr(c.shutil,'disk_usage',lambda p:NS(free=20<<30))
    def authenticate(**kw):
        events.append('gates')
        if fail.get('gate'):raise fail['gate']
        return auth
    monkeypatch.setattr(c.worker,'authenticate',authenticate)
    monkeypatch.setitem(sys.modules,'file_stages',NS(MODEL_ID='fixed',REVISION='pinned'))
    original=c.BorrowedDirectory
    class Borrowed(original):
        def __init__(self,path):super().__init__(path);owners.append(self)
        def check(self):
            super().check()
            if fail.get('check'):raise fail['check']
        def close(self):
            events.append('close');super().close()
            if fail.get('close'):raise fail['close']
    monkeypatch.setattr(c,'BorrowedDirectory',Borrowed)
    request=dict(sample_count=1,tile_count=8,tiles=inputs['tiles'],public_batch_sha256='b'*64,input_digest='d'*64)
    response=arithmetic()
    def launch(command,directory,*,timeout,cleanup_grace_s):
        events.append('launch');assert timeout is None and cleanup_grace_s>=30
        assert Path(command[1]).parent.name=='execution_source'
        assert not any('private' in x or 'snapshot' in x for x in command[2::2])
        (dirs['io']/'download').mkdir();(dirs['io']/'server_report').write_text('{}')
        if fail.get('launch'):raise fail['launch']
        before=c.hashes(raw)
        proof=dict(passed=True,actual_rotations_started=True,all_rotations_completed=True,
            request_readback_complete=True,upload_payloads_unchanged=True,source_files_unchanged=True,compiled_owner_cleanup_completed=True,loader_cleanup_completed=True,
            keygen_codecs_native_private_model_forbidden=True,terminal_scope_completed=True,forbidden_attempts=[],source_hashes_start=before,source_hashes_end=before,
            prerequisite=c.worker.prerequisite_summary(*auth),reviewed_sources={'ready':True},public_plan=public_plan(),
            request_readback=request,arithmetic=response,elapsed_s=1.)
        proof.update(fail.get('proof',{}));c.write(directory/'worker_report.json',proof)
        record=dict(returncode=0,process_reaped=True,public_key_store_removed=True,owned_directory_handle_closed=True,
            result=dict(passed=True,context_id='c'*64,model_id='m'*64,public_batch_sha256='b'*64,
                worker_report_sha256=c.process.sha256(directory/'worker_report.json')))
        record.update(fail.get('process',{}));c.write(directory/'process.json',record)
        if fail.get('source'):raw['marker.py']=b'# changed\n'
        if fail.get('late_check'):fail['check']=fail['late_check']
        return record
    monkeypatch.setattr(c.process,'run_case',launch)
    args=dict(output=tmp_path/'evidence',key_evidence=dirs['ke'],key_report_sha256='1'*64,key_workspace=dirs['keys'],
        model_evidence=dirs['me'],model_report_sha256='3'*64,model_workspace=dirs['model'],stage3_evidence=dirs['se'],
        stage3_report_sha256='5'*64,io_workspace=dirs['io'],execute=True)
    return NS(args=args,dirs=dirs,events=events,fail=fail,owners=owners,raw=raw,request=request,response=response)

def report(case):return json.loads((case.args['output']/'report.json').read_text())

def test_success_retains_all_external_artifacts(case):
    r=c.run(**case.args)
    assert r['passed'] and r['actual_rotation_parity'] and r['all_processes_reaped'] and r['source_archive_verified']
    assert case.events==['gates','launch','gates','close'] and case.owners[0].fd is None
    assert (case.dirs['io']/'download').is_dir() and (case.dirs['keys']/'private-sentinel').read_bytes()==b'untouched'

def test_noexecute_never_launches_or_removes_external_upload(case):
    case.args['execute']=False;r=c.run(**case.args)
    assert r['passed'] and not r['actual_rotation_parity'] and not r['actual_execution_attempted']
    assert 'launch' not in case.events and case.owners[0].fd is None and (case.dirs['io']/'upload').is_dir()

@pytest.mark.parametrize('which',['keys','model','io','ke','me','se'])
@pytest.mark.parametrize('alias',[False,True])
def test_output_containment_before_mkdir(case,tmp_path,which,alias):
    base=case.dirs[which]
    if alias:
        target=tmp_path/'alias';target.symlink_to(base,target_is_directory=True);base=target
    case.args['output']=base/'must-not-exist'
    with pytest.raises(ValueError,match='outside'):c.run(**case.args)
    assert not case.args['output'].exists() and not case.events

def test_pending_dependencies_precede_gates(case,monkeypatch):
    monkeypatch.setattr(c.group_protocol,'check_sources',lambda **kw:(_ for _ in ()).throw(RuntimeError('pending cleanup')))
    with pytest.raises(RuntimeError,match='pending'):c.run(**case.args)
    assert not case.events and not case.owners and not report(case)['passed']

def test_failed_gate_precedes_borrowing_or_launch(case):
    case.fail['gate']=ValueError('stage3 pending')
    with pytest.raises(ValueError,match='pending'):c.run(**case.args)
    assert case.events==['gates'] and not case.owners and not report(case)['actual_execution_attempted']

@pytest.mark.parametrize('flag',['process_reaped','public_key_store_removed','owned_directory_handle_closed'])
def test_process_cleanup_failclosed_but_retains_outputs(case,flag):
    case.fail['process']={flag:False}
    with pytest.raises(ValueError,match='worker failed'):c.run(**case.args)
    r=report(case);assert not r['passed'] and not r['all_processes_reaped'] and r['io_workspace_retained']
    assert (case.dirs['io']/'download').is_dir() and case.owners[0].fd is None

def test_unknown_child_cancellation_primary_retained(case):
    error=KeyboardInterrupt('cancel');case.fail['launch']=error
    with pytest.raises(KeyboardInterrupt) as caught:c.run(**case.args)
    assert caught.value is error and any('active process evidence' in n for n in error.__notes__)
    r=report(case);assert not r['all_processes_reaped'] and r['io_workspace_retained'] and not r['passed']
    assert (case.dirs['io']/'download').is_dir() and case.owners[0].fd is None

def test_worker_source_binding_failure(case):
    case.fail['proof']={'source_hashes_end':{}}
    with pytest.raises(ValueError,match='worker/source'):c.run(**case.args)
    assert not report(case)['passed'] and case.owners[0].fd is None

def test_wrong_response_level_rejected(case):
    case.response['comparisons'][0]['mismatched_words']=1
    with pytest.raises(ValueError,match='rotation parity/binding'):c.run(**case.args)
    assert not report(case)['passed']

def test_source_mutation_cannot_publish_pass(case):
    case.fail['source']=True;r=c.run(**case.args)
    assert not r['passed'] and not r['source_files_unchanged'] and r['io_workspace_retained']

def test_initial_publication_failure_recovers_failed_report(case,monkeypatch):
    original=c.write;calls=[]
    def write(path,value):
        calls.append(1)
        if len(calls)==1:raise OSError('initial publication')
        original(path,value)
    monkeypatch.setattr(c,'write',write)
    with pytest.raises(OSError,match='initial publication'):c.run(**case.args)
    assert not report(case)['passed'] and not case.events and not case.owners

def test_final_publication_preserves_cancellation(case,monkeypatch):
    original=c.write;error=KeyboardInterrupt('primary');case.fail['launch']=error
    def write(path,value):
        if path.name=='report.json' and 'source_hashes_end' in value:raise OSError('publication')
        original(path,value)
    monkeypatch.setattr(c,'write',write)
    with pytest.raises(KeyboardInterrupt) as caught:c.run(**case.args)
    assert caught.value is error and any('terminal publication' in n for n in error.__notes__) and case.owners[0].fd is None

def test_cleanup_cancellation_does_not_skip_close(case):
    case.fail['late_check']=ValueError('replaced');error=KeyboardInterrupt('close cancel');case.fail['close']=error
    with pytest.raises(ValueError,match='replaced') as caught:c.run(**case.args)
    assert case.owners[0].fd is None and any('close cancel' in n for n in caught.value.__notes__)

def test_metadata_failure_closes_borrowed_dir_retains_external(case,monkeypatch):
    monkeypatch.setattr(c.process,'software_versions',lambda:(_ for _ in ()).throw(ValueError('metadata')))
    with pytest.raises(ValueError,match='metadata'):c.run(**case.args)
    assert case.owners[0].fd is None and (case.dirs['io']/'upload').is_dir() and not report(case)['passed']

def test_reviewed_dependency_source_closure():
    import group_protocol
    assert group_protocol.check_sources(allow_pending=True)['parent_source_count']==231
    sources=c.sources()
    assert c.hashes(sources)['runtime/stage/full_model_engine.py']=='a5384484269139f67db18ea90b3a99f2952265321d84bc04964433d9dede3f53'
    assert 'joined/encrypted_bert_full_model.py' in sources and 'parity_origin.json' in sources
    assert all(not x.endswith('.npz') for x in sources)

@pytest.mark.parametrize('failure_type',[OSError,KeyboardInterrupt,SystemExit])
def test_cleanup_scope_exit_invalidates_persisted_success(case,monkeypatch,failure_type):
    from contextlib import contextmanager
    error=failure_type('cleanup exit')
    @contextmanager
    def shield():
        yield
        raise error
    monkeypatch.setattr(c,'cleanup_signals',shield)
    with pytest.raises(failure_type) as caught:c.run(**case.args)
    assert caught.value is error
    r=report(case);assert not r['passed'] and not r['terminal_scope_completed'] and not r['cleanup_completed']
    assert r['terminal_failure_republished'] and case.owners[0].fd is None

def test_cleanup_scope_cancellation_wins_ordinary_cleanup_failure(case,monkeypatch):
    from contextlib import contextmanager
    case.fail['close']=OSError('ordinary close');error=KeyboardInterrupt('scope cancellation')
    @contextmanager
    def shield():
        try:yield
        finally:raise error
    monkeypatch.setattr(c,'cleanup_signals',shield)
    with pytest.raises(KeyboardInterrupt) as caught:c.run(**case.args)
    assert caught.value is error and not report(case)['passed'] and case.owners[0].fd is None

def test_worker_terminal_scope_flag_required(case):
    case.fail['proof']={'terminal_scope_completed':False}
    with pytest.raises(ValueError,match='worker/source/phase'):c.run(**case.args)
    assert not report(case)['passed']

def test_intentional_main_exit_is_outside_termination_scope():
    import ast
    for name in ('group_worker.py','group_collect.py'):
        tree=ast.parse((HERE/name).read_text());main=tree.body[-1];assert isinstance(main,ast.If)
        scopes=[node for node in main.body if isinstance(node,ast.With)]
        assert len(scopes)==1 and not any(isinstance(node,ast.Raise) for node in ast.walk(scopes[0]))
        assert isinstance(main.body[-1],ast.Raise)


def test_existing_sibling_server_outputs_are_allowed(case):
    (case.dirs['io']/'download').mkdir();(case.dirs['io']/'server_report').write_text('root-owned')
    case.args['execute']=False
    assert c.run(**case.args)['passed']
    assert (case.dirs['io']/'server_report').read_text()=='root-owned'

@pytest.mark.parametrize('field,value',[('actual_rotation_calls',23),('input_unchanged',False),('level_reduction',{'passed':False})])
def test_incomplete_parity_cannot_publish_success(case,field,value):
    case.response[field]=value
    with pytest.raises(ValueError,match='parity/binding'):c.run(**case.args)

@pytest.mark.parametrize('field,value',[('compared_words',1),('zero_alias',False),('right',{'hash':'wrong'})])
def test_allword_and_budget_evidence_required(case,field,value):
    case.response['comparisons'][0][field]=value
    with pytest.raises(ValueError,match='all-word'):c.run(**case.args)


def test_borrowed_constructor_preserves_primary_on_close_failure(tmp_path,monkeypatch):
    directory=tmp_path/'borrowed';directory.mkdir();primary=ValueError('replaced');secondary=OSError('close failed')
    real_close=c.os.close;held=[]
    original_open=c.os.open
    def opening(*a,**kw):
        fd=original_open(*a,**kw);held.append(fd);return fd
    monkeypatch.setattr(c.os,'open',opening)
    monkeypatch.setattr(c.BorrowedDirectory,'check',lambda self:(_ for _ in ()).throw(primary))
    monkeypatch.setattr(c.os,'close',lambda fd:(_ for _ in ()).throw(secondary))
    try:
        with pytest.raises(ValueError) as caught:c.BorrowedDirectory(directory)
        assert caught.value is primary and any('close failed' in n for n in primary.__notes__)
        owner=c._BORROWED_QUARANTINE[-1];assert owner.fd==held[0] and owner._close_failure is secondary
    finally:
        for fd in held:real_close(fd)
        c._BORROWED_QUARANTINE.clear()


def test_borrowed_close_failure_keeps_uncertain_ownership(tmp_path,monkeypatch):
    directory=tmp_path/'borrowed';directory.mkdir();owner=c.BorrowedDirectory(directory);fd=owner.fd
    real_close=c.os.close;failure=OSError('close failed')
    monkeypatch.setattr(c.os,'close',lambda n:(_ for _ in ()).throw(failure))
    try:
        with pytest.raises(OSError) as caught:owner.close()
        assert caught.value is failure and owner.fd==fd and owner in c._BORROWED_QUARANTINE
    finally:real_close(fd);c._BORROWED_QUARANTINE.clear()
