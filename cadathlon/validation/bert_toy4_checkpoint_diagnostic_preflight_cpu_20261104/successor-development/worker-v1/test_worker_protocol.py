from contextlib import contextmanager
from pathlib import Path
import copy,json,sys,tarfile,io
import numpy as np
import pytest
import component_worker as w
import component_protocol as p
import collect as collector
import finish as outer

@pytest.mark.parametrize('bad',[np.full(65536,np.nan),np.full(65536,np.inf),np.zeros(65535),np.zeros(65536,dtype=complex)])
def test_numeric_rejects_nonreal_nonfinite_wrongshape(bad):
    with pytest.raises(ValueError):w.numeric_errors(bad,np.zeros(65536))

def test_extreme_finite_metric_remains_failed_json():
    r=w.numeric_errors(np.full(65536,np.finfo(float).max),np.full(65536,-np.finfo(float).max))
    assert r['passed']is False and r['max_abs_error']is None
    json.dumps(r,allow_nan=False)

@pytest.mark.parametrize('field,value',[('group_empty_verified',False),('direct_child_reaped',False),('cleanup_completed',False),('direct_returncode',True)])
def test_prior_role_gate_rejects_incomplete_child_before_worker_read(tmp_path,monkeypatch,field,value):
    workspace=tmp_path/'owned';workspace.mkdir(mode=0o700);evidence=tmp_path/'report.json'
    r=dict(stage='server',passed=True,complete=True,source_verified=True,worker_verified=True,cleanup_completed=True,terminal_scope_completed=True,termination_scope_completed=True,
        child=dict(group_empty_verified=True,direct_child_reaped=True,cleanup_completed=True,direct_returncode=0),workspace=str(workspace),workspace_identity=p.identity(workspace),source_hashes={'x':'a'*64})
    r['child'][field]=value;evidence.write_bytes(p.canonical(r));original=p.read;opened=[]
    def read(path,*a):opened.append(Path(path));return original(path,*a)
    monkeypatch.setattr(p,'read',read)
    with pytest.raises(ValueError,match='direct wait'):p.terminal_gate(evidence,p.sha(evidence.read_bytes()),stage='server',expected_sources=r['source_hashes'],workspace=workspace,workspace_identity=r['workspace_identity'])
    assert opened==[evidence]

def test_phase_gate_before_device_or_private(tmp_path,monkeypatch):
    workspace=tmp_path/'owned';workspace.mkdir(mode=0o700);out=tmp_path/'evidence'
    from publication import OwnedOutput
    owner=OwnedOutput(out).acquire()
    config=dict(stage='validate',workspace=str(workspace),workspace_identity=p.identity(workspace),nonce='b'*32,
        setup_receipt=str(tmp_path/'setup.json'),setup_sha256='a'*64,server_receipt=str(tmp_path/'server.json'),server_sha256='a'*64,server_root_review=str(tmp_path/'review.json'),server_root_review_sha256='a'*64)
    events=[];monkeypatch.setattr(p,'check_sources',lambda x:None);monkeypatch.setattr(p,'sources',lambda:{'x':b'x'});expected=p.hashes(p.sources())
    monkeypatch.setattr(p,'imported_sources',lambda expected:{});monkeypatch.setattr(w,'device',lambda:events.append('device'))
    with pytest.raises(FileNotFoundError):w.run(config,expected,out,publication=owner)
    assert events==[] and json.loads((out/'report.json').read_bytes())['passed']is False

@pytest.mark.parametrize('module',[collector,w])
def test_output_collision_preserves_old_receipt(tmp_path,monkeypatch,module):
    out=tmp_path/'old';out.mkdir();old=b'{"old":true}\n';(out/'report.json').write_bytes(old)
    source=tmp_path/'sources';source.write_text('{}');monkeypatch.setattr(module,'_BOOTSTRAP',({},{}))
    args=['command','--output',str(out),'--source-manifest',str(source),'--source-manifest-sha256',p.sha(source.read_bytes())]
    if module is w:
        config=tmp_path/'config';config.write_text('{}');source=tmp_path/'sources';source.write_text('{}')
        monkeypatch.setattr(p,'require_ready',lambda:None);monkeypatch.setattr(w,'config_check',lambda c:c)
        monkeypatch.setattr(w,'separate',lambda *a:None)
        args+=['--execute','--config',str(config),'--config-sha256',p.sha(config.read_bytes()),'--source-manifest',str(source)]
    monkeypatch.setattr(sys,'argv',args)
    with pytest.raises(FileExistsError):module.main()
    assert(out/'report.json').read_bytes()==old

def test_relative_collector_output_is_absolute_before_child_work(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);seen=[]
    def body(out,execute,parent,*,publication,**kwargs):
        assert out==tmp_path/'relative'and out.is_absolute()and parent.is_absolute();seen.append(out)
        report=dict(passed=True,terminal_scope_completed=True);publication.write(out/'report.json',report);return report
    monkeypatch.setattr(collector,'collect',body);monkeypatch.setattr(collector,'_BOOTSTRAP',({},{}))
    source=tmp_path/'sources';source.write_text('{}')
    monkeypatch.setattr(sys,'argv',['collector','--output','relative','--workspace-parent','.', '--source-manifest',str(source),'--source-manifest-sha256',p.sha(source.read_bytes())])
    assert collector.main()==0 and seen==[tmp_path/'relative']

@pytest.mark.parametrize('error',[RuntimeError('restore'),KeyboardInterrupt('restore'),SystemExit('restore')])
def test_failure_after_outer_scope_republishes_false(tmp_path,error):
    path=tmp_path/'report.json'
    @contextmanager
    def scope():yield;raise error
    def body():
        r=dict(passed=True,terminal_scope_completed=True,termination_scope_completed=False);p.write(path,r);return r
    with pytest.raises(type(error))as caught:outer.finish(path,body,p.write,p.read,termination=scope)
    assert caught.value is error
    r=json.loads(path.read_bytes());assert r['passed']is False and r['termination_scope_completed']is False

@pytest.mark.parametrize('mutation',['duplicate','unknown','oversize'])
def test_source_archive_fail_closed(mutation):
    b=io.BytesIO()
    with tarfile.open(fileobj=b,mode='w:gz')as tar:
        for name in(['x','x']if mutation=='duplicate'else['wrong']if mutation=='unknown'else['x']):
            info=tarfile.TarInfo(name);raw=b'x'*(8*1024*1024+1 if mutation=='oversize'else 1);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
    with pytest.raises(ValueError):p.verify_archive(b.getvalue(),{'x':p.sha(b'x')})

def test_disabled_execution_authorization():
    with pytest.raises(ValueError,match='unauthorized'):p.require_ready()

@pytest.mark.parametrize('primary',[None,ValueError('body')])
@pytest.mark.parametrize('cancel',[KeyboardInterrupt,SystemExit])
def test_finalizer_attempt_all_and_primary_precedence(primary,cancel):
    f=p.Finalizer(primary);events=[];later=cancel('cleanup cancellation')
    def ordinary():events.append('ordinary');raise OSError('first')
    def cancellation():events.append('cancel');raise later
    f.attempt('first',ordinary);f.attempt('second',cancellation);f.attempt('last',lambda:events.append('last'))
    chosen=primary if primary is not None else later
    with pytest.raises(type(chosen))as caught:f.raise_failure()
    assert caught.value is chosen and events==['ordinary','cancel','last']

def test_public_weight_export_readback_immutable(tmp_path):
    import fixture_assets as f
    result,references=f.prepare_client_reference();assert len(references)==96
    path=tmp_path/'public';f.export_public_weights(path,result['weights']);loaded=f.load_public_weights(path)
    assert set(loaded)==set(result['weights'])
    for name,value in loaded.items():
        assert np.array_equal(value,result['weights'][name])and not value.flags.writeable
        with pytest.raises(ValueError):value.setflags(write=True)

def test_resource_secret_is_actual_rns_uint32(tmp_path):
    import resource_plan
    result=resource_plan.report(tmp_path)
    assert result['keys']['private_array_bytes']==100*131072*4==52428800
    assert result['keys']['private_required_bytes']==53481472
    assert result['keys']['rotation_count']==199 and result['keys']['public_array_count']==401
    assert result['workspace_required_bytes']>result['keys']['public_required_bytes']+3758096384
