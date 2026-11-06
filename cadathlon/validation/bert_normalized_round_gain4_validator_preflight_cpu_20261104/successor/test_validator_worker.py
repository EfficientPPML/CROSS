"""Validator-only source/ordering/lifetime tests with explicit synthetic services."""
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
import copy,json,sys
import pytest
import validator_worker as w
import validator_common as c
import collect_validator as collector
from test_component_worker import config


def runtime(monkeypatch,events,failure=None):
    expected={'stub.py':'ab'*32}
    monkeypatch.setenv('JAX_PLATFORMS','cpu')
    monkeypatch.setattr(c,'validate_config',lambda v:v)
    monkeypatch.setattr(c,'separate',lambda *a:None)
    monkeypatch.setattr(c,'require_ready',lambda:events.append('authorization'))
    monkeypatch.setattr(c,'inspect_sources',lambda e:(b'profile',b'plan',{}))
    monkeypatch.setattr(w.protocol,'sources',lambda:{})
    monkeypatch.setattr(w.protocol,'hashes',lambda x:expected)
    monkeypatch.setattr(w.protocol,'imported_sources',lambda e:{})
    monkeypatch.setattr(w.protocol,'check_identity',lambda *a:None)
    def gates(cfg):
        events.append('producer_gates')
        if failure=='gate':raise ValueError('producer rejected')
        return {'fixture':{},'server':{}}
    monkeypatch.setattr(c,'prior_gates',gates)
    def keys(*a,**k):events.append('public_keys');return {'context_id':'context'}
    monkeypatch.setattr(w.protocol,'authenticate_keys',keys)
    @contextmanager
    def role(name,**kw):
        assert name=='validate';events.append('boundary')
        yield
        events.append('restored')
    module=ModuleType('component_boundaries');module.role_scope=role;monkeypatch.setitem(sys.modules,'component_boundaries',module)
    def body(cfg,keys,p,plan,report,publish,receipts):
        events.append('validate')
        if failure=='body':raise KeyboardInterrupt('entered body')
        report['loader_cleanup_completed']=True
    monkeypatch.setattr(w.original,'validate',body)
    for name in ('fixture','server','run','main'):
        monkeypatch.setattr(w.original,name,lambda *a,**k:pytest.fail('legacy dispatch must not run'))
    return expected


def test_only_validate_after_old_producer_gate(tmp_path,monkeypatch):
    tmp_path.chmod(0o700);events=[];expected=runtime(monkeypatch,events)
    r=w.run(config('validate'),expected,tmp_path/'report.json')
    assert r['passed']and r['terminal_scope_completed']and r['phase_complete']
    assert events.index('producer_gates')<events.index('public_keys')<events.index('boundary')<events.index('validate')<events.index('restored')
    assert events.count('producer_gates')==2


@pytest.mark.parametrize('failure',['gate','body'])
def test_failure_preserved_and_no_early_private_work(tmp_path,monkeypatch,failure):
    tmp_path.chmod(0o700);events=[];expected=runtime(monkeypatch,events,failure)
    with pytest.raises(ValueError if failure=='gate'else KeyboardInterrupt):w.run(config('validate'),expected,tmp_path/'report.json')
    r=json.loads((tmp_path/'report.json').read_bytes());assert r['passed']is False and r['terminal_scope_completed']is False
    if failure=='gate':assert 'public_keys'not in events and'validate'not in events


@pytest.mark.parametrize('stage',['fixture','server',None,True,1])
def test_nonvalidator_rejected_before_publication(tmp_path,monkeypatch,stage):
    cfg=config('validate');cfg['stage']=stage
    monkeypatch.setattr(w.protocol,'check_identity',lambda *a:None)
    with pytest.raises(ValueError):w.run(cfg,{},tmp_path/'report.json')
    assert not (tmp_path/'report.json').exists()


def test_pending_rejects_before_output_or_payload(tmp_path,monkeypatch):
    cfg=config('validate');monkeypatch.setattr(w.protocol,'check_identity',lambda *a:None)
    monkeypatch.setattr(c,'inspect_sources',lambda *a:pytest.fail('consumer work after disabled flag'))
    with pytest.raises(ValueError,match='disabled'):w.run(cfg,{},tmp_path/'report.json')
    assert not (tmp_path/'report.json').exists()


@pytest.mark.parametrize('field',['workspace','key_workspace','key_evidence','fixture_evidence','server_evidence'])
def test_standalone_output_cannot_mutate_borrowed_root(tmp_path,monkeypatch,field):
    cfg=config('validate');cfg[field]=str(tmp_path/'borrowed');Path(cfg[field]).mkdir()
    monkeypatch.setattr(c,'validate_config',lambda v:v)
    with pytest.raises(ValueError,match='separate'):w.run(cfg,{},Path(cfg[field])/'report.json')
    assert not list(Path(cfg[field]).iterdir())


@pytest.mark.parametrize('module',[w,collector])
@pytest.mark.parametrize('kind',[None,RuntimeError,KeyboardInterrupt,SystemExit])
def test_outer_scope_restoration_controls_final_success(tmp_path,monkeypatch,module,kind):
    tmp_path.chmod(0o700);path=tmp_path/('worker.json'if module is w else'report.json')
    primary=None if kind is None else kind('restore')
    @contextmanager
    def scope():
        yield
        assert json.loads(path.read_bytes())['termination_scope_completed']is False
        if primary is not None:raise primary
    monkeypatch.setattr(module,'termination_signals',scope)
    def body(*args,**kw):
        r=dict(passed=True,terminal_scope_completed=True,termination_scope_completed=False)
        if module is w:kw['publication'].begin(path,r);return r
        kw['_publication_owner'].claim(tmp_path);module.protocol.write(path,r);return 0
    if module is w:
        source=tmp_path/'sources.json';source.write_text('{}');cfg=tmp_path/'config.json';cfg.write_text('{}')
        monkeypatch.setattr(c,'require_ready',lambda:None);monkeypatch.setattr(c,'validate_config',lambda x:x);monkeypatch.setattr(c,'protected',lambda x:[])
        monkeypatch.setattr(w,'run',body)
        args=['worker','--config',str(cfg),'--config-sha256',w.protocol.sha(cfg.read_bytes()),'--source-manifest',str(source),'--output',str(path),'--execute']
    else:
        monkeypatch.setattr(collector,'collect',body)
        args=['collector','--output',str(tmp_path),'--key-evidence','/public/proof','--key-workspace','/retained/keys']
    monkeypatch.setattr(sys,'argv',args)
    if primary is None:assert module.main()==0
    else:
        with pytest.raises(kind)as seen:module.main()
        assert seen.value is primary
    r=json.loads(path.read_bytes());assert r['passed']is(kind is None)and r['termination_scope_completed']is(kind is None)


def test_collector_output_collision_is_unchanged(tmp_path,monkeypatch):
    out=tmp_path/'existing';out.mkdir(mode=0o700);path=out/'report.json';raw=b'{"foreign":true}';path.write_bytes(raw)
    monkeypatch.setattr(sys,'argv',['collector','--output',str(out),'--key-evidence',str(tmp_path/'proof'),'--key-workspace',str(tmp_path/'keys')])
    with pytest.raises(FileExistsError):collector.main()
    assert path.read_bytes()==raw


def test_disabled_collector_has_no_output_side_effect(tmp_path):
    out=tmp_path/'proof'
    with pytest.raises(ValueError,match='disabled'):
        collector.collect(output=out,key_evidence=tmp_path/'keys-proof',key_workspace=tmp_path/'keys',execute=True)
    assert not out.exists()


def test_existing_worker_report_is_not_republished(tmp_path,monkeypatch):
    path=tmp_path/'report.json';raw=b'{"foreign":true}';path.write_bytes(raw)
    source=tmp_path/'sources.json';source.write_text('{}');cfg=tmp_path/'config.json';cfg.write_text('{}')
    monkeypatch.setattr(c,'require_ready',lambda:None);monkeypatch.setattr(c,'validate_config',lambda x:x);monkeypatch.setattr(c,'protected',lambda x:[])
    monkeypatch.setattr(sys,'argv',['worker','--config',str(cfg),'--config-sha256',w.protocol.sha(cfg.read_bytes()),'--source-manifest',str(source),'--output',str(path),'--execute'])
    with pytest.raises(FileExistsError):w.main()
    assert path.read_bytes()==raw
