from contextlib import contextmanager,nullcontext
import json
from types import SimpleNamespace as NS
import pytest
import ranked_terminal as terminal,ranked_worker as worker


def write(p,r):p.write_text(json.dumps(r)+'\n')


def test_termination_restoration_failure_rewrites_prior_success(tmp_path):
    path=tmp_path/'report.json';r=dict(passed=True,terminal_scope_completed=True,cleanup_completed=True)
    @contextmanager
    def termination():
        yield
        raise OSError('outer restore')
    def call():write(path,r);return r
    with pytest.raises(OSError):terminal.finish(path,call,write,termination=termination,shield=nullcontext)
    report=json.loads(path.read_text());assert not report['passed'] and not report['termination_scope_completed'] and not report['cleanup_completed']


@pytest.mark.parametrize('error',[KeyboardInterrupt('private body'),SystemExit(143),ValueError('private body')])
def test_primary_identity_and_redaction_with_publication_error(tmp_path,error):
    path=tmp_path/'report.json'
    def call():raise error
    def broken(p,r):raise OSError('publication')
    with pytest.raises(type(error)) as got:terminal.finish(path,call,broken,termination=nullcontext,shield=nullcontext)
    assert got.value is error
    assert any('republication' in s for s in error.__notes__)


def test_outer_completion_written_only_after_scope(tmp_path):
    path=tmp_path/'report.json';events=[]
    @contextmanager
    def termination():
        events.append('enter');yield;events.append('exit')
    def output(p,r):assert events==['enter','body','exit'];write(p,r)
    def call():events.append('body');return {'passed':True}
    assert terminal.finish(path,call,output,termination=termination)['termination_scope_completed']


def test_virtual_torch_module_paths_are_not_misread_as_local_sources(monkeypatch):
    monkeypatch.setitem(worker.sys.modules,'synthetic_torch_virtual',NS(__file__='_ops.py'))
    # Supply exact hashes of genuinely loaded local sources, not fabricated
    #placeholders. The virtual relative name must not become a path read.
    expected={}
    for m in tuple(worker.sys.modules.values()):
        f=getattr(m,'__file__',None)
        if f and worker.Path(f).is_absolute():
            p=worker.Path(f).resolve()
            if worker.HERE in p.parents and p.suffix=='.py':expected[p.relative_to(worker.HERE).as_posix()]=worker.key_gate.digest(p.read_bytes())
    assert 'synthetic_torch_virtual' not in worker.local_modules(expected)


@pytest.mark.parametrize('primary',[KeyboardInterrupt('write body'),SystemExit(143)])
def test_initializer_write_primary_survives_uncertain_close(tmp_path,monkeypatch,primary):
    import ranked_execution as ex,ranked_case_gate as cases
    h=worker.HERE
    case=cases.authenticate(h/'public_evidence/fixture',cases.FIXTURE_REPORT_SHA,h/'public_evidence/reference',cases.REFERENCE_REPORT_SHA).case(1,42)
    monkeypatch.setattr(ex.os,'write',lambda *args:(_ for _ in ()).throw(primary))
    original=ex.os.close
    def close(fd):original(fd);raise OSError('close')
    monkeypatch.setattr(ex.os,'close',close)
    with pytest.raises(type(primary)) as got:ex.initialize_attempt(tmp_path/'owned',case)
    assert got.value is primary and any('descriptor close' in n for n in primary.__notes__)
    assert ex.QUARANTINE[-1]['close_state']=='uncertain; never retry'
