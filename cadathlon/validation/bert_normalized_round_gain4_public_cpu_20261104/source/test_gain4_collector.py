"""Narrow fixed-source, publication and child lifecycle checks; no backend."""
import copy,io,json,tarfile
from pathlib import Path
import pytest
import collect_gain4 as c

def valid_encoding():
    rows=[dict(gain=g,role=role,exact_integer_headroom_passed=True,nonzero_quantized_coefficients=1)for g in (1,4)for role in ('mask','anchor','dummy_bias')for _ in range(6)]
    return dict(passed=True,nonuniform_records=rows,uniform_records=[dict(headroom_passed=True,nonzero=True)for _ in range(24)],limitations=['No HE error guarantee'])

@pytest.mark.parametrize('field',('passed','nonuniform_records','uniform_records','limitations'))
def test_encoder_evidence_rejects_incomplete(field):
    value=valid_encoding();value[field]=False if field=='passed'else[]
    with pytest.raises(ValueError):c.plan_gate(value)

def test_fixed114_source_substitution_rejected():
    raw=c.sources();c.public_gate(raw)
    pins=json.loads(raw['gain4_source_origin.json'])['predecessor_source_hashes']
    name=next(iter(pins));raw[name]+=b'\n'
    with pytest.raises(ValueError,match='frozen public source'):c.public_gate(raw)

def test_archive_duplicate_member_rejected(tmp_path):
    path=tmp_path/'a.tar.gz'
    with tarfile.open(path,'w:gz')as t:
        for _ in range(2):
            m=tarfile.TarInfo('x');m.size=1;t.addfile(m,io.BytesIO(b'x'))
    with pytest.raises(ValueError):c.archive_check(path,{'x':c.e.digest(b'x')})

def test_nested_output_is_excluded_from_fixed_inventory(tmp_path):
    names=list(c.sources());inventory=json.loads((c.HERE/'source_inventory.json').read_bytes())
    assert names==inventory and not any(n.startswith(('source-bound','development','negative'))for n in names)

def fake_children(monkeypatch,output,*,failure=None,close_failure=None):
    made=[]
    class Child:
        def __init__(self,command,*,cwd,stdout,env):
            self.record=dict(pid=10+len(made),direct_child_reaped=False,group_empty_verified=False,cleanup_completed=False)
            self.command=command;self.stream=stdout;made.append(self)
        def wait(self):
            if failure is not None:raise failure
            if '-m'in self.command:
                (output/'tests.xml').write_text(f'<testsuites><testsuite tests="{c.TEST_COUNT}" errors="0" failures="0" skipped="0"/></testsuites>')
                rows=[dict(kind='all_K_fullslot',key_count=K,valid_rows=n,slots=65536,updates=12,output_level=8,fixed_gate=1e-6,passed=True,max_abs_error=0.,primitive_counts=dict(pt=3,rotate=14,multiply=24,rescale=27))for K in range(1,129)for n in (1,7,16)]
                rows.extend(dict(kind='original_fixture',name=name,slots=65536,checkpoint_count=27,checkpoints={str(i):dict(passed=True)for i in range(27)},derived_checks={'x':dict(passed=True)},fixed_gate=1e-6,passed=True)for name in ('K128_rows16','K15_rows9'))
                (output/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in rows))
            else:self.stream.write(json.dumps(valid_encoding()).encode())
            self.record.update(direct_returncode=0,direct_child_reaped=True,group_empty_verified=True,cleanup_completed=True)
            return 0
        def close(self,primary=None):
            self.record.update(direct_child_reaped=True,group_empty_verified=True,cleanup_completed=close_failure is None)
            if close_failure is not None:raise close_failure
    monkeypatch.setattr(c,'OwnedChild',Child)
    return made

def test_success_requires_two_groups_sources_and_terminal_flag(tmp_path,monkeypatch):
    output=tmp_path/'proof';made=fake_children(monkeypatch,output)
    result=c.run(output)
    assert result['passed']is True and result['terminal_scope_completed']is True
    assert len(made)==2 and result['all_owned_groups_empty']is True
    assert json.loads((output/'report.json').read_bytes())==result

def test_late_source_drift_publishes_failure(tmp_path,monkeypatch):
    output=tmp_path/'proof';fake_children(monkeypatch,output)
    real=c.sources;calls=[]
    def source():
        raw=real();calls.append(1)
        if len(calls)>1:raw['README_GAIN4.md']+=b'\nchanged'
        return raw
    monkeypatch.setattr(c,'sources',source)
    result=c.run(output)
    assert result['passed']is False and result['sources_unchanged']is False
    assert json.loads((output/'report.json').read_bytes())['passed']is False

@pytest.mark.parametrize('kind',(RuntimeError,KeyboardInterrupt,SystemExit))
def test_wait_primary_preserved_and_child_cleanup_attempted(tmp_path,monkeypatch,kind):
    output=tmp_path/'proof';primary=kind('wait failure')
    made=fake_children(monkeypatch,output,failure=primary,close_failure=OSError('close failure'))
    with pytest.raises(kind)as seen:c.run(output)
    assert seen.value is primary and made[0].record['direct_child_reaped']is True
    report=json.loads((output/'report.json').read_bytes())
    assert report['passed']is False and report['terminal_scope_completed']is False and report['cleanup_errors']

def test_outer_finish_restoration_failure_republishes(tmp_path,monkeypatch):
    from contextlib import contextmanager
    import finish as f
    output=tmp_path/'report.json';primary=KeyboardInterrupt('outer restore')
    @contextmanager
    def scope():yield;raise primary
    def success():
        r=dict(passed=True,complete=True);c.e.write_json(output,r);return r
    with pytest.raises(KeyboardInterrupt)as seen:f.finish(output,success,c.e.write_json,c.e.public_report,termination=scope)
    assert seen.value is primary and json.loads(output.read_bytes())['passed']is False
