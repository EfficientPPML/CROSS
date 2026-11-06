"""Source metadata/archive resolution tests only; no numerical imports."""
from pathlib import Path
import io,json,tarfile
import pytest
import assemble_sources as a


def archive(path,rows):
    with tarfile.open(path,'w:gz') as out:
        for name,raw in rows:
            member=tarfile.TarInfo(name);member.size=len(raw);out.addfile(member,io.BytesIO(raw))
    return a.digest(path.read_bytes())


def test_complete_196_file_assembly_and_only_overlay():
    r=json.loads((a.HERE/'assembly_report.json').read_text())
    assert r['passed'] and r['source_file_count']==196 and r['joined_ideal_proof_complete']
    assert r['stage3_runtime_files']==59 and r['joined_snapshot_files']==137 and r['shared_core_file_count']==30
    for n,h in r['source_hashes'].items():assert a.digest((a.HERE/n).read_bytes())==h
    assert r['sole_runtime_overlay']==dict(path='runtime/stage/full_model_engine.py',before=a.ENGINE_BASE,after=a.ENGINE_AFTER)
    source=a.source_archive(a.HERE/'origins/stage3_sources.tar.gz',a.STAGE3_TAR,76)
    for n,raw in source.items():
        if n.startswith('runtime/'):
            assert a.digest((a.HERE/n).read_bytes())==(a.ENGINE_AFTER if n=='runtime/stage/full_model_engine.py' else a.digest(raw))


def test_static_resolution_reproduces_without_executing_loaders():
    report=a.audit_resolution(a.HERE)
    assert report==json.loads((a.HERE/'import_resolution.json').read_text())
    assert set(report['equal_basename_collisions'])=={'server_public_model','support'}
    assert len(report['reachable_local_modules'])==75


def test_finder_never_executes_module(tmp_path):
    (tmp_path/'forbidden.py').write_text('raise AssertionError("must not execute")\n')
    assert a.resolve_module('forbidden',[tmp_path])==tmp_path/'forbidden.py'


def test_conflicting_local_basename_rejected(tmp_path):
    for root in a.resolver_roots(tmp_path):root.mkdir(parents=True,exist_ok=True)
    (tmp_path/'runtime/jaxite_word/__init__.py').write_text('pass\n')
    (tmp_path/'runtime/stage/support.py').write_text('ONE=1\n')
    (tmp_path/'joined/support.py').write_text('ONE=2\n')
    with pytest.raises(ValueError,match='conflicting import basename'):a.audit_resolution(tmp_path)


def test_actual_public_plan_larger_than_old_limit_is_preserved():
    source=a.source_archive(a.HERE/'origins/stage3_sources.tar.gz',a.STAGE3_TAR,76)
    raw=source['runtime/stage/server_inputs/twelve_layer_plan.json']
    assert len(raw)==6461050 and 4<<20<len(raw)<8<<20
    assert sum(map(len,source.values()))==8314834


@pytest.mark.parametrize('kind',['digest','duplicate','traversal','member_limit','total_limit'])
def test_bound_and_identity_rejections(tmp_path,kind):
    p=tmp_path/'source.tar.gz';rows=[('one.py',b'pass\n')];count=1
    if kind=='duplicate':rows*=2;count=2
    elif kind=='traversal':rows=[('../foreign.py',b'pass\n')]
    elif kind=='member_limit':rows=[('huge.py',b'x'*((8<<20)+1))]
    elif kind=='total_limit':rows=[(str(i)+'.py',b'x'*(6<<20)) for i in range(3)];count=3
    digest=archive(p,rows)
    if kind=='digest':digest='0'*64
    with pytest.raises(ValueError):a.source_archive(p,digest,count)
