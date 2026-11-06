import io,json,tarfile
import pytest
import collect_component as c
import freeze_io
from cleanup_origin import select,protect_body

def test_existing_output_preserved(tmp_path):
    p=tmp_path/'old';p.mkdir();f=p/'report.json';f.write_bytes(b'previous evidence\n')
    with pytest.raises(FileExistsError):c.collect(p)
    assert f.read_bytes()==b'previous evidence\n'and sorted(x.name for x in p.iterdir())==['report.json']

def test_duplicate_source_archive_rejected():
    data=io.BytesIO()
    with tarfile.open(fileobj=data,mode='w:gz')as t:
        for _ in range(2):m=tarfile.TarInfo('x.py');m.size=1;t.addfile(m,io.BytesIO(b'x'))
    with pytest.raises(ValueError):c.archive_map(data.getvalue())

def test_copied_source_substitution_rejected(tmp_path):
    (tmp_path/'x.py').write_bytes(b'new');(tmp_path/'origins.json').write_text(json.dumps({'x.py':{'sha256':'0'*64}}))
    with pytest.raises(ValueError):c.verify_origins(tmp_path)

def test_child_exit_without_group_absence_not_accepted():
    row=dict(direct_returncode=0,direct_child_reaped=True,group_empty_verified=False,cleanup_completed=True)
    assert not c.child_ok(row);row['group_empty_verified']=True;assert c.child_ok(row)
    row['direct_returncode']=False;assert not c.child_ok(row)

@pytest.mark.parametrize('later',[KeyboardInterrupt,SystemExit])
def test_close_only_failure_remains_cleanup_through_outer_layer(later):
    first=OSError('close')
    with pytest.raises(OSError)as caught:freeze_io.finish(None,lambda:(_ for _ in()).throw(first))
    assert select(caught.value,[('later close',later())]).__class__is later

def test_sticky_genuine_body_stays_primary_across_io_cleanup():
    body=protect_body(ValueError('body'))
    freeze_io.finish(body,lambda:(_ for _ in()).throw(body))
    assert select(body,[('later close',KeyboardInterrupt())])is body
