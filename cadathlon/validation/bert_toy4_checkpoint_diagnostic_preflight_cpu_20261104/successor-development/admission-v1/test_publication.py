"""Synthetic owned-inode/primary publication tests; no actual role data."""
from contextlib import contextmanager
from pathlib import Path
import os,json,sys
import pytest
from publication import OwnedOutput,separate
from finish import finish

@pytest.mark.parametrize('failure',[RuntimeError,KeyboardInterrupt,SystemExit])
@pytest.mark.parametrize('replace_directory',[True,False])
def test_replaced_owned_destination_never_overwrites_foreign(tmp_path,failure,replace_directory):
 owner=OwnedOutput(tmp_path/'owned').acquire();path=owner.path/'report.json';owner.write(path,dict(passed=False))
 sentinel=b'{"foreign":true}\n';primary=failure('body')
 def body():
  if replace_directory:owner.path.rename(tmp_path/'retained');owner.path.mkdir(mode=0o700)
  else:path.rename(owner.path/'retained.json')
  path.write_bytes(sentinel)
  raise primary
 with pytest.raises(failure)as caught:finish(path,body,owner.write,owner.read)
 assert caught.value is primary and path.read_bytes()==sentinel
 assert any('republication' in n for n in primary.__notes__)

def test_unacquired_owner_cannot_republish_collision(tmp_path):
 path=tmp_path/'old';path.mkdir();target=path/'report.json';target.write_bytes(b'foreign')
 owner=OwnedOutput(path)
 with pytest.raises(FileExistsError):owner.acquire()
 with pytest.raises(ValueError):owner.write(target,dict(passed=False))
 assert target.read_bytes()==b'foreign'

def test_own_successive_atomic_receipts_stay_readable(tmp_path):
 owner=OwnedOutput(tmp_path/'owned').acquire();path=owner.path/'report.json'
 for i in range(4):owner.write(path,dict(passed=True,value=i));assert json.loads(owner.read(path))['value']==i

def test_parent_rename_replacement_rejected(tmp_path):
 parent=tmp_path/'parent';parent.mkdir();owner=OwnedOutput(parent/'owned').acquire();path=owner.path/'report.json';owner.write(path,{})
 parent.rename(tmp_path/'retained');parent.mkdir();owner.path.mkdir(mode=0o700);path.write_bytes(b'foreign')
 with pytest.raises(ValueError):owner.write(path,{})
 assert path.read_bytes()==b'foreign'

@pytest.mark.parametrize('relationship',['equal','descendant','ancestor','symlink'])
def test_protected_paths_cannot_be_used_as_outputs(tmp_path,relationship):
 protected=tmp_path/'source';protected.mkdir();inside=protected/'child'
 if relationship=='equal':output=protected
 elif relationship=='descendant':output=inside
 elif relationship=='ancestor':output=tmp_path
 else:
  output=tmp_path/'link';output.symlink_to(protected,target_is_directory=True)
 with pytest.raises(ValueError):separate(output,[protected])

@pytest.mark.parametrize('cancel',[KeyboardInterrupt,SystemExit])
def test_outer_restore_failure_updates_owned_report_false(tmp_path,cancel):
 owner=OwnedOutput(tmp_path/'owned').acquire();path=owner.path/'report.json'
 @contextmanager
 def scope():yield;raise cancel('restore')
 def body():r=dict(passed=True,terminal_scope_completed=True);owner.write(path,r);return r
 with pytest.raises(cancel):finish(path,body,owner.write,owner.read,termination=scope)
 r=json.loads(owner.read(path));assert r['passed']is False and r['termination_scope_completed']is False
