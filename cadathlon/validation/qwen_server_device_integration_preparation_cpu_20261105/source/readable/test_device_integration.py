"""One original owned subprocess per case; no candidate import in pytest."""
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import pytest

HERE=Path(__file__).resolve().parent
CASES=tuple(json.loads((HERE/'selected_tests.json').read_bytes())['cases'])

def _dump(path,value):
    raw=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    assert len(raw)<=1<<20;path.write_bytes(raw)
def _ticks(pid):
    text=(Path('/proc')/str(pid)/'stat').read_text()
    return int(text[text.rfind(')')+2:].split()[19])
def _gone(pgid):
    try:os.killpg(pgid,0)
    except ProcessLookupError:return True
    return False

def _read_record(path,limit=2<<20):
    """Bounded regular owned record after the direct child has retired."""
    fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW|os.O_NONBLOCK);primary=None
    try:
        before=os.fstat(fd)
        assert stat.S_ISREG(before.st_mode)and 0<=before.st_size<=limit
        chunks=[];size=0
        while size<=limit:
            block=os.read(fd,min(65536,limit+1-size))
            if not block:break
            chunks.append(block);size+=len(block)
        after=os.fstat(fd);named=path.lstat()
        state=lambda v:(v.st_dev,v.st_ino,v.st_size,v.st_mtime_ns,v.st_ctime_ns,v.st_mode)
        assert state(before)==state(after)==state(named)and size==before.st_size and size<=limit
        return b''.join(chunks)
    except BaseException as error:primary=error;raise
    finally:
        try:os.close(fd)
        except BaseException as cleanup:
            if primary is None:raise
            if cleanup is not primary:primary.add_note('owned record close: '+repr(cleanup))

def _fixtures(root,case,result):
    selection=json.loads((HERE/'ARTIFACT_SELECTION.json').read_bytes())
    expected=set(selection['fixture_files'][case]);records=set(selection['records_per_child'])
    found={};directories=[]
    def walk(path,prefix=''):
        with os.scandir(path)as entries:
            for entry in entries:
                name=prefix+entry.name;info=entry.stat(follow_symlinks=False)
                assert not stat.S_ISLNK(info.st_mode),'child fixture alias not selected'
                if stat.S_ISDIR(info.st_mode):
                    directories.append(name);walk(entry.path,name+'/')
                else:
                    assert stat.S_ISREG(info.st_mode),'nonregular fixture is never opened'
                    if name not in records:
                        assert name in expected,'undeclared owned fixture'
                        found[name]=dict(bytes=info.st_size,device=info.st_dev,inode=info.st_ino,mode=stat.S_IMODE(info.st_mode))
    walk(root);assert set(found)==expected
    # Writer argument hashes are provenance, not a new readback of fixture
    # bodies by this metadata inspector.
    declared=result.get('details',{}).get('process_evidence_files',[])
    for row in declared:
        assert row['path']in found and found[row['path']]['bytes']==row['bytes']
        assert row['recording_fixture_only']is True
        found[row['path']]['producer_argument_sha256']=row['sha256']
    return dict(files=found,directories=sorted(directories),body_read=False,body_copied=False)

@pytest.mark.parametrize('case',CASES,ids=CASES)
def test_device_integration(case,tmp_path):
    source_map=os.environ['FULL28_FOCUSED_SOURCE_MAP'];source_sha=os.environ['FULL28_FOCUSED_SOURCE_MAP_SHA256']
    env=dict(os.environ)
    for name in('PYTHONPATH','PYTHONHOME','PYTEST_ADDOPTS','PYTEST_PLUGINS'):env.pop(name,None)
    env['PYTHONDONTWRITEBYTECODE']='1'
    observed_pgid=None;start=None;code=None;waited=False;primary=None;group_absent=False
    with(tmp_path/'child.log').open('xb')as log:
        child=subprocess.Popen([sys.executable,'-I','-B',str(HERE/'owned_child.py'),case,str(HERE),str(tmp_path),source_map,source_sha],
            cwd=tmp_path,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            try:observed_pgid=os.getpgid(child.pid)
            except ProcessLookupError:pass
            try:start=_ticks(child.pid)
            except FileNotFoundError:pass
            if observed_pgid is not None:assert observed_pgid==child.pid
            _dump(tmp_path/'parent-terminal.json',dict(schema='full28-device-parent-v1',case=case,pid=child.pid,
                owned_session_pgid=child.pid,observed_pgid=observed_pgid,starttime_ticks=start,status='waiting',direct_wait_completed=False))
            code=child.wait();waited=True
        except BaseException as error:
            primary=error
            try:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                except BaseException as cleanup:
                    if cleanup is not primary:primary.add_note('owned group cleanup: '+repr(cleanup))
            finally:
                try:code=child.wait();waited=True
                except BaseException as cleanup:
                    if cleanup is not primary:primary.add_note('owned direct wait: '+repr(cleanup))
            raise
        finally:
            try:
                group_absent=_gone(child.pid)
                _dump(tmp_path/'parent-terminal.json',dict(schema='full28-device-parent-v1',case=case,pid=child.pid,
                    owned_session_pgid=child.pid,observed_pgid=observed_pgid,starttime_ticks=start,status='terminal',
                    exit_code=code,direct_wait_completed=waited,reaped=waited,group_absent=group_absent,
                    child_report_read=False,primary_type=None if primary is None else type(primary).__name__))
            except BaseException as cleanup:
                if primary is None:raise
                if cleanup is not primary:primary.add_note('terminal observation/publication: '+repr(cleanup))
    assert waited and group_absent and code==0,(case,code,_read_record(tmp_path/'child.log').decode(errors='replace')[-8192:])
    result_raw=_read_record(tmp_path/'result.json')
    terminal_raw=_read_record(tmp_path/'child-terminal.json')
    value=json.loads(result_raw);terminal=json.loads(terminal_raw)
    assert value['case']==terminal['case']==case and value['passed']is True
    assert type(value['pid'])is int and value['pid']==child.pid==terminal['pid']
    assert type(value['pgid'])is int and type(terminal['pgid'])is int and value['pgid']==terminal['pgid']==child.pid
    if observed_pgid is not None:assert value['pgid']==observed_pgid
    assert type(value['starttime_ticks'])is int and value['starttime_ticks']==terminal['starttime_ticks']
    if start is not None:assert value['starttime_ticks']==start
    assert terminal['normal_return']is True and terminal['status']=='normal_return'
    assert terminal['result_sha256']==hashlib.sha256(result_raw).hexdigest()
    for name in('original_main_restored','temporary_names_removed','profile_restored','declared_exports_restored','source_unchanged','definition_only_entry'):
        assert value[name]is True
    for name in('genuine_cli_bootstrap','model_key_ciphertext_body_reads','native_backend_imported','capacity_measured','actual_execution_authorized'):
        assert value[name]is False
    assert value['backend_denials']==terminal['backend_denials']==[]
    # Original wait/terminal is already persisted. Append only child-result
    # joins and lstat metadata; no fixture file is opened by this inspector.
    parent=json.loads(_read_record(tmp_path/'parent-terminal.json'))
    parent.update(child_report_read=True,child_terminal_sha256=hashlib.sha256(terminal_raw).hexdigest(),
        result_sha256=hashlib.sha256(result_raw).hexdigest(),fixture_metadata=_fixtures(tmp_path,case,value))
    _dump(tmp_path/'parent-terminal.json',parent)
