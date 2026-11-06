"""Bounded public child records only; fixture paths are lstat-only metadata."""
from pathlib import Path
import hashlib,json,os,stat

def sha(raw):return hashlib.sha256(raw).hexdigest()

def read_record(path,limit):
    before=path.lstat();assert stat.S_ISREG(before.st_mode)and before.st_uid==os.geteuid()and 0<=before.st_size<=limit
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC);primary=None
    state=lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_uid,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    try:
        opened=os.fstat(fd);assert state(opened)==state(before)
        chunks=[];size=0
        while size<=limit:
            block=os.read(fd,min(65536,limit+1-size))
            if not block:break
            chunks.append(block);size+=len(block)
        assert size==before.st_size and size<=limit
        assert state(os.fstat(fd))==state(before)==state(path.lstat())
        return b''.join(chunks)
    except BaseException as error:primary=error;raise
    finally:
        try:os.close(fd)
        except BaseException as cleanup:
            if primary is None:raise
            if cleanup is not primary:primary.add_note('child record close: '+repr(cleanup))

def absent(pid,group=False):
    try:(os.killpg if group else os.kill)(pid,0)
    except ProcessLookupError:return True
    return False

def inspect(out,config):
    base=out/'pytest-owned';tmp=out/'tmp'
    assert base.is_dir()and not base.is_symlink()and tmp.is_dir()and not tmp.is_symlink()and not list(tmp.iterdir())
    plan=config['artifact_plan'];expected=config['selected_cases'];rows={};metadata={};aliases={};total=0;fixture_total=0
    for child in sorted(base.iterdir()):
        info=child.lstat()
        if stat.S_ISLNK(info.st_mode):
            assert child.name.startswith('test_device_integration_')and child.name.endswith('current')and info.st_uid==os.geteuid()
            target=os.readlink(child)
            # The only selected target is one lexical sibling basename. Never
            # resolve an alias or follow a target symlink during validation.
            assert type(target)is str and 0<len(target)<=255 and target not in('.', '..')
            assert '/'not in target and '\\'not in target and Path(target).name==target
            named=child.parent/target;assert named.parent==base and named!=child
            target_info=named.lstat()
            assert stat.S_ISDIR(target_info.st_mode)and target_info.st_uid==os.geteuid()
            prefix=child.name[:-len('current')]
            assert named.name.startswith(prefix)and named.name[len(prefix):].isdigit()
            after=child.lstat()
            assert(info.st_dev,info.st_ino,info.st_mode,info.st_uid)==(after.st_dev,after.st_ino,after.st_mode,after.st_uid)
            assert os.readlink(child)==target
            aliases[child.relative_to(out).as_posix()]=dict(kind='outer_pytest_alias',target=target,followed=False)
            continue
        assert stat.S_ISDIR(info.st_mode)and info.st_uid==os.geteuid()
        raw={name:read_record(child/name,plan['per_record_limit_bytes'])for name in plan['records_per_child']}
        parent=json.loads(raw['parent-terminal.json']);terminal=json.loads(raw['child-terminal.json']);result=json.loads(raw['result.json'])
        case=result['case'];assert case in expected and case not in rows and parent['case']==terminal['case']==case
        pid=result['pid'];assert type(pid)is int and pid>0
        assert type(result['pgid'])is int and result['pgid']==pid
        assert parent['schema']=='full28-device-parent-v1'and terminal['schema']=='full28-device-owned-child-v1'
        assert type(parent['pid'])is int and parent['pid']==terminal['pid']==pid
        assert type(terminal['pgid'])is int and terminal['pgid']==parent['owned_session_pgid']==pid
        assert parent['observed_pgid']is None or(type(parent['observed_pgid'])is int and parent['observed_pgid']==pid)
        assert parent['exit_code']==0 and type(parent['exit_code'])is int
        assert parent['direct_wait_completed']is True and parent['reaped']is True and parent['group_absent']is True
        assert parent['primary_type']is None and parent['child_report_read']is True
        assert type(result['starttime_ticks'])is int and result['starttime_ticks']>0 and result['starttime_ticks']==terminal['starttime_ticks']
        assert parent['starttime_ticks']is None or parent['starttime_ticks']==result['starttime_ticks']
        assert terminal['normal_return']is True and terminal['status']=='normal_return'and result['passed']is True
        assert terminal['result_sha256']==parent['result_sha256']==sha(raw['result.json'])
        assert parent['child_terminal_sha256']==sha(raw['child-terminal.json'])
        assert result['backend_denials']==terminal['backend_denials']==[]
        for name in('definition_only_entry','original_main_restored','temporary_names_removed','profile_restored','declared_exports_restored','source_unchanged'):
            assert result[name]is True
        for name in('genuine_cli_bootstrap','model_key_ciphertext_body_reads','native_backend_imported','capacity_measured','actual_execution_authorized'):
            assert result[name]is False
        assert absent(pid)and absent(pid,True)
        fixture_names=set(plan['fixture_files'][case]);seen_fixtures={};seen_records=set();dirs=[]
        def walk(path,prefix=''):
            with os.scandir(path)as entries:
                for entry in entries:
                    name=prefix+entry.name;observed=entry.stat(follow_symlinks=False)
                    assert not stat.S_ISLNK(observed.st_mode)
                    if stat.S_ISDIR(observed.st_mode):dirs.append(name);walk(entry.path,name+'/');continue
                    assert stat.S_ISREG(observed.st_mode),'nonregular fixture not read'
                    if name in raw:
                        assert prefix=='';seen_records.add(name)
                    else:
                        assert name in fixture_names
                        seen_fixtures[name]=dict(bytes=observed.st_size,device=observed.st_dev,inode=observed.st_ino,mode=stat.S_IMODE(observed.st_mode))
        walk(child)
        assert seen_records==set(raw)and set(seen_fixtures)==fixture_names
        declared=parent['fixture_metadata'];assert declared['body_read']is False and declared['body_copied']is False
        assert set(declared['files'])==fixture_names and declared['directories']==sorted(dirs)
        for name,row in seen_fixtures.items():
            assert all(declared['files'][name][k]==v for k,v in row.items())
            metadata[(child/name).relative_to(out).as_posix()]=dict(kind='owned_fixture_metadata',**row,body_read=False)
        fixture_total+=len(seen_fixtures)
        record_pins={}
        for name,data in raw.items():
            total+=len(data);assert total<=plan['total_record_limit_bytes']
            path=(child/name).relative_to(out).as_posix();pin=dict(bytes=len(data),sha256=sha(data),kind='bounded_record')
            metadata[path]=pin;record_pins[name]=dict(path=path,**pin)
        rows[case]=dict(case=case,pid=pid,pgid=pid,starttime_ticks=result['starttime_ticks'],original_wait_completed=True,
            independent_post_wait_pid_absent=True,independent_post_wait_group_absent=True,record_pins=record_pins,fixture_metadata=seen_fixtures)
    assert set(rows)==set(expected)and len(rows)==59
    assert len({row['pid']for row in rows.values()})==59
    assert fixture_total==plan['fixture_regular_count']==14
    assert sum(row['kind']=='bounded_record'for row in metadata.values())==plan['expected_regular_record_count']==236
    return dict(passed=True,validated_children=len(rows),regular_files=236,fixture_metadata_files=fixture_total,
        records=[rows[name]for name in expected],metadata=metadata,aliases=aliases,record_bytes=total,
        fixture_bodies_read=False,fixture_bodies_copied=False,aliases_followed=False,
        actual_role=False,capacity_measurement=False,native_backend_imported=False)
