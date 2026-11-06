"""Bounded post-run receipt inspection. Never opens fixture bodies or aliases."""
from pathlib import Path
import hashlib, json, os, stat


def _state(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_uid,
            value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _directory(path):
    value = path.lstat()
    assert stat.S_ISDIR(value.st_mode) and value.st_uid == os.geteuid()
    assert path.resolve() == path
    return value


def _read(path, limit):
    # O_NONBLOCK avoids blocking even if a named regular receipt is replaced
    # with a FIFO between lstat and the original descriptor acquisition.
    before = path.lstat()
    assert stat.S_ISREG(before.st_mode) and before.st_uid == os.geteuid()
    assert 0 <= before.st_size <= limit
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    primary = None
    try:
        opened = os.fstat(fd)
        assert _state(opened) == _state(before)
        parts = []; count = 0
        while count <= limit:
            part = os.read(fd, min(65536, limit + 1 - count))
            if not part: break
            parts.append(part); count += len(part)
        raw = b''.join(parts)
        assert len(raw) == before.st_size <= limit
        assert _state(os.fstat(fd)) == _state(before) == _state(path.lstat())
        return raw, dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                         device=before.st_dev, inode=before.st_ino, mode=before.st_mode)
    except BaseException as error:
        primary = error
        raise
    finally:
        try: os.close(fd)
        except BaseException as error:
            if primary is None: raise
            try: primary.add_note('receipt descriptor close: ' + type(error).__name__)
            except BaseException: pass


def _absent(pid):
    for operation in (os.kill, os.killpg):
        try: operation(pid, 0)
        except ProcessLookupError: pass
        else: return False
    return True


def inspect(out, config):
    base = out / 'pytest-owned'; _directory(base)
    temporary = out / 'tmp'; _directory(temporary)
    assert not list(temporary.iterdir()), 'undeclared worker temporary artifacts'
    plan = config['artifact_plan']
    names = plan['fixed_child_file_names']; assert len(names) == 7
    expected = {(r['nodeid'].split('::')[1], r['variant']): r for r in plan['children']}
    assert len(expected) == 56
    metadata = {}; aliases = []; fixtures = []; records = {}; errors = []
    directories = []; total = 0; discovered = 0
    def note(label, error): errors.append(dict(path=label, error_type=type(error).__name__, message=str(error)[:512]))
    def fixture_metadata(root):
        # Descriptor-only walk of declared child-owned roots, never read_bytes,
        # open, hash, or resolve a fixture (including the deliberately left FIFO).
        stack = [(root, 0)]
        while stack:
            path, depth = stack.pop()
            assert len(fixtures) < 4096 and depth <= 12
            value = path.lstat()
            assert value.st_uid == os.geteuid()
            item = dict(path=path.relative_to(out).as_posix(), mode=value.st_mode,
                        device=value.st_dev, inode=value.st_ino, bytes=value.st_size,
                        mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns,
                        body_read=False)
            if stat.S_ISLNK(value.st_mode):
                item['kind'] = 'symlink'; item['target'] = os.readlink(path)
            elif stat.S_ISDIR(value.st_mode):
                item['kind'] = 'directory'
                entries = sorted(path.iterdir())
                assert len(entries) <= 2048
                stack.extend((p, depth + 1) for p in reversed(entries))
            elif stat.S_ISREG(value.st_mode): item['kind'] = 'regular_metadata_only'
            elif stat.S_ISFIFO(value.st_mode): item['kind'] = 'fifo_metadata_only'
            else: raise ValueError('unexpected owned fixture descriptor kind')
            assert _state(path.lstat()) == _state(value)
            fixtures.append(item)
    entries = sorted(base.iterdir()); assert len(entries) <= 128
    for path in entries:
        value = path.lstat()
        assert value.st_uid == os.geteuid()
        if stat.S_ISLNK(value.st_mode):
            target = os.readlink(path)
            assert path.name.startswith('test_') and path.name.endswith('current')
            assert len(target) <= 4096
            target_path = Path(target)
            # Authenticate a direct owned sibling by metadata; do not follow
            # the alias or read any content through its target spelling.
            named = target_path if target_path.is_absolute() else base / target_path
            assert named.parent == base and named.name not in ('', '.', '..')
            assert '..' not in target_path.parts
            named_state = named.lstat()
            assert stat.S_ISDIR(named_state.st_mode) and named_state.st_uid == os.geteuid()
            assert _state(path.lstat()) == _state(value) and os.readlink(path) == target
            aliases.append(dict(path=path.relative_to(out).as_posix(), target=target,
                                followed=False, device=value.st_dev, inode=value.st_ino))
        else:
            _directory(path); assert path.name.startswith('test_')
            directories.append(path)
    for test_directory in directories:
        children = sorted(test_directory.iterdir()); assert 1 <= len(children) <= 2
        for child_directory in children:
            label = child_directory.relative_to(out).as_posix()
            try:
                _directory(child_directory)
                assert child_directory.name in {'child-case', 'child-help', 'child-usage'}
                discovered += 1; assert discovered <= 56
                listed = sorted(child_directory.iterdir())
                assert len(listed) <= 9
                assert {p.name for p in listed} <= set(names) | set(plan['fixture_roots_metadata_only'])
                blobs = {}; pins = {}
                for name in names:
                    path = child_directory / name
                    try:
                        raw, pin = _read(path, plan['per_file_maximum_bytes'])
                        total += len(raw); assert total <= config['max_regular_bytes']
                        pin['path'] = path.relative_to(out).as_posix()
                        metadata[pin['path']] = pin; pins[name] = pin; blobs[name] = raw
                    except BaseException as error: note(label + '/' + name, error)
                for name in plan['fixture_roots_metadata_only']:
                    path = child_directory / name
                    if os.path.lexists(path): fixture_metadata(path)
                assert len(blobs) == 7
                parent = json.loads(blobs['parent-terminal.json'])
                live = json.loads(blobs['parent-live.json'])
                early = json.loads(blobs['child-early.json'])
                terminal = json.loads(blobs['child-terminal.json'])
                result = json.loads(blobs['result.json'])
                key = (parent['case'], parent['variant'])
                assert key in expected and key not in records
                intended = expected[key]; assert child_directory.name == intended['output_child']
                pid = parent['pid']; ticks = early['start_ticks']
                assert type(pid) is int and pid > 0 and type(ticks) is int and ticks > 0
                assert parent['direct_wait_completed'] is True and parent['group_absent'] is True
                assert parent['primary_type'] is None and parent['cleanup_types'] == []
                assert type(parent['exit_code']) is int and parent['exit_code'] == intended['expected_exit']
                for field in ('pid', 'pgid'):
                    assert type(early[field]) is int and early[field] == pid
                    assert type(terminal[field]) is int and terminal[field] == pid
                assert type(live['pid']) is int and live['pid'] == pid
                for value in (parent['observed_pgid'], live['pgid']):
                    assert value is None or (type(value) is int and value == pid)
                for value in (parent['start_ticks'], live['start_ticks']):
                    assert value is None or (type(value) is int and value == ticks)
                assert terminal['start_ticks'] == ticks and type(terminal['start_ticks']) is int
                assert parent['observed_pgid'] == live['pgid'] and parent['start_ticks'] == live['start_ticks']
                assert all(row['case'] == key[0] for row in (early, terminal, result))
                assert early['variant'] == result['variant'] == key[1]
                assert early['runtime_map_sha256'] == result['runtime_map_sha256'] == config['runtime_map_sha256']
                assert terminal['passed'] is True and result['passed'] is True and result['error_type'] is None
                assert terminal['forbidden_imports'] == [] and terminal['recording_only'] is True
                assert terminal['actual_authority'] is False
                wanted_argv = [config['python'], '-I', '-B', str(out/'execution'/'case_child.py'),
                               '--case', key[0], '--variant', key[1], '--output', str(child_directory)]
                assert live['argv'] == wanted_argv
                allowed = key[0] in config['numpy_allowed_cases']
                assert terminal['numpy_source_pins_checked'] is allowed
                assert type(terminal['numpy_origins']) is dict
                assert set(terminal['numpy_origins']) <= set(config['numpy_pins']['origins'])
                assert all(value == config['numpy_pins']['origins'][name]
                           for name,value in terminal['numpy_origins'].items())
                if allowed: assert terminal['numpy_origins'].get('numpy') == config['numpy_pins']['origins']['numpy']
                else: assert terminal['numpy_origins'] == {}
                origins = terminal['helper_origins']; assert type(origins) is dict
                assert set(origins) <= set(config['selected_helpers'])
                for name,path in origins.items(): assert path == str(out/'execution'/'runtime'/(name+'.py'))
                if intended['expected_exit'] != 0 or key[1]: assert origins == {}
                assert _absent(pid)
                records[key] = dict(nodeid=intended['nodeid'], variant=key[1], directory=label,
                    parent_live=live, parent_terminal=parent, child_early=early,
                    child_terminal=terminal, result=result, files=pins,
                    independently_observed_pid_and_group_absent=True)
            except BaseException as error: note(label, error)
    ordered = [records[key] for row in plan['children']
               if (key := (row['nodeid'].split('::')[1], row['variant'])) in records]
    complete = (not errors and len(records) == discovered == 56 and len(metadata) == 392)
    return dict(passed=complete, expected_children=56, discovered_children=discovered,
                validated_children=len(records), expected_regular_files=392,
                regular_files=len(metadata), regular_bytes=total, errors=errors,
                files=metadata, aliases=aliases, fixture_metadata=fixtures, records=ordered,
                fixture_bodies_read=False, aliases_followed=False)
