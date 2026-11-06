"""Read-only source/AST/archive check of held device6; no candidate imports."""
import ast,difflib,hashlib,json,pathlib,stat,tarfile
HERE=pathlib.Path(__file__).resolve().parent
ROOT=pathlib.Path('/tmp/qwen-server-device-admission-source-v2-20261104')
OLD=pathlib.Path('/tmp/qwen-server-device-admission-source-20261104')
HAND=pathlib.Path('/tmp/qwen-server-device-admission-handoff-v2-20261104')
OLDHAND=pathlib.Path('/tmp/qwen-server-device-admission-handoff-20261104')
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):
    a=p.lstat();assert stat.S_ISREG(a.st_mode) and not p.is_symlink()
    raw=p.read_bytes();z=p.lstat()
    assert(a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns)
    return raw
def doc(p):return json.loads(read(p))
def canonical(d):return(json.dumps(d,sort_keys=True,separators=(',',':'))+'\n').encode()
def files(root):
    result={}
    for p in root.rglob('*'):
        assert not p.is_symlink()
        if p.is_file():result[p.relative_to(root).as_posix()]=sha(read(p))
        else:assert p.is_dir()
    return result
def archive(path,expected):
    result={}
    with tarfile.open(path,'r:gz')as t:
        for m in t.getmembers():
            assert m.isfile() and m.name in expected and m.name not in result
            f=t.extractfile(m)
            with f:b=f.read()
            assert len(b)==m.size;result[m.name]=sha(b)
    assert result==expected

h=doc(HAND/'handoff.json');m=doc(HAND/'sources.json');history=doc(ROOT/'successor_history.json')
assert len(m)==6 and sha(read(HAND/'sources.json'))=='9531060742e091394d0632cf43546817fe9bb121ea93b1f5670bb2eb176d121d'
assert read(HAND/'sources.json')==canonical(m) and files(ROOT)==m
assert sha(read(HAND/'sources.tar.gz'))==h['source_archive_sha256']=='d35074520b61c5e4d39ae2f32c3c83b6553f480bfc46d7adf34b335679336137'
archive(HAND/'sources.tar.gz',m)
old=history['parent_sources'];assert len(old)==5 and files(OLD)==old
assert doc(OLDHAND/'sources.json')==old and sha(read(OLDHAND/'sources.json'))==history['parent_manifest_sha256']
assert sha(read(OLDHAND/'sources.tar.gz'))==history['parent_archive_sha256'];archive(OLDHAND/'sources.tar.gz',old)
assert sorted(set(m)-set(old))==['successor_history.json']
changed=sorted(n for n in old if old[n]!=m[n]);assert changed==['API.md','qwen_device_capacity.py']
assert m['qwen_device_policy.py']=='7b2d69456fca8a07052a37d33bcd2dfda1fb769e340755e97b98b4fce4c26c45'
peer=pathlib.Path(history['peer_finding']['path']);assert sha(read(peer))==history['peer_finding']['sha256']=='2965d5612888d945b1717c4f6bba43a0d473655bae67b83589b484fe5a25a752'
patch=''.join(''.join(difflib.unified_diff(read(OLD/n).decode().splitlines(True),read(ROOT/n).decode().splitlines(True),fromfile='v1/'+n,tofile='v2/'+n))for n in ['qwen_device_capacity.py','API.md'])
assert patch.encode()==read(HAND/'source.patch') and sha(read(HAND/'source.patch'))==h['patch_sha256']

def functions(raw):return{n.name:n for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
before=functions(read(OLD/'qwen_device_capacity.py'));after=functions(read(ROOT/'qwen_device_capacity.py'))
dump=lambda n:ast.dump(n,include_attributes=False)
new=sorted(set(after)-set(before));modified=sorted(n for n in before if dump(before[n])!=dump(after[n]))
assert new==['_capture_policy','_policy_guard','_reauthenticate']
assert modified==['_caller','_context','_device_guard','_hook','_host','_jax_binding','issue_initialization','observe_capacity']
assert all(dump(before[n])==dump(after[n]) for n in ['_lookup','_before','initialize_once','require_current','retire','create_actual_server','_origin','_functions','InitializationPermit','CapacityAdmission'])

# Bind the specific source conclusions without executing any candidate code.
code=read(ROOT/'qwen_device_capacity.py').decode()
segment=lambda n:ast.get_source_segment(code,after[n])
capture=segment('_capture_policy');guard=segment('_policy_guard');reauth=segment('_reauthenticate');hook=segment('_hook');issue=segment('issue_initialization')
assert "fields=object.__getattribute__(policy,'__dict__');record['policy_dict']=fields" in capture
assert "record['policy_fields']=(fields['phase'],fields['blobs'],fields['expected_pins'])" in capture
assert "fields is not record['policy_dict']" in guard and 'fields[name]is not value' in guard
assert "record['state']='failed'" in guard
assert reauth.index('_policy_guard(record)')<reauth.index('authenticate(phase,blobs,expected_pins=pins)')<reauth.rindex('_policy_guard(record)')
assert hook.index('_policy_guard(record)')<hook.index('result=fn(phase,blobs,*args)')<hook.index("record['returned'].append(result)")<hook.rindex('_policy_guard(record)')
assert "if record['state']!='busy':raise RuntimeError" in hook
assert issue.index('_REGISTRY[id(permit)]=record')<issue.index('_capture_policy(record)')<issue.index('_reauthenticate(record)')<issue.index('_caller(_policy_guard(record)[0])')
assert not any(isinstance(n,ast.Attribute)and isinstance(n.value,ast.Subscript)and isinstance(n.value.value,ast.Name)and n.value.value.id=='record'and isinstance(n.value.slice,ast.Constant)and n.value.slice.value=='policy' for n in ast.walk(ast.parse(code)))
assert "raise PermissionError('device observations are not an actual server role factory')" in segment('create_actual_server')
assert files(ROOT)==m and files(OLD)==old
out={
 'schema':'qwen-device-identity-source-peer-review-v2','review_completed':True,'source_checks_passed':True,'no_additional_concrete_blocker_in_requested_scope':True,
 'source_root':str(ROOT),'source_count':6,'source_hashes':m,'source_map_sha256':sha(read(HAND/'sources.json')),'source_archive_sha256':sha(read(HAND/'sources.tar.gz')),
 'handoff_sha256':sha(read(HAND/'handoff.json')),'patch_sha256':sha(read(HAND/'source.patch')),
 'old5_current_and_archive_preserved':True,'old5_map_sha256':history['parent_manifest_sha256'],'old5_archive_sha256':history['parent_archive_sha256'],
 'old_peer_finding_sha256':sha(read(peer)),'changed_source_files':changed,'added_source_files':['successor_history.json'],
 'pure_policy_bytes_unchanged':True,'new_functions':new,'changed_existing_functions':modified,
 'source_conclusions':[
  'Registry insertion precedes capture/authentication/callback work. Capture retains the exact policy object, its original dictionary and original phase/blobs/pins objects; immutable tuple/bytes row types are checked.',
  'The guard rejects dictionary replacement, extra/missing fields and any changed field identity, including equal-content replacement. Its exception path sets failed; the public transition catches retain the record and original objects.',
  'Reauthentication uses the captured phase/blobs/pins through the unchanged pure authenticate function, with guards before and after. Selected process/helper lookup is followed by another guard.',
  'Hooks receive retained original phase/blobs. Returned owners are retained before post-callback checks; sticky busy-state and original policy identities are checked before continuation. Existing source/hook retention remains in force.',
  'Policy-derived allocator/workspace/host/document pins now come from guarded retained blobs/pins. No pure document schema, numerical policy, stage order, capacity equation, native owner or actual-entry authority changed.',
  'Original5 remains intact. The exact delta is one runtime module, API documentation, and one history file; other3 predecessor files including pure policy are exact.'
 ],
 'limits':[
  'Source/AST/archive review only; no actual policy instance, synthetic case, callback probe, candidate import, backend, capacity observation, body read or grant.',
  'Nested phase-object integrity still depends on full phase authentication and selected process snapshot/binding hooks; this does not assert an immutable nested graph or arbitrary-interpreter/native containment.',
  'A mutate-and-restore action wholly inside a trusted callback is not monitored continuously by boundary identity checks; no such broader guarantee is claimed.',
  'Required real process hooks, factual resource/device/allocator/access/workspace evidence, lease and later integration/authority remain absent and separately required. Current selected server is unchanged and still fails closed.'
 ],
 'candidate_imports':False,'tests_or_collection':False,'device_or_capacity_probes':False,'body_reads':False,'actual_grants':False,'current_server_edits':False,
 'review_script_sha256':sha(read(pathlib.Path(__file__)))}
(HERE/'review.json').write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
print(json.dumps({'passed':True,'receipt':str(HERE/'review.json'),'receipt_sha256':sha(read(HERE/'review.json')),'source_count':6,'old5_preserved':True,'candidate_execution':False},sort_keys=True))
