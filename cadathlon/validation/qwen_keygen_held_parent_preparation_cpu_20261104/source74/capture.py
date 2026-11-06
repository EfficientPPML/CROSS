"""Static source capture only; no candidate import or test collection."""
from pathlib import Path
import ast,difflib,gzip,hashlib,io,json,tarfile

OLD=Path('/tmp/qwen-key-generation-producer-v2-20261104')
NEW=Path('/tmp/qwen-key-generation-held-parent-20261104')
OUT=Path(__file__).parent
MAP=Path('/tmp/qwen-keygen-held-parent-sources-v1-20261104.json')
PARENT_MAP=Path('/tmp/qwen-key-producer-review-sources-v2-20261104.json')
AUTH=Path('/tmp/qwen-key-generation-actual-root-review-20261104/held-parent-source-authorization.json')
sha=lambda b:hashlib.sha256(b).hexdigest()
canonical=lambda v:(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()

def inventory(root):
    paths=sorted(root.rglob('*'))
    assert all(not p.is_symlink() and(p.is_file()or p.is_dir())for p in paths)
    return{p.relative_to(root).as_posix():sha(p.read_bytes())for p in paths if p.is_file()}

def main():
    raw=PARENT_MAP.read_bytes();oldmap=json.loads(raw)
    assert sha(raw)=='6b8a842e72694d691ea2f4a2d9831331b14e929ad521d701fe445e85f939d874'
    assert inventory(OLD)==oldmap and len(oldmap)==71
    assert sha(AUTH.read_bytes())=='335a51f23a5145fad72da05e36ea37de384b6b12e8677e982e226546d1df6086'
    old=(OLD/'qwen_key_store.py').read_text();new=(NEW/'qwen_key_store.py').read_text()
    assert sha(old.encode())=='2e60908c9773debaa85a5d325df1367013ffb99c6ddead2f587dafca78efb0bd'
    assert sha(new.encode())=='96a20067b0034a4f9ecd394e2b56e29fe5389c996505230701d9cbd883a2d67d'
    before=ast.parse(old);after=ast.parse(new)
    cls=lambda tree,name:next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name==name)
    oldclass=cls(before,'KeyStore');newclass=cls(after,'KeyStore')
    changed=[]
    for node in oldclass.body:
        if not isinstance(node,ast.FunctionDef):continue
        current=next(n for n in newclass.body if isinstance(n,ast.FunctionDef)and n.name==node.name)
        if ast.dump(node,include_attributes=False)!=ast.dump(current,include_attributes=False):changed.append(node.name)
    assert changed==['__init__','_directory_check','check']
    for name in ('_NpySink','_Descriptor','_StoredFile'):
        assert ast.get_source_segment(old,cls(before,name))==ast.get_source_segment(new,cls(after,name))
    for node in before.body:
        if not isinstance(node,ast.FunctionDef):continue
        current=next(n for n in after.body if isinstance(n,ast.FunctionDef)and n.name==node.name)
        assert ast.get_source_segment(old,node)==ast.get_source_segment(new,current)
    init=next(n for n in newclass.body if isinstance(n,ast.FunctionDef)and n.name=='__init__')
    assert init.args.kwonlyargs[0].arg=='expected_parent'and init.args.kw_defaults[0]is None
    calls=[n for n in ast.walk(init)if isinstance(n,ast.Call)]
    parentcall=next(n for n in calls if ast.unparse(n.func)=='self._parent_check')
    mkdir=next(n for n in calls if ast.unparse(n.func)=='os.mkdir')
    opened=next(n for n in calls if ast.unparse(n.func)=='os.open')
    assert opened.lineno<parentcall.lineno<mkdir.lineno
    core=json.loads((NEW/'core_source_manifest.json').read_bytes())
    assert len(core)==34 and all(sha((NEW/n).read_bytes())==h for n,h in core.items())
    assert sha((NEW/'qwen_key_producer.py').read_bytes())=='f85277dc7fe0d7dfcbb01e14500d745d975bb41017199cef6e948d6247427529'
    (OUT/'writer.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(OLD/'qwen_key_store.py'),tofile=str(NEW/'qwen_key_store.py'))))
    provenance=dict(schema='qwen-key-store-held-parent-source-successor-v1',source_only=True,parent_root=str(OLD),parent_count=71,
        parent_source_manifest_sha256=sha(raw),root_source_authorization_path=str(AUTH),root_source_authorization_sha256=sha(AUTH.read_bytes()),
        proposal_path='/tmp/qwen-actual-keygen-resource-source-review-20261104/HELD_PARENT_PROPOSAL.md',
        proposal_sha256='fd89e6a6d81756a1ddcc4a9e49857fe035787f38d43f3e3085199c8a33a2a5d4',
        overlay=dict(file='qwen_key_store.py',before_sha256=sha(old.encode()),after_sha256=sha(new.encode()),changed_existing_methods=changed,new_helpers=['_expected_parent','KeyStore._parent_check']),
        additions=['HELD_PARENT.md','held_parent_provenance.json','parent71_sources.json'],
        exact_core34=True,exact_producer=True,exact_NpySink_and_file_close_classes=True,exact_serialization_and_close_bodies=True,
        required_api='KeyStore(root, specs, *, expected_parent, private=False, shield=cleanup_signals)',
        expected_parent='canonical immutable bytes for existing path/device/inode/mode grant object; no None fallback',
        pure_authority_minimum_pin_modified=False,old_authority_correctly_rejects_overlay=True,
        inherited_tests_selected=False,candidate_imported=False,tests_collected=False,tests_executed=False,
        body_or_capacity_metadata_acquired=False,actual_key_generation_authorized=False)
    (NEW/'held_parent_provenance.json').write_bytes(canonical(provenance))
    current=inventory(NEW)
    assert len(current)==74
    assert[n for n in oldmap if current[n]!=oldmap[n]]==['qwen_key_store.py']
    assert set(current)-set(oldmap)==set(provenance['additions'])
    parsed=[]
    for name in current:
        if name.endswith('.py'):ast.parse((NEW/name).read_bytes(),filename=name);parsed.append(name)
    manifest=canonical(current);MAP.write_bytes(manifest);(OUT/'sources.json').write_bytes(manifest)
    with (OUT/'sources.tar.gz').open('xb')as rawfile:
        with gzip.GzipFile(fileobj=rawfile,mode='wb',filename='',mtime=0)as zipped:
            with tarfile.open(fileobj=zipped,mode='w')as archive:
                for name in sorted(current):
                    data=(NEW/name).read_bytes();member=tarfile.TarInfo(name);member.size=len(data);member.mode=0o644;member.mtime=0
                    archive.addfile(member,io.BytesIO(data))
    with tarfile.open(OUT/'sources.tar.gz')as archive:
        members=archive.getmembers();assert len(members)==len({m.name for m in members})==74
        assert all(m.isfile()for m in members)
        assert{m.name:sha(archive.extractfile(m).read())for m in members}==current
    assert inventory(OLD)==oldmap and inventory(NEW)==current
    report=dict(schema='qwen-key-store-held-parent-static-review-v1',passed=True,source_count=74,source_root=str(NEW),
        source_manifest_path=str(MAP),source_manifest_sha256=sha(manifest),source_archive_sha256=sha((OUT/'sources.tar.gz').read_bytes()),
        parent71_unchanged=True,inherited70_exact=True,core34_exact=True,producer_exact=True,
        writer_sha256=sha(new.encode()),writer_patch_sha256=sha((OUT/'writer.patch').read_bytes()),
        required_expected_parent=True,original_parent_open_before_external_check_before_mkdir=True,
        changed_existing_methods=changed,all_other_writer_function_and_class_bodies_exact=True,
        parsed_python_files=parsed,only_stdlib_source_capture_executed=True,candidate_imports=False,
        tests_or_collection=False,body_or_capacity_metadata_acquired=False,actual_authority=False,
        capture_script_sha256=sha(Path(__file__).read_bytes()))
    (OUT/'static-review.json').write_bytes(canonical(report))
    handoff=dict(schema='qwen-key-store-held-parent-source-handoff-v1',source_root=str(NEW),source_count=74,
        source_manifest_path=str(MAP),source_manifest_sha256=sha(manifest),source_archive_path=str(OUT/'sources.tar.gz'),
        source_archive_sha256=sha((OUT/'sources.tar.gz').read_bytes()),writer_sha256=sha(new.encode()),
        static_review_sha256=sha((OUT/'static-review.json').read_bytes()),writer_patch_sha256=sha((OUT/'writer.patch').read_bytes()),
        source_only=True,parent71_unchanged=True,all70_other_inherited_exact=True,
        tests_selected=[],tests_executed=0,candidate_imports=False,actual_grant=False,
        focused_test_source_review_required_later=True)
    (OUT/'handoff.json').write_bytes(canonical(handoff))
    print(json.dumps(dict(source_count=74,writer_sha256=sha(new.encode()),source_manifest_sha256=sha(manifest),
        source_archive_sha256=sha((OUT/'sources.tar.gz').read_bytes()),static_review_sha256=sha((OUT/'static-review.json').read_bytes()),handoff_sha256=sha((OUT/'handoff.json').read_bytes())),sort_keys=True))

if __name__=='__main__':main()
