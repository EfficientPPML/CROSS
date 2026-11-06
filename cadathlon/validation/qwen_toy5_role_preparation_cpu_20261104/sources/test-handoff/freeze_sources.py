"""Public source/AST closure only. Does not import or execute selected modules."""
from pathlib import Path
import ast,collections,gzip,hashlib,io,json,os,stat,tarfile,datetime
ROOT=Path('/tmp/qwen-toy5-role-focused-tests-20261104')
OUT=Path('/tmp/qwen-toy5-role-focused-test-handoff-20261104')
MAP=Path('/tmp/qwen-toy5-role-focused-test-sources-20261104.json')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def compact(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def pretty(v):return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
parent_raw=(ROOT/'runtime-map.json').read_bytes();parent=json.loads(parent_raw)
assert len(parent)==389 and sha(parent_raw)=='d82d68101c1f3ddc7f18b6ae4492aa7618d779012babaf9ed85355dfab927fea'
selection=json.loads((ROOT/'selected_tests.json').read_bytes());nodes=selection['nodes'];assert len(nodes)==55 and len(set(nodes))==55
observed=[]
for relative in selection['files']:
    tree=ast.parse((ROOT/relative).read_text(),filename=relative)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    for node in functions:
        assert node.name.startswith('test_')and not node.decorator_list
        assert len(node.body)==1 and isinstance(node.body[0],ast.Expr)
        call=node.body[0].value;assert isinstance(call,ast.Call)and isinstance(call.func,ast.Name)and call.func.id=='run_case'
        assert len(call.args)==2 and isinstance(call.args[0],ast.Constant)and call.args[0].value==node.name
        assert isinstance(call.args[1],ast.Name)and call.args[1].id=='tmp_path'and not call.keywords
        observed.append(relative+'::'+node.name)
    handler='cases_'+relative.removeprefix('test_q5_')
    handler_tree=ast.parse((ROOT/handler).read_text(),filename=handler)
    assert [n.name for n in handler_tree.body if isinstance(n,ast.FunctionDef)and n.name.startswith('test_')]==[n.name for n in functions]
assert observed==nodes
for path in ROOT.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
full={};sizes={}
for path in sorted(ROOT.rglob('*')):
    assert not path.is_symlink(),path
    if path.is_dir():continue
    assert stat.S_ISREG(path.lstat().st_mode)and path.suffix!='.pyc'and '__pycache__'not in path.parts,path
    rel=str(path.relative_to(ROOT));raw=path.read_bytes();full[rel]=sha(raw);sizes[rel]=len(raw)
assert {n.removeprefix('runtime/'):v for n,v in full.items()if n.startswith('runtime/')}==parent
assert full['cases_owned_publication.py']=='c8c77ffd887a5257e5a553439c5fbef982290930bda987a9ab563384a734592f'
for name,digest in parent.items():assert sha((Path('/tmp/qwen-toy5-actual-checkpoint-runtime-v4-20261104')/name).read_bytes())==digest
numpy_pins=json.loads((ROOT/'numpy-pins.json').read_bytes())
for path,digest in numpy_pins['sources'].items():assert sha(Path(path).read_bytes())==digest
raw_map=compact(full);MAP.write_bytes(raw_map);(OUT/'sources.json').write_bytes(raw_map)
with (OUT/'sources.tar.gz').open('wb')as raw:
    with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0)as zipped:
        with tarfile.open(fileobj=zipped,mode='w')as archive:
            for name in full:
                data=(ROOT/name).read_bytes();assert sha(data)==full[name]
                info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;info.mtime=0;info.uid=info.gid=0;info.uname=info.gname=''
                archive.addfile(info,io.BytesIO(data))
with tarfile.open(OUT/'sources.tar.gz','r:gz')as archive:
    members=archive.getmembers();assert [m.name for m in members]==list(full)
    assert all(m.isfile()and not m.issym()and not m.islnk()for m in members)
    assert {m.name:sha(archive.extractfile(m).read())for m in members}==full
assert {name:sha((ROOT/name).read_bytes())for name in full}==full
checks={'schema':'qwen-toy5-role-focused-static-source-review-v1','passed':True,'source_count':len(full),'runtime_parent_count':389,'new_test_provenance_files':len(full)-389,'test_count_ast':55,'prospective_child_count':56,'runtime389_all_exact':True,'archive_current_exact':True,'numpy4_sources_equal':True,'numpy_imported':False,'candidate_imported':False,'pytest_collection':False,'tests_executed':False,'native_backend_body_or_capacity_calls':False,'selected_nodes':nodes,'groups':dict(collections.Counter(n.split('::')[0]for n in nodes))}
(OUT/'static-review.json').write_bytes(pretty(checks))
handoff={'schema':'qwen-toy5-role-focused-source-handoff-v1','held':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_root':str(ROOT),'source_count':len(full),'source_bytes':sum(sizes.values()),'runtime_parent_count':389,'runtime_map_sha256':sha(parent_raw),'source_manifest':str(MAP),'source_manifest_sha256':sha(raw_map),'source_manifest_encoding':'compact sorted JSON + newline','source_archive':str(OUT/'sources.tar.gz'),'source_archive_sha256':sha((OUT/'sources.tar.gz').read_bytes()),'selection':str(ROOT/'selected_tests.json'),'selection_sha256':full['selected_tests.json'],'test_count_ast':55,'prospective_child_count':56,'test_files':selection['files'],'child_source_sha256':full['case_child.py'],'parent_source_sha256':full['test_support.py'],'fixture_source_sha256':full['case_fixtures.py'],'helper_selection_sha256':full['helper-selection.json'],'retention_plan_sha256':full['ARTIFACT_SELECTION.json'],'numpy_pins_sha256':full['numpy-pins.json'],'static_review_sha256':sha((OUT/'static-review.json').read_bytes()),'freeze_script_sha256':sha(Path(__file__).read_bytes()),'root_owned_publication_sha256':full['cases_owned_publication.py'],'planned_affinity':[108,109,110,111],'execution_authorized':False,'actual_role_authorized':False,'candidate_imports':False,'pytest_collection':False,'tests_executed':False,'current_source_only_findings':'Moving-draft source corrections and AST-only authoring errors are disclosed in AUTHORING_HISTORY.md; no product/test runtime failure is claimed.','next':'Independent root/peer source review; separately reviewed enclosing focused harness and explicit later execution grant required.'}
(OUT/'handoff.json').write_bytes(pretty(handoff))
print(json.dumps({'source_count':len(full),'new_files':len(full)-389,'map':sha(raw_map),'tar':handoff['source_archive_sha256'],'handoff':sha((OUT/'handoff.json').read_bytes()),'child':full['case_child.py'],'parent':full['test_support.py'],'selection':full['selected_tests.json'],'source_bytes':sum(sizes.values()),'scope':'stdlib source/hash/AST only; no selected imports/collection/tests'},sort_keys=True))
