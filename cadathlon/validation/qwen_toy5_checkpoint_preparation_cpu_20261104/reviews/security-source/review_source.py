"""Independent stdlib-only source, archive and static selection review."""
from pathlib import Path
import ast
import hashlib
import json
import tarfile

ROOT=Path('/tmp/qwen-toy5-checkpoint-tests-20261104')
PARENT=Path('/tmp/qwen-toy5-checkpoint-component-20261104')
HANDOFF=Path('/tmp/qwen-toy5-checkpoint-test-source-handoff-20261104')
OUT=Path(__file__).resolve().parent

def sha(data):return hashlib.sha256(data).hexdigest()
def read_json(path):return json.loads(path.read_bytes())
def canonical_map(value):return(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode()
def snapshot(root):
    result={}
    for path in sorted(root.rglob('*')):
        assert not path.is_symlink(),str(path)
        if path.is_file():result[str(path.relative_to(root))]=sha(path.read_bytes())
    return result

def main():
    handoff=read_json(HANDOFF/'handoff.json')
    raw_map=Path(handoff['source_map']).read_bytes();expected=json.loads(raw_map)
    assert len(expected)==261 and sha(raw_map)==handoff['source_manifest_sha256']
    assert raw_map==canonical_map(expected)
    start=snapshot(ROOT);assert start==expected
    parent=read_json(ROOT/'runtime_sources.json')
    assert len(parent)==250 and sha((ROOT/'runtime_sources.json').read_bytes())==handoff['parent_source_manifest_sha256']
    assert snapshot(PARENT)==parent
    assert {n.removeprefix('runtime/'):h for n,h in expected.items()if n.startswith('runtime/')}==parent
    assert {n:h for n,h in expected.items()if not n.startswith('runtime/')}==handoff['new_files']
    archive=HANDOFF/'sources.tar.gz'
    assert sha(archive.read_bytes())==handoff['source_archive_sha256']
    archived={}
    with tarfile.open(archive,'r:gz')as tar:
        for item in tar.getmembers():
            assert item.isfile()and not item.name.startswith('/')and'..'not in Path(item.name).parts
            name=item.name.removeprefix('./');assert name not in archived
            with tar.extractfile(item)as stream:archived[name]=sha(stream.read())
    assert archived==expected
    selected=read_json(ROOT/'selected_tests.json')
    filenames=['test_q5_mapping.py','test_q5_metadata.py','test_q5_checkpoint_state.py','test_q5_entry.py']
    nodes=[];counts={}
    for name in filenames:
        count=0;tree=ast.parse((ROOT/name).read_bytes(),filename=name)
        for node in tree.body:
            if not isinstance(node,ast.FunctionDef)or not node.name.startswith('test_'):continue
            decorators=[d for d in node.decorator_list if isinstance(d,ast.Call)and isinstance(d.func,ast.Attribute)and d.func.attr=='parametrize']
            if decorators:
                assert len(decorators)==1
                d=decorators[0];ids=ast.literal_eval(next(k.value for k in d.keywords if k.arg=='ids'))
                assert len(ids)==len(ast.literal_eval(d.args[1]))and len(ids)==len(set(ids))
                cases=[node.name+'['+i+']'for i in ids]
            else:cases=[node.name]
            nodes.extend(name+'::'+case for case in cases);count+=len(cases)
        counts[name]=count
    assert counts==handoff['counts']==selected['counts']
    assert len(nodes)==48 and nodes==handoff['ordered_nodes']==selected['ordered_nodes']
    assert selected['expected_tests']==48
    assert sha((ROOT/'selected_tests.json').read_bytes())==handoff['selection_sha256']
    pins=read_json(ROOT/'numpy_pins.json')
    assert len(pins['sources'])==4 and set(pins['required_origins'])=={'numpy'}
    assert (ROOT/'numpy_pins.json').read_bytes()==(ROOT/'runtime/reference/runtime_pins.json').read_bytes()
    for path,digest in pins['sources'].items():assert sha(Path(path).read_bytes())==digest
    conftest=ast.parse((ROOT/'conftest.py').read_bytes())
    forbidden=ast.literal_eval(next(n.value for n in conftest.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='FORBIDDEN'for t in n.targets)))
    assert {'jax','jaxlib','libtpu','jaxite_word','torch','tensorflow','key_gen','polynomial'}<=forbidden
    assert snapshot(ROOT)==start and snapshot(PARENT)==parent
    targets=filenames+['conftest.py','q5_test_fixtures.py','numpy_pins.json','FOCUSED_TEST_PLAN.md','test_provenance.json',
        'runtime/q5_checkpoint_oracle.py','runtime/q5_actual_metadata_replay.py','runtime/q5_checkpoint_contract.py',
        'runtime/q5_checkpoint_arithmetic.py','runtime/q5_checkpoint_entry.py','runtime/reference/literal_graph.py','runtime/reference/dense_reference.py']
    report={
        'schema':'qwen-toy5-checkpoint-security-source-review-v1',
        'scope':'Source-only review of held261 and statically selected48; no test result or actual authority.',
        'source_manifest_sha256':sha(raw_map),'source_archive_sha256':handoff['source_archive_sha256'],
        'handoff_sha256':sha((HANDOFF/'handoff.json').read_bytes()),'source_count':261,'parent_count':250,
        'new_files':11,'all_parent_bytes_exact':True,'current_before_after_archive_equal':True,
        'static_selection':{'count':48,'counts':counts,'ordered_nodes':nodes,'selection_sha256':handoff['selection_sha256']},
        'read_targets':{n:expected[n]for n in targets},
        'numpy':{'source_hash_count':4,'required_origin_names':['numpy'],'pins_sha256':sha((ROOT/'numpy_pins.json').read_bytes()),
            'scope':'Four genuine installed Python source hashes plus root origin; not full native-library closure; no import performed.'},
        'findings':[],
        'source_conclusions':[
            'Fresh literal original-graph evaluation and separately implemented dense formulas; every47 selected65536-slot vector checked over all64 replicas with literal expected216-trace indices.',
            'Manual19 dense-refresh mapping and distinguishable key-before-query fixtures at5/6 and13/14; no inherited140PASS transferred.',
            'Synthetic binary64 returns use an independent original operation-order expression and bounded search requiring nonnominal output; future failure of that search must remain a real fixture failure, not an assumed result.',
            'Rational enclosure expected values enumerate neighboring representable products independently; final actual-alignment26to25 obligation and nonnormal/ordering/export failures are targeted.',
            'Ordinary later output-scale corruption intentionally passes refresh-export validation before testing complete scalar replay; all19 returns and47 rows remain explicit.',
            'Recording Ledger negatives retain original payload and primary exception identity; wrong attribution rejects before callback. These do not execute actual arithmetic, transport or filesystem ownership.',
            'Actual dispatch immediately denies; reserved70 catches escapedBaseException and diagnostic-print failure. These are direct function cases, not CLI bootstrap proof.',
            'Exact static48 selection and backend-deny source are coherent; a separately reviewed isolated harness must bind sources before imports and collect exact order before fixtures.'
        ],
        'limits':['No imports/collection/tests/probes or toy weight generation performed in this review.',
                  'No model/private/token/key/ciphertext body reads or real transport/decoder/context claim.',
                  'Synthetic ideal identity refresh slots provide no bootstrap/noise/HE accuracy evidence.',
                  'Original native final0.10NRMSE/0.20maxabs gate remains separate from dense diagnostics.',
                  'Runtime numerical success and execution authorization remain pending independent harness/root review.'],
        'candidate_imports':False,'collection':False,'execution':False,'actual_execution_authorized':False,
        'review_script_sha256':sha(Path(__file__).read_bytes())}
    raw=(json.dumps(report,sort_keys=True,indent=2)+'\n').encode()
    with(OUT/'review.json').open('xb')as stream:stream.write(raw)
    print(json.dumps({'status':'source_review_pass','report_sha256':sha(raw),'source_count':261,'static_cases':48}))

if __name__=='__main__':main()
