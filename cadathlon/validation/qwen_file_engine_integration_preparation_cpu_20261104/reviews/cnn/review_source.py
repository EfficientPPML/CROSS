"""Static public source selection check; no reviewed-module imports or tests."""
from pathlib import Path
import ast,hashlib,json

root=Path('/tmp/qwen-organizer-file-engine-integration-20261104')
reader=Path('/tmp/qwen-key-context-adapter-v2-20261104')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
selection_path=root/'integration_selection.json'
assert sha(selection_path)=='ff28797f6e0dfc5edf554aadf24f662602a08eb3b6425f40649f8e0910b65b38'
selection=json.loads(selection_path.read_bytes())
overlays=selection['overlays'];copied=selection['copied_sources']
assert len(copied)==337 and set(overlays)=={'qwen_backend_composition.py','qwen_file_contract.py','qwen_file_lifecycle.py','qwen_public_parameters.py'}
for name,row in copied.items():
    if name not in overlays:assert sha(root/name)==row['sha256']
for name,row in overlays.items():
    before=Path(copied[name]['origin']).read_bytes()
    assert hashlib.sha256(before).hexdigest()==row['before_sha256']
    text=before.decode()
    for change in row['replacements']:
        assert text.count(change['before'])==1
        text=text.replace(change['before'],change['after'])
    assert (root/name).read_text()==text and sha(root/name)==row['after_sha256']
unchanged_reader=('qwen_key_contract.py','qwen_key_adapter.py','qwen_key_files_admission.py','qwen_key_readers.py','qwen_key_loader.py','key_descriptor_io.py','cleanup_origin.py')
assert all(sha(root/n)==sha(reader/n)for n in unchanged_reader)
assert sha(root/'cleanup_origin.py')=='4543b26bf1168e29bc21375c2f80a9aa99648205a9fd9aea6d5a8059c6d18175'
assert sha(root/'qwen_public_parse.py')==sha(reader/'reused_client_io.py')=='d29919969c9950a86d2e663f5392c401742e6306a0ed7c9ad7ab3ea2f4e830f0'
assert sha(root/'qwen_finalizer.py')=='46e6566a275a84b4ae4596e066c592a4aa3a60e41b3afb4ec30bb96b72551240'
for name in overlays:
    imports=[n for n in ast.walk(ast.parse((root/name).read_text()))if isinstance(n,ast.ImportFrom)]
    assert not any(n.module=='reused_client_io'and any(a.name in ('Finalizer','parse')for a in n.names)for n in imports)
review=dict(schema='qwen-integration-selection-cnn-source-review-v1',passed=True,concrete_findings=[],
    scope='Bounded static source/overlay/cleanup-policy inspection only. No selected module imports, tests, backend/context calls or actual key/model/token/ciphertext/private body reads.',
    selection_sha256=sha(selection_path),copied_source_count=337,declared_overlay_count=4,
    all_nonoverlaid_copied_sources_exact=True,overlays=overlays,
    retained_reader_sha256={n:sha(root/n)for n in unchanged_reader},
    parser_sha256=sha(root/'qwen_public_parse.py'),finalizer_sha256=sha(root/'qwen_finalizer.py'),
    conclusions=[
      'The four overlays reconstruct exactly from declared predecessor bytes and literal replacements; no other copied source differs at this checkpoint.',
      'File-contract and parameter parsing now select the explicit exact parser namespace. Engine, parameter and file lifecycle owners select the explicit new finalizer; no active import in these four files selects the old Finalizer or parse by search-path fallback.',
      'The shared Toy4v4 cleanup_origin is exact. Finalizer protects genuine primary identity before actions, marks caught cleanup failures, attempts shield fallback, and selects through the common rule. Engine quarantine now records final.selected(), matching the error later raised.',
      'The exact166/335 reader/admission/loader code remains unchanged from admitted43. Its full PublicContract/external expected IDs precede readers; key owner, context/maps/transfers and serial role/quarantine policy are retained.',
      'Borrowed engine ownership and numerical body are unchanged except Finalizer import and selected-error query. Original run-local owner/array/callback inventories and consumer-before-key-retirement requirements remain intact.'
    ],limitations=[
      'The future role-level qwen_stage_composition bridge was not part of this held source read; no joined loader/publisher/engine path or complete integration closure is admitted.',
      'HeldDirectory/state remain explicitly selected legacy helpers; this is not an all-reader robustness or arbitrary Python mutation guarantee.',
      'The actual factories remain disabled; full335 real bodies, backend security/context construction, signals/device readiness, source bootstrap and external execution grants are outside this review.',
      'Peer focused Finalizer regression results are not a test rerun or terminal admission by this reviewer.'
    ],review_script_sha256=sha(Path(__file__)))
path=Path(__file__).with_name('review.json');path.write_text(json.dumps(review,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(passed=True,review_sha256=sha(path),copied_sources=337,overlays=4)))
