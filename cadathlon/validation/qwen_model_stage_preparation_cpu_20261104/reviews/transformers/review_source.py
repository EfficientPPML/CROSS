"""Source-only model-admission handoff review. No candidate imports or tests."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path('/tmp/qwen-model-stage-aggregate-20261104')
MAP = Path('/tmp/qwen-model-stage-aggregate-review-sources-v1-20261104.json')
OUT = Path(__file__).resolve().parent
MAP_SHA = '613be1bd3e69ea5284882a2241913c02bbbbcdf11665e04fc42db0801d30ee6a'

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def read_json(path):
    return json.loads(path.read_bytes())

assert sha(MAP.read_bytes()) == MAP_SHA
expected = read_json(MAP)
assert len(expected) == 29
observed = {}
for name in sorted(expected):
    path = ROOT / name
    assert path.is_file() and not path.is_symlink(), name
    observed[name] = sha(path.read_bytes())
assert observed == expected
assert {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()} == set(expected)
assert not any(p.is_symlink() for p in ROOT.rglob('*'))

origin = read_json(ROOT / 'source_origin.json')
exact = []
for name, descriptor in origin['base_files'].items():
    parent = Path(descriptor['path'])
    assert sha(parent.read_bytes()) == descriptor['sha256'], name
    if name != 'qwen_public_parameters.py':
        assert (ROOT / name).read_bytes() == parent.read_bytes(), name
        exact.append(name)
assert len(exact) == 19
assert sum(n.endswith('.py') for n in exact) == 7
assert sum(n.startswith('public_inputs/') for n in exact) == 12

parent = Path(origin['base_files']['qwen_public_parameters.py']['path'])
patch = ''.join(difflib.unified_diff(
    parent.read_text().splitlines(True),
    (ROOT / 'qwen_public_parameters.py').read_text().splitlines(True),
    fromfile='selected366/qwen_public_parameters.py',
    tofile='model-stage-successor/qwen_public_parameters.py'))
assert patch.encode() == (ROOT / 'parameter_identity.patch.txt').read_bytes()
assert sha(patch.encode()) == origin['declared_overlay']['patch_sha256']

# Parse only: no evaluation or import of these candidate modules.
for name in ('model_admission.py', 'model_store_validation.py', 'qwen_public_parameters.py'):
    ast.parse((ROOT / name).read_bytes(), filename=name)
pure = ('model_admission.py', 'qwen_file_contract.py', 'qwen_public_parse.py', 'public_source_pins.py')
pure_imports = {}
for name in pure:
    tree = ast.parse((ROOT / name).read_bytes(), filename=name)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
    assert not any(n.split('.')[0] in {'numpy', 'jax', 'jaxlib', 'jaxite_word', 'torch', 'transformers', 'model_store_validation', 'qwen_public_parameters'} for n in imports)
    pure_imports[name] = imports

handoff_paths = (
    '/tmp/qwen-model-stage-aggregate-plan-v2-20261104/PLAN.md',
    '/tmp/qwen-model-stage-aggregate-plan-v2-20261104/HELD_IDENTITY.md',
    '/tmp/qwen-key-generation-producer-v2-20261104/qwen_key_producer.py',
    '/tmp/qwen-client-stage-glue-v2-20261104/qwen_key_contract.py',
    '/tmp/qwen-client-stage-glue-v2-20261104/qwen_key_adapter.py',
    '/tmp/qwen-client-stage-glue-v2-20261104/qwen_key_files_admission.py',
    '/tmp/qwen-client-stage-glue-v2-20261104/client_stage.py',
    '/tmp/qwen-organizer-file-engine-integration-v2-20261104/qwen_stage_composition.py',
    '/tmp/qwen-organizer-file-engine-integration-v2-20261104/qwen_backend_composition.py',
)
handoff_pins = {p: sha(Path(p).read_bytes()) for p in handoff_paths}
assert handoff_pins[handoff_paths[2]] == 'f85277dc7fe0d7dfcbb01e14500d745d975bb41017199cef6e948d6247427529'

assert {name: sha((ROOT / name).read_bytes()) for name in expected} == expected
report = {
    'schema': 'qwen-model-stage-transformers-source-review-v1',
    'passed': True,
    'scope': 'Independent source and public metadata handoff review only; no candidate imports, tests, body validation, grants or execution.',
    'source_root': str(ROOT),
    'source_manifest': str(MAP),
    'source_manifest_sha256': MAP_SHA,
    'source_count': 29,
    'source_hashes': observed,
    'exact_predecessor_copies': {'python': 7, 'public_metadata': 12},
    'identity_overlay_matches_declared_patch': True,
    'pure_imports_ast': pure_imports,
    'handoff_source_pins': handoff_pins,
    'findings': [
        'No concrete API, identity or ordering mismatch found in the reviewed scope.',
        'GenerationIntent.model_id, ExpectedIds and request/response identities carry one content model_id; crypto context and producer/consumer source identities remain separate.',
        'authenticate_model reauthenticates fixed aggregate/content/catalog, complete source maps, recorded store inventory and independently pinned outer/root evidence; require_admitted separately checks the selected consumer source.',
        'All three held roots are checked before parameter file opens; held/named file states are compared before the first hash, header or mmap on that file.',
        'Recording validator captures original owner callbacks, visits all198 logical/197 physical descriptors and113 gammas, discards returned views and invokes original readback/close under shared cleanup selection.',
        'Actual entrypoints deny and recording completion cannot satisfy the positive model-validation schema.'
    ],
    'explicit_limits': [
        'Source review, not independent execution or proof of current model body integrity.',
        'No actual stores, keys, ciphertexts, model or token bodies opened; no backend or candidate module imported.',
        'Future actual wrappers must require ModelAdmission and matching model_id before RNG or key/file loader entry. Existing366 server enters its recording key loader before parameter_factory, so a parameter-only hook is insufficient for that future outer prerequisite.',
        'Direct ExpectedStoreIdentity and recording factories are trusted component inputs, not execution capabilities.',
        'The pure gate does not inspect source archives; it binds the independently trusted root archive-member attestation.',
        'No broad lifecycle, arbitrary Python mutation, source-bootstrap, process or actual-role proof is claimed.'
    ],
    'tests_executed': False,
    'candidate_modules_imported': False,
    'actual_execution_authorized': False,
    'review_script_sha256': sha(Path(__file__).read_bytes()),
}
target = OUT / 'review.json'
with target.open('xb') as stream:
    stream.write((json.dumps(report, indent=2, sort_keys=True) + '\n').encode())
print(json.dumps({'passed': True, 'sources': 29, 'review': str(target), 'review_sha256': sha(target.read_bytes())}))
