"""Read only fixed public source/maps; no actual attempt or private data IO."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD_MAP = 'ccd5834b83b9dfec2d3ea988fb395a61ec4b05ce3f2f54621f29200cbbb8e06a'
OLD_GRANT = 'c3983fd975047bba16066106e85f9ae0768ed1e8c0fa30bef4da02b53253cd80'
def sha(raw): return hashlib.sha256(raw).hexdigest()
def read(path): return Path(path).read_bytes()
def mapping(path): return json.loads(read(path))
def constant(path, name):
    tree = ast.parse(read(path))
    found = [n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(found) == 1
    return ast.literal_eval(found[0])
def node(path, name):
    return next(n for n in ast.parse(read(path)).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)

specs = {
    'candidate139': ('/tmp/qwen-expanded-candidate-checker-ready-v2-20261104', '/tmp/qwen-expanded-candidate-checker-ready-sources-v2-20261104.json', OLD_MAP, 139),
    'candidate141': ('/tmp/qwen-expanded-candidate-checker-terminal-exit-successor-20261104', '/tmp/qwen-expanded-candidate-checker-terminal-exit-sources-v1-20261104.json', '45cd92473d9ba558288651d59d4f089c7fb9ba37650802848cad24687beaa437', 141),
    'issuer6': ('/tmp/qwen-candidate1-freeze-issuer-v2-20261104', '/tmp/qwen-candidate1-freeze-issuer-preflight-v2-20261104/source_manifest.json', '35df92d84625d2943a1366de82e48686fa5f16513cefeb6f75820c60c583cc17', 6),
    'reconstructor44': ('/tmp/qwen-candidate-terminal-reconstruction-v2-20261104', '/tmp/qwen-candidate-terminal-reconstruction-sources-v2-20261104.json', 'c0a00954ae17a8262df609a872b3081eee83e441fd9f8ad4187387a64c0d8439', 44),
    'heldout171': ('/tmp/qwen-heldout1088-runtime-v5-20261104', '/tmp/qwen-heldout1088-runtime-sources-v5-20261104.json', '9985aad6205af33a429fba3802bac415397973418e568ea3414d61197f03deeb', 171),
}
maps = {}; roots = {}; identities = {}
for key, (root, path, digest, count) in specs.items():
    raw = read(path); assert sha(raw) == digest
    value = json.loads(raw); assert type(value) is dict and len(value) == count
    maps[key] = value; roots[key] = Path(root)
    identities[key] = dict(root=root, manifest=path, manifest_sha256=digest, source_count=count)

selected = {
    'candidate139': ('entry.py', 'checker_process.py', 'execution_authorization.json'),
    'candidate141': ('entry.py', 'checker_process.py', 'execution_authorization.json', 'TERMINAL_SUCCESSOR.md'),
    'issuer6': ('freeze_issuer.py', 'freeze_admission.py'),
    'reconstructor44': ('contract.py', 'producer_admission.py', 'reconstruction_process.py', 'execution_authorization.json', 'RUNNABLE_DESIGN.md'),
    'heldout171': ('heldout_authorization.py', 'freeze_admission.py', 'heldout_process.py', 'heldout_entry.py', 'heldout_runtime.py', 'execution_authorization.json'),
}
source_hashes = {}
for key, names in selected.items():
    source_hashes[key] = {name: sha(read(roots[key]/name)) for name in names}
    assert all(value == maps[key][name] for name, value in source_hashes[key].items())
for key in ('candidate139', 'candidate141', 'heldout171'):
    assert mapping(roots[key]/'execution_authorization.json') == {'execute': False}

issuer = roots['issuer6']/'freeze_issuer.py'
assert constant(issuer, 'SOURCE') == OLD_MAP and constant(issuer, 'GRANT') == OLD_GRANT
assert constant(roots['reconstructor44']/'contract.py', 'SOURCE_MANIFEST_SHA256') == OLD_MAP
ha = roots['heldout171']/'heldout_authorization.py'
assert constant(ha, 'CALIBRATION_SOURCE') == OLD_MAP and constant(ha, 'CALIBRATION_GRANT') == OLD_GRANT
assert read(roots['issuer6']/'freeze_admission.py') == read(roots['heldout171']/'freeze_admission.py')
assert sha(read(roots['issuer6']/'freeze_admission.py')) == 'a4828b8073dcaad3950ce5cc733fbdbe614a073c7664c3766001b63b176577b6'
assert read(roots['candidate139']/'checker_process.py') == read(roots['candidate141']/'checker_process.py')
assert ast.dump(node(roots['candidate141']/'entry.py','terminal_status'), include_attributes=False) == ast.dump(node(roots['heldout171']/'heldout_entry.py','terminal_status'), include_attributes=False)
numeric = ('checker_runtime.py', 'candidate_domains.py', 'strict_metrics.py', 'test_checker_runtime.py', 'test_candidate_domains.py', 'test_strict_metrics.py')
assert all(maps['candidate139'][name] == maps['candidate141'][name] == maps['heldout171'][name] for name in numeric)
assert all(maps['heldout171'][name] == value for name, value in maps['candidate139'].items())
assert not any(name.endswith('ckks_ctx.py') for key in specs for name in maps[key])

# Static source evidence, not synthetic acceptance of fabricated actual proof.
recon = read(roots['reconstructor44']/'producer_admission.py').decode()
assert 'len(source)!=139' in recon and 'code=0 if passed else 1' in recon
assert "if not passed:" in recon and "same(worker.get('revision_eligible_ledger'),True)" in recon
assert "'reconstruction_source_manifest_sha256','reconstruction_source_archive_sha256','reconstruction_script_sha256'" in recon
grant = ast.get_source_segment(read(roots['candidate141']/'checker_process.py').decode(), node(roots['candidate141']/'checker_process.py','grant_check'))
assert 'source_manifest_sha256=args.manifest_sha256' in grant and 'Independent one-candidate execution grant required' in grant
heldout = read(ha).decode()
assert "receipt['attempts'][0]['source_manifest_sha256'], CALIBRATION_SOURCE" in heldout
assert "receipt['attempts'][0]['grant_sha256'], CALIBRATION_GRANT" in heldout
assert "{'execute': False}" in heldout
assert "'complete_scientific', 'independent_row0'" in heldout
assert "'none', 'terminal'" in heldout

rows = [
    dict(consumer='candidate141 entry/collector', valid_existing_path='Source-ready preparation only; selected candidate/numerics unchanged.', new_requirements='Fresh independently reviewed141 map/preflight/outputs grant; old139 grant cannot match args.manifest_sha256. This does not authorize a retry or reclassify old results.'),
    dict(consumer='reconstructor44', valid_existing_path='Exact original139 PASS can be admitted after independent actual outer exit0/reap/group absence and an explicit reconstruction grant binding all44 sources/archive/script/request.', new_requirements='141 producer is rejected by fixed139 map/count. A future141 consumer needs a separate reviewed producer-map/terminal-contract successor and grant.', restriction='Frozen code still has a failed139 branch; root policy must withhold failed139 reconstruction/revision grants because old exit1 is ambiguous. No new consumer can supply missing old producer normal-return evidence.'),
    dict(consumer='issuer6', valid_existing_path='Pure public-byte issuer accepts candidate1 original139 PASS plus its fixed c398 grant and independently pinned reconstruction review/outer/three closure pins. Returns freeze bytes, not execution permission.', new_requirements='New141 producer and new producer grant are deliberately rejected by SOURCE/GRANT constants; any extension requires a reviewed issuer successor. Later candidates are outside this candidate1 issuer.'),
    dict(consumer='heldout171', valid_existing_path='Consumes first candidate1 original139 PASS freeze with separate producer/reconstructor outer/script/closure pins; requires a distinct171 heldout or row0 grant before input/model reads.', new_requirements='A141-derived freeze is rejected by fixed CALIBRATION_SOURCE/CALIBRATION_GRANT despite identical numerical files. A separate reviewed authorization successor is required.', restriction='Optional complete_scientific prior-heldout path binds the same current consumer map. Independent row0 root receipt permits honest no-prior-attempt or retired terminal attempt without pretending scientific success, and gives no revision/selection authority.'),
    dict(consumer='decoder207 versus these Qwen consumers', valid_existing_path='No ckks_ctx.py in any of the five reviewed Qwen source maps; no decoder dependency or map change is required for their existing clear model/reference path.', new_requirements='Decoder207 establishes isolated clear CRT/FFT evidence only. It does not authorize BERT client decryption, ranked/full-model runs, or broader Qwen tasks.'),
]
result = dict(schema='bounded-consumer-readiness-source-audit-v1', reviewed_at_utc=datetime.now(timezone.utc).isoformat(), passed=True,
    scope='Read-only fixed source/maps and static bindings; no actual candidate outcome, private journal, calibration/validation token, model, key or ciphertext bodies read; no imports, grants, synthetic execution, or frozen-source edits.',
    identities=identities, examined_source_hashes=source_hashes, rows=rows,
    original139_pass_scope='Ordinary reviewed-code scope accepted by root; absence of stderr is not normal-return proof and no arbitrary injected exit/native-finalizer guarantee is claimed.',
    map_compatibility_findings=['141 is not interchangeable with139 in44/6/171.', 'Code70 is infrastructure-only at new141/171 actual entry boundary; old terminal evidence remains unchanged.', 'Source archive content must match the producer map and actual externally pinned archive bytes; a preflight gzip digest is not substituted for actual archive provenance.'],
    peer_scope={'cnn_performance':'Original122/combined125 decoder import/overlay and original producer-map preservation review; separate receipt pending.', 'transformers':'Toy4 future decoder comparison and original260 producer/new consumer-map separation.'},
    remaining_user_scope='Original full all-five/ranked/calibration tasks remain open; this audit adds no actual execution authority.')
(HERE/'review.json').write_text(json.dumps(result, sort_keys=True, indent=2)+'\n')
print(json.dumps(dict(passed=True, reviewed_source_files=sum(map(len,selected.values())), review_sha256=sha(read(HERE/'review.json'))),sort_keys=True))
