"""Independent stdlib-only readback of public P3 test1 evidence."""
import ast
import hashlib
import json
import math
import os
import stat
import tarfile
from datetime import datetime, timezone
from pathlib import Path

base = Path('/tmp/p3-original-six-test1-checkpoint-20261105')
out = Path('/tmp/p3-original-six-test1-root-readback-20261105.json')
assert not out.exists()
sha = lambda b: hashlib.sha256(b).hexdigest()
read = lambda name: json.loads((base / name).read_text())
index_raw = (base / 'artifact_hashes.json').read_bytes()
assert sha(index_raw) == '16216ead6d529093181dc56fe9cec6a38eba3e24ae7976792690c7fd06beee2d'
index = json.loads(index_raw)
assert len(index) == 22
assert {p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file()} == set(index) | {'artifact_hashes.json'}
for name, row in index.items():
    p = base / name
    assert not p.is_symlink() and stat.S_ISREG(p.stat().st_mode)
    b = p.read_bytes()
    assert sha(b) == row['sha256'] and len(b) == row['bytes']
origins = read('origins.json')
for name, row in origins.items():
    b = (base / name).read_bytes()
    assert sha(b) == row['sha256'] and len(b) == row['bytes']
    if not row.get('mutable_origin'):
        assert Path(row['path']).read_bytes() == b
    else:
        assert row['snapshot_only'] is True
sources = read('preparation/sources.json')
aux = read('preparation/auxiliary.json')
assert len(sources) == 128 and len(aux) == 1
assert sources == read('raw/source-start.json') == read('raw/test1/source-after.json')
archives = []
for name, expected in [('sources/actual-128.tar.gz', sources), ('sources/prepared-129.tar.gz', {**sources, **aux})]:
    data = {}
    with tarfile.open(base / name, 'r:gz') as t:
        for member in t.getmembers():
            assert member.isfile() and member.name in expected and member.name not in data
            assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
            body = t.extractfile(member).read()
            assert sha(body) == expected[member.name]
            data[member.name] = body
    assert set(data) == set(expected)
    archives.append(data)
assert all(archives[0][n] == archives[1][n] for n in sources)
audit = read('source-audit.json')
assert len(audit['current_sources']) == 129
for row in audit['current_sources']:
    assert sha(Path(row['original_path']).read_bytes()) == sha(Path(row['prepared_path']).read_bytes()) == row['sha256'] == {**sources, **aux}[row['name']]
definitions = {}
for label, row in audit['source_functions'].items():
    name, fun = label.rsplit(':', 1)
    tree = ast.parse(archives[0][name].decode())
    node, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fun]
    assert row == dict(line=node.lineno, end_line=node.end_lineno, AST_sha256=sha(ast.dump(node, include_attributes=False).encode()))
    definitions[fun] = node
assert len(definitions) == 14
dump = lambda node: ast.dump(node, include_attributes=False)
expr = lambda text: dump(ast.parse(text, mode='eval').body)
tests = {dump(n.test) for n in ast.walk(definitions['run']) if isinstance(n, ast.If) and any(isinstance(x, ast.Raise) for x in n.body)}
assert expr('not np.all(np.abs(output - expected) <= 0.02 + 0.01 * np.abs(expected))') in tests
assert expr('errors["top1_agreement"] < 0.99') in tests
process = read('raw/test1/process.json')
result = read('raw/test1/stdout.json')
capture = read('capture.json')
parent = read(capture['parent_partial_snapshot'])
assert process == parent['cases']['test1.in'] and result == process['result']
assert parent['active_case'] == 'test3.in' and parent['passed'] is False
assert process['returncode'] == 0
for key in ('passed', 'process_reaped', 'public_key_store_removed', 'owned_directory_handle_closed', 'source_files_unchanged'):
    assert process[key] is True
ring, = result['execution_rings']
product = math.prod(ring['q_towers'] + ring['p_towers'])
assert product.bit_length() == 605
tree = ast.parse(archives[0]['repo/jaxite_word/he_params.py'].decode())
table, = [n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_OPENFHE_CEILINGS_RAW' for t in n.targets)]
mapping, = [v for k, v in zip(table.keys, table.values) if dump(k) == expr('(CostModel.CLASSICAL, 128)')]
ceiling = ast.literal_eval(mapping)[32768]
assert ceiling == 881 and product.bit_length() <= ceiling
assert dump(definitions['modulus_product_is_secure'].body[-1].value) == expr('openfhe_logq_bits(modulus_product) <= security_ceiling(degree, target)')
assert dump(definitions['openfhe_logq_bits'].body[-1].value) == expr('modulus_product.bit_length()')
absence = {}
for ident in (1598337, 1598338, -1598337):
    try:
        os.kill(ident, 0)
    except ProcessLookupError:
        absence[str(ident)] = 'absent_ESRCH'
    else:
        raise AssertionError('completed child PID/group still exists')
assert not os.path.lexists(process['owned_public_key_directory'])
key_parent = Path('/dev/shm/cross-p3-original-six-key-parent-f383y30c')
s = key_parent.lstat()
assert (s.st_dev, s.st_ino, stat.S_IMODE(s.st_mode)) == (24, 4064, 0o700)
summary = read('summary.json')
assert summary['completed_cases_admitted'] == 1 and summary['sweep_complete'] is False
assert (base / 'artifact_hashes.json').read_bytes() == index_raw
record = dict(observed_utc=datetime.now(timezone.utc).isoformat(), status='PASS test1 public checkpoint only',
    original_session=81235, latest_root_poll='78efe7 live', indexed_files=22, origins=len(origins),
    actual_sources=128, prepared_sources=129, source_definitions=14,
    artifact_hashes_sha256=sha(index_raw), recorded_result=result, process_absence=absence,
    key_child_absent=True, key_parent_preserved=True, integer_QP_bits=605, ceiling=881,
    subject_execution=False, independent_numeric_recomputation=False, sweep_complete=False,
    recorded_parent_snapshot_active_case='test3.in',
    verifier_path=str(Path(__file__)), verifier_sha256=sha(Path(__file__).read_bytes()))
out.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
print(json.dumps(dict(path=str(out), sha256=sha(out.read_bytes()), indexed_files=22, sources=128, definitions=14, status=record['status'])))
