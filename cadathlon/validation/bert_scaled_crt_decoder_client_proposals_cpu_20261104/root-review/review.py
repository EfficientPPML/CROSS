"""Read-only public source proposal verification; imports no candidate modules."""
import ast
import hashlib
import json
import math
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/tmp/bert-scaled-crt-decoder-client-proposals-20261104')
PARENT = Path('/tmp/bert-client-clean-launch-source-proposals-20261104')
OUT = Path('/tmp/bert-decoder-client-proposal-root-review-20261104')
OLD = 'bc978cb7412b55416f7ed634a689ae616a263b0f47fd9eef996ba244f2505db6'
NEW = '9675b300d507fd9c6a3d03656c9df325d8da68a9331b4c92def0512784d0dc26'
PINS = {
    'original122': (122, '49ff2d27e11ce773d27baec448fc1bbb0672fbf2ff6e9e8b94ca14ceda126a56', '6e3e30ad643921e9546be144e7da0dce485e9dd1be088f31f62f35cd2aa4c5f9', '20fb4e7b818ad26fa0e69037c2558b017f995f7e5401c74390abca31fa84ec6f'),
    'combined125': (125, 'f5fd949507f210a7b40b0b0cf508268f6a625911c7dcb6526cb0870a9dc269d7', '6b2b1985c1f9e7298bf47cf8559fd8a2122b0ad854289fed8674c2c05e8475a0', '6fd87ede77b605bac56c53f8b636d76aeacc1fc9160749e9c056a476e56d00fb'),
}
sha = lambda b: hashlib.sha256(b).hexdigest()
dump = lambda node: ast.dump(node, include_attributes=False)

def guard_without_overlay(tree):
    found = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'OVERLAYS' for t in node.targets):
            found.append(ast.literal_eval(node.value))
            node.value = ast.Constant(value=None)
    assert len(found) == 1
    return found[0], dump(tree)

class ReverseDecoder(ast.NodeTransformer):
    calls = 0
    helpers = 0

    def visit_FunctionDef(self, node):
        if node.name == '_scale_centered_crt_coefficient':
            self.helpers += 1
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == '_scale_centered_crt_coefficient':
            assert len(node.args) == 3 and not node.keywords
            assert dump(node.args[1]) == dump(ast.Name(id='scale', ctx=ast.Load()))
            assert dump(node.args[2]) == dump(ast.Name(id='scale_inv', ctx=ast.Load()))
            self.calls += 1
            return ast.BinOp(left=ast.Call(func=ast.Name(id='float', ctx=ast.Load()), args=[node.args[0]], keywords=[]), op=ast.Mult(), right=node.args[2])
        return self.generic_visit(node)

report_bytes = (ROOT/'report.json').read_bytes()
assert sha(report_bytes) == '2251a4bc580ba2f008a638fca74803d77a48fa666e12c25a1b93f0b4794b2e97'
report = json.loads(report_bytes)
assert report['source_only'] is True
for key in ('actual_client', 'backend_imported', 'full_preflight_executed', 'launch_authorized', 'private_payload_read'):
    assert report[key] is False
rows = {}
for name, (count, parent_pin, map_pin, tar_pin) in PINS.items():
    p = ROOT/name
    manifest_bytes = (p/'source_manifest.json').read_bytes()
    parent_bytes = (p/'parent_source_manifest.json').read_bytes()
    assert sha(manifest_bytes) == map_pin and sha(parent_bytes) == parent_pin
    manifest, parent = json.loads(manifest_bytes), json.loads(parent_bytes)
    assert len(manifest) == count and set(manifest) == set(parent)
    assert sha((p/'sources.tar.gz').read_bytes()) == tar_pin
    changed = {n for n in manifest if manifest[n] != parent[n]}
    assert changed == {'README_CLIENT.md', 'client_source_guard.py', 'test_client_source_guard.py', 'runtime/jaxite_word/ckks_ctx.py'}
    for n, h in manifest.items():
        assert sha((p/'source'/n).read_bytes()) == h
        assert sha((PARENT/name/'source'/n).read_bytes()) == parent[n]
    with tarfile.open(p/'sources.tar.gz') as archive:
        members = archive.getmembers()
        assert len(members) == count and all(m.isfile() for m in members)
        assert {m.name for m in members} == set(manifest)
        for m in members:
            assert sha(archive.extractfile(m).read()) == manifest[m.name]
    decoder_path = 'runtime/jaxite_word/ckks_ctx.py'
    assert parent[decoder_path] == OLD and manifest[decoder_path] == NEW
    reverse = ReverseDecoder()
    reverted = reverse.visit(ast.parse((p/'source'/decoder_path).read_bytes()))
    assert reverse.helpers == 1 and reverse.calls == 2
    assert dump(reverted) == dump(ast.parse((PARENT/name/'source'/decoder_path).read_bytes()))
    old_overlay, old_guard = guard_without_overlay(ast.parse((PARENT/name/'source/client_source_guard.py').read_bytes()))
    new_overlay, new_guard = guard_without_overlay(ast.parse((p/'source/client_source_guard.py').read_bytes()))
    assert new_guard == old_guard
    assert new_overlay == {**old_overlay, decoder_path: NEW}
    assert decoder_path not in old_overlay
    rows[name] = {'source_count': count, 'changed_count': 4, 'unchanged_count': count-4, 'source_manifest_sha256': map_pin, 'source_archive_sha256': tar_pin, 'current_parent_archive_exact': True, 'guard_delta_only_explicit_overlay': True, 'whole_decoder_AST_reversal_exact': True, 'tests_run': False}

q = [2146959361, 860618753, 983826433, 862978049, 998244353, 1118568449]
for name in PINS:
    profile = json.loads((ROOT/name/'source/runtime/stage/p60_profile_and_refresh.json').read_bytes())
    assert profile['profile']['q'][:6] == q
product = math.prod(q)
assert report['final_BERT_response']['q'] == q
assert str(product) == report['final_BERT_response']['product']
assert product.bit_length() == 181 and (product//2).bit_length() == 180
assert math.isfinite(float(product//2))
OUT.mkdir(exist_ok=False)
result = {'schema': 'bert-decoder-client-source-review-v1', 'reviewed_utc': datetime.now(timezone.utc).isoformat(), 'passed': True, 'scope': 'source/hash/AST/public-integer readback only; no imports, tests, context, key, ciphertext or model access', 'clients': rows, 'final_q6_product_bits': 181, 'final_centered_coefficient_max_bits': 180, 'original_clients_remain_launch_defaults': True, 'new_client_preflight': False, 'new_launch_grant': False, 'actual_BERT_result': False, 'proposal_report_sha256': sha(report_bytes)}
(OUT/'review.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
(OUT/'review.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(result, sort_keys=True))
