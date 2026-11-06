"""Offline public artifact/source verifier; no numerical imports or execution."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import tarfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent

def sha(data):
    return hashlib.sha256(data).hexdigest()

def load(name):
    return json.loads((ROOT / name).read_bytes())

def tar_contents(name):
    with tarfile.open(ROOT / name, 'r:gz') as archive:
        members = archive.getmembers()
        assert all(m.isfile() for m in members)
        assert len(set(m.name for m in members)) == len(members)
        return {m.name: archive.extractfile(m).read() for m in members}

def xml_counts(name):
    cases = ET.parse(ROOT / name).getroot().findall('.//testcase')
    return len(cases), sum(c.find('failure') is not None for c in cases), sum(c.find('error') is not None for c in cases), sum(c.find('skipped') is not None for c in cases)

def main(current):
    index = load('files.sha256.json')
    actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and p.name != 'files.sha256.json'}
    assert actual == set(index)
    assert all(sha((ROOT / name).read_bytes()) == digest for name, digest in index.items())
    expected = load('proof/sources-before.json')
    assert len(expected) == 207 and expected == load('proof/sources-after.json')
    bodies = tar_contents('proof/sources.tar.gz')
    assert {n: sha(b) for n, b in bodies.items()} == expected
    report = load('proof/report.json')
    assert report['passed'] is True and report['tests'] == 35 and report['source_count'] == 207
    assert report['source_archive_sha256'] == sha((ROOT / 'proof/sources.tar.gz').read_bytes())
    assert report['source_manifest_sha256'] == sha((ROOT / 'proof/sources-before.json').read_bytes())
    assert report['worker_report_sha256'] == sha((ROOT / 'proof/worker/report.json').read_bytes())
    worker = load('proof/worker/report.json')
    assert worker['passed'] is True and worker['pytest_exit_code'] == 0
    assert worker['jax_backend'] == 'cpu' and worker['device_platforms'] == ['cpu']
    assert worker['import_guard_denials'] == ['libtpu'] and worker['device_access_denials'] == []
    assert worker['real_crt'] is True and worker['real_fft'] is True
    for key in ['ciphertext_decryption', 'model_forward', 'key_generation', 'encrypted_failure_recovered']:
        assert worker[key] is False
    assert all(r['sha256'] == expected[r['path']] for r in worker['loaded_local_modules'].values())
    assert xml_counts('proof/worker/tests.xml') == (35, 0, 0, 0)
    assert xml_counts('history/development-v1/tests.xml') == (35, 1, 0, 0)
    assert xml_counts('history/development-v2/tests.xml') == (35, 0, 0, 0)
    assert sum(r.get('slots', 0) for r in worker['records']) == 196744
    owner = load('owner.json')
    assert owner['session'] == report['session'] == 86719
    assert owner['exit_code'] == 0 and owner['reaped'] is True
    assert owner['authoritative_tool_reap_chunk'] == '733ce7'
    assert owner['pid'] == worker['pid'] == 3000289 and owner['pgid'] == worker['pgid'] == 3000287
    assert owner['wrapper_pid'] is None and owner['live_external_pid_row_captured'] is False
    old = load('original199/snapshot_manifest.json')['source_hashes']
    original = tar_contents('original199/sources.tar.gz')
    assert len(old) == len(original) == 199
    assert {n: sha(b) for n, b in original.items()} == old
    changed = [n for n in old if sha(bodies['checkout/' + n]) != old[n]]
    assert changed == ['jaxite_word/ckks_ctx.py']
    assert bodies['before/ckks_ctx.py'] == original[changed[0]]
    tree_old = ast.parse(original[changed[0]])
    tree_new = ast.parse(bodies['checkout/' + changed[0]])
    helper = [n for n in tree_new.body if isinstance(n, ast.FunctionDef) and n.name == '_scale_centered_crt_coefficient']
    assert len(helper) == 1
    tree_new.body.remove(helper[0])
    class Undo(ast.NodeTransformer):
        count = 0
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == '_scale_centered_crt_coefficient':
                assert not node.keywords and [a.id for a in node.args] in [['r', 'scale', 'scale_inv'], ['im', 'scale', 'scale_inv']]
                self.count += 1
                return ast.BinOp(left=ast.Call(func=ast.Name(id='float', ctx=ast.Load()), args=[node.args[0]], keywords=[]), op=ast.Mult(), right=node.args[2])
            return self.generic_visit(node)
    undo = Undo()
    tree_new = undo.visit(tree_new)
    assert undo.count == 2 and ast.dump(tree_old, include_attributes=False) == ast.dump(tree_new, include_attributes=False)
    geo = json.loads(bodies['public_geometry.json'])
    raw = (ROOT / 'original199/bert-test5-security_device.json').read_bytes()
    assert sha(raw) == geo['source_security_sha256']
    event = json.loads(raw)['events'][-1]
    ct = event['ciphertexts'][0]
    assert geo['scale'] == ct['scale'] and geo['q_towers'] == event['actual_rings'][0]['q_towers'][:40]
    correction = load('readback-correction/report.json')
    assert correction['passed'] is True and correction['measured207_unchanged'] is True
    assert correction['measured_report_sha256'] == sha((ROOT / 'proof/report.json').read_bytes())
    assert (ROOT / 'readback-correction/proof-original.py.txt').read_bytes() == bodies['proof.py']
    if current:
        lab = Path('/tmp/ckks-scaled-crt-decoder-candidate-v2-20261104')
        execution = lab / 'source-bound-v1/execution'
        for parent in [lab, execution]:
            assert all(sha((parent / n).read_bytes()) == h for n, h in expected.items())
        frozen = Path('/tmp/cross-optimized-twelve-20261103/checkout')
        assert all(sha((frozen / n).read_bytes()) == h for n, h in old.items())
    print(json.dumps({'passed': True, 'sources': 207, 'tests': 35, 'real_crt_fft': True,
        'original_failure_recovered': False, 'current_checked': current, 'artifact_files': len(index)}, sort_keys=True))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--current', action='store_true')
    main(parser.parse_args().current)
