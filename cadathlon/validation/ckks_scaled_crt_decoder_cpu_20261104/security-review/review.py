"""Independent public/source readback; never imports the candidate or backend."""
import ast
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import tarfile
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
LAB = Path('/tmp/ckks-scaled-crt-decoder-candidate-v2-20261104')
RUN = LAB / 'source-bound-v1'
OLD = Path('/tmp/cross-optimized-twelve-20261103')
CORRECTION = Path('/tmp/ckks-scaled-crt-decoder-readback-20261104')
ENV = Path('/home/jianming_gatech/miniconda3/envs/jaxite/lib/python3.13/site-packages')

def sha(data):
    return hashlib.sha256(data).hexdigest()

def read(path):
    return path.read_bytes()

def document(path):
    return json.loads(read(path))

def hashes(root, names):
    return {name: sha(read(root / name)) for name in names}

def archive_hashes(path):
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        assert all(m.isfile() for m in members)
        assert len({m.name for m in members}) == len(members)
        return {m.name: sha(archive.extractfile(m).read()) for m in members}

def absent(pid, group=False):
    try:
        (os.killpg if group else os.kill)(pid, 0)
    except ProcessLookupError:
        return True
    return False

before = document(RUN / 'sources-before.json')
assert len(before) == 207
assert sha(read(RUN / 'sources-before.json')) == '9d9190b2a3c65f831f69fe30af6ece7b0b94f05bf83e9a21310ca32f99fda2aa'
assert sha(read(RUN / 'sources.tar.gz')) == '1bd06ce0d036698bffcbc06eabe4e131107b30aee5fcb7ba08768b9dd0ee0afb'
assert before == document(RUN / 'sources-after.json') == hashes(LAB, before)
assert before == hashes(RUN / 'execution', before) == archive_hashes(RUN / 'sources.tar.gz')
origin = document(LAB / 'source_origin.json')['source_hashes']
assert len(origin) == 199
assert origin == document(OLD / 'evidence/snapshot_manifest.json')['source_hashes']
assert origin == hashes(OLD / 'checkout', origin) == archive_hashes(OLD / 'evidence/sources.tar.gz')
changed = [name for name in origin if sha(read(LAB / 'checkout' / name)) != origin[name]]
assert changed == ['jaxite_word/ckks_ctx.py']
assert sha(read(LAB / 'before/ckks_ctx.py')) == origin[changed[0]]
assert before['checkout/jaxite_word/ckks_ctx.py'] == '9675b300d507fd9c6a3d03656c9df325d8da68a9331b4c92def0512784d0dc26'
old_ast = ast.parse(read(LAB / 'before/ckks_ctx.py'))
new_ast = ast.parse(read(LAB / 'checkout/jaxite_word/ckks_ctx.py'))
helpers = [node for node in new_ast.body if isinstance(node, ast.FunctionDef) and node.name == '_scale_centered_crt_coefficient']
assert len(helpers) == 1
new_ast.body.remove(helpers[0])
class Restore(ast.NodeTransformer):
    count = 0
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == '_scale_centered_crt_coefficient':
            assert len(node.args) == 3 and not node.keywords
            assert [arg.id for arg in node.args] in (['r', 'scale', 'scale_inv'], ['im', 'scale', 'scale_inv'])
            self.count += 1
            return ast.parse('float(' + node.args[0].id + ') * scale_inv', mode='eval').body
        return self.generic_visit(node)
restore = Restore()
new_ast = restore.visit(new_ast)
assert restore.count == 2
assert ast.dump(old_ast, include_attributes=False) == ast.dump(new_ast, include_attributes=False)

report = document(RUN / 'report.json')
worker = document(RUN / 'worker/report.json')
assert report['passed'] is True and report['tests'] == 35 and report['source_count'] == 207
assert report['source_manifest_sha256'] == sha(read(RUN / 'sources-before.json'))
assert report['source_archive_sha256'] == sha(read(RUN / 'sources.tar.gz'))
assert report['worker_report_sha256'] == sha(read(RUN / 'worker/report.json'))
assert report['session'] == 86719 and report['owner_reported_exit_code'] == 0 and report['owner_reported_reaped'] is True
assert report['reap_tool_chunk'] == '733ce7'
assert worker['passed'] is True and worker['pytest_exit_code'] == 0
assert worker['jax_backend'] == 'cpu' and worker['device_platforms'] == ['cpu']
assert worker['import_guard_denials'] == ['libtpu'] and worker['device_access_denials'] == []
assert len(worker['loaded_local_modules']) == 19
for row in worker['loaded_local_modules'].values():
    assert row['sha256'] == before[row['path']]
assert worker['loaded_local_modules']['ckks_ctx']['path'] == 'checkout/jaxite_word/ckks_ctx.py'
assert worker['loaded_local_modules']['jaxite_word.ckks_ctx']['sha256'] == worker['loaded_local_modules']['ckks_ctx']['sha256']
for key in ('ciphertext_decryption', 'model_forward', 'key_generation', 'encrypted_failure_recovered'):
    assert worker[key] is False
assert report['actual_toy5_throw_site_proved'] is False and report['actual_toy5_recovered'] is False
assert worker['pid'] == report['observed_pid'] == 3000289
assert worker['pgid'] == report['observed_pgid'] == 3000287
assert absent(worker['pid']) and absent(worker['pgid'], group=True)
cases = ET.parse(RUN / 'worker/tests.xml').getroot().findall('.//testcase')
assert len(cases) == 35 and all(not any(c.find(k) is not None for k in ('failure', 'error', 'skipped')) for c in cases)
names = Counter(row['name'] for row in worker['records'])
assert names == {'large_symmetric_constant': 4, 'zero_full_and_trim': 2, 'legacy_constant': 4, 'legacy_mixed': 3, 'asymmetry_rejected': 4}
geometry = document(LAB / 'public_geometry.json')
assert len(geometry['q_towers']) == 40 and math.prod(geometry['q_towers']).bit_length() == 1096
assert geometry['scale'].hex() == '0x1.ea3b603ea0001p+55'
assert geometry['source_security_sha256'] == sha(read(OLD / 'evidence/bert/test5/security_device.json'))
for row in worker['records']:
    if row['name'] == 'large_symmetric_constant':
        expected = float(Fraction(row['sign'] * 2**1024) / Fraction(geometry['scale'])).hex()
        assert row['minimum_hex'] == row['maximum_hex'] == row['independent_fraction_expected_hex'] == expected
        assert row['slots'] == row['degree'] // 2
    if row['name'] == 'asymmetry_rejected':
        assert row['unchanged_limit'] == 1/32

original_proof = read(LAB / 'proof.py').decode()
expected_correction = original_proof.replace('ROOT = Path(__file__).resolve().parent', "ROOT = Path('/tmp/ckks-scaled-crt-decoder-candidate-v2-20261104')")
expected_correction = expected_correction.replace("original_root = Path(original['source_root'])", "original_root = Path('/tmp/cross-optimized-twelve-20261103/checkout')")
expected_correction = expected_correction.replace("'original199_current_unchanged': True, 'current_execution_archive_equal': True,", "'frozen199_current_unchanged': True, 'frozen199_root': str(original_root), 'current_execution_archive_equal': True,")
assert read(CORRECTION / 'readback.py').decode() == expected_correction
assert read(CORRECTION / 'proof-original.py.txt') == read(LAB / 'proof.py')
assert 'FileNotFoundError' in read(RUN / 'finalization-first-attempt.log').decode()
assert '/home/jianming_gatech/CROSS/validation_sweep.py' in read(RUN / 'finalization-first-attempt.log').decode()
correction = document(CORRECTION / 'report.json')
assert correction['passed'] is True and correction['measured207_unchanged'] is True
assert correction['measured_report_sha256'] == sha(read(RUN / 'report.json'))
assert document(CORRECTION / 'sources-before.json') == document(CORRECTION / 'sources-after.json')

test_source = read(LAB / 'test_real_decode.py').decode()
assert test_source.index('sys.meta_path.insert(0, CPU_IMPORT_GUARD)') < test_source.index('import numpy as np')
assert test_source.index('sys.addaudithook(device_audit)') < test_source.index('import numpy as np')
assert "assert not _jax.pjrt_plugin_loaded('tpu')" in test_source
installed_paths = ['jax/__init__.py', 'jax/_src/cloud_tpu_init.py', 'libtpu/__init__.py']
installed_hashes = hashes(ENV, installed_paths)
assert "import libtpu" in read(ENV / installed_paths[1]).decode()
assert "os.environ['TPU_LIBRARY_PATH'] = get_library_path()" in read(ENV / installed_paths[2]).decode()

assert before == hashes(LAB, before) == hashes(RUN / 'execution', before)
assert origin == hashes(OLD / 'checkout', origin)
result = {
    'schema': 'ckks-scaled-crt-independent-source-review-v1',
    'reviewed_at_utc': datetime.now(timezone.utc).isoformat(), 'passed': True,
    'scope': 'Public source and terminal evidence only; no backend import or numerical rerun by reviewer',
    'source_count': 207, 'source_manifest_sha256': sha(read(RUN / 'sources-before.json')),
    'source_archive_sha256': sha(read(RUN / 'sources.tar.gz')),
    'report_sha256': sha(read(RUN / 'report.json')),
    'worker_report_sha256': sha(read(RUN / 'worker/report.json')),
    'source_hashes': before,
    'frozen199_current_archive_equal': True, 'runtime_changed_files': changed,
    'runtime_delta': 'One conversion helper plus exactly two calls; restoring them gives identical whole-module AST',
    'current_execution_before_after_archive_equal': True,
    'tests': 35, 'numerical_tests_rerun': False, 'loaded_local_module_names': 19,
    'worker_pid_absent': True, 'worker_group_absent': True,
    'authoritative_reap': {'scope': 'Peer-owned tool observation, independently checked PID/group absence', 'session': 86719, 'exit_code': 0, 'tool_chunk': '733ce7'},
    'cpu_evidence': {'backend': 'cpu', 'device_platforms': ['cpu'], 'optional_libtpu_import_denials': 1, 'audited_device_access_denials': 0,
                     'plugin_absence_scope': 'Asserted in the first CPU/source test; not separately sampled again in final report'},
    'guard_scope': 'Python import/audit guards precede backend imports; not an OS sandbox or proof against arbitrary native loads. Crypto entry guards install after module import; selected tests call clear CRT/FFT only.',
    'original_development_libtpu_scope': 'Installed source shows JAX cloud initialization can import libtpu for version/library-path metadata and mutate TPU_LIBRARY_PATH before CPU backend selection. Original module-presence failure alone does not prove TPU execution or plugin loading; no original plugin-state trace retained.',
    'installed_source_review_hashes': installed_hashes,
    'separate_readback_correction': {'path': str(CORRECTION), 'sha256': sha(read(CORRECTION / 'readback.py')), 'scope': 'Only candidate/frozen source root resolution and report field label; original failed finalization log retained, measured207 unchanged'},
    'limitations': ['Installed dependencies are not a hermetic source closure.', 'No retained B5 ciphertext was read; actual throw-site remains an inference.', 'No genuine HE precision, model recovery, key generation, TPU execution, or runtime deployment claim.'],
}
(HERE / 'review.json').write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n')
print(json.dumps({'passed': True, 'source_count': 207, 'tests': 35, 'review_sha256': sha(read(HERE / 'review.json'))}, sort_keys=True))
