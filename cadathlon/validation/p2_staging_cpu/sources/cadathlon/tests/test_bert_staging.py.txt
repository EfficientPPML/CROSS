"""P2 staged import/provenance checks; no encryption or backend initialization."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'cadathlon/agent_pset/p2_bert/src'


@pytest.fixture
def staging(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SOURCE))
    monkeypatch.delitem(sys.modules, 'benchmark', raising=False)
    spec = importlib.util.spec_from_file_location('bert_staging_under_test', SOURCE / 'stage_benchmark.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    upstream = tmp_path / 'upstream'
    (upstream / 'harness').mkdir(parents=True)
    (upstream / 'harness/run_submission.py').write_text('# fake harness for import-only staging tests\n')
    (upstream / 'LICENSE.md').write_text('upstream license fixture\n')
    manifest = {'harness/run_submission.py': hashlib.sha256((upstream / 'harness/run_submission.py').read_bytes()).hexdigest()}
    monkeypatch.setattr(module, 'HARNESS_SHA256', manifest)
    monkeypatch.setattr(module, 'check_upstream', lambda root: {'commit': 'test-only', 'harness_files': 1})
    return module, upstream, tmp_path / 'staged'


def command(destination, *arguments, script='runner.py'):
    # Any accidental backend initialization must fail. Imports and interface
    # inspection do not need JAX devices or actual CROSS client keys.
    env = dict(os.environ, PYTHONPATH='', CROSS_ROOT=str(ROOT),
               JAX_PLATFORMS='bert_staging_test_no_backend')
    return subprocess.run([sys.executable, script, *arguments],
                          cwd=destination/'submissions', env=env, capture_output=True, text=True)


def test_staged_runtime_is_complete_byte_identical_and_recorded(staging):
    stage, upstream, destination = staging
    original = (upstream/'harness/run_submission.py').read_bytes()
    stage.stage(upstream, destination)
    manifest = json.loads((destination/'cross_staging_manifest.json').read_text())
    assert manifest['common_directory'] == 'submissions/_cross_common'
    common = destination/manifest['common_directory']
    names = set(manifest['common_sha256'])
    assert {'support.py','transformer_runner.py','encrypted_reference.py','encrypted_compilation.py',
            'encrypted_primitives.py','encrypted_bounded_transformer.py'} <= names
    for name, expected in manifest['common_sha256'].items():
        assert hashlib.sha256((common/name).read_bytes()).hexdigest() == expected
        assert (common/name).read_bytes() == (ROOT/'cadathlon/common'/name).read_bytes()
        for node in ast.walk(ast.parse((common/name).read_text())):
            modules = ([node.module] if isinstance(node, ast.ImportFrom) else
                       [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            for module in modules:
                dependency = module.split('.')[0]+'.py' if module else None
                if dependency and (ROOT/'cadathlon/common'/dependency).is_file():
                    assert dependency in names, f'{name} needs {dependency}'
    assert (upstream/'harness/run_submission.py').read_bytes() == original
    assert 'p2_runtime.py' in manifest['starter_sha256']


@pytest.mark.parametrize('adapter', ['bert_reference.py','bert_packed_reference.py','bert_bounded_reference.py'])
def test_staged_adapters_import_but_do_not_claim_missing_full_stages(staging, adapter):
    stage, upstream, destination = staging
    stage.stage(upstream, destination)
    result = command(destination, '--preflight', '--implementation', adapter)
    assert result.returncode == 1, result.stdout+result.stderr
    report = json.loads(result.stdout)
    assert report['implementation_status'] == 'experimental-unvalidated'
    assert report['encryption_validated'] is False and report['ready_for_execution'] is False
    assert 'create_session' not in report['missing_interfaces']
    assert set(report['missing_interfaces']) == {'client_key_generation','server_preprocess_model',
        'client_encode_encrypt_input','server_encrypted_compute','client_decrypt_decode'}


@pytest.mark.parametrize('damage', ['changed','missing','no-manifest'])
def test_staged_provenance_failure_never_falls_back_to_checkout(staging, damage):
    stage, upstream, destination = staging
    stage.stage(upstream, destination)
    helper = destination/'submissions/_cross_common/encrypted_primitives.py'
    if damage == 'changed':
        helper.write_text(helper.read_text()+'\n# stale staged helper\n')
    elif damage == 'missing':
        helper.unlink()
    else:
        (destination/'cross_staging_manifest.json').unlink()
    result = command(destination, '--preflight', '--implementation', 'bert_reference.py')
    assert result.returncode != 0
    assert 'restage the benchmark' in result.stderr
    assert 'experimental-unvalidated' not in result.stdout


def test_file_stage_imports_local_shared_runtime_without_private_paths(staging):
    stage, upstream, destination = staging
    stage.stage(upstream, destination)
    # An import-only handler demonstrates the file process's dependency path.
    # It does not emit ciphertexts or pretend to implement BERT.
    (destination/'submissions/bert_he.py').write_text('''import json
import encrypted_reference

def server_preprocess_model(paths):
    print(json.dumps({'import_only': True, 'helper': encrypted_reference.__file__, 'paths': sorted(paths)}))
''')
    env = dict(os.environ, PYTHONPATH='', CROSS_ROOT=str(ROOT), JAX_PLATFORMS='bert_staging_test_no_backend')
    result = subprocess.run([sys.executable,'submissions/server_preprocess_model.py','0'],
                            cwd=destination, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    report = json.loads(result.stdout)
    assert report['import_only'] is True
    assert Path(report['helper']) == destination/'submissions/_cross_common/encrypted_reference.py'
    assert {'client_private','test_input','preprocessed_input','decrypted_results'}.isdisjoint(report['paths'])


def test_missing_declared_helper_fails_before_destination_exists(staging, tmp_path, monkeypatch):
    stage, upstream, destination = staging
    common = tmp_path/'incomplete-common'
    common.mkdir()
    shutil.copyfile(ROOT/'cadathlon/common/package.py', common/'package.py')
    monkeypatch.setattr(stage, 'common_directory', lambda source_file: common)
    with pytest.raises(FileNotFoundError, match='Missing declared archive dependency'):
        stage.stage(upstream, destination)
    assert not destination.exists()


@pytest.mark.parametrize('changed', [False, True])
def test_checkout_runner_cannot_bypass_staged_adapter_provenance(staging, changed):
    stage, upstream, destination = staging
    stage.stage(upstream, destination)
    if changed:
        helper = destination/'submissions/_cross_common/encrypted_primitives.py'
        helper.write_text(helper.read_text()+'\n# changed staged dependency\n')
    env = dict(os.environ, PYTHONPATH='', CROSS_ROOT=str(ROOT), JAX_PLATFORMS='bert_staging_test_no_backend')
    result = subprocess.run([sys.executable, str(SOURCE/'runner.py'), '--preflight',
        '--implementation', str(destination/'submissions/bert_reference.py')],
        cwd=SOURCE, env=env, capture_output=True, text=True)
    assert result.returncode == 1
    report = json.loads(result.stdout)
    expected = 'Staged P2 helper changed' if changed else 'use the matching staged runner'
    assert expected in report['error']
    assert report['passed'] is False


def test_staged_runner_rejects_an_adapter_from_another_helper_root(staging):
    stage, upstream, destination = staging
    stage.stage(upstream, destination)
    result = command(destination, '--preflight', '--implementation', str(SOURCE/'bert_reference.py'))
    assert result.returncode == 1
    assert 'use the matching staged runner' in json.loads(result.stdout)['error']
