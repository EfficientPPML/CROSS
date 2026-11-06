"""Synthetic public proofs only; no retained private/HE payload read."""
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
import copy
import io
import json
import os
import tarfile
import pytest
import launch_authorization as gate


def tar_bytes(sources, *, duplicate=False, symlink=False):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for name, data in sources.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        if duplicate:
            name, data = next(iter(sources.items()))
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        if symlink:
            info = tarfile.TarInfo('foreign')
            info.type = tarfile.SYMTYPE
            info.linkname = '/private/not-read'
            archive.addfile(info)
    return buffer.getvalue()


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    assert not gate._UNCERTAIN_FDS
    evidence = tmp_path / 'evidence'
    (evidence / 'worker').mkdir(parents=True)
    parent = tmp_path / 'staged'
    parent.mkdir()
    sources = {n: ('synthetic:' + n).encode() for n in
               ('runtime/stage/key_contexts.py', 'cleanup.py', 'terminal.py', 'worker.py')}
    expected = {n: gate.digest(v) for n, v in sources.items()}
    monkeypatch.setattr(gate, 'SIZE0_SOURCE_COUNT', len(expected))
    monkeypatch.setattr(gate, 'SIZE0_SOURCE_MAP_SHA256', gate.digest(gate.canonical(expected)))
    (evidence / 'sources.tar.gz').write_bytes(tar_bytes(sources))
    hashes = ('context_id', 'key_report_sha256', 'key_source_archive_sha256', 'model_id',
              'model_report_sha256', 'model_manifest_sha256', 'stage3_report_sha256',
              'public_batch_sha256', 'input_digest', 'stage4_report_sha256', 'input_sha256',
              'fixture_report_sha256', 'reference_report_sha256', 'reference_source_archive_sha256')
    prerequisite = {name: gate.digest(name.encode()) for name in hashes}
    prerequisite.update(input_sha256=gate.INPUT_SHA256, fixture_report_sha256=gate.FIXTURE_SHA256)
    deps = dict(ready=True, key_contexts_sha256=expected['runtime/stage/key_contexts.py'],
                cleanup_sha256=expected['cleanup.py'], terminal_sha256=expected['terminal.py'])
    metrics = dict(passed=True, original_P2_logit_gate=True, prediction_agreement=True, finite_differences=True,
                   he_vs_official_max_abs=0.001, he_vs_frozen_approx_max_abs=0.00001,
                   he_vs_frozen_approx_nrmse=0.00003)
    readback = dict(sample_count=1, private_receipt_validated=True, result_permissions_0600=True,
                    public_batch_sha256=prerequisite['public_batch_sha256'], metrics=metrics,
                    decoded_result_sha256='d' * 64, completed_receipt_sha256='c' * 64)
    worker = {n: True for n in ('passed', 'operation_complete', 'actual_client_decode_started',
              'actual_client_decode_completed', 'readback_complete', 'source_files_unchanged',
              'keygen_encrypt_native_server_forbidden', 'terminal_scope_completed',
              'external_workspaces_never_removed')}
    worker.update({n: False for n in ('keygen', 'encryption', 'server_compute', 'model_forward')})
    worker.update(source_hashes_start=expected, source_hashes_end=expected, prerequisite=prerequisite,
                  reviewed_dependencies=deps, readback=readback, forbidden_attempts=[],
                  context_id=prerequisite['context_id'], model_id=prerequisite['model_id'])
    report = {n: True for n in ('passed', 'execute', 'actual_execution_attempted', 'actual_client_decode',
              'operation_complete', 'numeric_gates_passed', 'all_processes_reaped', 'source_files_unchanged',
              'source_archive_verified', 'cleanup_completed', 'terminal_scope_completed',
              'external_workspaces_never_removed', 'private_outputs_never_archived', 'private_result_workspace_retained')}
    report.update(source_hashes_start=expected, source_hashes_end=expected, execution_source_hashes_end=expected,
                  source_archive_sha256=gate.digest((evidence / 'sources.tar.gz').read_bytes()),
                  prerequisite=prerequisite, reviewed_dependencies=deps, readback=readback,
                  worker_process=dict(returncode=0, process_reaped=True, public_key_store_removed=True,
                                      owned_directory_handle_closed=True,
                                      result=dict(passed=True, operation_complete=True, worker_report_sha256='a' * 64)))
    case = dict(size=1, seed=42, count=100, input_sha256='1' * 64, target_order_sha256='2' * 64,
                fixture_report_sha256='3' * 64, reference_report_sha256='4' * 64,
                candidate_sha256=gate.CANDIDATE_SHA256, model_revision=gate.MODEL_REVISION)
    state = parent.stat()
    grant = dict(schema='cross-bert-ranked-root-launch-grant-v1', case_binding=case,
                 source_inventory_sha256='b' * 64, staged_parent=str(parent),
                 staged_parent_identity=[state.st_dev, state.st_ino], fresh_keys=True,
                 retained_key_mode=False, execute_authorized=True, size0_report_sha256='a' * 64,
                 size0_outer_session_id=1234, size0_outer_exit_code=0, size0_outer_session_reaped=True,
                 raw_public_key_bytes=gate.PUBLIC_KEY_BYTES, upload_bytes=100 * gate.UPLOAD_BYTES_PER_SAMPLE,
                 response_bytes=100 * gate.RESPONSE_BYTES_PER_SAMPLE,
                 private_model_margin_bytes=gate.MINIMUM_PRIVATE_MODEL_MARGIN_BYTES)
    grant['required_bytes'] = sum(grant[n] for n in ('raw_public_key_bytes', 'upload_bytes', 'response_bytes', 'private_model_margin_bytes'))
    f = SimpleNamespace(evidence=evidence, parent=parent, sources=sources, expected=expected, report=report,
                        worker=worker, case=case, grant=grant, storage=tmp_path / 'grant.json')

    def save():
        raw = gate.canonical(f.worker)
        (evidence / 'worker/worker_report.json').write_bytes(raw)
        f.report['worker_process']['result']['worker_report_sha256'] = gate.digest(raw)
        raw = gate.canonical(f.report)
        (evidence / 'report.json').write_bytes(raw)
        report_sha = gate.digest(raw)
        f.grant['size0_report_sha256'] = report_sha
        f.storage.write_bytes(gate.canonical(f.grant))
        return dict(size0_evidence=evidence, size0_report_sha256=report_sha, storage_path=f.storage,
                    storage_sha256=gate.digest(f.storage.read_bytes()), case_binding=f.case,
                    source_inventory_sha256='b' * 64, staged_parent=parent)
    f.save = save
    monkeypatch.setattr(gate.os, 'fstatvfs', lambda _: SimpleNamespace(f_bavail=1 << 50, f_frsize=1))
    yield f
    gate._UNCERTAIN_FDS.clear()


def test_fixed_ready_provenance_is_local_public():
    path = Path(__file__).parent / 'authorization_inputs/size0_ready_preflight.json'
    raw = path.read_bytes()
    assert gate.digest(raw) == gate.SIZE0_PREFLIGHT_SHA256
    report = gate.parse(raw)
    assert report['passed'] is True and report['source_count'] == gate.SIZE0_SOURCE_COUNT == 89
    assert gate.digest(gate.canonical(report['source_hashes_before'])) == gate.SIZE0_SOURCE_MAP_SHA256
    assert report['source_hashes_before'] == report['source_hashes_after'] == report['snapshot_hashes']


@pytest.mark.parametrize('size,count', [(1, 100), (2, 200), (3, 408)])
def test_all_counts_exact_public_authorization(fixture, size, count):
    f = fixture
    f.case.update(size=size, count=count)
    f.grant.update(upload_bytes=count * gate.UPLOAD_BYTES_PER_SAMPLE, response_bytes=count * gate.RESPONSE_BYTES_PER_SAMPLE)
    f.grant['required_bytes'] = sum(f.grant[n] for n in ('raw_public_key_bytes', 'upload_bytes', 'response_bytes', 'private_model_margin_bytes'))
    result = gate.authenticate(**f.save())
    assert result.required_bytes == gate.PUBLIC_KEY_BYTES + count * (gate.UPLOAD_BYTES_PER_SAMPLE + gate.RESPONSE_BYTES_PER_SAMPLE) + (8 << 30)
    assert result.storage_checked is True and result.outer_session_id_attested_by_root == 1234
    assert json.loads(result.case_binding_bytes)['count'] == count
    with pytest.raises(FrozenInstanceError):
        result.required_bytes = 1
    assert list(f.parent.iterdir()) == []


@pytest.mark.parametrize('name', ['passed', 'execute', 'actual_client_decode', 'operation_complete',
                                 'numeric_gates_passed', 'cleanup_completed', 'terminal_scope_completed',
                                 'private_result_workspace_retained', 'source_archive_verified'])
def test_report_boolean_alias_and_incomplete_rejected(fixture, name):
    fixture.report[name] = 1
    with pytest.raises(ValueError):
        gate.authenticate(**fixture.save())


@pytest.mark.parametrize('name', ['operation_complete', 'actual_client_decode_completed', 'readback_complete',
                                 'terminal_scope_completed', 'keygen_encrypt_native_server_forbidden'])
def test_worker_complete_flags_required(fixture, name):
    fixture.worker[name] = False
    with pytest.raises(ValueError):
        gate.authenticate(**fixture.save())


@pytest.mark.parametrize('field,value', [('returncode', True), ('returncode', 1), ('process_reaped', 1),
                                      ('owned_directory_handle_closed', False), ('public_key_store_removed', False)])
def test_worker_terminal_and_cleanup_required(fixture, field, value):
    fixture.report['worker_process'][field] = value
    with pytest.raises(ValueError):
        gate.authenticate(**fixture.save())


@pytest.mark.parametrize('field', ['original_P2_logit_gate', 'prediction_agreement', 'finite_differences'])
def test_original_numeric_and_prediction_gates_cannot_be_replaced(fixture, field):
    fixture.report['readback']['metrics'][field] = False
    with pytest.raises(ValueError):
        gate.authenticate(**fixture.save())


def test_diagnostic_nrmse_not_acceptance_gate(fixture):
    fixture.report['readback']['metrics']['he_vs_frozen_approx_nrmse'] = None
    assert gate.authenticate(**fixture.save()).storage_checked


@pytest.mark.parametrize('field,value', [('sample_count', True), ('sample_count', 2), ('private_receipt_validated', False),
                                      ('result_permissions_0600', 1), ('public_batch_sha256', '9' * 64)])
def test_receipt_and_same_batch_binding(fixture, field, value):
    fixture.report['readback'][field] = value
    with pytest.raises(ValueError):
        gate.authenticate(**fixture.save())


def test_mutually_rehashed_source_substitution_rejected(fixture):
    f = fixture
    f.sources['worker.py'] = b'substituted'
    altered = {n: gate.digest(v) for n, v in f.sources.items()}
    for target, fields in ((f.report, ('source_hashes_start', 'source_hashes_end', 'execution_source_hashes_end')),
                           (f.worker, ('source_hashes_start', 'source_hashes_end'))):
        for name in fields:
            target[name] = altered
    raw = tar_bytes(f.sources)
    (f.evidence / 'sources.tar.gz').write_bytes(raw)
    f.report['source_archive_sha256'] = gate.digest(raw)
    with pytest.raises(ValueError, match='externally reviewed'):
        gate.authenticate(**f.save())


@pytest.mark.parametrize('kind', ['duplicate', 'symlink'])
def test_archive_inventory_rejected(fixture, kind):
    raw = tar_bytes(fixture.sources, **{kind: True})
    (fixture.evidence / 'sources.tar.gz').write_bytes(raw)
    fixture.report['source_archive_sha256'] = gate.digest(raw)
    with pytest.raises(ValueError, match='archive'):
        gate.authenticate(**fixture.save())


def test_bad_external_size0_hash_reads_no_later_public_files(fixture, monkeypatch):
    args = fixture.save()
    args['size0_report_sha256'] = '0' * 64
    calls = []
    original = gate.read_public
    monkeypatch.setattr(gate, 'read_public', lambda p, *a: (calls.append(Path(p)), original(p, *a))[1])
    with pytest.raises(ValueError, match='external size0'):
        gate.authenticate(**args)
    assert calls == [fixture.evidence / 'report.json']


@pytest.mark.parametrize('field,value', [('fresh_keys', 1), ('retained_key_mode', True), ('execute_authorized', 1),
                                      ('size0_outer_exit_code', True), ('size0_outer_exit_code', 1),
                                      ('size0_outer_session_reaped', 1), ('size0_outer_session_id', True),
                                      ('raw_public_key_bytes', 1), ('upload_bytes', 1), ('response_bytes', 1),
                                      ('private_model_margin_bytes', (8 << 30) - 1), ('required_bytes', True),
                                      ('source_inventory_sha256', 'e' * 64), ('staged_parent_identity', [1, 2])])
def test_external_grant_exact_fields_and_accounting(fixture, field, value):
    fixture.grant[field] = value
    with pytest.raises(ValueError):
        gate.authenticate(**fixture.save())


def test_wrong_grant_case_or_hash_rejected(fixture):
    args = fixture.save()
    args['storage_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='external root'):
        gate.authenticate(**args)
    fixture.grant['case_binding'] = copy.deepcopy(fixture.case)
    fixture.grant['case_binding']['seed'] = 43
    with pytest.raises(ValueError, match='root grant'):
        gate.authenticate(**fixture.save())


def test_initial_storage_check_and_later_incremental_policy(fixture, monkeypatch):
    monkeypatch.setattr(gate.os, 'fstatvfs', lambda _: SimpleNamespace(f_bavail=1, f_frsize=1))
    args = fixture.save()
    with pytest.raises(OSError, match='insufficient'):
        gate.authenticate(**args)
    assert gate.authenticate(**args, check_storage=False).storage_checked is False
    with pytest.raises(ValueError, match='boolean'):
        gate.authenticate(**args, check_storage=0)


def test_parent_replacement_and_alias_rejected(fixture, monkeypatch):
    args = fixture.save()
    original = gate.os.fstatvfs
    def replace(fd):
        value = original(fd)
        fixture.parent.rename(fixture.parent.with_name('moved'))
        fixture.parent.mkdir()
        return value
    monkeypatch.setattr(gate.os, 'fstatvfs', replace)
    with pytest.raises(ValueError, match='moved'):
        gate.authenticate(**args)


def test_no_private_body_or_payload_reads(fixture, monkeypatch):
    args = fixture.save()
    allowed = {fixture.evidence / 'report.json', fixture.evidence / 'sources.tar.gz',
               fixture.evidence / 'worker/worker_report.json', fixture.storage}
    reads = []
    original = gate.read_public
    def observed(path, *rest):
        assert Path(path) in allowed
        reads.append(Path(path))
        return original(path, *rest)
    monkeypatch.setattr(gate, 'read_public', observed)
    gate.authenticate(**args)
    assert set(reads) == allowed


def test_public_fd_body_primary_preserved_and_uncertain_close_not_retried(monkeypatch):
    body = RuntimeError('body')
    closed = []
    monkeypatch.setattr(gate.os, 'open', lambda *a: 999999)
    monkeypatch.setattr(gate.os, 'fstat', lambda _: (_ for _ in ()).throw(body))
    def fail_close(fd):
        closed.append(fd)
        raise KeyboardInterrupt('close')
    monkeypatch.setattr(gate.os, 'close', fail_close)
    try:
        with pytest.raises(RuntimeError) as caught:
            gate.read_public('/public/synthetic')
        assert caught.value is body and closed == [999999] and body.__notes__
        with pytest.raises(RuntimeError, match='previously uncertain'):
            gate.read_public('/public/synthetic')
        assert closed == [999999]
    finally:
        gate._UNCERTAIN_FDS.clear()


def test_cleanup_only_cancellation_escapes(monkeypatch, tmp_path):
    path = tmp_path / 'public.json'
    path.write_bytes(b'{}')
    original = gate.os.close
    captured = []
    def fail_close(fd):
        captured.append(fd)
        raise SystemExit(7)
    monkeypatch.setattr(gate.os, 'close', fail_close)
    try:
        with pytest.raises(SystemExit) as error:
            gate.read_public(path)
        assert error.value.code == 7 and len(captured) == 1
    finally:
        for fd in captured:
            original(fd)  # This test's injected setter did not actually close it.
        gate._UNCERTAIN_FDS.clear()
