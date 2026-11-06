"""Public launch gate; completed size0 plus an externally pinned root grant.

This module neither creates a grant nor opens any private input/reference,
matrix, key or ciphertext body. Later phases may skip the initial full-store
free-space test only after their caller authenticates predecessor stages and
checks that phase's incremental allocation separately.
"""
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import hashlib
import io
import json
import math
import os
import stat
import tarfile

SIZE0_SOURCE_COUNT = 89
SIZE0_SOURCE_MAP_SHA256 = 'fc331420c0e7cfb61301dd628bc552feab585434fc3bef2690283526e6887959'
SIZE0_PREFLIGHT_SHA256 = '274ec333531f42e0ca76cd8b6ec2ef508dcf0dd9c1366ac9350c3e52f093c424'
CANDIDATE_SHA256 = '4e2277beea7189aa1d77b168c500c8a175f6c630afa520fcc6b38cb5a179cff9'
FIXTURE_SHA256 = '2218282c9d4ef468a448d00d6e7afeebc9c3a9fa0e9f3834227afe299ea28e10'
INPUT_SHA256 = 'c30ffc79d8150cf6cf3d544bca23a07150d8a0d0f7b7924788e91df3aa55991f'
MODEL_REVISION = 'f150c1d609d1e50dd5e2e5408661cfac8339277c'
COUNTS = {1: 100, 2: 200, 3: 408}
PUBLIC_KEY_BYTES = 125380363136
UPLOAD_BYTES_PER_SAMPLE = 838860800
RESPONSE_BYTES_PER_SAMPLE = 6291456
MINIMUM_PRIVATE_MODEL_MARGIN_BYTES = 8 << 30
MAX_PUBLIC_BYTES = 16 << 20
_UNCERTAIN_FDS = []


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha_field(value):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('externally authenticated SHA256 required')
    return value


def parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate public evidence field')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite public evidence')))


def _close(fd, primary):
    try:
        os.close(fd)
    except BaseException as error:
        # No blind retry: close may already have retired/reassigned the fd.
        _UNCERTAIN_FDS.append(fd)
        if primary is not None:
            primary.add_note('public authorization descriptor close: ' + repr(error))
        else:
            raise


def read_public(path, maximum=MAX_PUBLIC_BYTES):
    if _UNCERTAIN_FDS:
        raise RuntimeError('public descriptor cleanup previously uncertain')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    primary = None
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise ValueError('bounded regular public evidence required')
        parts, total = [], 0
        while total <= maximum:
            block = os.read(fd, min(1 << 20, maximum + 1 - total))
            if not block:
                break
            parts.append(block)
            total += len(block)
        raw = b''.join(parts)
        after = os.fstat(fd)
        state = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if len(raw) != before.st_size or state(before) != state(after):
            raise ValueError('public evidence changed or exceeded its byte bound')
        return raw
    except BaseException as error:
        primary = error
        raise
    finally:
        _close(fd, primary)


def _true(value, names):
    if type(value) is not dict or any(value.get(n) is not True for n in names):
        raise ValueError('exact complete public proof flags required')
    if value.get('error') or value.get('errors') or value.get('cleanup_errors') or value.get('collector_cleanup_errors'):
        raise ValueError('failed public proof cannot authorize execution')


def _source_archive(raw, expected):
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        members = archive.getmembers()
        if (len(members) != len(expected) or len({m.name for m in members}) != len(members)
                or {m.name for m in members} != set(expected)
                or any(not m.isfile() or m.size < 0 or m.size > 8 << 20 for m in members)
                or sum(m.size for m in members) > MAX_PUBLIC_BYTES):
            raise ValueError('bounded exact reviewed size0 source archive required')
        if {m.name: digest(archive.extractfile(m).read()) for m in members} != expected:
            raise ValueError('size0 source archive bytes differ')


def _size0(evidence, expected_report_sha256):
    expected_report_sha256 = sha_field(expected_report_sha256)
    root = Path(evidence)
    raw = read_public(root / 'report.json')
    if digest(raw) != expected_report_sha256:
        raise ValueError('external size0 terminal report differs')
    report = parse(raw)
    _true(report, ('passed', 'execute', 'actual_execution_attempted', 'actual_client_decode',
                  'operation_complete', 'numeric_gates_passed', 'all_processes_reaped',
                  'source_files_unchanged', 'source_archive_verified', 'cleanup_completed',
                  'terminal_scope_completed', 'external_workspaces_never_removed',
                  'private_outputs_never_archived', 'private_result_workspace_retained'))
    expected = report.get('source_hashes_start')
    if type(expected) is not dict or len(expected) != SIZE0_SOURCE_COUNT:
        raise ValueError('complete reviewed size0 source inventory required')
    for name, hash_ in expected.items():
        path = PurePosixPath(name) if type(name) is str else None
        if path is None or path.is_absolute() or '..' in path.parts or str(path) != name:
            raise ValueError('canonical relative size0 source path required')
        sha_field(hash_)
    if digest(canonical(expected)) != SIZE0_SOURCE_MAP_SHA256:
        raise ValueError('size0 sources do not match externally reviewed ready closure')
    if any(report.get(n) != expected for n in ('source_hashes_end', 'execution_source_hashes_end')):
        raise ValueError('size0 source closure changed')
    source = read_public(root / 'sources.tar.gz')
    source_sha = sha_field(report.get('source_archive_sha256'))
    if digest(source) != source_sha:
        raise ValueError('size0 archive digest differs')
    _source_archive(source, expected)
    process = report.get('worker_process')
    _true(process, ('process_reaped', 'public_key_store_removed', 'owned_directory_handle_closed'))
    if type(process.get('returncode')) is not int or process['returncode'] != 0:
        raise ValueError('actual size0 worker exit zero required')
    result = process.get('result')
    _true(result, ('passed', 'operation_complete'))
    worker_raw = read_public(root / 'worker/worker_report.json')
    if digest(worker_raw) != sha_field(result.get('worker_report_sha256')):
        raise ValueError('size0 worker report binding differs')
    worker = parse(worker_raw)
    _true(worker, ('passed', 'operation_complete', 'actual_client_decode_started',
                  'actual_client_decode_completed', 'readback_complete', 'source_files_unchanged',
                  'keygen_encrypt_native_server_forbidden', 'terminal_scope_completed',
                  'external_workspaces_never_removed'))
    if (worker.get('forbidden_attempts') or any(worker.get(n) is not False for n in
            ('keygen', 'encryption', 'server_compute', 'model_forward'))
            or any(worker.get(n) != expected for n in ('source_hashes_start', 'source_hashes_end'))):
        raise ValueError('actual size0 client phase/source proof required')
    prerequisite = report.get('prerequisite')
    dependencies = report.get('reviewed_dependencies')
    _true(dependencies, ('ready',))
    for field, source in (('key_contexts_sha256', 'runtime/stage/key_contexts.py'),
                          ('cleanup_sha256', 'cleanup.py'), ('terminal_sha256', 'terminal.py')):
        if dependencies.get(field) != expected.get(source):
            raise ValueError('reviewed size0 cleanup dependency differs')
    fields = {'context_id', 'key_report_sha256', 'key_source_archive_sha256', 'model_id',
              'model_report_sha256', 'model_manifest_sha256', 'stage3_report_sha256',
              'public_batch_sha256', 'input_digest', 'stage4_report_sha256', 'input_sha256',
              'fixture_report_sha256', 'reference_report_sha256', 'reference_source_archive_sha256'}
    if type(prerequisite) is not dict or set(prerequisite) != fields:
        raise ValueError('complete same-input size0 prerequisite chain required')
    for field in fields:
        sha_field(prerequisite[field])
    if (prerequisite['input_sha256'] != INPUT_SHA256 or prerequisite['fixture_report_sha256'] != FIXTURE_SHA256
            or canonical(worker.get('prerequisite')) != canonical(prerequisite)
            or worker.get('context_id') != prerequisite['context_id'] or worker.get('model_id') != prerequisite['model_id']
            or canonical(worker.get('reviewed_dependencies')) != canonical(report.get('reviewed_dependencies'))):
        raise ValueError('fixed size0 input/target/context binding differs')
    readback = report.get('readback')
    _true(readback, ('private_receipt_validated', 'result_permissions_0600'))
    if (type(readback.get('sample_count')) is not int or readback['sample_count'] != 1
            or readback.get('public_batch_sha256') != prerequisite['public_batch_sha256']
            or canonical(worker.get('readback')) != canonical(readback)):
        raise ValueError('complete same-input size0 private receipt summary required')
    for name in ('decoded_result_sha256', 'completed_receipt_sha256'):
        sha_field(readback.get(name))
    metrics = readback.get('metrics')
    _true(metrics, ('passed', 'original_P2_logit_gate', 'prediction_agreement', 'finite_differences'))
    for name in ('he_vs_official_max_abs', 'he_vs_frozen_approx_max_abs'):
        value = metrics.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('finite size0 metric aggregates required')
    # nRMSE is a diagnostic, never an additional or replacement acceptance gate.
    value = metrics.get('he_vs_frozen_approx_nrmse')
    if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
        raise ValueError('finite-or-null diagnostic nRMSE required')
    return source_sha


def _case(value):
    fields = {'size', 'seed', 'count', 'input_sha256', 'target_order_sha256',
              'fixture_report_sha256', 'reference_report_sha256', 'candidate_sha256', 'model_revision'}
    if (type(value) is not dict or set(value) != fields or type(value['size']) is not int
            or value['size'] not in COUNTS or type(value['seed']) is not int or value['seed'] not in (42, 43, 44)
            or type(value['count']) is not int or value['count'] != COUNTS[value['size']]
            or value['candidate_sha256'] != CANDIDATE_SHA256 or value['model_revision'] != MODEL_REVISION):
        raise ValueError('one exact authenticated ranked case required')
    for name in fields - {'size', 'seed', 'count', 'model_revision'}:
        sha_field(value[name])
    return canonical(value)


@dataclass(frozen=True)
class LaunchAuthorization:
    size0_report_sha256: str
    size0_source_archive_sha256: str
    storage_grant_sha256: str
    source_inventory_sha256: str
    case_binding_bytes: bytes
    staged_parent: str
    staged_parent_identity: tuple
    required_bytes: int
    observed_free_bytes: int
    storage_checked: bool
    outer_session_id_attested_by_root: int


def authenticate(size0_evidence, size0_report_sha256, storage_path, storage_sha256, *,
                 case_binding, source_inventory_sha256, staged_parent, check_storage=True):
    if type(check_storage) is not bool:
        raise ValueError('explicit boolean initial-storage policy required')
    case_bytes = _case(case_binding)
    source_inventory_sha256 = sha_field(source_inventory_sha256)
    storage_sha256 = sha_field(storage_sha256)
    source_sha = _size0(size0_evidence, size0_report_sha256)
    grant_raw = read_public(Path(storage_path))
    if digest(grant_raw) != storage_sha256:
        raise ValueError('external root execution/storage grant differs')
    grant = parse(grant_raw)
    fields = {'schema', 'case_binding', 'source_inventory_sha256', 'staged_parent', 'staged_parent_identity',
              'fresh_keys', 'retained_key_mode', 'execute_authorized', 'size0_report_sha256',
              'size0_outer_session_id', 'size0_outer_exit_code', 'size0_outer_session_reaped',
              'raw_public_key_bytes', 'upload_bytes', 'response_bytes', 'private_model_margin_bytes', 'required_bytes'}
    if (type(grant) is not dict or set(grant) != fields or grant['schema'] != 'cross-bert-ranked-root-launch-grant-v1'
            or grant['fresh_keys'] is not True or grant['retained_key_mode'] is not False
            or grant['execute_authorized'] is not True or grant['size0_outer_session_reaped'] is not True
            or type(grant['size0_outer_exit_code']) is not int or grant['size0_outer_exit_code'] != 0
            or type(grant['size0_outer_session_id']) is not int or grant['size0_outer_session_id'] <= 0
            or grant['size0_report_sha256'] != size0_report_sha256
            or grant['source_inventory_sha256'] != source_inventory_sha256
            or canonical(grant['case_binding']) != case_bytes):
        raise ValueError('explicit root grant and authoritative size0 outer-reap attestation required')
    count = case_binding['count']
    quantities = ('raw_public_key_bytes', 'upload_bytes', 'response_bytes', 'private_model_margin_bytes', 'required_bytes')
    if any(type(grant[n]) is not int or grant[n] <= 0 for n in quantities):
        raise ValueError('exact positive integer full-store accounting required')
    if (grant['raw_public_key_bytes'] != PUBLIC_KEY_BYTES or grant['upload_bytes'] != count * UPLOAD_BYTES_PER_SAMPLE
            or grant['response_bytes'] != count * RESPONSE_BYTES_PER_SAMPLE
            or grant['private_model_margin_bytes'] < MINIMUM_PRIVATE_MODEL_MARGIN_BYTES
            or grant['required_bytes'] != sum(grant[n] for n in quantities[:-1])):
        raise ValueError('full-count fresh-key/upload/response/reserve accounting differs')
    parent = Path(staged_parent)
    if (not parent.is_absolute() or '..' in parent.parts or parent.resolve(strict=True) != parent
            or type(grant['staged_parent']) is not str or grant['staged_parent'] != str(parent)):
        raise ValueError('exact canonical existing staged parent required')
    identity = grant['staged_parent_identity']
    if type(identity) is not list or len(identity) != 2 or any(type(n) is not int or n < 0 for n in identity):
        raise ValueError('exact staged parent device/inode binding required')
    fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    primary = None
    try:
        held = os.fstat(fd)
        if [held.st_dev, held.st_ino] != identity:
            raise ValueError('staged parent identity differs from root grant')
        usage = os.fstatvfs(fd)
        free = usage.f_bavail * usage.f_frsize
        if check_storage and free < grant['required_bytes']:
            raise OSError('insufficient free storage for the complete fresh ranked case')
        current = parent.lstat()
        if not stat.S_ISDIR(current.st_mode) or [current.st_dev, current.st_ino] != identity:
            raise ValueError('staged parent moved during public authorization')
    except BaseException as error:
        primary = error
        raise
    finally:
        _close(fd, primary)
    return LaunchAuthorization(size0_report_sha256, source_sha, storage_sha256, source_inventory_sha256,
                               case_bytes, str(parent), tuple(identity), grant['required_bytes'], free,
                               check_storage, grant['size0_outer_session_id'])
