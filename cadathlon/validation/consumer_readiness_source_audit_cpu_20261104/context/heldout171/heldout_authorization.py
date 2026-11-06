"""Pure public-byte admission for one frozen heldout run or later row0 run.

The external grant digest is a caller trust anchor, never created here. Source
bootstrap and the process reader own filesystem identity/symlink checks. This
module opens nothing, validates lexical path separation, and replays immutable
receipt/grant bytes at the numerical entry boundary.
"""
from dataclasses import dataclass, fields as dataclass_fields
from pathlib import PurePosixPath
import json
import math

import freeze_admission as freeze
import input_gate
from heldout_input_gate import inspect_validation

SCHEMA = 'qwen-heldout-workflow-grant-v1'
CANDIDATE = '344954b3873a8847ececba320646f52612ca7277412c7de81a7313894e21799a'
CALIBRATION_SOURCE = 'ccd5834b83b9dfec2d3ea988fb395a61ec4b05ce3f2f54621f29200cbbb8e06a'
CALIBRATION_GRANT = 'c3983fd975047bba16066106e85f9ae0768ed1e8c0fa30bef4da02b53253cd80'
CLOSURE = ('reconstruction_source_manifest_sha256', 'reconstruction_source_archive_sha256', 'reconstruction_grant_sha256')
BASE_ROLES = ('freeze', 'producer_outer_1', 'reconstruction_outer_1')
ROW0_ROLES = ('heldout_report', 'heldout_outer')
GRANT_FIELDS = ('schema', 'execute', 'mode', 'scope', 'candidate_ordinal',
    'candidate_manifest_sha256', 'freeze_receipt_sha256', 'model_pins_sha256',
    'protocol_sha256', 'tokenizer_report_sha256', 'tokenizer_review_sha256',
    'source_manifest_sha256', 'runtime_policy', 'evidence_output', 'private_output',
    'model_forward', 'coefficient_selection', 'encrypted_execution',
    'selection_permanently_closed', 'validation_read', 'public_evidence',
    'reviewed_script_sha256', 'reviewed_reconstruction_pins',
    'preflight_report_sha256', 'independent_review_sha256', 'prior_heldout_policy')


def _parse(raw, limit=1 << 20):
    if type(raw) is not bytes or not 0 < len(raw) <= limit:
        raise ValueError('Bounded immutable public bytes required')
    def pairs(rows):
        out = {}
        for key, value in rows:
            if key in out: raise ValueError('Duplicate public field')
            out[key] = value
        return out
    def invalid(value): raise ValueError('Nonfinite public value')
    def number(value):
        result = float(value)
        if not math.isfinite(result): invalid(value)
        return result
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid, parse_float=number)


def _equal(left, right):
    if not input_gate.typed_equal(left, right): raise ValueError('Exact typed public binding differs')


def _path(value):
    if type(value) is not str or not 1 < len(value) <= 4096 or '\x00' in value or '\\' in value:
        raise ValueError('Canonical absolute public path required')
    path = PurePosixPath(value)
    if not value.startswith('/') or value.startswith('//') or '..' in path.parts or str(path) != value:
        raise ValueError('Canonical absolute public path required')
    return path


def _overlap(left, right):
    return left == right or left in right.parents or right in left.parents


def _policy():
    return dict(schema='qwen-heldout-runtime-policy-v1', cores=[108, 109, 110, 111],
        torch_threads=4, torch_interop_threads=1, heldout_entries=1088,
        heldout_native_forwards=1120, row0_entries=1, row0_native_forwards=2,
        coefficient_selection=False, candidate_revision=False)


def _grant(raw):
    row = _parse(raw); freeze.fields(row, GRANT_FIELDS)
    mode = row['mode']
    if type(mode) is not str or mode not in ('heldout', 'row0'):
        raise ValueError('Exact heldout or row0 grant required')
    fixed = dict(schema=SCHEMA, execute=True, mode=mode,
        scope='first-frozen-candidate-heldout1088' if mode == 'heldout' else 'first-frozen-candidate-row0',
        candidate_ordinal=1, candidate_manifest_sha256=CANDIDATE, protocol_sha256=freeze.PROTOCOL,
        runtime_policy=_policy(), model_forward=True, coefficient_selection=False,
        encrypted_execution=False, selection_permanently_closed=True, validation_read=mode == 'heldout')
    for key, value in fixed.items(): _equal(row[key], value)
    for key in GRANT_FIELDS:
        if key.endswith('_sha256') and key != 'reviewed_script_sha256': freeze.digest(row[key])
    public = _path(row['evidence_output']); private = _path(row['private_output'])
    if _overlap(public, private): raise ValueError('Separate public/private output roots required')
    scripts = row['reviewed_script_sha256']; closures = row['reviewed_reconstruction_pins']
    if type(scripts) is not list or len(scripts) != 1 or type(closures) is not list or len(closures) != 1:
        raise ValueError('One independently reviewed candidate1 reconstruction required')
    freeze.digest(scripts[0]); freeze.fields(closures[0], CLOSURE)
    for key in CLOSURE: freeze.digest(closures[0][key])
    policy = row['prior_heldout_policy']
    if type(policy) is not str or (mode == 'heldout' and policy != 'none') or (
            mode == 'row0' and policy not in ('complete_scientific', 'independent_row0')):
        raise ValueError('Explicit mode-specific prior heldout policy required')
    roles = BASE_ROLES + (ROW0_ROLES if policy == 'complete_scientific' else
                         ('row0_review',) if policy == 'independent_row0' else ())
    evidence = row['public_evidence']; freeze.fields(evidence, roles)
    seen = []; total = 0
    for role in roles:
        descriptor = evidence[role]; freeze.fields(descriptor, ('path', 'sha256', 'bytes'))
        path = _path(descriptor['path']); freeze.digest(descriptor['sha256'])
        freeze.integer(descriptor['bytes'], 1, (16 if role == 'heldout_report' else 1) << 20)
        total += descriptor['bytes']
        if any(_overlap(path, other) for other in (*seen, public, private)):
            raise ValueError('Public proof roles and owned outputs must be disjoint')
        seen.append(path)
    if total > 20 << 20: raise ValueError('Public proof aggregate exceeds bound')
    _equal(evidence['freeze']['sha256'], row['freeze_receipt_sha256'])
    return row


def public_requests(grant_bytes):
    """Validate role schema/limits before reading; caller checks real paths.

    This is not admission. The process must reject symlinks and all protected
    input/model/source/output overlaps before opening these bounded public roles.
    """
    return _grant(grant_bytes)['public_evidence']


def _source_state(blobs):
    if type(blobs) is not dict or not blobs: raise ValueError('Authenticated complete source blobs required')
    for name, raw in blobs.items():
        if type(name) is not str or type(raw) is not bytes: raise ValueError('Immutable source bytes required')
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or str(path) != name or not name:
            raise ValueError('Canonical relative source member required')
    return tuple(sorted((name, freeze.sha(raw)) for name, raw in blobs.items()))


def _protect_paths(row, blobs, external):
    guards = [_path(external['model_root']), *(_path(p) for p in external['protected_paths']),
              PurePosixPath(__file__).parent]
    copied = _parse(blobs['checker_protected_paths.json'])
    freeze.fields(copied, ('schema', 'paths'))
    if copied['schema'] != 'qwen-candidate-protected-paths-v1' or type(copied['paths']) is not list or len(copied['paths']) < 5:
        raise ValueError('Protected prior observation/selection trees required')
    guards += [_path(p) for p in copied['paths']]
    predecessor = _parse(blobs['heldout_protected_paths.json'])
    freeze.fields(predecessor, ('schema', 'paths'))
    if (predecessor['schema'] != 'qwen-heldout-protected-predecessors-v1' or
            type(predecessor['paths']) is not list or len(predecessor['paths']) != 2):
        raise ValueError('Two protected candidate calibration trees required')
    guards += [_path(p) for p in predecessor['paths']]
    paths = [_path(row['evidence_output']), _path(row['private_output'])]
    paths += [_path(d['path']) for d in row['public_evidence'].values()]
    if any(_overlap(path, guard) for path in paths for guard in guards):
        raise ValueError('Grant public roles/output overlap a protected borrowed tree')


TERMINAL_FLAGS = ('complete', 'cleanup_completed', 'signal_restoration_completed', 'source_files_unchanged',
    'terminal_after_body_cleanup_and_signal_restoration_attempts',
    'acceptance_requires_independent_exit0_reap_and_exact_report_hash', 'public_proofs_reauthenticated')
WORKER_FLAGS = ('private_directory_handle_closed', 'private_outputs_retained', 'external_inputs_never_removed',
    'model_asset_postchecks_completed', 'model_cleanup_completed', 'native_capture_cleanup_completed',
    'ledger_complete', 'selection_permanently_closed', 'model_loaded', 'validation_reauthenticated',
    'validation_read_attempted', 'validation_input_authenticated')


def _prior_heldout(report_raw, outer_raw, frozen, source_manifest_sha256):
    """Bind completed prior heldout, including scientific failure, before row0."""
    parent = _parse(report_raw, 16 << 20); outer = _parse(outer_raw)
    if type(parent) is not dict or type(parent.get('worker')) is not dict:
        raise ValueError('Completed heldout parent and worker required')
    worker = parent['worker']; passed = parent.get('passed')
    if type(passed) is not bool or worker.get('passed') is not passed:
        raise ValueError('Exact prior scientific outcome required')
    code = 0 if passed else 1
    for record in (parent, worker):
        if any(record.get(k) is not True for k in TERMINAL_FLAGS) or record.get('audit_denials') != [] or 'error_type' in record:
            raise ValueError('Prior heldout infrastructure/cleanup incomplete')
        fixed = dict(workload_mode='heldout', source_manifest_sha256=source_manifest_sha256,
            manifest_sha256=frozen.manifest_sha256, freeze_receipt_sha256=frozen.receipt_sha256,
            selection_permanently_closed=True, revision_eligible_ledger=False)
        for key, value in fixed.items(): _equal(record.get(key), value)
        freeze.digest(record.get('grant_sha256'))
    _equal(parent.get('schema'), 'qwen-heldout-workflow-parent-v1')
    _equal(worker.get('schema'), 'qwen-heldout-workflow-worker-v1')
    _equal(worker['grant_sha256'], parent['grant_sha256'])
    if any(worker.get(k) is not True for k in WORKER_FLAGS): raise ValueError('Prior heldout body postchecks incomplete')
    for key, value in dict(candidate_ordinal=1, retained_entries=1088, expected_entries=1088,
        coefficient_selection_performed=False, encrypted_execution=False, tokenizer_execution=False,
        validation_inputs_read=True, validation_read_status='confirmed', contains_tokens_logits_or_failure_text=False).items():
        _equal(worker.get(key), value)
    if worker.get('candidate_validation_admissible') is True or worker.get('failed_attempt_terminal_candidate') is True:
        raise ValueError('Heldout cannot enable revision/selection')
    expected = parent.get('source_hashes_start')
    if type(expected) is not dict or not expected: raise ValueError('Prior complete source map required')
    for name, digest in expected.items():
        if type(name) is not str or not name or str(PurePosixPath(name)) != name or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:
            raise ValueError('Canonical prior source member required')
        freeze.digest(digest)
    for record in (parent, worker):
        _equal(record.get('source_hashes_start'), expected); _equal(record.get('source_hashes_end'), expected)
    # This is the unchanged producer's exact persisted worker serialization.
    worker_raw = (json.dumps(worker, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    _equal(parent.get('worker_report_sha256'), freeze.sha(worker_raw))
    archive = parent.get('source_archive')
    if type(archive) is not dict: raise ValueError('Prior source archive descriptor required')
    archive_sha = freeze.digest(archive.get('sha256'))
    freeze.integer(parent.get('child_returncode'), code)
    if parent.get('child_reaped') is not True or parent.get('child_group_absent') is not True:
        raise ValueError('Prior direct child and owned group must be retired')
    counts = worker.get('native_forward_accounting')
    freeze.fields(counts, ('expected', 'started', 'completed', 'passed'))
    expected_calls = freeze.integer(counts['expected'], 0, 1120)
    calls = freeze.integer(counts['started'], 1, 1120)
    completed = freeze.integer(counts['completed'], 0, calls)
    freeze.integer(worker.get('native_forward_started'), calls)
    freeze.integer(worker.get('native_forward_completed'), completed)
    _equal(counts['passed'], calls == completed == expected_calls)
    if counts['passed'] is not True:
        # Known nonfinite native/exact observations retain a completed failed
        # ledger, including actual partial call counts. They never become PASS.
        # The independently pinned outer still attests exit1 and full cleanup.
        if passed: raise ValueError('Prior PASS forward accounting failed')
        kinds = worker.get('by_kind'); freeze.fields(kinds, ('prefill', 'teacher', 'free'))
        comparison_invalid = 0
        for name, total in (('prefill', 1024), ('teacher', 32), ('free', 32)):
            part = kinds[name]
            freeze.fields(part, ('expected', 'records', 'complete', 'candidate_invalid',
                'comparison_invalid', 'unmeasured', 'top1_matches', 'top1_denominator'))
            for key in ('expected', 'records', 'top1_denominator'): freeze.integer(part[key], total)
            statuses = [freeze.integer(part[key], 0, total) for key in
                ('complete', 'candidate_invalid', 'comparison_invalid', 'unmeasured')]
            if sum(statuses) != total: raise ValueError('Complete heldout failure classification differs')
            freeze.integer(part['top1_matches'], 0, total)
            comparison_invalid += part['comparison_invalid']
        if comparison_invalid == 0: raise ValueError('Unaccounted calls without classified comparison failure')
    if passed:
        freeze.integer(calls, 1120)
        for key in ('trajectories_complete', 'all_length_invariance', 'complete_measured_context_coverage'):
            if worker.get(key) is not True: raise ValueError('Prior PASS coverage incomplete')
        _equal(worker.get('phase_counts'), dict(native=1088, exact=1088, candidate=1088))
    fixed_outer = dict(schema='qwen-heldout-workflow-outer-v1', mode='heldout',
        report_sha256=freeze.sha(report_raw), worker_report_sha256=parent['worker_report_sha256'],
        source_manifest_sha256=source_manifest_sha256, source_archive_sha256=archive_sha,
        candidate_manifest_sha256=frozen.manifest_sha256, freeze_receipt_sha256=frozen.receipt_sha256,
        grant_sha256=parent['grant_sha256'], exit_code=code, worker_exit_code=code,
        reaped=True, process_group_absent=True, worker_reaped=True, worker_group_absent=True,
        publication_completed=True)
    _equal(outer, fixed_outer)


def _independent_row0(raw, frozen, grant):
    """An independently pinned root observation removes no scientific gate.

    Row0 requires a completed calibration freeze, never a heldout PASS. A prior
    attempt, when present, must have known process retirement. Missing source,
    cleanup, publication or read evidence stays explicitly unknown; this route
    never admits the prior attempt as a scientific result or authorizes a retry.
    """
    row = _parse(raw)
    if type(row) is not dict: raise ValueError('Independent row0 root receipt required')
    fixed = dict(schema='qwen-independent-row0-admission-v1',
        candidate_manifest_sha256=frozen.manifest_sha256, freeze_receipt_sha256=frozen.receipt_sha256,
        protocol_sha256=grant['protocol_sha256'], model_pins_sha256=grant['model_pins_sha256'],
        selection_permanently_closed=True, scientific_result_admitted=False,
        retry_authorized=False, feedback_authorized=False, coefficient_selection_authorized=False,
        candidate_revision_authorized=False, validation_execution_authorized=False,
        row0_execution_authorized=False)
    prior = row.get('prior_attempt')
    if type(prior) is not str or prior not in ('none', 'terminal'):
        raise ValueError('Prior attempt must be explicitly absent or terminal')
    extras = () if prior == 'none' else ('heldout_grant_sha256', 'expected_source_manifest_sha256',
        'parent', 'worker', 'observed_outcome', 'validation_read')
    freeze.fields(row, (*fixed, 'prior_attempt', *extras))
    for key, value in fixed.items(): _equal(row[key], value)
    if prior == 'none': return
    for key in ('heldout_grant_sha256', 'expected_source_manifest_sha256'): freeze.digest(row[key])
    if row['observed_outcome'] not in ('startup_failure', 'infrastructure_failure', 'complete_scientific'):
        raise ValueError('Exact observed prior outcome required')
    def retirement(value, parent=False):
        names = ('pid', 'pgid', 'exit_code', 'reaped', 'group_absent', 'report_sha256')
        more = ('source_archive_sha256', 'source_verified', 'cleanup_completed', 'publication_completed') if parent else ('state',)
        freeze.fields(value, (*names, *more))
        for key in ('pid', 'pgid'): freeze.integer(value[key], 1, (1 << 31)-1)
        freeze.integer(value['exit_code'], -255, 255)
        if value['reaped'] is not True or value['group_absent'] is not True:
            raise ValueError('Known owned-process retirement required before independent row0')
        for key in ('report_sha256', 'source_archive_sha256') if parent else ('report_sha256',):
            if value[key] is not None: freeze.digest(value[key])
        if parent:
            for key in ('source_verified', 'cleanup_completed', 'publication_completed'):
                if value[key] is not None and type(value[key]) is not bool:
                    raise ValueError('Honest typed or unknown prior terminal observation required')
    retirement(row['parent'], parent=True)
    worker = row['worker']
    if type(worker) is not dict: raise ValueError('Known worker existence required')
    if worker.get('state') == 'not_created': _equal(worker, {'state': 'not_created'})
    elif worker.get('state') == 'retired': retirement(worker)
    else: raise ValueError('Unknown worker existence is not process retirement')
    read = row['validation_read']; freeze.fields(read, ('status', 'attempted', 'authenticated', 'reauthenticated'))
    status = read['status']
    if type(status) is not str or status not in ('not_started', 'attempted', 'confirmed', 'unknown'):
        raise ValueError('Explicit prior validation-read observation required')
    for key in ('attempted', 'authenticated', 'reauthenticated'):
        if read[key] is not None and type(read[key]) is not bool: raise ValueError('Typed or unknown read observation required')
    if status == 'not_started': _equal(read, dict(status=status, attempted=False, authenticated=False, reauthenticated=False))
    elif status == 'unknown': _equal(read, dict(status=status, attempted=None, authenticated=None, reauthenticated=None))
    elif status == 'confirmed':
        if read['attempted'] is not True or read['authenticated'] is not True:
            raise ValueError('Confirmed read requires observed authenticated access')
    elif read['attempted'] is not True or read['authenticated'] is True or read['reauthenticated'] is True:
        raise ValueError('Attempted read cannot fabricate completed authentication')


@dataclass(frozen=True, slots=True)
class Authorization:
    frozen: freeze.FrozenCandidate
    mode: str
    manifest_bytes: bytes
    manifest_sha256: str
    grant_sha256: str
    source_manifest_sha256: str
    evidence_output: str
    private_output: str
    grant_bytes: bytes
    evidence_bytes: tuple
    source_hashes: tuple
    policy_bytes: bytes
    validation_bytes: bytes

    @property
    def runtime_policy(self): return _parse(self.policy_bytes)
    @property
    def validation_descriptor(self): return _parse(self.validation_bytes)
    @property
    def public_evidence(self): return public_requests(self.grant_bytes)


def authenticate(grant_bytes, grant_sha256, public_evidence, *, mode,
                 source_manifest_sha256, evidence_output, private_output, blobs):
    freeze.digest(grant_sha256)
    if type(grant_bytes) is not bytes or freeze.sha(grant_bytes) != grant_sha256:
        raise ValueError('External grant digest differs')
    row = _grant(grant_bytes)
    for name, value in dict(mode=mode, source_manifest_sha256=source_manifest_sha256,
                            evidence_output=evidence_output, private_output=private_output).items():
        _equal(row[name], value)
    freeze.digest(source_manifest_sha256)
    source_hashes = _source_state(blobs)
    _equal(_parse(blobs['execution_authorization.json']), {'execute': False})
    _equal(_parse(blobs['heldout_runtime_policy.json']), _policy())
    external = input_gate.inspect(blobs); _protect_paths(row, blobs, external)
    roles = row['public_evidence']; freeze.fields(public_evidence, roles)
    for name, descriptor in roles.items():
        raw = public_evidence[name]
        if type(raw) is not bytes or len(raw) != descriptor['bytes'] or freeze.sha(raw) != descriptor['sha256']:
            raise ValueError('External public role bytes differ: ' + name)
    frozen = freeze.authenticate_complete_freeze(public_evidence['freeze'], row['freeze_receipt_sha256'],
        producer_outers=(public_evidence['producer_outer_1'],),
        reconstruction_outers=(public_evidence['reconstruction_outer_1'],),
        reviewed_script_sha256=tuple(row['reviewed_script_sha256']),
        reviewed_reconstruction_pins=tuple(row['reviewed_reconstruction_pins']))
    receipt = freeze.parse(frozen.receipt_bytes)
    _equal(frozen.ordinal, 1); _equal(frozen.manifest_sha256, CANDIDATE)
    _equal(receipt['attempts'][0]['source_manifest_sha256'], CALIBRATION_SOURCE)
    _equal(receipt['attempts'][0]['grant_sha256'], CALIBRATION_GRANT)
    manifest = blobs['candidate_proof/candidate-manifest.json']
    _equal(freeze.sha(manifest), CANDIDATE)
    for key, value in dict(model_pins_sha256=freeze.sha(blobs['public/model_pins.json']),
        protocol_sha256=freeze.PROTOCOL, tokenizer_report_sha256=external['producer']['tokenizer_report_sha256'],
        tokenizer_review_sha256=external['producer']['tokenizer_root_review_sha256']).items():
        _equal(row[key], value); _equal(receipt[key], value)
    validation = inspect_validation(blobs, frozen)
    if row['prior_heldout_policy'] == 'complete_scientific':
        _prior_heldout(public_evidence['heldout_report'], public_evidence['heldout_outer'], frozen, source_manifest_sha256)
    elif row['prior_heldout_policy'] == 'independent_row0':
        _independent_row0(public_evidence['row0_review'], frozen, row)
    return Authorization(frozen, mode, manifest, CANDIDATE, grant_sha256, source_manifest_sha256,
        evidence_output, private_output, grant_bytes, tuple(sorted(public_evidence.items())),
        source_hashes, freeze.canonical(row['runtime_policy']), freeze.canonical(validation))


def require_authorized(value, blobs):
    if type(value) is not Authorization: raise ValueError('Authenticated external workflow grant required')
    freeze.require_frozen(value.frozen)
    if type(value.evidence_bytes) is not tuple or any(type(row) is not tuple or len(row) != 2 or
        type(row[0]) is not str or type(row[1]) is not bytes for row in value.evidence_bytes):
        raise ValueError('Immutable retained public evidence required')
    if len(dict(value.evidence_bytes)) != len(value.evidence_bytes): raise ValueError('Duplicate retained evidence role')
    expected = authenticate(value.grant_bytes, value.grant_sha256, dict(value.evidence_bytes),
        mode=value.mode, source_manifest_sha256=value.source_manifest_sha256,
        evidence_output=value.evidence_output, private_output=value.private_output, blobs=blobs)
    for field in dataclass_fields(Authorization):
        left, right = getattr(value, field.name), getattr(expected, field.name)
        if type(left) is not type(right) or left != right:
            raise ValueError('Retained workflow authorization changed: ' + field.name)
    return expected
