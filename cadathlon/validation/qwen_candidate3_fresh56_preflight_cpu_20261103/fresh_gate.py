"""Authenticate the prior input freeze; input bodies are read only at runtime."""
from pathlib import Path
import hashlib
import json
import tarfile


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError('Fresh input freeze prerequisite failed: ' + message)


def verify_metadata(root, requirements):
    root = Path(root)
    pins = requirements['fresh_freeze']
    report_path = root / 'fresh_freeze/report.json'
    require(sha(report_path) == pins['report_sha256'], 'report hash')
    report = json.loads(report_path.read_bytes())
    require(report.get('passed') is True and report.get('complete') is True, 'complete tokenizer freeze')
    require(report.get('model_forward') is False and report.get('coefficient_selection') is False
            and report.get('no_model_backend_imported') is True, 'input-only preparation')
    before = report['source_hashes_before']
    require(len(before) == 11 and before == report['source_hashes_after']
            and report['source_files_unchanged'] is True, 'eleven unchanged sources')
    require(report['source_archive_sha256'] == pins['archive_sha256'], 'source archive binding')
    require(report['tokenized_sha256'] == pins['tokenized_sha256']
            == requirements['validation_input_sha256'], 'token file binding')
    require(report['records'] == report['distinct_texts'] == report['distinct_token_sequences'] == 56
            and report['prefills'] == 48 and report['decode_sequences'] == 8
            and report['contexts_if_eventually_measured'] == 112, 'frozen workload counts')
    require(all(report[k] == 0 for k in ('exact_text_overlap', 'exact_truncated_sequence_overlap',
                                       'exact_full_sequence_overlap')), 'recorded equality disjointness')
    require(report['assets_unchanged'] is True
            and report['asset_sha256_before'] == report['asset_sha256_after'], 'asset hash agreement')
    for name in ('protocol.json', 'input_pins.json', 'model_pins.json'):
        require(sha(root / 'fresh_freeze' / name) == before[name], 'public metadata ' + name)
    return report


def verify_inputs(root, requirements, archive_path, token_path, *, on_read=lambda: None):
    """Called only by the authorized actual validation, after calibration gates."""
    report = verify_metadata(root, requirements)
    pins = requirements['fresh_freeze']
    def tracked_sha(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            while chunk := stream.read(1024**2):
                on_read()
                digest.update(chunk)
        return digest.hexdigest()
    require(tracked_sha(archive_path) == pins['archive_sha256'], 'actual source archive hash')
    expected = report['source_hashes_before']
    with tarfile.open(archive_path) as tar:
        members = tar.getmembers()
        require(len(members) == len(expected) == len({m.name for m in members}), 'unique archive inventory')
        require(all(m.isfile() and 0 <= m.size <= 8 * 1024**2 for m in members)
                and sum(m.size for m in members) <= 16 * 1024**2, 'bounded regular source members')
        actual = {m.name: hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in members}
    require(actual == expected, 'every archived source byte')
    require(tracked_sha(token_path) == pins['tokenized_sha256'], 'actual frozen token file hash')
    return dict(verified=True, report_sha256=pins['report_sha256'],
                source_archive_sha256=pins['archive_sha256'], tokenized_sha256=pins['tokenized_sha256'],
                source_count=11, tokenized_records=56, planned_contexts=112)
