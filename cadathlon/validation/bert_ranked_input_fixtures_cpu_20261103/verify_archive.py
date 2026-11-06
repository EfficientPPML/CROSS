"""Offline public fixture proof; no external assets/private paths are opened."""
from pathlib import Path
import hashlib, json, tarfile
ROOT = Path(__file__).resolve().parent
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    inventory = json.loads((ROOT / 'artifact_hashes.json').read_bytes())
    actual = {p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json' and '__pycache__' not in p.parts}
    assert actual == inventory
    assert sha(ROOT / 'report.json') == 'ad1458f5183c6ccf2631c2a8771d0da92196f5f9c690016e2a0b12fa7b88f4d7'
    r = json.loads((ROOT / 'report.json').read_bytes())
    assert all(r[n] is True for n in ('passed', 'terminal_scope_completed', 'cleanup_completed', 'private_workspace_retained', 'sources_unchanged', 'assets_unchanged', 'upstream_originals_unchanged', 'source_archive_verified', 'model_construction_and_forward_guards'))
    assert r['model_forward'] is False and r['encrypted_execution'] is False and r['cleanup_errors'] == []
    sources = r['source_hashes_before']
    assert len(sources) == 27 and sources == r['source_hashes_after']
    assert r['asset_file_sha256_before'] == r['asset_file_sha256_after']
    assert r['upstream_original_sha256_before'] == r['upstream_original_sha256_after']
    assert sha(ROOT / 'sources.tar.gz') == r['source_archive_sha256'] == 'ea2c9e0ef102181b47bc0e453f771868ac4b1adc366ffbd5374222bd35e4c20a'
    with tarfile.open(ROOT / 'sources.tar.gz') as archive:
        members = archive.getmembers()
        assert len(members) == len({m.name for m in members}) == 27
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        assert {m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in members} == sources
    assert [(c['size'], c['count'], c['base_seed']) for c in r['cases']] == [(size,count,seed) for size,count in ((1,100),(2,200),(3,408)) for seed in (42,43,44)]
    assert sum(c['record_count'] for c in r['cases']) == r['selected_records_expected'] == 2124
    assert len(r['cases']) == r['ranked_cases_expected'] == 9
    for c in r['cases']:
        assert c['record_count'] == c['count'] and c['fixed_token_length'] == 128
        assert len(c['input_file_sha256']) == 6 and sum(c['valid_token_count_histogram'].values()) == c['count']
        assert all(1 <= int(k) <= 128 for k in c['valid_token_count_histogram'])
        assert all(c[n] is True for n in ('selection_byte_parity', 'tokenization_byte_parity', 'labels_private', 'all_private_files_0600', 'all_private_directories_0700'))
    full = [c for c in r['cases'] if c['count'] == 408]
    assert len({c['private_target_order_sha256'] for c in full}) == len({c['private_row_order_sha256'] for c in full}) == 1
    readback = json.loads((ROOT / 'private_readback.json').read_bytes())
    assert readback == dict(passed=True, cases=9, selected_records=2124, private_files=54, full408_inputs_identical=True, selection_and_tokenization_byte_parity=True, private_permissions_verified=True, model_forward=False, encrypted_execution=False)
    verification = json.loads((ROOT / 'root_verification.json').read_bytes())
    assert verification['current_source_hashes'] == sources
    assert verification['source_map_equals_preflight'] is True
    assert verification['private_readback_session_id'] == 74483 and verification['private_readback_exit_code'] == 0 and verification['private_readback_reaped'] is True
    terminal = json.loads((ROOT / 'terminal.json').read_bytes())
    assert terminal['authoritative_session_id'] == 61991 and terminal['observed_exit_code'] is None and terminal['session_reaped'] is None
    print(json.dumps(dict(passed=True, sources=27, cases=9, selected_records=2124, private_readback_verified=True, model_forward=False, encrypted_execution=False)))

if __name__ == '__main__':
    main()
