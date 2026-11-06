"""Offline integrity and declared-scope verification; no crypto imports."""
from pathlib import Path
import hashlib
import json
import tarfile

ROOT = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify_sources(archive, report):
    before = report['source_hashes_before']
    assert before == report['source_hashes_after']
    assert report['source_files_unchanged'] is True
    with tarfile.open(ROOT / archive) as handle:
        members = handle.getmembers()
        assert len(members) == len(before)
        assert {member.name for member in members} == set(before)
        for member in members:
            assert member.isfile() and not member.issym() and not member.islnk()
            assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
            assert hashlib.sha256(handle.extractfile(member).read()).hexdigest() == before[member.name]
    return len(before)

def main():
    audit = json.loads((ROOT / 'audit.json').read_text())
    candidate = json.loads((ROOT / 'candidate_files.json').read_text())
    assert sha(ROOT / 'candidate.patch') == candidate['patch_sha256'] == audit['patch_sha256']
    assert candidate['source_count'] == len(candidate['source_sha256']) == 6
    for name, digest in candidate['source_sha256'].items():
        assert sha(ROOT / 'candidate' / name) == digest
    focused = json.loads((ROOT / 'focused.json').read_text())
    final = json.loads((ROOT / 'secure-v2/report.json').read_text())
    assert focused['passed'] and focused['returncode'] == 0
    assert focused['source_archive_sha256'] == sha(ROOT / 'focused-sources.tar.gz')
    assert verify_sources('focused-sources.tar.gz', focused) == 434
    assert verify_sources('secure-v2/sources.tar.gz', final) == 101
    for record in audit['archives']:
        assert sha(ROOT / record['archive']) == record['sha256']
    assert final['passed'] and final['cleanup_errors'] == [] and final['input_immutability']
    assert final['fixed_absolute_error_gate'] == 1e-6
    assert len(final['checks']) == 18 and all(check['passed'] for check in final['checks'])
    assert max(check.get('max_abs_error', 0) for check in final['checks']) < 1e-6
    assert final['ring']['QP_bit_length'] <= final['ring']['security_bit_length_ceiling']
    terminal = json.loads((ROOT / 'secure-v2/terminal.json').read_text())
    assert terminal['authoritatively_reaped'] and terminal['exit_code'] == 0
    prior = json.loads((ROOT / 'secure-v1/report.json').read_text())
    prior_focused = json.loads((ROOT / 'pre-guard-focused/focused.json').read_text())
    verify_sources('secure-v1/sources.tar.gz', prior)
    verify_sources('pre-guard-focused/focused-sources.tar.gz', prior_focused)
    assert prior['source_hashes_before'] != final['source_hashes_before']
    intersection = json.loads((ROOT / 'measured_source_intersection.json').read_text())
    assert not any(intersection['intersections'].values())
    print(json.dumps({'passed': True, 'candidate_files': 6, 'focused_sources': 434,
                      'secure_sources': 101, 'secure_updates': 2, 'secure_checks': 18,
                      'full13_final_source_claim': False}, sort_keys=True))

if __name__ == '__main__':
    main()
