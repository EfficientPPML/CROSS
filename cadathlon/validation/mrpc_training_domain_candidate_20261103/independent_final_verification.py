"""Read back completed public evidence; do not execute or retune the model."""
from pathlib import Path
import hashlib
import json
import tarfile

ROOT = Path(__file__).resolve().parent
CANDIDATE = '4e2277beea7189aa1d77b168c500c8a175f6c630afa520fcc6b38cb5a179cff9'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_run(directory, run):
    for name, expected in json.loads((directory / 'artifact_sha256.json').read_text()).items():
        assert digest(directory / name) == expected, name
    report = json.loads((directory / run / 'report.json').read_text())
    assert report['passed'] and report['source_files_unchanged']
    before = report['source_hashes_before']
    assert before == report['source_hashes_after']
    archive = directory / run / 'sources.tar.gz'
    assert digest(archive) == report['source_archive_sha256']
    with tarfile.open(archive) as source:
        members = source.getmembers()
        assert all(m.isfile() for m in members)
        archived = {m.name: hashlib.sha256(source.extractfile(m).read()).hexdigest() for m in members}
    assert len(archived) == len(members)
    assert archived == {name.lstrip('/'): value for name, value in before.items()}
    for name, expected in before.items():
        assert digest(Path(name)) == expected, name
    assert report['candidate_manifest_sha256'] == CANDIDATE
    assert digest(directory / 'candidate-v2/candidate_manifest.json') == CANDIDATE
    return report


def main():
    train_dir = ROOT / 'explicit_v2_complete_training'
    valid_dir = ROOT / 'explicit_v2_heldout_validation'
    train = read_run(train_dir, 'approximate-train3668-v2')
    valid = read_run(valid_dir, 'approximate-validation408-v2')
    rounded = read_run(train_dir, 'rounded-reciprocal-v2')
    assert train['completed_examples'] == train['complete_model_outputs'] == 3668
    assert train['approximate_trajectory_coverage'] and not train['domain_failures']
    assert not train['numeric_failures'] and train['all_training_examples_evaluated']
    assert not train['training_labels_loaded'] and not train['validation_evaluated']
    assert valid['completed_examples'] == 408 and valid['domain_coverage_passed']
    assert not valid['domain_failures'] and not valid['screening_differences']
    assert valid['labels_loaded_after_outputs']
    assert valid['training_report_sha256'] == digest(train_dir / 'approximate-train3668-v2/report.json')
    for report in (train, valid):
        assert report['no_jax_imported'] and not report['encrypted_execution']
    assert rounded['rule_count'] == 130 and rounded['candidate_unchanged']
    assert float(rounded['worst_upper']) <= 1e-12
    expected = {42: {100: 82, 200: 162, 408: 345},
                43: {100: 79, 200: 172, 408: 345},
                44: {100: 84, 200: 166, 408: 345}}
    records = valid['ranked']['subsets']
    assert len(records) == 9
    assert {(r['base_seed'], r['size']) for r in records} == {
        (seed, size) for seed in expected for size in expected[seed]}
    for row in records:
        size = row['size']
        assert row['complete_outputs'] == size
        assert len(row['positions']) == len(set(row['positions'])) == size
        assert len(row['target_ids']) == len(set(row['target_ids'])) == size
        assert row['native_correct'] == row['approximate_correct'] == expected[row['base_seed']][size]
        assert 4 * row['approximate_correct'] >= 3 * size
        assert 100 * (row['native_correct'] - row['approximate_correct']) <= 2 * size
        assert row['accuracy_drop_percentage_points'] == 0
        assert row['ranked_accuracy_pass'] and row['finite_logits_pass'] and row['domain_coverage_pass']
    prior = ROOT.parent / 'mrpc408_fixed_domain_audit_20261103'
    candidates = [json.loads(p.read_text()) for p in prior.rglob('report.json')]
    assert any(r.get('tokenization_sha256') == valid['tokenization_sha256'] for r in candidates)
    result = dict(passed=True, scope='Independent archive, source, fixed-domain and ranked-count readback; no new model execution.',
                  candidate_manifest_sha256=CANDIDATE, training_examples=3668,
                  heldout_examples=408, ranked_subsets=9, rounded_reciprocal_rules=130,
                  training_sources=len(train['source_hashes_before']), heldout_sources=len(valid['source_hashes_before']),
                  current_sources_match_measured=True, encrypted_execution=False)
    (ROOT / 'independent_final_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
