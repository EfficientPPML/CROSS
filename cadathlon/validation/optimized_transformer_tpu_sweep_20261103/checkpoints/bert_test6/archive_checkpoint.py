"""Archive fixed public BERT6 evidence only; no subject imports or process/key operations."""
from pathlib import Path
import datetime
import json
import shutil
import sys

AUTHOR = Path(__file__).resolve().parent
sys.path.insert(0, str(AUTHOR))
from verify import (ARCHIVE_SHA, ROOT_RECEIPT_SHA, CASE_NAMES, CURRENT,
                    definitions, digest, log_observations, raw, read, source_members)

ROOT = Path('/home/jianming_gatech/CROSS/cadathlon/validation/optimized_transformer_tpu_sweep_20261103')
DEST = ROOT / 'checkpoints/bert_test6'
EV = CURRENT / 'evidence'
SELECTED = {
    'cadathlon/agent_pset/p2_bert/test/test6.in': [],
    'cadathlon/agent_pset/p2_bert/src/bert_optimized_reference.py': ['create_session'],
    'cadathlon/agent_pset/p2_bert/src/runner.py': ['run', 'main'],
    'cadathlon/common/transformer_runner.py': ['benchmark_session'],
    'cadathlon/common/encrypted_reference.py': ['ToyReferenceSession.evaluate', 'ToyReferenceSession.decrypt'],
    'cadathlon/common/encrypted_packed_transformer.py': ['PackedTransformer._bert'],
    'cadathlon/common/encrypted_bounded_arithmetic.py': ['BoundedArithmetic._ensure_capacity'],
    'jaxite_word/ckks_ctx.py': ['CKKSContext.decrypt_slots', '_fast_decrypt_decode', '_fast_decode',
                              '_decode_approximation_stddev', '_validate_decode_approximation', '_check_decode_stddev'],
    'validation_sweep.py': ['instrumented_child'],
}


def dump(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def main():
    assert not DEST.exists(), 'Never overwrite a checkpoint'
    expected, bodies = source_members(ROOT)
    current_expected, _ = source_members(EV)
    assert current_expected == expected
    assert raw(ROOT / 'snapshot_manifest.json') == raw(EV / 'snapshot_manifest.json')
    assert {name: digest(raw(CURRENT / 'checkout' / name)) for name in expected} == expected
    origins = {name: EV / 'bert/test6' / name for name in CASE_NAMES}
    origins.update({'parent_report_final.json': EV / 'report.json',
                    'root-terminal-admission.json': EV / 'root-terminal-admission-20261105.json'})
    assert {p.name for p in (EV / 'bert/test6').iterdir()} == set(CASE_NAMES)
    contents = {name: raw(path) for name, path in origins.items()}
    before = {name: digest(body) for name, body in contents.items()}
    assert before['root-terminal-admission.json'] == ROOT_RECEIPT_SHA
    process = json.loads(contents['process.json'])
    parent = json.loads(contents['parent_report_final.json'])
    terminal = json.loads(contents['root-terminal-admission.json'])
    assert terminal['report_sha256'] == before['parent_report_final.json']
    assert process == parent['models']['bert']['cases']['test6.in']
    assert parent['source_hashes_start'] == parent['source_hashes_end'] == expected
    assert process['passed'] is False and process['returncode'] == 1
    observed = log_observations(contents['stderr.log'].decode())
    refs = {}
    for name, selected in SELECTED.items():
        refs[name] = {'sha256': expected[name],
                      'definitions': definitions(bodies[name], selected) if selected else {}}
    # The one JSON input is source metadata; it is not parsed as Python.
    case = json.loads(bodies['cadathlon/agent_pset/p2_bert/test/test6.in'])
    summary = {
        'schema': 'optimized-tpu-bert-test6-negative-checkpoint-v1',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': 'Completed public toy BERT6 negative; original full twelve-case sweep terminal failure.',
        'case_passed': False, 'case_returncode': 1, 'case_pid': process['pid'],
        'parent_session': 12657, 'parent_terminal_exit': 1, 'parent_terminal_tool_chunk': '27218b',
        'completed_cases': 12, 'passed_cases': 7, 'failed_cases': 5,
        'source_count': 199, 'source_archive_sha256': ARCHIVE_SHA, 'source_file_sha256': expected,
        'source_archive_and_current_verified': True,
        'case_public': {'purpose': case['purpose'], 'weight_seed': case['weight_seed'],
                        'batch_size': len(case['input_ids']), 'sequence_length': len(case['input_ids'][0]),
                        'config': case['config'], 'valid_keys': [sum(row) for row in case['attention_mask']]},
        'numerical_result': json.loads(contents['stdout.json']),
        'external_case_elapsed_s': process['process_elapsed_s'],
        'resources': process['resources'], 'refresh_pairs': 8, 'layer_milestones': 6,
        'logit_metrics_available': False, 'successful_runner_timings_available': False,
        'source_diagnosis_only': True,
        'archival_reviewer_observed_processes_or_key_paths': False,
        'cleanup_authority': 'Original process/parent reports plus copied root independent terminal receipt; no archival process/key probe.',
        'root_terminal_receipt_sha256': ROOT_RECEIPT_SHA,
        'limitations': [
            'Synthetic dimensions only; no pretrained/MRPC/ranked or full-model accuracy claim.',
            'Conjugate-symmetry estimate is not a plaintext-logit error measurement.',
            'Completed refresh and finite metadata do not establish numerical correctness.',
            'First numerical divergence is not localized by logs or this source-only inference.',
            'Resource figures are sampled RSS or device checkpoints, not capacity/continuous peak guarantees.',
            'No subject execution or numerical rerun; no private/key/ciphertext bodies read or copied.',
            'Root receipt TPU-holder data is historical observation only; no queue or external job authority.'],
    }
    after = {name: digest(raw(path)) for name, path in origins.items()}
    assert before == after
    DEST.mkdir()
    for name, body in contents.items():
        (DEST / name).write_bytes(body)
    dump(DEST / 'copy_origins.json', {name: {'source': str(path), 'sha256': before[name],
                                           'source_sha256_before': before[name], 'source_sha256_after': after[name]}
                                     for name, path in origins.items()})
    dump(DEST / 'summary.json', summary)
    dump(DEST / 'milestones.json', observed)
    dump(DEST / 'source_refs.json', refs)
    for name in ('README.md', 'DIAGNOSIS.md', 'verify.py', 'archive_checkpoint.py'):
        shutil.copyfile(AUTHOR / name, DEST / name)
    dump(DEST / 'artifact_hashes.json', {p.name: digest(raw(p)) for p in sorted(DEST.iterdir())})
    print(json.dumps({'destination': str(DEST), 'files': len(list(DEST.iterdir())),
                      'index_sha256': digest(raw(DEST / 'artifact_hashes.json')),
                      'readonly_public_archive_created': True}, sort_keys=True))


if __name__ == '__main__':
    main()
