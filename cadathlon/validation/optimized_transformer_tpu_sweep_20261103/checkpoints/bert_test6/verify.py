"""Verify public BERT6 negative evidence; never imports the archived subjects."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import tarfile

HERE = Path(__file__).resolve().parent
CURRENT = Path('/tmp/cross-optimized-twelve-20261103')
ARCHIVE_SHA = 'bc0ad1b2a3796fe4ecb5944d68f3ca65561ad5740357470a1e634cc243ec387b'
ROOT_RECEIPT_SHA = 'b12038aa4400b1b8878570d6ef1dbd8e443570f36361de2dd4ec9e9ede38599f'
CASE_NAMES = ('process.json', 'resource_samples.jsonl', 'resources.json',
              'security_device.json', 'stderr.log', 'stdout.json')
ERROR = ('CKKS decryption failed because the approximation error is too high; '
         'estimated stddev=1.036e+26, limit=3.125e-02. Check the ring '
         'parameters, scale, and circuit depth.')


def raw(path, cap=16 * 1024 * 1024):
    before = path.lstat()
    assert stat.S_ISREG(before.st_mode) and before.st_size <= cap, path
    data = path.read_bytes()
    after = path.lstat()
    assert (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
            before.st_ctime_ns) == (after.st_dev, after.st_ino, after.st_size,
                                    after.st_mtime_ns, after.st_ctime_ns), path
    assert len(data) == before.st_size
    return data


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(raw(path))


def source_members(root):
    expected = read(root / 'snapshot_manifest.json')['source_hashes']
    assert len(expected) == 199
    assert digest(raw(root / 'sources.tar.gz')) == ARCHIVE_SHA
    with tarfile.open(root / 'sources.tar.gz', 'r:gz') as archive:
        members = archive.getmembers()
        assert len(members) == len({m.name for m in members}) == 199
        assert all(m.isfile() and 0 <= m.size <= 8 * 1024 * 1024 for m in members)
        assert {m.name for m in members} == set(expected)
        bodies = {m.name: archive.extractfile(m).read() for m in members}
    assert {name: digest(body) for name, body in bodies.items()} == expected
    return expected, bodies


def log_observations(text):
    refreshes = []
    pending = None
    milestones = []
    for number, line in enumerate(text.splitlines(), 1):
        match = re.fullmatch(r'\[experimental-bert\] bounded homomorphic refresh '
                             r'(starting|complete); elapsed ([0-9.]+)s, device resident ([0-9.]+) GiB', line)
        if match:
            item = {'line': number, 'elapsed_s_rounded': float(match[2]),
                    'device_resident_gib_rounded': float(match[3])}
            if match[1] == 'starting':
                assert pending is None
                pending = item
            else:
                assert pending is not None
                refreshes.append({'ordinal': len(refreshes) + 1, 'start': pending,
                                  'complete': item,
                                  'duration_s_from_rounded_log': round(item['elapsed_s_rounded'] - pending['elapsed_s_rounded'], 1)})
                pending = None
        match = re.fullmatch(r'\[experimental-bert\] packed BERT layer ([0-9]+): '
                             r'(attention/normalization|output) ready; elapsed ([0-9.]+)s, device resident ([0-9.]+) GiB', line)
        if match:
            milestones.append({'line': number, 'layer': int(match[1]), 'stage': match[2],
                               'elapsed_s_rounded': float(match[3]),
                               'device_resident_gib_rounded': float(match[4])})
    assert pending is None and len(refreshes) == 8
    assert [(v['layer'], v['stage']) for v in milestones] == [
        (layer, phase) for layer in range(3) for phase in ('attention/normalization', 'output')]
    assert refreshes[-1]['start']['elapsed_s_rounded'] == 37289.6
    assert refreshes[-1]['complete']['elapsed_s_rounded'] == 41795.6
    assert refreshes[-1]['duration_s_from_rounded_log'] == 4506.0
    assert milestones[-1]['elapsed_s_rounded'] == 37269.2
    return {'refresh_start_count': 8, 'refresh_complete_count': 8,
            'layer_milestone_count': 6, 'refreshes': refreshes, 'milestones': milestones,
            'traceback_present': 'Traceback (most recent call last)' in text,
            'constant_capture_warning_count': text.count('UserWarning: A large amount of constants were captured during lowering')}


def definitions(data, selected):
    if not selected:
        return {}
    tree = ast.parse(data)
    out = {}
    def walk(body, prefix=''):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + node.name
                if name in selected:
                    out[name] = {'lineno': node.lineno, 'end_lineno': node.end_lineno,
                                 'ast_sha256': digest(ast.dump(node, include_attributes=False).encode())}
                if isinstance(node, ast.ClassDef):
                    walk(node.body, name + '.')
    walk(tree.body)
    assert set(out) == set(selected)
    return out


def verify(current=False):
    index = read(HERE / 'artifact_hashes.json')
    entries = list(HERE.iterdir())
    assert all(stat.S_ISREG(p.lstat().st_mode) for p in entries)
    assert {p.name for p in entries} == set(index) | {'artifact_hashes.json'}
    assert {name: digest(raw(HERE / name)) for name in index} == index
    origins = read(HERE / 'copy_origins.json')
    for name, item in origins.items():
        assert digest(raw(HERE / name)) == item['sha256'] == item['source_sha256_before'] == item['source_sha256_after']
    expected, bodies = source_members(HERE.parents[1])
    summary = read(HERE / 'summary.json')
    assert summary['source_file_sha256'] == expected
    assert summary['source_archive_sha256'] == ARCHIVE_SHA
    refs = read(HERE / 'source_refs.json')
    for name, item in refs.items():
        assert digest(bodies[name]) == item['sha256'] == expected[name]
        assert definitions(bodies[name], item['definitions']) == item['definitions']
    process = read(HERE / 'process.json')
    result = read(HERE / 'stdout.json')
    parent = read(HERE / 'parent_report_final.json')
    terminal = read(HERE / 'root-terminal-admission.json')
    assert digest(raw(HERE / 'root-terminal-admission.json')) == ROOT_RECEIPT_SHA
    assert terminal['report_sha256'] == digest(raw(HERE / 'parent_report_final.json'))
    assert terminal['session_id'] == 12657 and terminal['exit_code'] == 1
    assert terminal['authoritative_tool_chunk'] == '27218b' and terminal['finished'] is True
    assert terminal['all_cleanup_flags_true'] is True
    assert terminal['all_owned_case_and_parent_key_paths_absent'] is True
    assert terminal['original_parent_child_pids_and_groups_absent'] is True
    assert terminal['source_start_end_current_archive_exact'] is True
    assert process == parent['models']['bert']['cases']['test6.in']
    assert process['pid'] == 897099 and process['returncode'] == 1 and process['passed'] is False
    assert process['result'] == result == summary['numerical_result']
    assert result['case'] == 'test6.in' and result['passed'] is False
    assert result['error_type'] == 'ApproximationErrorTooHigh' and result['error'] == ERROR
    assert result['notes'] == []
    assert set(result) == {'case', 'passed', 'error', 'error_type', 'notes'}
    assert process['process_elapsed_s'] == summary['external_case_elapsed_s'] == 41889.48500001902
    for flag in ('process_reaped', 'public_key_store_removed', 'owned_directory_handle_closed', 'source_files_unchanged'):
        assert process[flag] is True
    cases = [c for model in parent['models'].values() for c in model['cases'].values()]
    assert len(cases) == terminal['completed'] == 12
    assert sum(c['passed'] is True for c in cases) == terminal['passed'] == 7
    assert sum(c['passed'] is False for c in cases) == terminal['failed'] == 5
    assert parent['passed'] is False
    assert parent['source_hashes_start'] == parent['source_hashes_end'] == expected
    assert parent['source_archive_sha256'] == ARCHIVE_SHA
    assert parent['cleanup_errors'] == [] and parent['inner_cleanup_complete'] is True
    assert summary['case_passed'] is False and summary['logit_metrics_available'] is False
    assert summary['archival_reviewer_observed_processes_or_key_paths'] is False
    obs = log_observations(raw(HERE / 'stderr.log').decode())
    assert obs == read(HERE / 'milestones.json')
    assert obs['traceback_present'] is False and obs['constant_capture_warning_count'] == 4
    security = read(HERE / 'security_device.json')
    assert security['passed'] is False and security['error'] == 'SystemExit: 1'
    assert [e['boundary'] for e in security['events']] == ['input', 'output']
    for event in security['events']:
        assert len(event['ciphertexts']) == 1 and len(event['actual_rings']) == 2
        for ring in event['actual_rings']:
            assert ring['degree'] == 131072 and len(ring['q_towers']) == 100 and len(ring['p_towers']) == 24
            assert ring['security_bits'] == 128 and ring['security_model'] == 'classical'
            bits = sum(math.log2(v) for v in ring['q_towers'] + ring['p_towers'])
            assert abs(bits - ring['log2_QP']) < 1e-8 and bits < 3523
        assert all(d['platform'] == 'tpu' for c in event['ciphertexts'] for d in c['devices'])
    output = security['events'][1]['ciphertexts'][0]
    assert output['moduli_count'] == 40 and output['nsd'] == 1
    assert output['scale'] == 6.8994011570896904e16
    resources = read(HERE / 'resources.json')
    assert resources == process['resources'] == summary['resources']
    lines = raw(HERE / 'resource_samples.jsonl').splitlines()
    assert all(len(line) <= 8192 for line in lines)
    samples = [json.loads(line) for line in lines]
    assert len(samples) == resources['samples'] == 40088
    assert max(r['aggregate_rss_bytes'] for r in samples) == resources['sampled_peak_process_tree_rss_bytes'] == 119110234112
    case = json.loads(bodies['cadathlon/agent_pset/p2_bert/test/test6.in'])
    cfg = case['config']
    assert (cfg['hidden_size'], cfg['intermediate_size'], cfg['num_attention_heads'], cfg['num_hidden_layers'], cfg['num_labels']) == (24, 48, 4, 3, 2)
    assert case['weight_seed'] == 9506
    assert len(case['input_ids']) == 3 and all(len(row) == 16 for row in case['input_ids'])
    assert [sum(row) for row in case['attention_mask']] == [15, 14, 13]
    if current:
        assert {name: digest(raw(CURRENT / 'checkout' / name)) for name in expected} == expected
        assert digest(raw(CURRENT / 'evidence/sources.tar.gz')) == ARCHIVE_SHA
        for name, item in origins.items():
            assert digest(raw(Path(item['source']))) == item['sha256']
    return {'evidence_consistent': True, 'case_passed': False, 'source_count': 199,
            'case': 'bert/test6', 'completed_cases': 12, 'passed_cases': 7, 'failed_cases': 5,
            'refresh_pairs': 8, 'layer_milestones': 6, 'current_checked': current,
            'actual_subjects_executed': False, 'independent_cleanup_observation': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current', action='store_true')
    print(json.dumps(verify(parser.parse_args().current), sort_keys=True))
