"""Read-only, standard-library verification of the completed P3 test3 checkpoint.

This checks archived evidence, not numerical execution. It never imports archived
code, extracts files, polls processes, reads key bodies or rereads the live parent.
"""
import argparse
import ast
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import tarfile


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, cap=16 * 1024 * 1024):
    path = Path(path)
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode), 'not a regular file: ' + str(path))
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        opened = os.fstat(fd)
        require((before.st_dev, before.st_ino) == (opened.st_dev, opened.st_ino), 'open identity')
        require(stat.S_ISREG(opened.st_mode) and opened.st_size <= cap, 'file size/type')
        parts = []
        total = 0
        while True:
            block = os.read(fd, min(1024 * 1024, cap - total + 1))
            if not block:
                break
            parts.append(block)
            total += len(block)
            require(total <= cap, 'read cap')
        raw = b''.join(parts)
        after = os.fstat(fd)
        require((opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) ==
                (after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'file changed during read')
        require(len(raw) == opened.st_size, 'short read')
    finally:
        os.close(fd)
    named = path.lstat()
    require((named.st_dev, named.st_ino) == (opened.st_dev, opened.st_ino), 'name changed')
    return raw


def pairs(rows):
    result = {}
    for key, value in rows:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def decode(raw):
    def bad_constant(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad_constant)


def archive(raw, expected):
    result = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as stream:
        members = stream.getmembers()
        require(len(members) == len(expected), 'archive member count')
        total = 0
        for member in members:
            name = member.name
            require(name in expected and name not in result, 'archive member name/duplicate')
            require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'archive path')
            require(member.isfile() and not member.issym() and not member.islnk(), 'archive type')
            require(0 <= member.size <= 4 * 1024 * 1024, 'archive member bound')
            total += member.size
            require(total <= 8 * 1024 * 1024, 'archive total bound')
            raw_member = stream.extractfile(member).read(member.size + 1)
            require(len(raw_member) == member.size and sha(raw_member) == expected[name], 'archive bytes')
            result[name] = raw_member
    require(set(result) == set(expected), 'archive closure')
    return result


def finite(value, nonnegative=True):
    require(type(value) in (int, float) and math.isfinite(value), 'finite numeric field')
    require(not nonnegative or value >= 0, 'nonnegative numeric field')


def fn(tree, name):
    found = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    require(len(found) == 1, 'unique function ' + name)
    return found[0]


def expression(text):
    return ast.dump(ast.parse(text, mode='eval').body, include_attributes=False)


def verify(root, current):
    root = Path(root)
    require(stat.S_ISDIR(root.lstat().st_mode), 'capsule directory')
    manifest_raw = read(root / 'artifact_hashes.json')
    manifest = decode(manifest_raw)
    found = set()
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs:
            require(stat.S_ISDIR((Path(directory) / name).lstat().st_mode), 'directory alias')
        for name in files:
            path = Path(directory) / name
            require(stat.S_ISREG(path.lstat().st_mode), 'file alias')
            found.add(path.relative_to(root).as_posix())
    require(found == set(manifest) | {'artifact_hashes.json'}, 'closed capsule file set')
    raw = {}
    for name, record in manifest.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'manifest path')
        data = read(root / name)
        require(set(record) == {'sha256', 'bytes'} and sha(data) == record['sha256'] and
                len(data) == record['bytes'], 'manifest bytes: ' + name)
        raw[name] = data
    critical = {
        'preparation/preparation.json': '80e3b7fab9e439df7fe62a42ef98ef69288ff366f5a0b4ad07dcaedc3cd75a65',
        'preparation/sources.json': '7f1fa4a084c0a5a83216154a2d49ebdb0d7694c1fb9df67459fcad9ac615e748',
        'preparation/auxiliary.json': '95134510d4e0ad5b680679a3dcc93f5cdaae11c581c92c41682d2b68279b13b1',
        'sources/actual-128.tar.gz': 'a5e8b41586e683bb6eb23d57266195bfcb0206acb94f2f1a4af9fef6636074eb',
        'sources/prepared-129.tar.gz': '07450d3d707c263616b4ffc8e1150ff0c45a821fa4b9b1f5a2cd867d6cbffec1',
        'raw/source-start.json': '7ac559c9143ef60aa688884b7a78a21bda49d024a725d182361362e9b2b36ac3',
        'raw/test3/source-after.json': '7ac559c9143ef60aa688884b7a78a21bda49d024a725d182361362e9b2b36ac3',
        'raw/launch-v1.json': 'f219750cc03acb90d05283d2076f53cb9623df62e35f0ef7b9e5021434b3d5d5',
    }
    for name, digest in critical.items():
        require(sha(raw[name]) == digest, 'fixed origin pin: ' + name)
    get = lambda name: decode(raw[name])
    prep = get('preparation/preparation.json')
    sources = get('preparation/sources.json')
    auxiliary = get('preparation/auxiliary.json')
    require(sources == get('raw/source-start.json') == get('raw/test3/source-after.json'), 'source map joins')
    require(len(sources) == 128 and len(auxiliary) == 1, 'source counts')
    require(auxiliary == {'repo/jaxite_word/API_REFERENCE.md':
        'ed1c589f0204ce878d26f0cac7ddf1d3446ae361214a99191df562ef6ee4961f'}, 'auxiliary sentinel')
    actual = archive(raw['sources/actual-128.tar.gz'], sources)
    prepared = archive(raw['sources/prepared-129.tar.gz'], {**sources, **auxiliary})
    require(all(actual[name] == prepared[name] for name in actual), 'two archives join')
    require(prep['source_map_sha256'] == critical['preparation/sources.json'] and
            prep['archive_sha256'] == critical['sources/prepared-129.tar.gz'], 'preparation joins')
    require(prep['original_baseline_dynamic_constants_override'] is False, 'original baseline')
    p = get('raw/test3/process.json')
    result = get('raw/test3/stdout.json')
    capture = get('capture.json')
    summary = get('summary.json')
    parent_name = capture['parent_partial_snapshot']
    require(re.fullmatch(r'raw/parent-partial-\d{8}T\d{6}Z\.json', parent_name), 'partial report filename')
    parent = get(parent_name)
    require(sha(raw[parent_name]) == capture['parent_snapshot_sha256'], 'partial report pin')
    require(parent['cases']['test3.in'] == p and p['result'] == result, 'result/process/partial parent joins')
    require(parent['passed'] is False and parent.get('complete') is not True and
            parent['active_case'] != 'test3.in' and parent['collector_pid'] == 1598331, 'partial parent scope')
    require(parent['source_hashes_start'] == sources and
            parent['source_archive_sha256'] == critical['sources/actual-128.tar.gz'], 'parent sources')
    require(set(name.split('/')[-1] for name in raw if name.startswith('raw/test3/')) ==
            {'process.json', 'stdout.json', 'stderr.log', 'time.txt', 'source-after.json'}, 'exact five case files')
    for key in ('passed', 'process_reaped', 'public_key_store_removed', 'owned_directory_handle_closed', 'source_files_unchanged'):
        require(p[key] is True, 'case terminal ' + key)
    require(type(p['returncode']) is int and p['returncode'] == 0 and p['pid'] == 1606928 and 'error' not in p, 'case exit')
    require(p['timeout_s'] is None and p['cleanup_grace_s'] == 30.0, 'case timeout policy')
    base = '/tmp/p3-original-six-preparation-20261105'
    command = ['/usr/bin/time', '-v', '-o', base + '/run-v1/test3/time.txt',
        '/home/jianming_gatech/miniconda3/envs/jaxite/bin/python', base + '/collector/check_p3_original_six.py',
        '--repo', base + '/checkout', '--child-runner', base + '/checkout/cadathlon/non_agent_pset/p3_cnn/src/runner.py',
        '--', '--case', base + '/checkout/cadathlon/non_agent_pset/p3_cnn/test/test3.in',
        '--baseline', '--warmups', '0', '--repeats', '1']
    require(p['command'] == command, 'original cold baseline command')
    fixture = decode(actual['repo/cadathlon/non_agent_pset/p3_cnn/test/test3.in'])
    require(fixture == {'purpose': 'synthetic-arithmetic-not-trained-accuracy', 'model': 'lenet',
        'batch': 4, 'weight_seed': 7203, 'input_seed': 8303, 'input_kind': 'uniform'}, 'original fixture')
    fixture_pin = prep['fixtures'][2]
    require(fixture_pin['case'] == 'test3.in' and fixture_pin['model'] == 'lenet' and fixture_pin['batch'] == 4, 'fixture metadata')
    for suffix, key in [('in', 'input_sha256'), ('out', 'output_sha256')]:
        require(sha(actual['repo/cadathlon/non_agent_pset/p3_cnn/test/test3.' + suffix]) == fixture_pin[key], 'fixture source pin')
    reference = decode(actual['repo/cadathlon/non_agent_pset/p3_cnn/test/test3.out'])
    require(reference['kind'] == 'plaintext-numerical-reference' and len(reference['logits']) == 4 and all(len(row) == 10 for row in reference['logits']), 'reference scope')
    for row in reference['logits']:
        for value in row:
            finite(value, False)
    require(result['case'] == 'test3.in' and result['batch'] == 4 and result['mode'] == 'HE-synthetic-smoke' and
            result['implementation'] == 'baseline.py' and result['device'] == 'TPU_0(process=0,(0,0,0,0))', 'result scope')
    for key in ('passed', 'mapping_release_included', 'cold_start_included'):
        require(result[key] is True, 'runner flag ' + key)
    require(result['trained_accuracy_evaluated'] is False, 'no trained accuracy')
    expected_metrics = {'max_abs_error': 9.141108009425025e-13, 'nrmse': 3.2013200584170407e-11,
        'top1_agreement': 1.0, 'total_s': 635.2648103709798, 'setup_s': 630.4757404160046,
        'median_s': 3.0013872089912184}
    for key, value in expected_metrics.items():
        finite(result[key]); require(result[key] == value, 'unnormalized recorded metric: ' + key)
    require(result['warmup_s'] == result['warmup_server_s'] == [], 'zero warmups')
    require(result['end_to_end_s'] == [3.0013872089912184] and result['server_s'] == [1.456408719997853], 'one cold repeat')
    finite(p['process_elapsed_s']); require(p['process_elapsed_s'] == 651.4186651210184, 'external time')
    require(0 < result['server_s'][0] <= result['end_to_end_s'][0] <= result['total_s'] <= p['process_elapsed_s'], 'timing order')
    require(result['setup_s'] + result['end_to_end_s'][0] <= result['total_s'], 'setup/request within total')
    time_text = raw['raw/test3/time.txt'].decode()
    require('Command being timed: "' + ' '.join(command[4:]) + '"' in time_text, 'time command')
    require(re.search(r'Exit status: 0\s*$', time_text) is not None, 'external exit')
    require(int(re.search(r'Maximum resident set size \(kbytes\): (\d+)', time_text).group(1)) == p['peak_child_rss_kib'] == 129900924, 'RSS join')
    wall = re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)', time_text).group(1)
    fields = wall.split(':'); wall_s = sum(float(value) * (60 ** power) for power, value in enumerate(reversed(fields)))
    require(abs(wall_s - p['process_elapsed_s']) < 0.1, 'rounded external wall time')
    require('A large amount of constants were captured during lowering (12.36GB total)' in raw['raw/test3/stderr.log'].decode(), 'preserved warning')
    require('RAW: File offset incorrectly aligned for file-backed THP:' in raw['raw/test3/stderr.log'].decode(), 'preserved hugepage diagnostic')
    rings = result['execution_rings']; require(len(rings) == 1, 'one reported ring')
    ring = rings[0]
    for key, value in {'degree':32768, 'num_slots':16384, 'num_q':16, 'num_p':4, 'dnum':4, 'max_level':7, 'security_bits':128}.items():
        require(type(ring[key]) is int and ring[key] == value, 'actual ring ' + key)
    require(ring['security_model'] == 'classical', 'cost model')
    q, p_towers = ring['q_towers'], ring['p_towers']
    require(len(q) == 16 and len(p_towers) == 4 and all(type(n) is int and n > 1 for n in q + p_towers), 'moduli')
    product = math.prod(q + p_towers)
    he_tree = ast.parse(actual['repo/jaxite_word/he_params.py'].decode())
    ceilings = [n.value for n in he_tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_OPENFHE_CEILINGS_RAW' for t in n.targets)]
    require(len(ceilings) == 1 and isinstance(ceilings[0], ast.Dict), 'original ceiling table')
    selected = [v for k, v in zip(ceilings[0].keys, ceilings[0].values) if ast.dump(k, include_attributes=False) == expression('(CostModel.CLASSICAL, 128)')]
    require(len(selected) == 1, 'classical128 table')
    ceiling = ast.literal_eval(selected[0])[32768]
    require(ceiling == 881 and product.bit_length() == 605 <= ceiling, 'integer security ceiling')
    require(ast.dump(fn(he_tree, 'modulus_product_is_secure').body[-1].value, include_attributes=False) == expression('openfhe_logq_bits(modulus_product) <= security_ceiling(degree, target)'), 'inclusive security expression')
    require(ast.dump(fn(he_tree, 'openfhe_logq_bits').body[-1].value, include_attributes=False) == expression('modulus_product.bit_length()'), 'integer GetMSB expression')
    require(abs(math.log2(math.prod(q)) - ring['log2_Q']) < 1e-10 and abs(math.log2(product) - ring['log2_QP']) < 1e-10, 'reported floating log joins')
    audit = get('source-audit.json')
    require(audit['security_integer_check']['product_decimal'] == str(product), 'product receipt')
    for label, record in audit['source_functions'].items():
        name, function = label.rsplit(':', 1)
        node = fn(ast.parse(actual[name].decode()), function)
        require(record == {'line': node.lineno, 'end_line': node.end_lineno, 'AST_sha256': sha(ast.dump(node, include_attributes=False).encode())}, 'source function pin')
    require(len(audit['source_functions']) == 14, 'source definition count')
    runner = fn(ast.parse(actual['repo/cadathlon/non_agent_pset/p3_cnn/src/runner.py'].decode()), 'run')
    tests = {ast.dump(n.test, include_attributes=False) for n in ast.walk(runner) if isinstance(n, ast.If) and any(isinstance(x, ast.Raise) for x in n.body)}
    require(expression('not np.all(np.abs(output - expected) <= 0.02 + 0.01 * np.abs(expected))') in tests, 'every-logit gate')
    require(expression('errors["top1_agreement"] < 0.99') in tests, 'top1 gate')
    release_calls = [n.lineno for n in ast.walk(runner) if isinstance(n, ast.Call) and ast.dump(n.func, include_attributes=False) == expression('mapping.release')]
    total_writes = [n.lineno for n in runner.body if isinstance(n, ast.Assign) and any(ast.dump(t, include_attributes=False) == ast.dump(ast.parse('report["total_s"] = 0').body[0].targets[0], include_attributes=False) for t in n.targets)]
    require(len(release_calls) == len(total_writes) == 1 and release_calls[0] < total_writes[0], 'release before total report')
    live = get('raw/launch-v1-root-readback.json')
    require(live['original_session'] == 81235 and live['launch_tool_chunk'] == '58a623' and live['terminal'] is False, 'historical root live provenance')
    # The launch receipt is historical parent/test1 provenance, not test3 live identity.
    helper = fn(ast.parse(actual['collector/process_helpers.py'].decode()), 'run_case')
    popens = [n for n in ast.walk(helper) if isinstance(n, ast.Call) and
              isinstance(n.func, ast.Attribute) and ast.unparse(n.func) == 'subprocess.Popen']
    require(len(popens) == 1 and any(k.arg == 'start_new_session' and
            isinstance(k.value, ast.Constant) and k.value.value is True for k in popens[0].keywords),
            'original wrapper process-group ownership')
    absence = get('process-absence.json')
    require(absence['pids'] == {'1606928':'absent_ESRCH'} and
            absence['pgids'] == {'1606928':'absent_ESRCH'}, 'late wrapper absence observation')
    key_parent = '/dev/shm/cross-p3-original-six-key-parent-f383y30c'
    key_child = key_parent + '/cross-transformer-case-vg_ja2yj'
    require(p['owned_public_key_directory'] == absence['child_key_directory'] == key_child, 'key child identity')
    require(p['owned_public_key_directory_identity'] == {'device':24, 'inode':4067}, 'recorded child identity')
    require(absence['parent_key_directory'] == key_parent and absence['parent_directory_statted'] is False,
            'untouched key parent')
    require(absence['inner_pid_observed'] is False and absence['unknown_inner_pid_absence_claim'] is False,
            'no invented inner PID observation')
    for key in ('proc_entries_absent', 'child_key_directory_absent'):
        require(absence[key] is True, 'absence flag')
    for key in ('parent_directory_removed', 'parent_directory_contents_read', 'original_session_polled', 'wait_authority_inherited', 'live_observation_inherited'):
        require(absence[key] is False, 'scope flag')
    require(datetime.datetime.fromisoformat(absence['observed_utc']) >= datetime.datetime.fromisoformat(live['observed_utc']), 'observation chronology')
    require(capture['parent_terminal_claim'] is False and capture['active_case_numerical_or_key_body_read'] is False and capture['key_body_read'] is False and capture['candidate_import_or_execution'] is False, 'capture scope')
    require(summary['completed_cases_admitted'] == 1 and summary['scheduled_cases'] == 6 and summary['sweep_complete'] is False and summary['sweep_passed_claim'] is False and summary['trained_accuracy_evaluated'] is False, 'summary limits')
    require(summary['parent_snapshot'] == parent_name and summary['parent_snapshot_active_case'] == parent['active_case'], 'summary parent join')
    for target, source in [('max_abs_error','max_abs_error'), ('nrmse','nrmse'), ('top1_agreement','top1_agreement'), ('runner_total_s','total_s'), ('setup_s','setup_s')]:
        require(summary[target] == result[source], 'summary metric')
    require(summary['external_process_elapsed_s'] == p['process_elapsed_s'], 'summary external time')
    origins = get('origins.json')
    for name, origin in origins.items():
        require(name in raw and sha(raw[name]) == origin['sha256'] and len(raw[name]) == origin['bytes'], 'origin copy join')
        if origin.get('mutable_origin'):
            require(name == parent_name and origin['snapshot_only'] is True, 'only live origin')
        elif current:
            require(sha(read(origin['path'])) == origin['sha256'], 'current stable origin')
    require(len(audit['current_sources']) == 129, 'source path inventory')
    observed_names = set()
    for row in audit['current_sources']:
        require(row['name'] not in observed_names and {**sources, **auxiliary}[row['name']] == row['sha256'], 'current source inventory')
        observed_names.add(row['name'])
        if current:
            require(sha(read(row['prepared_path'])) == sha(read(row['original_path'])) == row['sha256'], 'current/original source equality')
    require(observed_names == set(prepared), 'current source closure')
    # Do not reread the mutable original report or perform any new process check.
    require(read(root / 'artifact_hashes.json') == manifest_raw, 'stable index')
    for name, data in raw.items():
        require(read(root / name) == data, 'stable capsule bytes')
    return {'passed':True, 'scope':'P3 original test3 only; third of six, one case admitted here', 'indexed_files':len(manifest),
        'artifact_hashes_sha256':sha(manifest_raw), 'actual_sources':128, 'prepared_sources':129,
        'source_definitions':14, 'result_process_parent_equal':True, 'product_bit_length':605,
        'security_ceiling':881, 'max_abs_error':result['max_abs_error'], 'nrmse':result['nrmse'],
        'top1_agreement':result['top1_agreement'], 'runner_total_s':result['total_s'],
        'external_process_elapsed_s':p['process_elapsed_s'], 'current_public_origins_checked':current,
        'live_parent_reread':False, 'new_process_poll':False, 'candidate_execution':False,
        'independent_numeric_recomputation':False, 'final_sweep_claim':False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--current', action='store_true')
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.root, arguments.current), sort_keys=True, allow_nan=False))
