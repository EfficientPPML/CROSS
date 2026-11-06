#!/usr/bin/env python3
"""Verify public evidence only; never follow model-store descriptors or execute sources."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import tarfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
LIMIT = 32 * 1024 * 1024
PINS = {
    'actual/source/sources.json': 'b73e669b0cc2bc34d7930f2a8a4c620be719fe663dfa6ae7b12df7cae22c2bc0',
    'actual/source/sources.tar.gz': '06a7ae13b6df2adb9808af69d7906eec1e86ed811c419da9c1d064479fe95a25',
    'focused/sources-before.json': '319b4503b63635f4ba9ddf675a97a2a481fe6b67df503919f91bec3ea356057d',
    'focused/sources.tar.gz': 'e8747c20a234cd1272d16a43a7fbc2c93b5555136e48d174843697c1cedf419e',
    'focused/report.json': '4fb5945fc2555d2afaaee21676817561aac7079dbe2041f1df379cffbc211630',
    'focused/owner-terminal.json': 'bc252201ce44537fc04cab38ed5dddc864c6b5be749e7e3a410c66920f9af9c2',
    'metadata/result/expected_stores.json': 'c3023d3782b3aa3b9b9607f07d291162383281a59b86be71fb9cb185fbcf8604',
    'actual/report.json': 'cc8aaab893268b96692d9998412b121b97f7cfdd5ea47c5f3a60949fe620203c',
    'actual/admission/admission.json': '38fa418426d4e14c3e06ac76e93b9f63953ce5bf747fe44fbde3f0d5bb716fed',
    'actual/admission/outer.json': 'fdf261a2ff2358a2da2cccb02d953b002ac13dab14884423586c9d983efbcb90',
    'actual/admission/root-review.json': '0bfd8b1d69c2724be0c52ac03a97150a48351f08dbd1696fe67b2ece0442580e',
    'actual/admission/expected-pins-authority28.json': '21eddb3ba457d262a14dc0391f357c23f476692231d641b6af1d4db014dce155',
    'actual/launch/verifier-terminal.json': 'f50e8656102a6ac46c9d7e69c28cc3b2da8699e53639540e48079d21cade1226',
}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()

def relative(name):
    p = PurePosixPath(name)
    require(type(name) is str and str(p) == name and not p.is_absolute() and '..' not in p.parts, 'unsafe relative name')
    return p

def file_bytes(path):
    s = path.lstat()
    require(stat.S_ISREG(s.st_mode) and not path.is_symlink() and s.st_size <= LIMIT, 'nonregular/large public file: '+str(path))
    return path.read_bytes()

def raw(name):
    relative(name)
    return file_bytes(ROOT/name)

def digest(name):
    return sha(raw(name))

def pairs(values):
    result = {}
    for key, value in values:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result

def decode(data):
    return json.loads(data, object_pairs_hook=pairs, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

def read(name):
    return decode(raw(name))

def hash_map(contents):
    return {name: sha(data) for name, data in contents.items()}

def archive(name, expected):
    contents = {}
    total = 0
    with tarfile.open(ROOT/name, 'r:gz') as tf:
        for member in tf:
            relative(member.name)
            require(member.isfile() and member.name not in contents and member.size <= LIMIT, 'invalid archive member')
            total += member.size
            require(total <= LIMIT, 'oversize source archive')
            data = tf.extractfile(member).read()
            require(len(data) == member.size, 'short archive member')
            contents[member.name] = data
    require(hash_map(contents) == expected, 'archive/map mismatch: '+name)
    return contents

def link(record, field, name):
    require(record[field] == digest(name), 'broken link '+field+' -> '+name)

def normal_zero(record, code, reaped):
    require(type(record[code]) is int and record[code] == 0 and record[reaped] is True, 'missing normal exit/reap')

def current_map(path, expected):
    # Only explicitly declared public source files; never recurse into other paths.
    base = Path(path)
    for name, expected_sha in expected.items():
        relative(name)
        require(sha(file_bytes(base/name)) == expected_sha, 'changed current source: '+str(base/name))

def main(current=False):
    index = read('files.sha256.json')
    whitelist = read('public-whitelist.json')
    names = whitelist['files']
    require(len(names) == 115 and len(set(names)) == 115, 'closed115 whitelist')
    require(set(index) == set(names)-{'files.sha256.json'}, 'index coverage')
    observed = set()
    for p in ROOT.rglob('*'):
        require(not p.is_symlink(), 'capsule symlink')
        if p.is_file():
            observed.add(str(p.relative_to(ROOT)))
    require(observed == set(names), 'unexpected/missing capsule file')
    for name, expected_sha in index.items():
        require(digest(name) == expected_sha, 'index digest: '+name)
    for name, expected_sha in PINS.items():
        require(digest(name) == expected_sha, 'external fixed pin: '+name)
    origins = read('copy-origins.json')['files']
    require(len(origins) == 108 and set(origins) <= set(index), 'copy origin closure')
    for name, row in origins.items():
        require(digest(name) == row['sha256'] and len(raw(name)) == row['bytes'], 'copy identity')
    source42 = read('actual/source/sources.json')
    source54 = read('focused/sources-before.json')
    require(len(source42) == 42 and len(source54) == 54, 'source counts')
    runtime = archive('actual/source/sources.tar.gz', source42)
    focused = archive('focused/sources.tar.gz', source54)
    require({n[8:]: h for n,h in source54.items() if n.startswith('runtime/')} == source42, 'runtime42 subtree')
    require(read('focused/sources-after.json') == read('focused/execution-after.json') == source54, 'focused source pre/post')
    for name in origins:
        if name.startswith('actual/source/readable/'):
            require(raw(name) == runtime[name.removeprefix('actual/source/readable/')], 'readable/runtime mismatch')
    predecessor = read('history/published-predecessor-links.json')
    parent29 = predecessor['parent29_sources']
    require(len(parent29) == 29 and sha(canonical(parent29)) == predecessor['parent29_manifest_sha256'], 'parent29 map')
    require(all(source42.get(n) == h for n,h in parent29.items()), 'parent29 changed')
    old40 = read('history/runtime40/sources.json')
    require(len(old40) == 40, 'old40 count')
    old40_bytes = {n:runtime[n] for n in old40}
    for n in ('model_entry.py','model_validation_process.py'):
        old40_bytes[n] = focused['history/source40-'+n+'.txt']
    old40_bytes['model_source_inventory.json'] = raw('history/runtime40/model_source_inventory.json')
    require(hash_map(old40_bytes) == old40, 'thin40 reconstruction')
    old54 = read('history/focused-v1/sources.json')
    old54_bytes = dict(focused)
    for n in ('README.md','test_provenance.json','tests/support.py'):
        old54_bytes[n] = raw('history/focused-v1/'+n)
    require(len(old54) == 54 and hash_map(old54_bytes) == old54, 'thin old54 reconstruction')
    config = read('focused/run_config.json')
    report = read('focused/report.json')
    audit = read('focused/worker-audit.json')
    owner = read('focused/owner-terminal.json')
    nodes = config['selected_nodes']
    require(len(nodes) == 62 and len(set(nodes)) == 62, '62 unique nodes')
    require(report['selected_nodes'] == audit['collected_nodes'] == nodes, 'exact node collection')
    require(sha(canonical(nodes)) == config['selected_nodes_sha256'], 'selected nodes digest')
    require(sha(canonical(config['selected_xml_cases'])) == config['selected_xml_cases_sha256'], 'XML case digest')
    cases = list(ET.fromstring(raw('focused/tests.xml')).iter('testcase'))
    observed_cases = [{'classname': x.get('classname'), 'name': x.get('name')} for x in cases]
    require(observed_cases == config['selected_xml_cases'] == report['selected_xml_cases'], 'exact ordered XML cases')
    require(len(cases) == 62 and all(not any(x.find(t) is not None for t in ('failure','error','skipped')) for x in cases), 'XML failures/skip')
    require(report['tests'] == owner['tests'] == {'tests':62,'errors':0,'failures':0,'skipped':0}, 'test totals')
    require(report['passed'] is True and report['synthetic_only'] is True and report['actual_execution_authorized'] is False, 'focused scope')
    normal_zero(report, 'child_returncode', 'child_reaped')
    require(report['child_group_absent'] is True, 'focused child group')
    normal_zero(owner, 'tool_exit_code', 'tool_session_reaped')
    require(owner['session_id'] == 20688 and owner['authoritative_wait_tool_chunk'] == '751336', 'owner terminal')
    require(audit['denied'] == audit['loaded_backend'] == [] and audit['pytest_exit_code'] == 0, 'focused audit')
    require(audit['affinity'] == [108,109,110,111], 'focused affinity')
    expected_numpy = config['runtime_source_expectations']
    require(len(expected_numpy) == 4 and audit['runtime_sources_before'] == audit['runtime_sources_after'] == expected_numpy, 'NumPy source4')
    require(set(audit['runtime_module_origins']) == set(config['runtime_module_origins']) and len(config['runtime_module_origins']) == 3, 'NumPy origin3')
    for mod,path in config['runtime_module_origins'].items():
        require(audit['runtime_module_origins'][mod] == {'file':path,'spec':path}, 'NumPy origin')
    for field,name in [('report_sha256','report.json'),('tests_xml_sha256','tests.xml'),('tests_log_sha256','tests.log'),('worker_audit_sha256','worker-audit.json'),('live_observation_sha256','live-observation.json')]:
        link(owner,field,'focused/'+name)
    for field,name in [('driver_sha256','executed_driver.py'),('worker_driver_sha256','worker_driver.py'),('config_sha256','run_config.json'),('source_manifest_sha256','sources-before.json'),('source_archive_sha256','sources.tar.gz')]:
        link(report,field,'focused/'+name)
    require(audit['pid'] == owner['observed_pids']['worker_pid'] and audit['pgid'] == owner['observed_pids']['worker_pgid'], 'owned worker identity')
    require(len(owner['child_terminals']) == 15, 'child receipt count')
    for row in owner['child_terminals']:
        name = 'focused/test-child-terminals/'+row['file']
        require(digest(name) == row['sha256'], 'child receipt hash')
        child = read(name)
        require(child['pid'] == row['pid'] and child['pgid'] == row['pgid'] and child['exit_code'] == row['exit_code'], 'child identity/exit')
        require(child['reaped'] is True and child['process_group_absent'] is True and row['independently_observed_absent_now'] is True, 'child terminal')
    rootfocused = read('reviews/root/focused/readonly-terminal-review.json')
    require(rootfocused['passed'] is True and rootfocused['independent_live_observation'] is False, 'focused root scope')
    link(rootfocused,'owner_terminal_sha256','focused/owner-terminal.json')
    link(rootfocused,'script_sha256','reviews/root/focused/verify_readonly_terminal.py')
    for prefix,count in [('metadata/original6',5),('metadata/executed8',7)]:
        metaindex = read(prefix+'/files.json')
        require(len(metaindex) == count, 'metadata source index')
        for n,h in metaindex.items():
            name = prefix+'/'+n
            if prefix.endswith('executed8') and n == 'planned_names.json':
                shared=read(prefix+'/shared-files.json')[n]
                name=shared['capsule_path'];require(shared['sha256']==h,'shared metadata source')
            require(digest(name) == h, 'metadata source hash')
    acquisition = read('metadata/result/acquisition.json')
    expected = read('metadata/result/expected_stores.json')
    require(len(expected['roots']) == 3 and len(expected['files']) == 198, 'expectation counts')
    require([(x['family'],x['name']) for x in expected['files']] == sorted({(x['family'],x['name']) for x in expected['files']}), 'sorted unique file expectations')
    require(acquisition['body_files_opened'] == 0 and acquisition['body_integrity_admitted'] is False and acquisition['body_hashes_recomputed'] is False and acquisition['observations'] == 2, 'metadata-only scope')
    link(acquisition,'expected_stores_sha256','metadata/result/expected_stores.json')
    link(acquisition,'script_sha256','metadata/executed8/acquire_expectations.py')
    link(acquisition,'source_pins_sha256','metadata/executed8/source_pins.json')
    metaterm = read('reviews/root/metadata-terminal-review.json')
    require(metaterm['passed'] is True and metaterm['direct_exec_terminal_chunk'] == '106464' and metaterm['live_observation_performed'] is False, 'metadata terminal scope')
    normal_zero(metaterm,'direct_exec_exit_code','direct_completion_reaped')
    link(metaterm,'acquisition_sha256','metadata/result/acquisition.json')
    link(metaterm,'expected_stores_sha256','metadata/result/expected_stores.json')
    link(metaterm,'source_index_sha256','metadata/executed8/files.json')
    actual = read('actual/report.json')
    require(actual['source_hashes_before'] == actual['source_hashes_after'] == source42, 'actual source checks')
    require(actual['passed'] is True and actual['complete'] is True and actual['audit_denials'] == [] and actual['error_type'] is None, 'actual report acceptance')
    require(all(x is True for x in actual['checks'].values()) and len(actual['checks']) == 10, 'actual integrity/cleanup checks')
    require(actual['counts'] == {'gamma_raw_bytes':524288,'gamma_vectors':113,'logical_matrices':198,'physical_matrices':197,'physical_matrix_raw_bytes':4767875072}, 'actual coverage')
    require(actual['stores'] == expected, 'observed store/expectation equality')
    require(actual['scope']['parameter_body_reads_performed'] is True and all(v is False for k,v in actual['scope'].items() if k != 'parameter_body_reads_performed'), 'actual readonly scope')
    for field,name in [('execution_grant_sha256','actual/launch/grant.json'),('source_manifest_sha256','actual/source/sources.json'),('source_archive_sha256','actual/source/sources.tar.gz')]:
        link(actual,field,name)
    grant = read('actual/launch/grant.json')
    require(grant['execute'] is True and grant['role'] == 'validate_existing_model_stores' and grant['parameter_body_read_only'] is True, 'actual grant role')
    require(grant['expected_stores']['sha256'] == digest('metadata/result/expected_stores.json'), 'grant expectations')
    require(grant['prelaunch_review']['sha256'] == digest('actual/launch/prelaunch-review.json'), 'grant prelaunch')
    outer = read('actual/admission/outer.json')
    normal_zero(outer,'exit_code','reaped')
    require(outer['normal_return'] is True and outer['process_group_absent'] is True and outer['pid']==outer['pgid']==actual['process']['pid']==actual['process']['pgid']==56991, 'actual normal terminal')
    for field,name in [('report_sha256','actual/report.json'),('execution_grant_sha256','actual/launch/grant.json'),('source_manifest_sha256','actual/source/sources.json'),('source_archive_sha256','actual/source/sources.tar.gz')]:
        link(outer,field,name)
    admission = read('actual/admission/admission.json')
    require(admission['passed'] is True and admission['model_integrity_admitted'] is True and admission['scientific_acceptance_claimed'] is False and admission['atomic_snapshot_claimed'] is False, 'admission scope')
    require(admission['tool_session_id'] == 27288 and admission['terminal_tool_chunk'] == '724ea7' and admission['post_terminal_bodies_read'] is False and admission['normal_zero_direct_reap'] is True, 'actual owner admission')
    require(admission['model_execution_authorized'] is False and admission['key_generation_authorized'] is False, 'no downstream execution')
    for field,name in [('actual_report_sha256','actual/report.json'),('outer_sha256','actual/admission/outer.json'),('root_review_sha256','actual/admission/root-review.json'),('expected_pins_sha256','actual/admission/expected-pins-authority28.json'),('script_sha256','actual/launch/verify_terminal_and_admit.py'),('independent_live_observation_sha256','actual/launch/live-observation.json'),('launch_review_sha256','actual/launch/execution-launch-review.json'),('store_binding_sha256','metadata/result/expected_stores.json')]:
        link(admission,field,name)
    expectedpins = read('actual/admission/expected-pins-authority28.json')
    require(expectedpins['producer_source_map'] == source42, 'admitted producer map')
    link(expectedpins,'consumer_source_manifest_sha256','actual/admission/consumer28-sources.json')
    for field,name in [('validation_report_sha256','actual/report.json'),('validation_outer_sha256','actual/admission/outer.json'),('root_review_sha256','actual/admission/root-review.json'),('execution_grant_sha256','actual/launch/grant.json')]:
        link(expectedpins,field,name)
    terminal = read('actual/launch/verifier-terminal.json')
    require(terminal['passed'] is True and terminal['authoritative_tool_chunk'] == '9fd0d3' and terminal['independent_live_observation'] is False and terminal['no_session_id_allocated'] is True, 'root verifier terminal scope')
    normal_zero(terminal,'direct_completion_exit_code','direct_process_reaped')
    require(len(terminal['files'])==4,'root output count')
    for n,h in terminal['files'].items():
        require(digest('actual/admission/'+n)==h,'root verifier output')
    link(terminal,'script_sha256','actual/launch/verify_terminal_and_admit.py')
    link(terminal,'source_review_sha256','reviews/transformers/admission/review.json')
    provenance = read('reviews/root/public-provenance-current-review.json')
    require(provenance['passed'] is True and provenance['public_pins_verified']==38 and provenance['borrowed_parameter_body_read'] is False, 'public provenance receipt')
    link(provenance,'pins_sha256','provenance/public-evidence-pins.json')
    if current:
        for name,row in origins.items():
            require(sha(file_bytes(Path(row['path']))) == row['sha256'], 'changed public copy origin: '+name)
        current_map('/tmp/qwen-model-store-readonly-v2-20261104',source42)
        current_map(config['source_root'],source54)
        current_map(str(Path(config['proof_root'])/'execution'),source54)
        for p,h in expected_numpy.items():
            require(sha(file_bytes(Path(p)))==h,'changed declared NumPy source')
    print(json.dumps({'passed':True,'current':current,'files':len(names),'indexed':len(index),'copied_public_files':len(origins),'actual_sources':42,'focused_sources':54,'focused_tests':62,'test_child_receipts':15,'model_store_descriptors_followed':False,'candidate_imports_or_execution':False,'scope':'Recorded readonly integrity proof and its public preparation; no future execution authority.'},sort_keys=True))

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current',action='store_true')
    main(parser.parse_args().current)
