#!/usr/bin/env python3
"""Read public capsule bytes only; never import or execute archived sources."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile
import xml.etree.ElementTree as ET


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    before = path.lstat()
    assert stat.S_ISREG(before.st_mode) and not path.is_symlink(), str(path)
    raw = path.read_bytes()
    after = path.lstat()
    assert (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    assert len(raw) == after.st_size
    return raw


def run(root, current=False):
    def raw(name):
        assert name and str(PurePosixPath(name)) == name and not name.startswith('/') and '..' not in PurePosixPath(name).parts
        return read(root / name)
    def obj(name):
        return json.loads(raw(name))
    whitelist = obj('WHITELIST.json')
    assert whitelist == sorted(set(whitelist))
    actual = []
    for base, dirs, files in os.walk(root, followlinks=False):
        assert all(not (Path(base) / d).is_symlink() for d in dirs)
        actual.extend(str((Path(base) / n).relative_to(root)) for n in files)
    assert sorted(actual) == whitelist
    sums = {}
    for line in raw('SHA256SUMS').decode().splitlines():
        sha, name = line.split('  ', 1)
        assert name not in sums
        sums[name] = sha
    assert set(sums) == set(whitelist) - {'SHA256SUMS'}
    for name, sha in sums.items():
        assert digest(raw(name)) == sha, name
    origins = obj('copy_origins.json')
    destinations = [row['destination'] for row in origins['copies']]
    assert len(destinations) == len(set(destinations))
    for row in origins['copies']:
        value = raw(row['destination'])
        assert digest(value) == row['sha256'] and len(value) == row['bytes']
    assert set(destinations) | set(origins['generated']) == set(whitelist)
    assert set(destinations).isdisjoint(origins['generated'])
    summary = obj('summary.json')
    archives = {}
    for label, count in [('client', 24), ('keygen', 32)]:
        entry = summary['proofs'][label]
        base = label + '/proof/'
        mapping = obj(base + 'sources-before.json')
        assert len(mapping) == entry['source_count']
        assert digest(raw(base + 'sources-before.json')) == entry['source_manifest_sha256']
        assert digest(raw(base + 'sources.tar.gz')) == entry['source_archive_sha256']
        assert raw(base + 'sources-before.json') == raw(base + 'sources-after.json') == raw(base + 'execution-after.json')
        blobs = {}
        with tarfile.open(root / (base + 'sources.tar.gz'), 'r:gz') as archive:
            for member in archive:
                assert member.isfile() and member.name in mapping and member.name not in blobs
                assert member.size <= 64 * 1024 * 1024
                value = archive.extractfile(member).read()
                assert len(value) == member.size and digest(value) == mapping[member.name]
                blobs[member.name] = value
        assert set(blobs) == set(mapping)
        archives[label] = blobs
        if current:
            source = Path(entry['current_source'])
            for name, sha in mapping.items():
                assert digest(read(source / name)) == sha, str(source / name)
        report = obj(base + 'report.json')
        assert report['passed'] is True and report['sources_unchanged'] is True
        assert report['source_manifest_sha256'] == entry['source_manifest_sha256']
        assert report['source_archive_sha256'] == entry['source_archive_sha256']
        assert type(report['child_returncode']) is int and report['child_returncode'] == 0
        assert report['child_reaped'] is True and report['child_group_absent'] is True
        assert report['tests'] == dict(tests=count, errors=0, failures=0, skipped=0)
        for key in ('actual_execution_authorized', 'actual_key_generation', 'backend_context_construction', 'model_execution', 'encrypted_execution', 'numpy_imported'):
            assert report[key] is False
        audit = obj(base + 'worker-audit.json')
        assert audit['denied'] == audit['loaded_backend'] == []
        assert audit['runtime_module_origins'] == audit['runtime_sources_before'] == audit['runtime_sources_after'] == {}
        assert audit['pytest_exit_code'] == 0 and audit['numpy_imported'] is False
        assert audit['affinity'] == entry['affinity']
        assert audit['collected_nodes'] == report['selected_nodes']
        live = obj(base + 'live-observation.json')
        assert (audit['pid'], audit['pgid']) == (live['worker_pid'], live['worker_pgid'])
        xml = ET.fromstring(raw(base + 'tests.xml'))
        cases = list(xml.iter('testcase'))
        assert len(cases) == count
        assert [{'classname': x.attrib['classname'], 'name': x.attrib['name']} for x in cases] == report['selected_xml_cases']
        assert all(not list(x) for x in cases)
        reviewbase = 'root-review/' + entry['root_label']
        review = obj(reviewbase + '-terminal-review.json')
        assert review['passed'] is True and review['independent_live_observation'] is False
        for name, sha in review['evidence'].items():
            assert digest(raw(base + name)) == sha
        for field, suffix in [('direct_owner_receipt_sha256', '-direct-owner.json'), ('root_preexecution_review_sha256', '-preexecution-review.json'), ('root_run_grant_sha256', '-run-grant.json')]:
            assert digest(raw(reviewbase + suffix)) == review[field]
        owner = obj(reviewbase + '-direct-owner.json')
        assert owner['exit_code'] == 0 and owner['direct_session_reaped'] is True
        assert owner['automatic_retry'] is False and owner['healthy_execution_deadline'] is False
        assert digest(raw('root-review/root_proc_self_terminal_20261104.py')) == review['review_script_sha256']
    for name, binding in summary['readable_source_bindings'].items():
        assert raw(name) == archives[binding['archive']][binding['member']]
    # Thin predecessor reconstruction: exact final archives plus explicit old files.
    for lineage in summary['lineages']:
        files = archives[lineage['archive']]
        prefix = lineage['prefix']
        prior = {name[len(prefix):]: value for name, value in files.items() if name.startswith(prefix)}
        for name, before in lineage['replace'].items():
            prior[name] = raw(before)
        expected = obj(lineage['map'])
        assert {name: digest(value) for name, value in prior.items()} == expected, lineage['label']
    # Exact retained tiny child evidence; never revisit original pytest trees.
    client_map = obj('client/children/files.json')
    assert len(client_map) == 96
    client_nodes = obj('client/proof/report.json')['selected_nodes']
    pids = []
    for i, node in enumerate(client_nodes):
        base = f'client/children/child-artifacts/{i:02d}/'
        receipt = obj(base + 'receipt.json')
        owner = obj(base + 'owner.json')
        assert receipt['passed'] is True and owner['waited'] is True and owner['returncode'] == 0 and owner['group_absent'] is True
        assert owner['receipt_sha256'] == digest(raw(base + 'receipt.json'))
        assert receipt['pid'] == receipt['pgid'] == owner['pid'] == owner['pgid']
        assert receipt['case'] == node.split('[', 1)[1][:-1]
        assert receipt['baseline'] == ('test_old464_' in node)
        assert receipt['body_content_recorded'] is False and receipt['backend_imported'] is False
        assert receipt['actual_capacity_check'] is False and receipt['real_role_admission'] is False
        assert receipt['original_open_restored'] is True
        pids.append(receipt['pid'])
    assert len(set(pids)) == 24
    for name, sha in client_map.items():
        assert digest(raw('client/children/' + name)) == sha
    keygen_map = obj('keygen/children/files.json')
    observations = sorted(name for name in keygen_map if name.startswith('children/'))
    assert len(observations) == 32
    keygen_nodes = obj('keygen/proof/report.json')['selected_nodes']
    pids = []
    for name, node in zip(observations, keygen_nodes):
        rec = obj('keygen/children/' + name)
        assert rec['passed'] is True and rec['scope_restored'] is True
        assert rec['pid'] == rec['pgid']
        assert rec['case'] == node.split('[', 1)[1][:-1]
        assert rec['backend_attempts'] == rec['backend_modules'] == []
        assert rec['resource_capacity_evaluated'] is False and rec['actual_generation_authorized'] is False
        pids.append(rec['pid'])
    assert len(set(pids)) == 32
    for name, sha in keygen_map.items():
        assert digest(raw('keygen/children/' + name)) == sha
    print(json.dumps({'passed': True, 'indexed_files': len(sums), 'total_files': len(whitelist), 'current_public_sources_checked': current, 'client_cases': 24, 'keygen_cases': 32, 'client_child_artifacts': 96, 'keygen_child_observations': 32, 'tests_rerun': False, 'backend_imports': False, 'capacity_observations': False, 'borrowed_body_reads': False}, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--current', action='store_true')
    args = parser.parse_args()
    run(Path(__file__).resolve().parent, args.current)
