"""Source-only harness-v2 preparation; never import or execute candidate files."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

H = Path(__file__).resolve().parent
B = H.parent / 'preparation-v1'
S = Path('/tmp/qwen-server-snapshot-integration-tests-v2-20261104')
SH = Path('/tmp/qwen-server-snapshot-integration-test-handoff-v2-20261104')
sha = lambda raw: hashlib.sha256(raw).hexdigest()
def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
def compact(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
def inventory(root):
    result = {}
    for path in sorted(root.rglob('*')):
        assert not path.is_symlink()
        if path.is_file(): result[path.relative_to(root).as_posix()] = sha(path.read_bytes())
    return result

expected_before = {
    'handoff.json': 'ad7b69f004da28a838094861bd384bc8759556951991252bff0f6691a86dedfe',
    'executed_driver.py': '77862d27858c55aee305223507e7feadf3d20376511d194f0d41b72dadb23057',
    'worker_driver.py': '9ee25a101d4611e17b376d9b4729dae44bcf9f804d34a72ae340096337d65361',
    'run_config.json': 'd804adc4a8b4f2d94bc21ec95976b5a3f43c2a003b0b03c7734e7125ad701090',
    'launch-proposal.json': '4526412ca6bfb7fbc06d46530ec820f353187285dc2249e290470da973b4fda3',
}
for name, digest in expected_before.items(): assert sha((B / name).read_bytes()) == digest
preserved_files = {p.name: sha(p.read_bytes()) for p in B.iterdir() if p.is_file()}
assert not any(p.is_symlink() for p in B.iterdir())
mapping_raw = (SH / 'sources.json').read_bytes()
mapping = json.loads(mapping_raw)
assert sha(mapping_raw) == '7c4f95e68dabba24ee489bfe05ddd77b5de0ca5e7ba1a3e77139ef57a410bbd4'
assert mapping_raw == compact(mapping) and len(mapping) == 405
assert inventory(S) == mapping
assert sha((SH / 'sources.tar.gz').read_bytes()) == '7814e934402820f57da5084c6cefc4d0a924cc5e68a06c219bdeacd47d03ee59'
assert sha((SH / 'handoff.json').read_bytes()) == '57498a8ea7a0b03d75f4088d173434ec4bca218094e982ed5531c183c748ffa5'
assert not (H / 'execution').exists()
(H / 'execution').mkdir()
for name, digest in mapping.items():
    data = (S / name).read_bytes()
    assert sha(data) == digest
    target = H / 'execution' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
for origin, target in [('sources.json', 'sources-before.json'), ('sources.tar.gz', 'sources.tar.gz'), ('handoff.json', 'source-handoff.json')]:
    shutil.copyfile(SH / origin, H / target)

old_config = json.loads((B / 'run_config.json').read_bytes())
config = dict(old_config)
config.update(source_root=str(S), proof_root=str(H), source_count=405,
              source_manifest_sha256=sha(mapping_raw),
              source_archive_sha256=sha((SH / 'sources.tar.gz').read_bytes()),
              pytest_basetemp=str(H / 'pytest-owned'), owned_tmpdir=str(H / 'tmp'))
changed_config_fields = sorted(k for k in old_config if config[k] != old_config[k])
assert changed_config_fields == ['owned_tmpdir', 'proof_root', 'pytest_basetemp', 'source_archive_sha256', 'source_count', 'source_manifest_sha256', 'source_root']
assert config['cpu_affinity'] == [112, 113, 114, 115]
assert len(config['selected_nodes']) == 36
selection = json.loads((S / 'selected_tests.json').read_bytes())
assert config['selected_nodes'] == selection['static_nodeids']
assert config['selected_files'] == selection['selected_files'] == ['test_snapshot_integration.py']
assert config['runtime_source_expectations'] == json.loads((S / 'runtime_sources.json').read_bytes())
assert len(config['runtime_source_expectations']) == 391
(H / 'run_config.json').write_bytes(canonical(config))
config_sha = sha((H / 'run_config.json').read_bytes())

old_driver = (B / 'executed_driver.py').read_text()
worker = (B / 'worker_driver.py').read_bytes()
driver = old_driver.replace("CONFIG_SHA256 = '" + expected_before['run_config.json'] + "'", "CONFIG_SHA256 = '" + config_sha + "'")
assert driver != old_driver
assert driver.replace(config_sha, expected_before['run_config.json']) == old_driver
(H / 'executed_driver.py').write_text(driver)
(H / 'worker_driver.py').write_bytes(worker)
(H / 'before_executed_driver.py').write_text(old_driver)
(H / 'before_worker_driver.py').write_bytes(worker)
old_ast, new_ast = ast.parse(old_driver), ast.parse(driver)
def remove_pin(tree):
    result = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CONFIG_SHA256' for t in node.targets):
            continue
        result.append(ast.dump(node, include_attributes=False))
    return result
assert remove_pin(old_ast) == remove_pin(new_ast)
assert sha(worker) == expected_before['worker_driver.py']
for name, before, after in [
    ('executed_driver.py', old_driver, driver),
    ('worker_driver.py', worker.decode(), worker.decode()),
    ('run_config.json', (B / 'run_config.json').read_text(), (H / 'run_config.json').read_text()),
]:
    (H / (name + '.patch')).write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile='preserved-preparation-v1/' + name, tofile='unrun-preparation-v2/' + name)))

failed_receipts = {
    '/tmp/qwen-server-snapshot-integration-root-test-review-20261104/failed-terminal-v1.json': '5505f782e2850765d27940190332fb0c8596622207c406832abd16df75ff0c49',
    '/tmp/qwen-server-snapshot-integration-root-test-review-20261104/direct-owner-failed-v1.json': '960d7fa7d259f6724d1c5e2a3dfefbe3aa4c2da76507e07a46132f0518176cc7',
    '/tmp/qwen-server-snapshot-integration-failed-security-review-20261104/review.json': 'afea0689bb3a0072b05d4e77b50e9140c40a00aac1b89a5307d8e9c09b69849d',
}
for name, digest in failed_receipts.items(): assert sha(Path(name).read_bytes()) == digest
provenance = dict(
    schema='qwen-snapshot-integration-harness-source-lineage-v2',
    base=str(B), base_pins=expected_before,
    changed_config_fields=changed_config_fields,
    driver_delta='Only CONFIG_SHA256 literal; all other driver bytes unchanged.',
    worker_bytes_exact=True, owned_Popen_cleanup_AST_exact=True,
    source_only_fixture_correction='Two plain local delegates in capture_runtime; runtime391 and all36 assertions unchanged.',
    source_handoff_sha256=sha((SH / 'handoff.json').read_bytes()),
    preserved_failed_run=dict(session=31389, launch_tool='4c43f9', terminal_tool='915588',
                             exit_code=1, reaped=True, xml_tests=36, failures=25, passed=11,
                             cause='Fixture bound methods rejected before25 direct assertions;6 process and5 terminal cases passed.',
                             report_absent=True, evidence_not_promoted_to_readiness=True),
    independent_failed_receipts=failed_receipts,
    imports=False, collection=False, execution=False, launch_authorized=False)
(H / 'driver_provenance.json').write_bytes(canonical(provenance))
(H / 'preserved-v1-files.json').write_bytes(canonical(preserved_files))
(H / 'README.md').write_text(
    'Held unrun405/36 harness-v2; no launch authority. Exact391 runtime and all36 assertion nodes are unchanged. '
    'Only source/proof paths, source count and source/config pins differ from preserved preparation-v1. '
    'Worker bytes, driver ownership/wait/cleanup and artifact checks are unchanged; CPU112-115 remains fixed. '
    'Source maps are canonical sorted compact JSON plus newline. Proposed success retains144 child artifacts '
    'and6 tiny newly owned public fixtures; optional owned pytest aliases are recorded without traversal. '
    'Preserved v1 produced25 setup failures and11 passing cases, with no final report after the strict artifact inspector rejected missing positive records. '
    'This source-only preparation makes no runtime PASS, backend, CLI bootstrap, science, borrowed-body or capacity claim. '
    'Exact root and peer failed-run receipts remain external provenance; no source or evidence from v1 was edited.\n')
proposal = json.loads((B / 'launch-proposal.json').read_bytes())
proposal.update(proof_root=str(H), source_root=str(S), source_count=405,
                source_manifest_sha256=config['source_manifest_sha256'],
                source_archive_sha256=config['source_archive_sha256'],
                driver_sha256=sha(driver.encode()), worker_sha256=sha(worker), config_sha256=config_sha)
proposal['argv'][-1] = str(H / 'executed_driver.py')
assert proposal['held_unrun'] and not proposal['launch_authorized'] and not proposal['execution']
(H / 'launch-proposal.json').write_bytes(canonical(proposal))

assert inventory(S) == inventory(H / 'execution') == mapping
with tarfile.open(H / 'sources.tar.gz', 'r:gz') as t:
    members = t.getmembers()
    assert len(members) == 405 and all(m.isfile() for m in members)
    assert {m.name: sha(t.extractfile(m).read()) for m in members} == mapping
assert {p.name: sha(p.read_bytes()) for p in B.iterdir() if p.is_file()} == preserved_files
names = ['README.md', 'before_executed_driver.py', 'before_worker_driver.py', 'driver_provenance.json',
         'executed_driver.py', 'executed_driver.py.patch', 'launch-proposal.json', 'prepare_source.py',
         'preserved-v1-files.json', 'run_config.json', 'run_config.json.patch', 'source-handoff.json',
         'sources-before.json', 'sources.tar.gz', 'worker_driver.py', 'worker_driver.py.patch']
handoff = dict(proposal, preparation_version=2,
               preserved_preparation_v1=True, final405_source_unchanged=True,
               worker_bytes_exact=True, driver_only_config_pin_changed=True,
               changed_config_fields=changed_config_fields,
               files={name: sha((H / name).read_bytes()) for name in names})
(H / 'handoff.json').write_bytes(canonical(handoff))
print(json.dumps(dict(handoff_sha256=sha((H / 'handoff.json').read_bytes()),
                     proposal_sha256=sha((H / 'launch-proposal.json').read_bytes()),
                     driver_sha256=proposal['driver_sha256'], worker_sha256=proposal['worker_sha256'],
                     config_sha256=config_sha, source_count=405, static_cases=36,
                     imports=False, collection=False, execution=False), sort_keys=True))
