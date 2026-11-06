"""Independent public-artifact checks; outer direct-wait evidence is separate."""
from pathlib import Path
import hashlib
import json
import os
import tarfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

ROOT = Path('/tmp/qwen-expanded-full-model-v3-root-review-20261104')
PROOF = Path('/tmp/qwen-expanded-full-model-preflight-v3-20261104')
SOURCE = Path('/tmp/qwen-expanded-full-model-runtime-final-v3-20261104')
OLD_PROOF = Path('/tmp/qwen-expanded-full-model-preflight-20261104')
OLD_SOURCE = Path('/tmp/qwen-expanded-full-model-runtime-final-20261104')
sha = lambda raw: hashlib.sha256(raw).hexdigest()
load = lambda path: json.loads(path.read_bytes())

raw = (PROOF/'sources.json').read_bytes()
assert sha(raw) == 'c9b9dfd19f6514d6c43fd39378d1ea54a188b3b9d57085985dfc4ce8ae0dfe83'
expected = json.loads(raw)
assert len(expected) == 199
assert sha((PROOF/'sources.tar.gz').read_bytes()) == '99de0e64165e26b9bd51352a1cbcd982187fdb3d093643f2a47afe3ddae26caa'
for root in (SOURCE, PROOF/'execution'):
    paths = list(root.rglob('*'))
    assert not any(p.is_symlink() for p in paths)
    assert {str(p.relative_to(root)):sha(p.read_bytes()) for p in paths if p.is_file()} == expected
with tarfile.open(PROOF/'sources.tar.gz', 'r:gz') as archive:
    members = archive.getmembers()
    assert len(members) == 199 and len({m.name for m in members}) == 199
    assert all(m.isfile() for m in members)
    assert {m.name:sha(archive.extractfile(m).read()) for m in members} == expected
old = load(OLD_PROOF/'sources.json')
assert len(old) == 198
assert {str(p.relative_to(OLD_SOURCE)):sha(p.read_bytes()) for p in OLD_SOURCE.rglob('*') if p.is_file()} == old
assert set(expected)-set(old) == {'test_wrapper_callback.py'} and not set(old)-set(expected)
changed = sorted(n for n in old if old[n] != expected[n])
assert changed == ['PREPARATION.md','checkout/cadathlon/common/encrypted_qwen_causal_softmax.py',
                   'driver_history.json','final_origin.json','prepare_expanded.py','selected_preparation.json','successor_inventory.json']
for name, pin in expected.items():
    if name.startswith('before87/'):
        p = Path('/tmp/qwen-full-model-runtime-v3-20261103')/name.removeprefix('before87/')
        assert sha(p.read_bytes()) == pin
assert sum(n.startswith('before87/') for n in expected) == 87

report = load(PROOF/'worker/report.json')
assert report['passed'] is True
assert report['synthetic_tests_started'] is True
assert report['synthetic_component_tests_completed'] is True
assert report['test_exit_code'] == 0
assert report['tests'] == dict(tests=180, failures=0, errors=0, skipped=0)
assert report['denied_attempts'] == []
assert report['import_cache']['empty_at_terminal'] is True
assert report['import_cache']['dont_write_bytecode'] is True
assert report['import_cache']['owned_empty_prefix'] == str(PROOF/'worker/bytecode')
assert not list((PROOF/'worker/bytecode').iterdir())
assert report['outer_receipt_required'] is True
for name in ('scientific_calibration_verified','actual_execution_authorized','learned_wholegraph_executed','encrypted_execution'):
    assert report[name] is False
for name in ('execution_sources_before','execution_sources_after','original_sources_before','original_sources_after'):
    assert report[name] == expected
assert report['source_manifest_sha256'] == sha(raw)
assert report['cpu_affinity'] == [112,113,114,115]
assert report['pid'] == 3299697 and report['ppid'] == report['pgid'] == 3299696
for name in ('derived_bindings.json','inputs/candidate_bridge.json','inputs/join_plan.json',
             'inputs/normalization_certificates.json','inputs/site_metadata.json'):
    assert report['fresh_derivation'][name] == expected[name]
assert len(report['fresh_derivation']) == 5
for item in report['loaded_local_modules'].values():
    assert expected[item['path']] == item['sha256']
assert sha((PROOF/'worker/public_plan.json').read_bytes()) == report['public_plan_sha256']
cases = list(ET.parse(PROOF/'worker/tests.xml').getroot().iter('testcase'))
assert len(cases) == 180
assert all(all(case.find(kind) is None for kind in ('failure','error','skipped')) for case in cases)
for pid in (3299697,3299696):
    assert not Path('/proc',str(pid)).exists()
try:
    os.killpg(3299696,0)
except ProcessLookupError:
    pass
else:
    raise AssertionError('preparation process group still exists')

result = dict(passed=True, utc=datetime.now(timezone.utc).isoformat(), sources=199, tests=180,
              source_manifest_sha256=sha(raw), worker_report_sha256=sha((PROOF/'worker/report.json').read_bytes()),
              source_archive_sha256=sha((PROOF/'sources.tar.gz').read_bytes()),
              verifier_sha256=sha(Path(__file__).read_bytes()), old198_preserved=True, old87_preserved=True,
              five_fresh_derivations_match=True, no_denied_attempts=True, worker_parent_group_absent=True,
              changed_parent_files=changed,
              scope='Public certificates/contracts and synthetic component/owner tests only. No learned wholegraph, calibration, model, key, ciphertext or HE execution. Numeric outer exit must be admitted separately from the owner direct-wait receipt.')
(ROOT/'review.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,sort_keys=True))
