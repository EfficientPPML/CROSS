"""Public source/terminal readback; never imports the measured implementation."""
import hashlib
import json
import os
from pathlib import Path
import tarfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

ROOT = Path(__file__).parent
PROOF = Path('/tmp/qwen-cohort-full-owner-combined-preflight-20261104/preflight-v1')
SOURCE = Path('/tmp/qwen-softmax-cohort-full-owner-integration-v2-20261104')
PARENT = Path('/tmp/qwen-expanded-full-model-runtime-final-v3-20261104')
PARENT_PROOF = Path('/tmp/qwen-expanded-full-model-preflight-v3-20261104')
PEER = Path('/tmp/qwen-cohort-combined-closure-transformers-review-20261104/review.json')
sha = lambda raw: hashlib.sha256(raw).hexdigest()
load = lambda path: json.loads(path.read_bytes())

raw = (PROOF/'sources.json').read_bytes()
assert sha(raw) == 'ff8d902f71497517609f3629c3adfe12f9fffa5b720ec22aa0db61b5b5bd5d5e'
expected = json.loads(raw)
assert len(expected) == 208
assert sha((PROOF/'sources.tar.gz').read_bytes()) == 'ffc73bde207c6007258abacfcb464e7a919614f3b41facf4822650c55ad77d16'
for root in (SOURCE, PROOF/'execution'):
    paths = list(root.rglob('*'))
    assert not any(p.is_symlink() for p in paths)
    assert {str(p.relative_to(root)):sha(p.read_bytes()) for p in paths if p.is_file()} == expected
with tarfile.open(PROOF/'sources.tar.gz', 'r:gz') as archive:
    members = archive.getmembers()
    assert len(members) == 208 and len({m.name for m in members}) == 208
    assert all(m.isfile() for m in members)
    assert {m.name:sha(archive.extractfile(m).read()) for m in members} == expected
parent_raw = (PARENT_PROOF/'sources.json').read_bytes()
assert sha(parent_raw) == 'c9b9dfd19f6514d6c43fd39378d1ea54a188b3b9d57085985dfc4ce8ae0dfe83'
parent = json.loads(parent_raw)
assert len(parent) == 199
assert {str(p.relative_to(PARENT)):sha(p.read_bytes()) for p in PARENT.rglob('*') if p.is_file()} == parent
assert sha(PEER.read_bytes()) == '671583d13090c1aae1b4f5ee653b7a5ab96d9eada5d5f70b3966dd2d7239c1b9'
peer = load(PEER)
changes = {name:dict(old=parent[name],new=expected[name]) for name in parent if expected[name] != parent[name]}
assert changes == peer['changed_parent_files'] and len(changes) == 6
assert sorted(set(expected)-set(parent)) == peer['added_files']
assert not set(parent)-set(expected)
assert expected['checkout/cadathlon/common/encrypted_qwen_causal_softmax_batched.py'] == 'd50cabad91574d4446ad563954ced1d90c9cf984ac6f2db3bf5f49aad1058e97'

report = load(PROOF/'worker/report.json')
assert report['passed'] is True
assert report['synthetic_tests_started'] is True
assert report['synthetic_component_tests_completed'] is True
assert report['test_exit_code'] == 0
assert report['tests'] == dict(tests=330, failures=0, errors=0, skipped=0)
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
assert report['pid'] == report['pgid'] == 3397122 and report['ppid'] == 3397120
assert len(report['fresh_derivation']) == 5
for name, pin in report['fresh_derivation'].items():
    assert expected[name] == pin
for item in report['loaded_local_modules'].values():
    assert expected[item['path']] == item['sha256']
assert sha((PROOF/'worker/public_plan.json').read_bytes()) == report['public_plan_sha256']
cases = list(ET.parse(PROOF/'worker/tests.xml').getroot().iter('testcase'))
assert len(cases) == 330
assert all(all(case.find(kind) is None for kind in ('failure','error','skipped')) for case in cases)

plan = load(PROOF/'worker/public_plan.json')
rules = load(SOURCE/'inputs/candidate_manifest.json')
certs = load(SOURCE/'inputs/certificates.json')
assert plan['execution_authorized'] is False and plan['scientific_calibration_verified'] is False
assert plan['reference']['executed'] is False
assert len(plan['workloads']) == 32 and len(plan['layers']) == 28
assert plan['final_norm']['degree'] == 256 and plan['final_norm']['raw_input_bound'] == 8192
assert plan['final_norm']['output_level'] == 3 and plan['vocabulary']['output_level'] == 2
assert plan['vocabulary']['all_logits'] == 151936 and len(plan['vocabulary']['blocks']) == 38
assert plan['resources']['rotation_keys'] == 166
for n, row in enumerate(plan['workloads'],1):
    assert row['sequence'] == n and row['softmax_strategy'] == 'tile-cohorts-v1'
    groups = []
    for tile in range((n+15)//16):
        grouped = {}
        for k in range(tile*16+1,min(n,(tile+1)*16)+1):
            if k == 1:
                continue
            key = (rules['reciprocal_by_keys'][str(k)]['iterations'], certs['softmax_K'][k-1]['existing_inverse_refresh_bound'])
            grouped.setdefault(key,[]).append(k)
        groups.append(grouped)
    group_count = sum(map(len,groups))
    per_query = sum(3+2*rules['softmax'][str(layer)]['squarings'] for layer in range(28))
    assert row['refresh_counts']['softmax'] == group_count*per_query
    assert row['literal_reference_softmax_refreshes'] == (n-1)*per_query
    assert row['refresh_counts']['total']-group_count*per_query == row['refresh_counts']['literal_reference_total']-(n-1)*per_query
    for layer in row['softmax_cohorts']:
        assert len(layer['tiles']) == len(groups)
        for actual, required in zip(layer['tiles'],groups):
            assert {(g['newton_iterations'],g['refresh_bound']):[q['k'] for q in g['queries']] for g in actual} == required
assert plan['workloads'][-1]['refresh_counts']['softmax'] == 3760
assert plan['workloads'][-1]['refresh_counts']['total'] == 5035
assert plan['workloads'][-1]['refresh_counts']['literal_reference_total'] == 12931
for pid in (3397120,3397122):
    assert not Path('/proc',str(pid)).exists()
    try:
        os.killpg(pid,0)
    except ProcessLookupError:
        pass
    else:
        raise AssertionError('owned process group still exists')
result = dict(passed=True,utc=datetime.now(timezone.utc).isoformat(),sources=208,tests=330,
              source_manifest_sha256=sha(raw),source_archive_sha256=sha((PROOF/'sources.tar.gz').read_bytes()),
              worker_report_sha256=sha((PROOF/'worker/report.json').read_bytes()),
              verifier_sha256=sha(Path(__file__).read_bytes()),parent199_preserved=True,
              six_explicit_overlays_exact=True,all32_cohort_counts_independently_checked=True,
              s32_softmax_refreshes=3760,s32_whole_plan_refreshes=5035,literal_whole_plan_refreshes=12931,
              five_fresh_derivations_match=True,no_denied_attempts=True,owned_process_groups_absent=True,
              scope='Public/source and synthetic owner preparation only. No model, private calibration, key, ciphertext or HE run. Outer numeric exit must be admitted separately from the owner direct-wait receipt.')
(ROOT/'review.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,sort_keys=True))
