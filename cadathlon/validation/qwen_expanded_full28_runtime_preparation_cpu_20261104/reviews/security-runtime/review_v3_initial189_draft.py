"""Read-only v3 delta receipt. No candidate imports or test execution."""
from pathlib import Path
import ast, datetime, hashlib, json

V1=Path('/tmp/qwen-expanded-full-model-runtime-20261104')
V3=Path('/tmp/qwen-expanded-full-model-runtime-v3-20261104')
DRIVER=Path('/tmp/qwen-expanded-full-preparation-driver-draft-20261104/prepare_expanded.py')
OUT=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
tree=lambda p:ast.parse(p.read_bytes())
def constants(p):
    result={}
    for n in tree(p).body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            try:result[n.targets[0].id]=ast.literal_eval(n.value)
            except (ValueError,TypeError):pass
    return result
def fn(p,name):return next(n for n in tree(p).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
def main():
    prior=(OUT/'review.json').read_bytes()
    assert sha(prior)=='ab49d2c4a855094fae27d26fa3bd31af3276e1fc2c952758a78042982caae48f'
    oldmap=json.loads(Path('/tmp/qwen-expanded-full-model-runtime-development-capture-20261104/source-map-during-run.json').read_bytes())
    assert all(sha((V1/n).read_bytes())==h for n,h in oldmap.items())
    changed={n:sha((V3/n).read_bytes()) for n,h in oldmap.items() if sha((V3/n).read_bytes())!=h}
    assert set(changed)=={'derive_expanded.py','join_ideal.py'}
    additions=sorted(p.relative_to(V3).as_posix() for p in V3.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.relative_to(V3).as_posix() not in oldmap)
    assert additions==['development_predecessor.json','test_expanded_fixture.py','v2_history.json']
    p87=json.loads((V3/'predecessor87.json').read_bytes())['sources']
    assert len(p87)==87 and all(sha((V3/'before87'/n).read_bytes())==h for n,h in p87.items())
    # The withdrawn queue interpretation has no surviving code change.
    assert ast.dump(fn(V3/'join_ideal.py','install_matrix'))==ast.dump(fn(V1/'join_ideal.py','install_matrix'))
    # Certificate/level/bound derivation itself is unchanged through report creation.
    oldfn,newfn=fn(V1/'derive_expanded.py','derive'),fn(V3/'derive_expanded.py','derive')
    prefix=[]
    for n in oldfn.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='out' for t in n.targets):break
        prefix.append(n)
    assert [ast.dump(n) for n in newfn.body[:len(prefix)]]==[ast.dump(n) for n in prefix]
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('write_bytes','write_text','mkdir','unlink','replace','rename') for n in ast.walk(tree(V3/'derive_expanded.py')))
    assert set(constants(DRIVER)['TESTS'])=={
        'test_expanded_contract.py','test_expanded_fixture.py','test_full_plan.py','test_full_owner.py',
        'test_profile.py','test_setup_scope.py','test_full_callback_negative.py','test_full_returned.py','test_full_entry_failure.py'}
    assert constants(DRIVER)['EXPECTED_TESTS']==176
    historical={p.stem for p in V3.iterdir() if p.is_file() and p.name.startswith(('collect_','verify_','audit_'))}
    assert historical <= set(constants(DRIVER)['HISTORICAL'])
    assert 'apply_successor' in constants(DRIVER)['HISTORICAL']
    test=tree(V3/'test_expanded_fixture.py')
    assert [n.name for n in test.body if isinstance(n,ast.FunctionDef)]==[
        'test_period_alias_is_real_noop','test_exact_inventory_and_unplanned_rotation_rejected','test_removing_planned_key_rejects_before_primitive']
    plan=(V3/'inputs/actual_key_plan.json').read_bytes()
    assert sha(plan)=='bac55debfe95752a16a9a2e849e66488f4afbb576959403a2af4698358dd32ca'
    union=json.loads(plan)
    assert len(set(union['combined_keys']))==union['combined_key_count']==166
    selected=['derive_expanded.py','join_ideal.py',*additions,'inputs/actual_key_plan.json','derived_bindings.json']
    hashes={n:sha((V3/n).read_bytes()) for n in selected}
    report=dict(schema='qwen-expanded-full-successor-security-v3-delta-v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Source and public metadata only; no runtime import, derivation execution, test run or private/model/input bodies.',
        prior_review_sha256=sha(prior),reviewed_root=str(V3),source_hashes=hashes,
        draft_entry=dict(path=str(DRIVER),sha256=sha(DRIVER.read_bytes()),review_scope='Only selected test inventory, expected count and historical entry exclusions; parent owns broader driver/bootstrap/audit/lifecycle review.'),
        no_new_concrete_blocker_in_reviewed_delta=True,held_v1_files=186,v3_source_files_observed=189,
        changes_from_held_v1=changed,additions_from_held_v1=additions,before87_exact=True,
        runtime_bridge_and_derived_inputs_unchanged_from_v1=True,queue_seam_unchanged_from_v1=True,
        findings=[
            'Synthetic Arithmetic authenticates the exact actual key plan, clears inherited broad placeholders and installs exactly166 key/map identities.',
            'Exact integer zero/period rotations are identity no-ops; nonzero rotations require both key and coefficient-map membership before the primitive.',
            'Five new fixture cases cover three no-op aliases, exact inventory/unplanned rejection plus one planned numeric roll, and removal of a planned key before execution.',
            'derive returns serialized artifacts in memory; verify_generated byte-compares all four derived public inputs plus derived_bindings.json without writes. The prior certificate/level/bound computation AST prefix is exact.',
            'Draft preparation selects nine named test modules and blocks all six copied collector/verifier/audit modules plus the source-edit script; inherited inventories are not accepted as the expanded proof.'
        ],
        development_history='V1 101/171 pass with70 resources-fixture failures preserved. V2 75pass/4withdrawn queue-test setup failures preserved; no queue fix claimed or adopted.',
        pending='Fresh source-bound preparation with final entry/complete closure and authoritative outer exit/reap is still required; this receipt is source review, not176 test outcomes.',
        limitations=['No calibration PASS, first-PASS freeze, execution grant, model/HE run or encrypted B8192/head-L2 guarantee.'])
    (OUT/'review-v3.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(dict(no_new_concrete_blocker=True,receipt_sha256=sha((OUT/'review-v3.json').read_bytes()),source_hashes=hashes,draft_entry_sha256=sha(DRIVER.read_bytes()))))
if __name__=='__main__':main()
