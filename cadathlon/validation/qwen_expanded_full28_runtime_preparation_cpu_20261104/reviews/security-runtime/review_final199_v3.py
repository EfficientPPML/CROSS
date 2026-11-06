"""Narrow source/smoke/closure readback; no imports of preparation/runtime code."""
from pathlib import Path
from datetime import datetime, timezone
import ast, hashlib, json, os, tarfile

OUT=Path(__file__).resolve().parent
ROOT=Path('/tmp/qwen-expanded-full-model-runtime-final-v3-20261104')
PROOF=Path('/tmp/qwen-expanded-full-model-preflight-v3-20261104')
OLD=Path('/tmp/qwen-expanded-full-model-preflight-v2-20261104')
SMOKE=Path('/tmp/qwen-expanded-preparation-startup-smoke-20261104')
MAP='c9b9dfd19f6514d6c43fd39378d1ea54a188b3b9d57085985dfc4ce8ae0dfe83'
TAR='99de0e64165e26b9bd51352a1cbcd982187fdb3d093643f2a47afe3ddae26caa'
DRIVER='3497a46a743adaf7cbd81eae507ab4dfcaf45f64d67f85fd611d6203a75ed1e9'

def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(path):return json.loads(path.read_bytes())
def archive(path,expected):
    with tarfile.open(path) as handle:
        members=handle.getmembers()
        assert len(members)==len(expected)==len({x.name for x in members})
        assert all(x.isfile() for x in members)
        assert {x.name:sha(handle.extractfile(x).read()) for x in members}==expected
def maps(root,expected):
    assert {n:sha((root/n).read_bytes()) for n in expected}==expected
def absent_group(pgid):
    try:os.killpg(pgid,0)
    except ProcessLookupError:return True
    return False

def main():
    expected=read(PROOF/'sources.json');old=read(OLD/'sources.json')
    assert sha((PROOF/'sources.json').read_bytes())==MAP
    assert sha((PROOF/'sources.tar.gz').read_bytes())==TAR
    assert len(expected)==len(old)==199 and set(expected)==set(old)
    changed=sorted(n for n in expected if expected[n]!=old[n])
    assert changed==['PREPARATION.md','driver_history.json','final_origin.json','prepare_expanded.py']
    maps(ROOT,expected);maps(PROOF/'execution',expected);archive(PROOF/'sources.tar.gz',expected)
    maps(Path('/tmp/qwen-expanded-full-model-runtime-final-v2-20261104'),old)
    driver=(ROOT/'prepare_expanded.py').read_text()
    assert sha(driver.encode())==DRIVER
    assert (Path('/tmp/qwen-expanded-full-preparation-driver-v6-20261104')/'prepare_expanded.py').read_bytes()==driver.encode()
    before=(OLD/'execution/prepare_expanded.py').read_text()
    def top_other(text):
        return [ast.dump(x,include_attributes=False) for x in ast.parse(text).body if not isinstance(x,ast.FunctionDef) or x.name!='run']
    assert top_other(before)==top_other(driver)
    smoke_map=read(SMOKE/'sources.json');smoke=read(SMOKE/'worker/report.json');scope=read(SMOKE/'scope.json');owner=read(SMOKE/'owner-terminal.json')
    assert len(smoke_map)==6
    maps(SMOKE/'source',smoke_map);maps(SMOKE/'execution',smoke_map);archive(SMOKE/'sources.tar.gz',smoke_map)
    for field in ('execution_sources_before','execution_sources_after','original_sources_before','original_sources_after'):assert smoke[field]==smoke_map
    assert sha((SMOKE/'sources.json').read_bytes())==scope['manifest_sha256']==smoke['source_manifest_sha256']
    assert sha((SMOKE/'sources.tar.gz').read_bytes())==scope['source_archive_sha256']
    assert sha((SMOKE/'worker/report.json').read_bytes())==owner['report_sha256']=='f84c069671ae7b64c0272e57b2fcf8a70f33f3ee4c6c2753c8f3546fad6d5a6f'
    smoke_driver=(SMOKE/'source/prepare_expanded.py').read_text()
    start=driver.index('TESTS=');end=driver.index('FORBIDDEN=')
    adapted=driver[:start]+"TESTS=('test_startup.py',)\nEXPECTED_TESTS=1\n"+driver[end:]
    assert adapted.replace('all180 selected checks required','single startup check required')==smoke_driver
    for name in ('freeze_io.py','cleanup_origin.py'):assert smoke_map[name]==expected[name]
    assert smoke['passed'] is True and smoke['denied_attempts']==[]
    assert smoke['tests']==dict(tests=1,failures=0,errors=0,skipped=0)
    assert smoke['import_cache']['dont_write_bytecode'] is True and smoke['import_cache']['empty_at_terminal'] is True
    assert not any(Path(smoke['import_cache']['owned_empty_prefix']).iterdir())
    assert all(not os.path.lexists(x) for x in smoke['import_cache']['removed_absent_stdlib_zip'])
    assert owner['exit_code']==0 and owner['tool_chunk']=='a36fc7' and owner['direct_exec_completed'] is True
    assert all(not Path('/proc',str(x)).exists() for x in (3288693,3288692)) and absent_group(3288692)
    launch=read(PROOF/'launch-draft.json')
    assert launch['expected_tests']==180 and launch['actual_execution_authorized'] is False
    assert launch['source_manifest_sha256']==MAP and launch['source_archive_sha256']==TAR
    assert '-I' in launch['command'] and '-B' in launch['command']
    assert 'SOFTMAX_REVIEW_SOURCE' in launch['outer_environment_unset']
    prior=read(OLD/'worker/report.json')
    assert prior['passed'] is False and prior['tests']==dict(tests=180,failures=0,errors=0,skipped=0)
    assert len(prior['denied_attempts'])==12
    result=dict(schema='qwen-expanded-full-final199-v3-security-review-v1',passed=True,observed_utc=datetime.now(timezone.utc).isoformat(),
        review_script_sha256=sha(Path(__file__).read_bytes()),source_manifest_sha256=MAP,source_archive_sha256=TAR,source_count=199,
        current_execution_archive_exact=True,changed_vs_previous199=changed,added_files=[],unchanged_arithmetic_generated_inputs_and_all180_test_sources=True,
        driver_sha256=DRIVER,driver_scope='Only run() startup/terminal cache handling changed; all other top-level AST including trusted-path/read-denial/terminal-exit policy unchanged.',
        source_findings=['Fresh owned empty bytecode directory becomes sys.pycache_prefix before pytest, with mandatory -B unchanged; terminal rechecks prefix identity string, bytecode-write prohibition and emptiness.',
            'Only entries resolving to the calculated default standard-library ZIP are removed; lexists must first show the exact ZIP absent. Matching importer-cache entries are removed; no allowlist expansion.',
            'Runtime and synthetic test selection remain unchanged at180; no scientific or actual execution authority is added.'],
        smoke=dict(source_count=6,tests=1,passed=True,audit_denials=0,tool_chunk='a36fc7',exit_code=0,recorded_process_group_currently_absent=True,report_sha256=owner['report_sha256'],source_manifest_sha256=scope['manifest_sha256'],source_archive_sha256=scope['source_archive_sha256'],source_before_after_current_execution_tar_exact=True,
            adaptations='Only TESTS/count/inactive error text change in driver; derive_expanded and plan_full are explicit startup stubs. Actual NumPy and pytest import/test run; not the full runtime or180 suite.'),
        preserved_previous199=dict(report_sha256=sha((OLD/'worker/report.json').read_bytes()),passed=False,tests_passed=180,sticky_denied_attempts=12,classification='Failed preparation despite all180 test assertions passing; unchanged historical report.'),
        launch='Exact closure matches parent-authorized public180 retry. Peer owns launch/reap; no actual model or HE grant.',
        limitations=['Reviewer ran only stdlib public source/evidence readback, no tests/backend/runtime imports or body reads.','Tiny startup smoke does not replace the pending source-bound180 suite.'])
    path=OUT/'review-final199-v3.json';path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,review_sha256=sha(path.read_bytes()),sources=199,changed=changed)))

if __name__=='__main__':main()
