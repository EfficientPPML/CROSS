"""Independent source-only entry-boundary review; hashes public source closure."""
import ast
import hashlib
import json
from pathlib import Path

OLD=Path('/tmp/qwen-expanded-candidate-checker-ready-v2-20261104')
NEW=Path('/tmp/qwen-expanded-candidate-checker-terminal-exit-successor-20261104')
EXPECTED=json.loads(Path('/tmp/qwen-expanded-candidate-checker-ready-sources-v2-20261104.json').read_bytes())
HERE=Path(__file__).parent
sha=lambda b:hashlib.sha256(b).hexdigest()
old={n:sha((OLD/n).read_bytes()) for n in EXPECTED}
assert old==EXPECTED
current={p.relative_to(NEW).as_posix():sha(p.read_bytes()) for p in NEW.rglob('*') if p.is_file()}
assert len(current)==141
changed=[n for n in old if current[n]!=old[n]]
added=sorted(set(current)-set(old))
assert changed==['entry.py']
assert added==['TERMINAL_SUCCESSOR.md','test_terminal_exit.py']
assert current['entry.py']=='ad5bc00c800db5e9df5594d86e1b714a7db7c67ecde5ae89fe1e113660b674bd'
assert current['test_terminal_exit.py']=='66a96ca79b901b93cd68cbcd71c2902df1b26976827bb721a50335cbfb1abca9'
numeric=('checker_runtime.py','strict_metrics.py','candidate_domains.py','test_checker_runtime.py','test_strict_metrics.py','test_candidate_domains.py')
assert all(current[n]==EXPECTED[n] for n in numeric)
old_tree=ast.parse((OLD/'entry.py').read_bytes())
new_tree=ast.parse((NEW/'entry.py').read_bytes())
lookup=lambda tree:{n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
o=lookup(old_tree);n=lookup(new_tree)
for name in set(o)-{'main'}:
    assert ast.dump(o[name],include_attributes=False)==ast.dump(n[name],include_attributes=False)
# Exact prior authenticated work is moved intact, not replaced with a weaker path.
assert ast.dump(ast.Module(body=o['main'].body[-4:],type_ignores=[]),include_attributes=False)==ast.dump(ast.Module(body=n['authenticated_dispatch'].body,type_ignores=[]),include_attributes=False)
reviewed=('entry.py','test_terminal_exit.py','TERMINAL_SUCCESSOR.md','checker_process.py','cleanup_origin.py','test_checker_entry.py')
after={name:sha((NEW/name).read_bytes()) for name in reviewed}
assert all(after[name]==current[name] for name in reviewed)
result={
 'schema':'qwen-candidate-terminal-successor-source-review-v1',
 'scope':'Independent source/diff/test-design review only; no test rerun, backend import, model/token/journal/validation or actual candidate outcome read.',
 'source_root':str(NEW),'source_count':141,
 'inherited_source_count':139,'inherited_unchanged':138,'overlays':changed,'new_files':added,
 'reviewed_sources':after,'reviewed_before_after_equal':True,
 'numerical_six_unchanged':{name:current[name] for name in numeric},
 'authenticated_dispatch_ast_equals_original_main_work':True,
 'all_other_existing_entry_helpers_ast_unchanged':True,
 'finding':'No concrete blocker in this minimal entry-boundary repair.',
 'checks':[
  'Source authentication, frozen import admission and worker/collector dispatch all occur inside terminal_status.',
  'Only a normally returned exact int0 or int1 is propagated. All escaped BaseException and malformed normal returns map to reserved70.',
  'Failure diagnostics reveal the exception class only; all diagnostic failures are caught and cannot replace70.',
  'argparse help0/usage2 precede authenticated work; intentional final SystemExit(status) is outside terminal_status.',
  'Real isolated entry probes authenticate synthetic helpers containing exact unchanged supervise/accumulate AST, then inject final-close OSError/SystemExit0/SystemExit1/KeyboardInterrupt/diagnostic failure.',
  'Probe covers both role modes, clean0/1, bool/None/float/otherinteger/string/intsubclass malformed returns, source failure before helper import, and help/usage.',
  'Production worker and collect are separately asserted to return unchanged supervise; the focused probes do not run the full production workers or a model.'
 ],
 'limits':[
  'Whole successor source-bound preflight and independent outer terminal verification remain separate requirements.',
  'No old139 result is repaired or reclassified. Root policy continues to withhold old failed139 reconstruction/revision.',
  'Injected SystemExit0 is broad contract coverage; prior source trace identified no concrete ordinary unmodified late SystemExit0 route.'
 ]
}
p=HERE/'successor-review.json';p.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'receipt':str(p),'sha256':sha(p.read_bytes()),'source_count':141,'overlays':changed}))
