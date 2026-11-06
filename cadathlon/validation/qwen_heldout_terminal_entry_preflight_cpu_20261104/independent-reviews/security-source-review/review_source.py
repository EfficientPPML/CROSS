"""Public-only source review of the shared heldout/row0 entry boundary."""
import ast,hashlib,json
from pathlib import Path
R=Path('/tmp/qwen-heldout1088-runtime-v5-20261104')
M=Path('/tmp/qwen-heldout1088-runtime-sources-v5-20261104.json')
OLD=Path('/tmp/qwen-heldout1088-runtime-v2-20261104')
CNN=Path('/tmp/qwen-expanded-candidate-checker-terminal-exit-successor-20261104')
OUT=Path(__file__).parent
sha=lambda b:hashlib.sha256(b).hexdigest()
raw=M.read_bytes();pin='9985aad6205af33a429fba3802bac415397973418e568ea3414d61197f03deeb';assert sha(raw)==pin
expected=json.loads(raw)
observed={p.relative_to(R).as_posix():sha(p.read_bytes()) for p in R.rglob('*') if p.is_file()}
assert expected==observed and len(expected)==171
origin=json.loads((R/'heldout_terminal_origin.json').read_bytes());old=origin['parent167_sources'];baseline=origin['inherited139_sources']
assert len(old)==167 and len(baseline)==139
assert all(sha((OLD/n).read_bytes())==h for n,h in old.items())
assert all(observed[n]==h for n,h in baseline.items())
changed=sorted(n for n,h in old.items()if observed[n]!=h)
assert changed==['heldout_entry.py','heldout_preflight_expectations.json','test_heldout_no_grant.py']
added=sorted(set(observed)-set(old));assert added==['TERMINAL_ENTRY_DESIGN.md','heldout_terminal_origin.json','shared_terminal_exit_probe.py.txt','test_heldout_terminal_exit.py']
lookup=lambda data:{n.name:n for n in ast.parse(data).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
o=lookup((OLD/'heldout_entry.py').read_bytes());n=lookup((R/'heldout_entry.py').read_bytes())
for key in set(o)-{'main'}:assert ast.dump(o[key],include_attributes=False)==ast.dump(n[key],include_attributes=False)
assert ast.dump(ast.Module(body=o['main'].body[-4:],type_ignores=[]),include_attributes=False)==ast.dump(ast.Module(body=n['authenticated_dispatch'].body,type_ignores=[]),include_attributes=False)
text=(R/'heldout_entry.py').read_text();guard=text[text.index('INFRASTRUCTURE_EXIT = 70'):text.index('def authenticated_dispatch(args):')]
cnn=(CNN/'entry.py').read_text();reference=cnn[cnn.index('INFRASTRUCTURE_EXIT = 70'):cnn.index('def authenticated_dispatch(args):')]
assert guard==reference==origin['shared_guard']
probe=(R/'shared_terminal_exit_probe.py.txt').read_text();assert sha(probe.encode())==origin['shared_probe_sha256']=='66a96ca79b901b93cd68cbcd71c2902df1b26976827bb721a50335cbfb1abca9'
for a,b in origin['probe_path_role_replacements']:
 assert probe.count(a)==1;probe=probe.replace(a,b)
assert (R/'test_heldout_terminal_exit.py').read_text().startswith(probe)
oldtest=(OLD/'test_heldout_no_grant.py').read_text();newtest=(R/'test_heldout_no_grant.py').read_text()
a="assert child.returncode!=0 and b'Independent frozen-workflow grant required'in stderr,stderr[-4000:]"
b="assert child.returncode==70 and b'Candidate infrastructure failure: PermissionError'in stderr,stderr[-4000:]"
assert oldtest.count(a)==1 and oldtest.replace(a,b)==newtest
oldexpect=json.loads((OLD/'heldout_preflight_expectations.json').read_bytes());newexpect=json.loads((R/'heldout_preflight_expectations.json').read_bytes())
assert oldexpect['synthetic_tests']==478 and newexpect['synthetic_tests']==563
assert oldexpect['installed_import_denials']==newexpect['installed_import_denials']==10
assert newexpect['tests']==oldexpect['tests']+['test_heldout_terminal_exit.py']
v4=json.loads(Path('/tmp/qwen-heldout1088-runtime-sources-v4-20261104.json').read_bytes())
assert [n for n in v4 if v4[n]!=observed[n]]==['TERMINAL_ENTRY_DESIGN.md']
reviewed=['heldout_entry.py','heldout_process.py','heldout_authorization.py','heldout_preflight_expectations.json',
 'test_heldout_no_grant.py','test_heldout_terminal_exit.py','shared_terminal_exit_probe.py.txt','heldout_terminal_origin.json','TERMINAL_ENTRY_DESIGN.md']
assert all(sha((R/p).read_bytes())==observed[p] for p in reviewed)
result=dict(schema='qwen-heldout-terminal-entry-source-review-v1',scope='Source/diff and synthetic test-design review only; no tests rerun, backend/model/input/validation/private journal or actual candidate outcome read.',
 source_root=str(R),source_manifest_sha256=pin,source_count=171,current_equals_manifest=True,
 inherited139_all_unchanged=True,parent167_unchanged_count=164,parent_overlays={p:observed[p]for p in changed},additions={p:observed[p]for p in added},
 reviewed_sources={p:observed[p]for p in reviewed},reviewed_before_after_equal=True,
 moved_authenticated_work_ast_identical=True,old_entry_helpers_ast_identical=True,literal_guard_matches_reviewed_candidate=True,
 shared_probe_identical_except_declared_path_role_replacements=True,
 no_grant_test_delta='Only prior message/nonzero assertion -> class-only PermissionError/status70. No-read/no-output/direct wait/group absence assertions remain exact.',
 expected_checks={'parent_synthetic':478,'new_boundary':85,'final_synthetic':563,'installed_denials':10},
 v4_to_v5_delta='Only scope document corrects three parent overlays and preserves development85+6 history.',
 finding='No concrete blocker in the reviewed terminal-entry successor.',
 limits=['Full source-bound563+10 terminal proof and independent outer observation remain separate requirements.',
 'Shared subprocess probes exercise exact supervise with authenticated synthetic helpers in four role modes, not actual heldout or row0 execution.',
 'Actual private read authorization and independent-row0 versus scientific-result contracts are unchanged; this review does not grant execution.',
 'Original139/167 results are neither repaired nor reclassified; old failed139 reconstruction/revision remains withheld.',
 'Arbitrary concurrent process signals or native finalizers are outside this narrow dispatch-exception guarantee.'])
p=OUT/'review.json';p.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'receipt':str(p),'sha256':sha(p.read_bytes()),'source_manifest_sha256':pin,'current_equal':True}))
