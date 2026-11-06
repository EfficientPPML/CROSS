from pathlib import Path
import ast,hashlib,json,os,tarfile,xml.etree.ElementTree as ET
ROOT=Path('/tmp/bert-toy4-checkpoint-component-v4-20261104')
BASE=Path('/tmp/bert-toy4-checkpoint-v4-source-bound-20261104')
P=BASE/'preflight-v1'
OUT=Path('/tmp/bert-toy4-v4-root-review-20261104')
sha=lambda b:hashlib.sha256(b).hexdigest()
r=json.loads((P/'report.json').read_bytes());expected=r['source_hashes_start']
assert sha((P/'report.json').read_bytes())=='ca68514369eaca65b4c334da3ccba1c2fc9bc1cd4cdd4e642b5ba12941d7b275'
assert sha((P/'sources.tar.gz').read_bytes())=='fd937837a2ffdebc1f9403230a22060da609afb629211755db9b73c69bfc6d86'
assert len(expected)==260 and expected==r['source_hashes_end']==r['execution_hashes_end']
roots=[ROOT,BASE/'execution_source',Path(r['execution_root'])]
for tree in roots:
 assert tree.is_dir(),str(tree)
 files={p.relative_to(tree).as_posix():p for p in tree.rglob('*') if p.is_file()}
 assert set(files)==set(expected),(str(tree),set(files)-set(expected))
 assert not any(p.is_symlink() for p in tree.rglob('*'))
 assert {n:sha(p.read_bytes()) for n,p in files.items()}==expected
with tarfile.open(P/'sources.tar.gz') as t:
 members=t.getmembers();assert len(members)==len({m.name for m in members})==260 and all(m.isfile() for m in members)
 raw={m.name:t.extractfile(m).read() for m in members}
assert {n:sha(b) for n,b in raw.items()}==expected
assert json.loads(raw['execution_authorization.json'])=={'ready':False}
for name in ('passed','complete','cleanup_completed','source_verified','terminal_scope_completed','termination_scope_completed','all_tests_executed_without_skip'):assert r[name] is True
assert r['execute'] is False and r['test_cases']==r['expected_test_cases']==371
assert r['preflight_audit']['backend_or_private_attempts']==[] and r['preflight_audit']['zero_forbidden_attempts'] is True
xml=ET.parse(P/'tests.xml');assert len(xml.findall('.//testcase'))==371 and not any(xml.findall('.//'+n) for n in ('error','failure','skipped'))
child=r['test_child'];assert child['direct_returncode']==0 and all(child[n] for n in ('cleanup_completed','direct_child_reaped','group_empty_verified'))
assert not Path('/proc',str(child['pid'])).exists()
try:os.killpg(child['process_group'],0)
except ProcessLookupError:pass
else:raise AssertionError('live process group')
old=Path('/tmp/bert-toy4-checkpoint-component-v2-20261104')
unchanged=[n for n in expected if n.startswith('runtime/')]
assert len(unchanged)==199 and all(raw[n]==(old/n).read_bytes() for n in unchanged)
a,b=ast.parse(raw['component_arithmetic.py']),ast.parse((old/'component_arithmetic.py').read_bytes())
dump=lambda n:ast.dump(n,include_attributes=False)
for node in b.body:
 if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
  other=next(q for q in a.body if type(q) is type(node) and q.name==node.name)
  if node.name!='exercise':assert dump(node)==dump(other)
  else:
   first=next(q for q in node.body if isinstance(q,ast.Try));second=next(q for q in other.body if isinstance(q,ast.Try))
   assert [dump(q) for q in first.body]==[dump(q) for q in second.body]
for name in ('profile.json','fixture_manifest.json','response_admission.py','scale_replay.py'):
 assert raw[name]==(old/name).read_bytes(),name
OUT.mkdir(mode=0o700)
review=dict(passed=True,source_count=260,tests=371,report_sha256=sha((P/'report.json').read_bytes()),source_archive_sha256=sha((P/'sources.tar.gz').read_bytes()),
 current_both_execution_start_end_archive_equal=True,original199_byte_identical=True,numerical_classes_and_exercise_body_ast_identical=True,
 profile_coefficients_admission_scale_replay_unchanged=True,test_child=child['pid'],test_child_absent=True,group_absent=True,
 outer_completion='Transformer authoritative session51618 exit0/reaped; root verifies child absence independently',
 reviewed_source_fixes=['independent file/spec/symlink/PathLike origin admission','nested cleanup origin preservation','genuine body identity protected against sticky same-object cleanup rethrows'],
 actual_execution_authorized=False,actual_encrypted_diagnostic=False,queue='Original TPU12 then original P3six, diagnostic afterward',
 review_script_sha256=sha(Path(__file__).read_bytes()))
(OUT/'review.json').write_text(json.dumps(review,indent=2)+'\n');print(json.dumps(review))
