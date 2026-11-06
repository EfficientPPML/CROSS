"""Stdlib source-only slice review; does not import the selected candidates."""
import ast,hashlib,json,stat
from pathlib import Path
new=Path('/tmp/qwen-full28-device-focused-tests-v3-20261105')
old=Path('/tmp/qwen-full28-device-focused-tests-v2-20261105')
out=Path('/tmp/qwen-full28-device459-successor-source-cnn-review-20261105')
assert out.is_dir()and not list(out.iterdir())
sha=lambda raw:hashlib.sha256(raw).hexdigest()
canon=lambda value:(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
pins={};trees={};raws={}
for name in ('owned_child.py','fixture_base.py','cases_process.py','cases_audit.py','cases_loader_engine.py','source_loader.py',
    'runtime/keygen_output.py','runtime/qwen_server_dependencies.py','runtime/qwen_server_device_owner.py'):
    raw=(new/name).read_bytes();raws[name]=raw;pins[name]=dict(path=str(new/name),sha256=sha(raw));trees[name]=ast.parse(raw)
assert pins['owned_child.py']['sha256']=='6b20b09cabea5c10d74e9a502107f1ae5a742f34556f6b87c59ef8f63829f613'
assert pins['fixture_base.py']['sha256']=='9272014ffd1b4697ff011237c39cac68027d3d6684f96041dbb4c83ec6d20ae7'
assert pins['cases_process.py']['sha256']=='5c75eb9464979dbee0271735b9be8011bad9d3b435b41a8357f1a409b7d1ee2f'
assert pins['cases_loader_engine.py']['sha256']=='434b33452878f5a3accbfcf733398c45ff6ce13ef96f009e38c56abaa8395569'
assert raws['cases_audit.py']==(old/'cases_audit.py').read_bytes()
assert raws['source_loader.py']==(old/'source_loader.py').read_bytes()
for name in ('cases_process.py','cases_audit.py'):
    before=[ast.dump(n,include_attributes=False)for n in ast.walk(ast.parse((old/name).read_bytes()))if isinstance(n,ast.Assert)]
    after=[ast.dump(n,include_attributes=False)for n in ast.walk(trees[name])if isinstance(n,ast.Assert)]
    assert before==after
runtime_expected=json.loads((old/'runtime_sources.json').read_bytes());observed={}
for path in sorted((new/'runtime').rglob('*')):
    info=path.lstat();assert not stat.S_ISLNK(info.st_mode)
    if stat.S_ISREG(info.st_mode):observed[path.relative_to(new/'runtime').as_posix()]=sha(path.read_bytes())
assert observed==runtime_expected and len(observed)==403
assert (new/'selected_tests.json').read_bytes()==(old/'selected_tests.json').read_bytes()
main=next(n for n in trees['owned_child.py'].body if isinstance(n,ast.FunctionDef)and n.name=='main')
cleanup=next(n for n in main.body if isinstance(n,ast.Try)and n.finalbody)
install=cleanup.body[0]
assert isinstance(install,ast.Assign)and len(install.targets)==1
assert ast.dump(install,include_attributes=False)==ast.dump(ast.parse('os.open=owned_open').body[0],include_attributes=False)
assert not any(isinstance(n,ast.Assign)and any(isinstance(t,ast.Attribute)and isinstance(t.value,ast.Name)and t.value.id=='os'and t.attr=='open'for t in n.targets)for n in main.body)
assert any(isinstance(n,ast.Try)and any(isinstance(t,ast.Assign)and ast.dump(t,include_attributes=False)==ast.dump(ast.parse('os.open=_ORIGINAL_OPEN').body[0],include_attributes=False)for t in n.finalbody)for n in ast.walk(cleanup))
# Verify the exact one-statement move against the initially observed8539 hash.
text=raws['owned_child.py'].decode()
before=text.replace('sys.meta_path.insert(0,finder);sys.addaudithook(audit)\n','sys.meta_path.insert(0,finder);sys.addaudithook(audit);os.open=owned_open\n')
before=before.replace('    try:\n        os.open=owned_open\n        sys.modules[\'__main__\']=entry;',"    try:\n        sys.modules['__main__']=entry;")
assert sha(before.encode())=='8539b66b431d6980666339dd3fc6f2e774f46208b78113a02126213596654fe0'
for name,raw in raws.items():assert(new/name).read_bytes()==raw
review=dict(schema='full28-test-successor-source-review-v2',reviewer='cnn_performance',
    scope='Bounded current mutable successor slices only; not a frozen complete test closure',passed=True,
    source_imports_tests=False,runtime403_exact=True,selection59_exact=True,old_AR_assertions_preserved=True,
    resolved_findings=[dict(id='wrapper_install_before_cleanup_ownership',
        before_sha256=sha(before.encode()),after_sha256=sha(raws['owned_child.py']),
        correction='Exact wrapper installation moved into first statement of cleanup-owned try; audit installation remains before source/provenance reads.',
        original_finding='Previous8539 installed os.open wrapper before intervening fallible provenance/entry-definition steps, outside restoring try.')],
    clear_within_scope=[
        'relative evidence write admits only exact report.json/failure.json, exact original Output flag tuple/mode and nonnegative exact-int directory FD matched by bounded owned-root metadata',
        'original fstat/lstat/scandir captures avoid synthetic descriptor metadata; no proc read needed; aliases denied',
        'pending audit allowance matches exact path/flags once and is reset in original-open finally; original os.open restored in cleanup finally',
        'recording_jax uses pinned source bytes direct compilation, genuine file/spec/loader remains named; installed numerical packages remain unavailable',
        'R setup adds existing owned empty NumPy definition before stage import; existing fixture cleanup restores temporary module names',
        'A case assertions unchanged; source edit removes unintended cache probe without clearing audit denials'],
    limits=['E source correction attributed to Security separate review, not independently re-audited here',
        'No native audit sandbox, concurrent filesystem attack guarantee, actual capacity or runtime success claim'],pins=pins)
(out/'review.json').write_bytes(canon(review));(out/'source-pins.json').write_bytes(canon(pins))
(out/'observed8539-reconstructed-exact.py.txt').write_text(before)
(out/'review_source.py').write_bytes(Path(__file__).read_bytes())
(out/'reviewer-history.json').write_bytes(canon(dict(initial_tool='9c13f9',initial_exit_code=1,
    cause='Root applied the notified installation-order correction while the initial receipt script still asserted old8539. The script rejected changed bytes before writing any receipt.',
    candidate_changed_by_reviewer=False,initial_receipt_created=False,final_scope='New6b20 slice explicitly re-read and exact statement move verified.')))
print('review',sha((out/'review.json').read_bytes()));print('resolved wrapper ownership;403 runtime and59 selection unchanged; no imports/tests')
