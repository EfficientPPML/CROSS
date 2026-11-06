"""Read-only verification of isolated executable decoder CPU arithmetic proof."""
from pathlib import Path
import ast,hashlib,json,os,tarfile,xml.etree.ElementTree as ET
LAB=Path('/tmp/ckks-scaled-crt-decoder-candidate-v2-20261104')
P=LAB/'source-bound-v1'
OUT=Path('/tmp/ckks-scaled-crt-decoder-root-review-20261104')
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_bytes())
raw=(P/'sources-before.json').read_bytes();expected=json.loads(raw)
assert sha(raw)=='9d9190b2a3c65f831f69fe30af6ece7b0b94f05bf83e9a21310ca32f99fda2aa' and len(expected)==207
assert expected==read(P/'sources-after.json')
assert all(sha((LAB/n).read_bytes())==h==sha((P/'execution'/n).read_bytes())for n,h in expected.items())
execution={}
for p in (P/'execution').rglob('*'):
    assert not p.is_symlink()
    if p.is_file():execution[p.relative_to(P/'execution').as_posix()]=sha(p.read_bytes())
    else:assert p.is_dir()
assert execution==expected
tar_pin='1bd06ce0d036698bffcbc06eabe4e131107b30aee5fcb7ba08768b9dd0ee0afb'
assert sha((P/'sources.tar.gz').read_bytes())==tar_pin
with tarfile.open(P/'sources.tar.gz')as tar:
    members=tar.getmembers();assert len(members)==len({m.name for m in members})==207
    assert all(m.isfile()and m.name in expected for m in members)
    assert {m.name:sha(tar.extractfile(m).read())for m in members}==expected
parent=read(Path('/tmp/cross-optimized-twelve-20261103/evidence/snapshot_manifest.json'))['source_hashes'];assert len(parent)==199
assert {n for n,h in parent.items()if expected['checkout/'+n]!=h}=={'jaxite_word/ckks_ctx.py'}
assert all(sha((Path('/tmp/cross-optimized-twelve-20261103/checkout')/n).read_bytes())==h for n,h in parent.items())
assert expected['before/ckks_ctx.py']==parent['jaxite_word/ckks_ctx.py']=='bc978cb7412b55416f7ed634a689ae616a263b0f47fd9eef996ba244f2505db6'
assert expected['checkout/jaxite_word/ckks_ctx.py']=='9675b300d507fd9c6a3d03656c9df325d8da68a9331b4c92def0512784d0dc26'
old=ast.parse((LAB/'before/ckks_ctx.py').read_bytes());new=ast.parse((LAB/'checkout/jaxite_word/ckks_ctx.py').read_bytes())
helpers=[n for n in new.body if isinstance(n,ast.FunctionDef)and n.name=='_scale_centered_crt_coefficient'];assert len(helpers)==1;new.body.remove(helpers[0])
class Restore(ast.NodeTransformer):
    count=0
    def visit_Call(self,node):
        if isinstance(node.func,ast.Name)and node.func.id=='_scale_centered_crt_coefficient':
            assert not node.keywords and len(node.args)==3
            assert isinstance(node.args[0],ast.Name)and node.args[0].id in ('r','im')
            assert [n.id for n in node.args[1:]]==['scale','scale_inv']
            self.count+=1
            return ast.BinOp(left=ast.Call(func=ast.Name(id='float',ctx=ast.Load()),args=[node.args[0]],keywords=[]),op=ast.Mult(),right=node.args[2])
        return self.generic_visit(node)
restore=Restore();new=restore.visit(new)
assert restore.count==2 and ast.dump(new,include_attributes=False)==ast.dump(old,include_attributes=False)
assert sha((P/'report.json').read_bytes())=='2b897eb81f65ab2ff5f1dc3ecb17f2092c8aa514a75dceaead415a7aa53b9503'
assert sha((P/'worker/report.json').read_bytes())=='ef1ec33c1b321d2c8d18fca06e39e836d20f96d4cccdfd6146454e953e1f3675'
r=read(P/'report.json');w=read(P/'worker/report.json');started=read(P/'worker/started.json')
assert r['passed']is True and r['tests']==35 and r['session']==86719 and r['owner_reported_exit_code']==0 and r['owner_reported_reaped']is True and r['reap_tool_chunk']=='733ce7'
assert w['passed']is True and w['pytest_exit_code']==0 and w['jax_backend']=='cpu' and w['device_platforms']==['cpu']
assert w['import_guard_denials']==['libtpu'] and w['device_access_denials']==[]
assert all(w[n]is False for n in ('ciphertext_decryption','key_generation','model_forward','encrypted_failure_recovered'))
assert w['real_crt']is True and w['real_fft']is True
assert len(w['loaded_local_modules'])==19 and all(expected[v['path']]==v['sha256']for v in w['loaded_local_modules'].values())
tree=ET.parse(P/'worker/tests.xml');assert len(tree.findall('.//testcase'))==35 and not any(tree.findall('.//'+n)for n in ('failure','error','skipped'))
large=[x for x in w['records']if x['name']=='large_symmetric_constant'and x['degree']==131072]
assert len(large)==2 and {x['sign']for x in large}=={-1,1}
assert all(x['slots']==65536 and x['all_finite']is True and x['old_conversion_overflow']is True and x['minimum_hex']==x['maximum_hex']==x['independent_fraction_expected_hex']for x in large)
assert started==dict(pid=3000289,pgid=3000287,scope='clear CPU unit arithmetic')
assert w['pid']==r['observed_pid']==started['pid'] and w['pgid']==r['observed_pgid']==started['pgid']
for func,number in ((os.kill,started['pid']),(os.killpg,started['pgid'])):
    try:func(number,0)
    except ProcessLookupError:pass
    else:raise AssertionError('Recorded worker process/group remains')
correction=Path('/tmp/ckks-scaled-crt-decoder-readback-20261104/report.json')
assert read(correction)['passed']is True and read(correction)['measured207_unchanged']is True
assert (P/'finalization-first-attempt.log').is_file()
report=dict(schema='ckks-scaled-crt-integration-independent-root-review-v1',review_passed=True,sources=207,tests=35,
    source_manifest_sha256=sha(raw),source_archive_sha256=tar_pin,current_execution_start_end_archive_equal=True,
    runtime_sha256=expected['checkout/jaxite_word/ckks_ctx.py'],parent199_unchanged=198,original_frozen199_current_unchanged=True,
    runtime_ast_delta='One scaling helper and only the two real/imaginary conversion call sites',
    cpu_only=True,libtpu_import_denied=True,device_access_denials=[],real_crt_and_fft=True,full_degree=131072,full_slots=65536,
    source_module_bindings=19,owner_session=86719,owner_reported_exit_code=0,owner_reported_reaped=True,
    worker_recorded_pid=started['pid'],worker_recorded_pgid=started['pgid'],independently_verified_recorded_pid_and_pgid_absent=True,
    identity_scope='Startup worker self-recording; no external live observation or separate wrapper PID was captured',
    readback_correction_sha256=sha(correction.read_bytes()),readback_correction_scope='Wrong original-source path corrected after test completion; arithmetic and measured207 unchanged',
    scope='Clear CPU arithmetic with genuine CRT/FFT; no ciphertext, key, model or TPU execution',
    actual_toy5_cause_proven=False,actual_toy5_recovered=False,candidate_installed=False,
    review_script_sha256=sha(Path(__file__).read_bytes()))
OUT.mkdir(mode=0o700)
with (OUT/'review.json').open('x')as f:json.dump(report,f,indent=2);f.write('\n')
with (OUT/'review.py').open('xb')as f:f.write(Path(__file__).read_bytes())
print(json.dumps(dict(review_passed=True,review=str(OUT/'review.json'),sha256=sha((OUT/'review.json').read_bytes()),sources=207,tests=35)))
