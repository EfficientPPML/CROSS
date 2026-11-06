"""Public source/AST/evidence review only; does not import diagnosis or a backend."""
import ast,hashlib,json,tarfile
from pathlib import Path
import xml.etree.ElementTree as ET
R=Path('/tmp/bert-toy5-overflow-diagnosis-v2-20261104');S=R/'sources';F=Path('/tmp/cross-optimized-twelve-20261103/checkout');OUT=Path(__file__).parent
sha=lambda b:hashlib.sha256(b).hexdigest()
expected=json.loads((R/'proof-sources-before.json').read_bytes());after=json.loads((R/'proof-sources-after.json').read_bytes())
assert expected==after and len(expected)==9
current={n:sha((R/n).read_bytes()) for n in expected};assert current==expected
origin=json.loads((S/'source_origin.json').read_bytes())
for name,row in origin.items():assert sha((S/name).read_bytes())==sha((F/row['frozen_path']).read_bytes())==row['sha256']
assert expected['failure-preserved/security_device.json']=='f2b0c6f8a5ed3c07d394f8b19a22eb3f6a433feb4f3549dd53299b23e288e318'
archive=R/'proof-sources.tar.gz'
with tarfile.open(archive,'r:gz')as tf:
 members=tf.getmembers();assert len(members)==9 and all(x.isfile()for x in members)
 assert {x.name:sha(tf.extractfile(x).read())for x in members}==expected
xml=ET.fromstring((R/'tests.xml').read_bytes());suites=xml.findall('testsuite');assert len(suites)==1
assert int(suites[0].get('tests'))==19 and len(xml.findall('.//testcase'))==19
for name in ('failures','errors','skipped'):assert int(suites[0].get(name))==0
assert not xml.findall('.//failure')and not xml.findall('.//error')and not xml.findall('.//skipped')
ckks=ast.parse((S/'ckks_ctx.py').read_bytes());decode=next(n for n in ckks.body if isinstance(n,ast.FunctionDef)and n.name=='_fast_decode')
matched=[]
for n in ast.walk(decode):
 if isinstance(n,ast.BinOp)and isinstance(n.op,ast.Mult)and isinstance(n.left,ast.Call)and isinstance(n.left.func,ast.Name)and n.left.func.id=='float'and len(n.left.args)==1 and isinstance(n.left.args[0],ast.Name)and n.left.args[0].id in('r','im')and isinstance(n.right,ast.Name)and n.right.id=='scale_inv':matched.append({'line':n.lineno,'expression':ast.unparse(n)})
assert len(matched)==2 and {x['expression']for x in matched}=={'float(r) * scale_inv','float(im) * scale_inv'}
revision=json.loads((R/'revision.json').read_bytes());assert revision['candidate_or_original_runtime_changed']is False and revision['predecessor_passed']==17 and revision['predecessor_failed']==1
assert {n:sha((R/n).read_bytes())for n in expected}==expected
report=dict(schema='bert-toy5-conversion-public-source-review-v1',scope='Independent source/AST and existing public19-test evidence readback only. No test rerun, diagnosis-module/backend import, model/CT/key/private body read, actual decode, or TPU execution.',
 diagnosis_root=str(R),source_count=9,source_hashes=expected,current_before_after_archive_equal=True,source_archive_sha256=sha(archive.read_bytes()),tests=19,errors=0,failures=0,skipped=0,
 frozen_runtime_copies_match_199_source_snapshot=True,reviewed_conversion_expressions=matched,
 original_stddev_threshold=0.03125,
 source_findings=[
 'No concrete blocker found in the isolated range-safe conversion candidate.',
 'For integers whose float conversion succeeds, the helper returns the literal old raw*scale_inv result; existing rounding behavior is preserved.',
 'Only OverflowError from float(value) activates the fallback. It derives the exact binary64 scale ratio and uses integer true division after scaling, avoiding the oversized intermediate float conversion.',
 'For the pinned positive finite output scale, a truly unrepresentable scaled result still raises OverflowError. The helper is not presented here as a newly validated general invalid-scale API.',
 'The AST transformer applies to only two r/im multiplication expressions inside _fast_decode. Conjugation, original1/32 stddev gate, averaging and FFT call remain unchanged.',
 'Synthetic tests bind actual public Q40 product1096bits/outputscale, reproduce old conversion failure for2**1024 before FFT, compare fallback against exactFraction for both signs, and retain giant-asymmetric-noise rejection.',
 'The corrected realindex1 rejection and separate realindex0 symmetry control address the preserved test-only wrong assumption; they do not weaken the original gate.'
 ],
 limits=[
 'CRT reconstruction and FFT are explicit synthetic seams. These tests establish conversion and guard classification, not real decryption, FFT precision, encrypted accuracy or application success.',
 'A large conjugate-symmetric constant can satisfy the unchanged noise gate; no new arbitrary magnitude rejection is added. Future actual output must still pass the original application gates.',
 'The actual BERT5 throw-site remains an inference: public case evidence has no traceback/payload proving this exact conversion failed.',
 'All original199/runtime bytes remain unchanged. No TPU rerun or live patch is authorized or performed.'
 ],
 preserved_history=dict(predecessor_tests=18,predecessor_passed=17,predecessor_failed=1,reason='Real coefficient index0 is conjugate symmetric; v2 moves rejection probe toindex1 and adds symmetric positivecontrol.'))
p=OUT/'review.json';p.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'receipt':str(p),'sha256':sha(p.read_bytes()),'source_archive_sha256':sha(archive.read_bytes()),'tests_readback':19}))
