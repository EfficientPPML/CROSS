from pathlib import Path
import hashlib,json,shutil,os,tarfile
P=Path('/tmp/bert-toy4-checkpoint-diagnostic-plan-20261104');R=P/'preflight-v1';D=Path('/home/jianming_gatech/CROSS/cadathlon/validation/bert_toy4_checkpoint_diagnostic_plan_cpu_20261104');D.mkdir(exist_ok=False)
report=json.loads((R/'report.json').read_bytes());before=report['source_hashes_before']
assert {n:hashlib.sha256((P/n).read_bytes()).hexdigest()for n in before}==before
assert {n:hashlib.sha256((R/'execution'/n).read_bytes()).hexdigest()for n in before}==before
for c in report['children']:
 for pid in (c['pid'],):
  try:os.kill(pid,0)
  except ProcessLookupError:pass
  else:raise ValueError('producer still present')
 try:os.killpg(c['process_group'],0)
 except ProcessLookupError:pass
 else:raise ValueError('group still present')
files={'DESIGN.md':P/'DESIGN.md','plan.py':P/'plan.py','public_graph.py':P/'public_graph.py','dense_reference.py':P/'dense_reference.py','protocol.py':P/'protocol.py','helper_provenance.json':P/'helper_provenance.json'}
for n in ('report.json','sources.tar.gz','plan.json','tests.xml','tests.log','plan.log','audit.json','plan-audit.json'):files['preflight/'+n]=R/n
for n in ('development-public-graph-v1.log','development-reference-v1.log','development-plan-v1.log','development-plan-v1.json','development-tests-v1.log','development-tests-v1.xml','development-test-audit-v1.json','development-tests-v2.log','development-tests-v2.xml','development-test-audit-v2.json'):files['development/'+n]=P/n
for name,source in files.items():
 destination=D/name;destination.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(source,destination)
shutil.copyfile(__file__,D/'archive.py');shutil.copyfile(Path(__file__).with_name('verify.py'),D/'verify.py')
(D/'README.md').write_text('''# Toy BERT4 diagnostic plan: public preflight only

The fixed public graph and independent dense oracle agree over all96 checkpoints/all65536 slots. The61-test preflight authenticates222 sources (including the exact original199), five refreshes, key/storage inventory, synthetic post-server decode admission, and owned-group/terminal cleanup. No HE, TPU or model framework was imported or run. The future actual diagnostic worker/transport remains unimplemented and unlaunched; see DESIGN.md.

Original BERT4 encrypted failure remains a failure: decoder stddev=inf, no decoded accuracy result. The extra1e-6 checkpoint diagnostic is predeclared and does not replace the original gates. Identity-refresh and public domain checks do not certify CKKS error or refresh range preservation.

`preflight/` is the source-bound final evidence. `development/` retains earlier successful graph/reference and44/61-test logs; those exploratory runs did not create startup source archives, so only their observed outcomes are claimed. No development failure was observed. The final source archive includes tests and lifecycle helpers. No model, key, ciphertext, token or plaintext array bodies are archived. Upstream public seeded toy fixture source files remain in the authenticated199-source archive.

Run `python -B verify.py` for portable public readback. The verifier checks immutable recorded evidence, not current machine resources or live jobs.
''')
(D/'artifact_hashes.json').write_text(json.dumps({str(f.relative_to(D)):hashlib.sha256(f.read_bytes()).hexdigest()for f in sorted(D.rglob('*'))if f.is_file()},indent=2)+'\n')
print(D)
