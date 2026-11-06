from pathlib import Path
import hashlib,json,shutil,os,subprocess
P=Path('/tmp/qwen-expanded-prefix-protocol-20261104');D=Path('/home/jianming_gatech/CROSS/cadathlon/validation/qwen_expanded_prefix_protocol_public_20261104');D.mkdir(exist_ok=False)
r=json.loads((P/'public-v3/report.json').read_bytes());h=r['source_hashes_before'];assert {n:hashlib.sha256((P/n).read_bytes()).hexdigest()for n in h}==h
for c in r['children']:
 try:os.kill(c['pid'],0)
 except ProcessLookupError:pass
 else:raise ValueError('producer present')
 try:os.killpg(c['process_group'],0)
 except ProcessLookupError:pass
 else:raise ValueError('group present')
files={n:P/n for n in ('protocol.py','test_protocol.py','DESIGN.md','input_pins.json','helper_provenance.json')}
for run,destination in (('public-v2','preserved-public-v2'),('public-v3','public-v3')):
 for name in ('report.json','sources.tar.gz','plan.json','tests.log','tests.xml','audit.json','plan-audit.json','plan.log'):files[destination+'/'+name]=P/run/name
 files[destination+'/collector.log']=P/(run+'.log')
for f in (P/'draft-v1').iterdir():
 if f.is_file():files['preserved-draft-v1/'+f.name]=f
for n in ('development-tests-v2.log','development-tests-v2.xml'):files['development/'+n]=P/n
for n,f in files.items():
 target=D/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
for n in ('archive.py','verify.py'):shutil.copyfile(Path(__file__).with_name(n),D/n)
(D/'README.md').write_text('''# Separate expanded-prefix Qwen protocol proposal

This public design declares3200 calibration entries (the same96 original prompts at every length1..32 plus original64teacher/64free steps) and binds the old416 subset. It preserves fixed gates and proposes a uniform113-site normalization-only selection policy. Existing candidate3 and all old results remain unchanged. No input/tokenizer, selection, certificate/model or HE execution occurred.

The new authored32-prompt/all-prefix validation recipe is frozen as a proposal, not yet tokenized or certified disjoint. Before any new calibration result or selection, a separately reviewed tokenizer phase must check expanded3072+originaldecode/oldsets/row0 full-sequence disjointness and report entries versus unique contexts by length. Entries/denominators remain fixed despite duplicates. Row0 tokenID0 is a post-freeze regression only. See DESIGN.md and public-v3/plan.json for exact rules and pending gates.

Final public-v3 passed68 tests and23 exact source/current/execution/archive checks with zero audited old input/model/backend reads. Preserved public-v2 has all68 tests passing but overallfailure because inventory erroneously expected67; the sole successor change is that count. Preserved draft-v1 predates expanded-prefix disjointness wording and has development27-test evidence only, without a startup archive claim.

The archive includes public original protocols/coefficient manifests/algorithm source and newly authored validation text design. It contains no old calibration/validation token bodies, checkpoint tensors, keys or ciphertext. No claim of1088 unique validation contexts or already achieved untouched token disjointness is made. Run `python -B verify.py` for offline public readback.
''')
(D/'artifact_hashes.json').write_text(json.dumps({str(x.relative_to(D)):hashlib.sha256(x.read_bytes()).hexdigest()for x in sorted(D.rglob('*'))if x.is_file()},indent=2)+'\n');print(D)
