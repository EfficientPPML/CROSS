"""Offline public preflight verification; no private/backend reads."""
from pathlib import Path
import hashlib,json,tarfile,xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
manifest=json.loads((HERE/'sha256.json').read_bytes())
assert {p.relative_to(HERE).as_posix() for p in HERE.rglob('*') if p.is_file() and p.name!='sha256.json' and '__pycache__' not in p.parts}==set(manifest)
assert {n:sha((HERE/n).read_bytes()) for n in manifest}==manifest
r=json.loads((HERE/'measured/report.json').read_bytes());expected=r['source_hashes_before']
assert r['passed']is True and r['sources_unchanged']is True and r['source_archive_verified']is True and r['termination_scope_completed']is True
assert len(expected)==272 and expected==r['source_hashes_after']
assert sha((HERE/'measured/sources.tar.gz').read_bytes())==r['source_archive_sha256']
with tarfile.open(HERE/'measured/sources.tar.gz') as t:
 members=t.getmembers();assert len(members)==len(expected)==len({m.name for m in members}) and all(m.isfile() for m in members)
 raw={m.name:t.extractfile(m).read() for m in members};assert {n:sha(b) for n,b in raw.items()}==expected
parent=json.loads(raw['parent_source_manifest.json']);assert len(parent)==241 and all(expected[n]==h for n,h in parent.items())
assert not any(Path(n).suffix in ('.npy','.npz','.safetensors') for n in raw)
counts={n:sum(int(s.attrib.get(n,0)) for s in ET.parse(HERE/'measured/tests.xml').getroot().iter('testsuite')) for n in ('tests','failures','errors','skipped')}
assert counts==r['tests']==dict(tests=294,failures=0,errors=0,skipped=0)
assert len(r['installed_probes'])==5 and sum(x['denied_count'] for x in r['installed_probes'])==106
for stage,count in ((1,15),(2,24),(3,21),(4,25),(5,21)):
 p=json.loads((HERE/f'measured/installed{stage}.json').read_bytes())
 assert p['passed']is True and p['all_restored']is True and p['allowed_handler_unchanged']is True and p['denied_count']==count
 assert p['context_constructed']is False and p['key_generation']is False and p['payload_or_model_reads']is False
 assert all(expected[x['path']]==x['sha256'] for x in p['loaded_local_modules'].values())
terminal=json.loads((HERE/'measured/terminal.json').read_bytes())
assert terminal['observed_exit_code']==0 and terminal['tool_session_reaped']is True and terminal['wrapper_pid_absent']is True
assert terminal['report_sha256']==sha((HERE/'measured/report.json').read_bytes())
for p in (HERE/'candidate').glob('*.py'):assert sha(p.read_bytes())==expected[p.name]
print(json.dumps({'passed':True,'tests':294,'installed_denials':106,'exact_sources':272,'parent_sources_unchanged':241,'actual_ranked_execution':False}))
