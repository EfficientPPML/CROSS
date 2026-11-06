from pathlib import Path
import hashlib,json,tarfile,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
manifest=json.loads((P/'archive_manifest.json').read_bytes())
assert {p.relative_to(P).as_posix():sha(p.read_bytes()) for p in P.rglob('*') if p.is_file() and p.name!='archive_manifest.json'}==manifest
r=json.loads((P/'measured/report.json').read_bytes());expected=r['source_hashes_start']
assert len(expected)==300 and r['source_hashes_end']==expected
assert all(r[n] is True for n in ('passed','synthetic','source_files_unchanged','source_archive_verified','cleanup_completed','terminal_scope_completed','termination_scope_completed','all_processes_reaped'))
assert r['genuine_eligible'] is False and r['actual_ranked_launch'] is False
assert sha((P/'measured/sources.tar.gz').read_bytes())==r['source_archive_sha256']
with tarfile.open(P/'measured/sources.tar.gz') as t:
 members=t.getmembers();assert len(members)==300==len({m.name for m in members}) and all(m.isfile() for m in members)
 assert {m.name:sha(t.extractfile(m).read()) for m in members}==expected
counts={k:sum(int(s.get(k,0)) for s in ET.parse(P/'measured/tests.xml').getroot().iter('testsuite')) for k in ('tests','failures','errors','skipped')}
assert counts==r['tests']==dict(tests=45,failures=0,errors=0,skipped=0)
assert [(x['size'],x['seed']) for x in r['cases']]==[(s,k) for s in (1,2,3) for k in (42,43,44)]
assert sum(x['count'] for x in r['cases'])==2124 and len({x['context_id'] for x in r['cases']})==9
assert all(x['num_runs']==1 and x['stage_count']==7 and x['synthetic_correct']==x['synthetic_native_correct']==x['count'] and x['timing']['timer_kind']=='component_diagnostic' for x in r['cases'])
assert r['explicit_repeat']['stage_count']==12 and r['explicit_repeat']['fresh_unique_request_count']==200
print(json.dumps({'passed':True,'source_count':300,'tests':45,'default_cases':9,'explicit_repeat':2,'genuine_eligible':False}))
