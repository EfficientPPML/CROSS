"""Offline public metadata/source proof; no model or tensor payload reads."""
from pathlib import Path
import hashlib,json,tarfile,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):return json.loads(path.read_bytes())
def check_archive(path,expected):
    with tarfile.open(path) as t:
        rows=t.getmembers();assert len(rows)==len({m.name for m in rows})==len(expected) and all(m.isfile() for m in rows)
        assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in rows}==expected
        assert not any(Path(m.name).suffix in ('.npy','.npz','.safetensors') for m in rows)
def counts(path):
    suites=list(ET.parse(path).getroot().iter('testsuite'))
    return {n:sum(int(s.attrib.get(n,0)) for s in suites) for n in ('tests','failures','errors','skipped')}
manifest=load(P/'sha256.json')
assert {p.relative_to(P).as_posix() for p in P.rglob('*') if p.is_file() and p.name!='sha256.json' and '__pycache__' not in p.parts}==set(manifest)
assert {n:sha(P/n) for n in manifest}==manifest
r=load(P/'actual/report.json');h=r['source_hashes_before'];assert len(h)==10
assert h==r['source_hashes_after']==r['execution_source_hashes_after']
assert all(r[n]is True for n in ('passed','execute','actual_execution_attempted','all_processes_reaped_and_owned_group_empty','sources_unchanged','source_archive_verified','terminal_scope_completed','termination_scope_completed'))
assert sha(P/'actual/report.json')=='0f9c089cc4e4e63b626f264ae5bdc6e5c0b1c8fa2c0af6c85fee01a81e888be6'
assert sha(P/'actual/sources.tar.gz')==r['source_archive_sha256']=='1b921363993f5a06eced87ffa5f44b284b51100a46af43dd58e6788cb806a175'
check_archive(P/'actual/sources.tar.gz',h)
assert {n:sha(P/'candidate'/n) for n in h}==h
proc=r['process'];assert proc['direct_returncode']==0 and all(proc[n]is True for n in ('direct_child_reaped','group_empty_verified','cleanup_completed'))
terminal=load(P/'actual/terminal.json');assert terminal['observed_exit_code']==0 and terminal['tool_session_reaped']is True and terminal['owned_group_absent']is True
assert terminal['report_sha256']==sha(P/'actual/report.json')
w=load(P/'actual/worker/report.json');assert sha(P/'actual/worker/report.json')==r['worker_report_sha256']
assert all(w[n]is True for n in ('passed','operation_complete','termination_scope_completed'))
m=load(P/'actual/worker/parameter_manifest.json')
assert sha(P/'actual/worker/parameter_manifest.json')==w['parameter_manifest_sha256']==r['parameter_manifest_sha256']=='60c556376e42358f227a2978edb6b2addf59dc9e694b65e17265b15e813d65c3'
assert m['matrices']==w['matrices'] and len(m['matrices'])==114 and len({row['file'] for row in m['matrices'].values()})==113
for layer in range(28):
    assert m['matrices'][f'model.layers.{layer}.self_attn.o_proj.weight']['shape']==[1024,2048]
    for kind,shape in (('gate',[3072,1024]),('up',[3072,1024]),('down',[1024,3072])):
        assert m['matrices'][f'model.layers.{layer}.mlp.{kind}_proj.weight']['shape']==shape
a=m['matrices']['lm_head.weight'];b=m['matrices']['model.embed_tokens.weight']
assert a['shape']==b['shape']==[151936,1024] and a['file']==b['file'] and a['sha256']==b['sha256'] and a['source_raw_sha256']==b['source_raw_sha256']
assert b['storage_alias_of']=='lm_head.weight' and b['exact_bf16_alias_verified']is True
assert all(row['promotion_exact']is True and row['source_dtype']=='BF16' and row['dtype']=='<f8' for row in m['matrices'].values())
f=load(P/'preflight/preflight-v3/report.json');assert f['source_hashes_before']==f['source_hashes_after']==f['execution_source_hashes_after']==h and f['passed']is True
check_archive(P/'preflight/preflight-v3/sources.tar.gz',h)
assert counts(P/'preflight/preflight-v3/tests.xml')==dict(tests=53,failures=0,errors=0,skipped=0)
assert counts(P/'history/preflight-v1/tests.xml')==dict(tests=47,failures=0,errors=0,skipped=0)
assert counts(P/'history/preflight-v2/tests.xml')==dict(tests=53,failures=1,errors=0,skipped=0)
assert counts(P/'independent/negative-v1/tests.xml')==dict(tests=2,failures=2,errors=0,skipped=0)
review=load(P/'independent/successor-v2/report.json');assert review['passed']is True and len(review['source_hashes_before'])==11
assert review['source_hashes_before']==review['source_hashes_after'] and all(review['source_hashes_before'][n]==v for n,v in h.items())
check_archive(P/'independent/successor-v2/sources.tar.gz',review['source_hashes_before'])
assert counts(P/'independent/successor-v2/tests.xml')==dict(tests=9,failures=0,errors=0,skipped=0)
root=load(P/'root_readback.json');assert root['passed']is True and root['unique_values_checked']==478543872 and root['named_values_checked']==634126336
assert root['parent_report_sha256']==sha(P/'actual/report.json')
print(json.dumps(dict(passed=True,named_matrices=114,physical_matrices=113,sources=10,synthetic_tests=53,independent_tests=9,model_forward=False,he=False)))
