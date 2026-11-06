import ast,hashlib,json,pathlib,tarfile
R=pathlib.Path('/tmp/qwen-server-snapshot-runtime-20261104'); D=pathlib.Path('/tmp/qwen-server-snapshot-runtime-handoff-20261104'); O=pathlib.Path(__file__).parent
H=lambda x:hashlib.sha256(x).hexdigest()
hraw=(D/'handoff.json').read_bytes();h=json.loads(hraw)
assert H(hraw)=='b1f197da134487a2d5c136320cd51d331408c3274d8e2f078a800e5ef4d2842c'
raw=(D/'sources.json').read_bytes();m=json.loads(raw);praw=(D/'parent387-sources.json').read_bytes();p=json.loads(praw)
assert len(m)==390 and len(p)==387 and H(raw)==h['source_map_sha256'] and H(praw)==h['parent_map_sha256']
def current(root):
    assert not any(x.is_symlink() for x in root.rglob('*'))
    return {x.relative_to(root).as_posix():H(x.read_bytes()) for x in root.rglob('*') if x.is_file()}
assert current(R)==m
for fname,wanted,hashkey in [('sources.tar.gz',m,'source_archive_sha256'),('parent387-sources.tar.gz',p,'parent_archive_sha256')]:
    path=D/fname;assert H(path.read_bytes())==h[hashkey]
    with tarfile.open(path) as t:
        members=t.getmembers();assert len(members)==len(wanted) and all(x.isfile() for x in members)
        assert {x.name:H(t.extractfile(x).read()) for x in members}==wanted
changed=sorted(k for k in p if m[k]!=p[k]);added=sorted(set(m)-set(p))
assert changed==['qwen_server_backend.py','qwen_server_entry.py','qwen_server_process.py']
assert added==['SNAPSHOT_INTEGRATION.md','qwen_admission_snapshot.py','snapshot_integration.json']
assert sum(m[k]==v for k,v in p.items())==384
assert m['qwen_admission_snapshot.py']=='8790711a1a402f4c21ef444ccdb76f773b4c62cd334f5be07a49d2cd4535b395'
proc=ast.parse((R/'qwen_server_process.py').read_text());back=ast.parse((R/'qwen_server_backend.py').read_text())
closure=ast.literal_eval(next(n.value for n in proc.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='SNAPSHOT_BINDING_SOURCES' for t in n.targets)))
assert len(closure)==20 and 'cleanup_origin.py'not in closure
cr=(D/'callback-source-closure.json').read_bytes();assert H(cr)==h['callback_closure_sha256'] and json.loads(cr)=={n:m[n]for n in closure}
lookup=next(n for n in back.body if isinstance(n,ast.FunctionDef)and n.name=='_runtime')
assert any(isinstance(n,ast.ImportFrom)and n.module=='qwen_server_process'and any(x.name=='require_runtime'for x in n.names)for n in lookup.body)
assert any(isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='require_runtime'for n in ast.walk(lookup))
# Assert entry differs only in its explicit allowed-local module set, without importing it.
with tarfile.open(D/'parent387-sources.tar.gz')as t:oldentry=ast.parse(t.extractfile('qwen_server_entry.py').read())
entry=ast.parse((R/'qwen_server_entry.py').read_text())
def strip(tree):
    tree.body=[n for n in tree.body if not(isinstance(n,ast.Assign)and any(isinstance(v,ast.Name)and v.id=='ROOT_MODULES'for v in n.targets))];return ast.dump(tree,include_attributes=False)
assert strip(oldentry)==strip(entry)
report=dict(schema='qwen-server-snapshot-integration-security-source-review-v1',handoff_sha256=H(hraw),source_manifest_sha256=H(raw),source_archive_sha256=h['source_archive_sha256'],parent_manifest_sha256=H(praw),source_count=390,parent_source_count=387,unchanged_parent_files=384,changed=changed,added=added,callback_source_count=20,source_hashes=m,disposition='two bounded source findings require successor; no execution claim',findings=[dict(id='callback_cleanup_dependency',description='check_bindings calls keygen_output.read -> finish -> original protect_body/select. Those functions use cleanup_origin.is_cleanup/mark globals, but cleanup_origin is absent from the20 source/module bindings. Imported alias capture does not bind these transitive exports/code.',recommendation='Add exact cleanup_origin4543 source/module capture for the actual read cleanup path; qwen_finalizer is not on that path and is not requested by this finding.'),dict(id='dispatcher_called_before_guard',description='backend._runtime dynamically imports current process.require_runtime and invokes it before retained guard. RuntimeAdmission methods similarly look up current dispatch exports. A replaced exported dispatcher can be invoked before later module-binding rejection.',recommendation='Retain original dispatchers at supported caller boundaries, compare current exports before invocation, and call retained originals. Preserve original390; no private-registry/interpreter-mutation scope expansion.')],other_source_conclusions=['Hot guard callback itself contains no runtime, validator, capacity, audit-transition or backend invocation. Finite full packet/source/public proof checks remain separate.','Runtime-control registry holds original instance/control and original field dictionary; metadata bytes/digest/ID field identities are retained. Pending key/body checks permit necessary original audit callbacks; nested transitions set sticky failure and outer transitions recheck activity.','Original335 map, original borrowed-file/keyless/security/reader arithmetic and audit code are unchanged. Capacity remains unconditionally denied; actual readiness false.','Finalizer attempts full source/audit/runtime/evidence/signal cleanup, skips snapshot retirement after selected failure, and quarantines original retained state. Snapshot retirement remains after original audit/evidence closure.','Historical source inventory metadata remains preserved, so this held source is not an executable actual-role closure/admission. No inferred actual-source readiness.'],candidate_imports=False,collection=False,tests=False,backend_or_capacity_probes=False,body_reads=False,actual_authority=False)
(O/'review.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n');print(H((O/'review.json').read_bytes()))
