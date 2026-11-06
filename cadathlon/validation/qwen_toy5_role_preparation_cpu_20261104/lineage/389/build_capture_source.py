"""Source-only immutable returned-table snapshot refinement; no runtime import."""
from pathlib import Path
import ast,json,hashlib,difflib,tarfile,io
OLD=Path('/tmp/qwen-toy5-actual-checkpoint-runtime-v3-20261104')
NEW=Path('/tmp/qwen-toy5-actual-checkpoint-runtime-v4-20261104')
OUT=Path('/tmp/qwen-toy5-actual-checkpoint-runtime-v4-handoff-20261104')
def sha(v):return hashlib.sha256(v).hexdigest()
def raw(v):return(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def put(p,b):
    with p.open('xb')as f:f.write(b)
mp=Path('/tmp/qwen-toy5-actual-checkpoint-runtime-v3-handoff-20261104/sources.json').read_bytes()
assert sha(mp)=='e4cfe702d42e44cc48468ea8c7954b43df49709f8d1f9ce0da16d64f361684fb'
table=json.loads(mp);assert len(table)==387;NEW.mkdir()
for n,pin in table.items():
    b=(OLD/n).read_bytes();assert sha(b)==pin
    p=NEW/n;p.parent.mkdir(parents=True,exist_ok=True);put(p,b)
def once(t,a,b):assert t.count(a)==1,a[:80];return t.replace(a,b)
name='q5_actual_process.py';before=(OLD/name).read_text();text=before
text=once(text,"            value=None,state=None,entered=False,closed=False,failure=None)\n", "            table_roots=(),table_dicts=(),table_held=(),table_snapshot=None,\n            value=None,state=None,entered=False,closed=False,failure=None)\n")
text=once(text,"                raise ValueError('original returned/attached finite-field owner changed')\n", "                raise ValueError('original returned/attached finite-field owner changed')\n        if r['table_snapshot']is not None:\n            from q5_checkpoint_arithmetic import working_snapshot\n            for owner,fields in r['table_dicts']:\n                if vars(owner)is not fields:raise ValueError('original setup NTT/field dictionary changed')\n            if working_snapshot(r['table_roots'])[1]!=r['table_snapshot']:\n                raise ValueError('original setup NTT/field table or control changed')\n")
text=once(text,"        self._check_retained(r)\n        ctx=r['fixed'][2];value=r['value'];attached=getattr(ctx,'_polynomial_ntt_ctx',None)\n", "        self._check_retained(r)\n        # Once the returned/attached graph is captured, it is immutable here.\n        # Never add replacement arrays or foreign waits on a later close.\n        if r['table_snapshot']is not None:return\n        ctx=r['fixed'][2];value=r['value'];attached=getattr(ctx,'_polynomial_ntt_ctx',None)\n")
text=once(text,"        for owner in tuple(r['ntt_owners']):visit(vars(owner))\n", "        for owner in tuple(r['ntt_owners']):visit(vars(owner))\n        from q5_checkpoint_arithmetic import working_snapshot\n        r['table_roots']=tuple(r['ntt_owners'])\n        r['table_dicts']=tuple((owner,vars(owner))for owner in r['table_roots'])\n        r['table_held'],r['table_snapshot']=working_snapshot(r['table_roots'])\n        self._check_retained(r)\n")
ast.parse(text);(NEW/name).write_text(text)
(NEW/'provenance'/'owner_table_snapshot.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),text.splitlines(True),fromfile='held387/'+name,tofile='successor/'+name)))
(NEW/'SOURCE_DESIGN.md').write_text((OLD/'SOURCE_DESIGN.md').read_text()+'''
### Original table/control graph refinement

The preserved intermediate387 adds direct returned-NTT and NTT-to-ff edges. Root's next source detail identified replacement of array/table fields within those unchanged objects. This successor captures original NTT/ff dictionaries and their complete reachable table/control identity snapshot at first returned/attached observation using the already selected working_snapshot helper. It retains that original graph and validates it before any later traversal. Once captured, retention does not add new arrays or wait callables; replacement is rejected and cleanup still attempts every originally captured barrier. The helper records array identity/shape/dtype and container/control structure, not numeric table-body equality; no arbitrary-global or inaccessible-constructor-local coverage is claimed. The original NTT/ff source objects have no excluded ctx/_param_cache fields in this traversal.

Root correction385 source review1dc0d72f accepted the CT/moduli and InputOwner tombstone fixes while retaining the setup graph-binding finding. This refinement has no new import/test evidence. The previously pinned RuntimeAdmission plan remains separate and unimplemented here.
''')
receipt=Path('/tmp/qwen-toy5-actual-runtime-root-review-20261104/correction385-review.json').read_bytes()
assert sha(receipt)=='1dc0d72f4823a4791423212db529b174d45a70a4e8aa63c283d574dcee952d3c'
history=dict(schema='qwen-toy5-original-table-snapshot-source-v1',parent387_map_sha256=sha(mp),
 parent387_archive_sha256='333dc4b2ab834791e42bf7e472ccd9b20a18f408a378709dee772351faa6f1af',
 correction385_root_review_sha256=sha(receipt),
 overlays=['q5_actual_process.py','SOURCE_DESIGN.md'],additions=['owner_table_snapshot_history.json','provenance/owner_table_snapshot.patch'],
 source_only=True,candidate_imports=False,tests=False,body_reads=False,actual_runtime_issuer=False,
 semantics='one original graph snapshot; no subsequent barrier target acquisition; original waits still attempted on replacement',
 scope_limits=['identity/shape/dtype/control structure, not numeric table-body equality','no inaccessible constructor-local ownership','no arbitrary global mutation claim'])
put(NEW/'owner_table_snapshot_history.json',raw(history))
files={p.relative_to(NEW).as_posix():sha(p.read_bytes())for p in sorted(NEW.rglob('*'))if p.is_file()}
assert len(files)==389
assert sorted(n for n,pin in table.items()if files[n]!=pin)==['SOURCE_DESIGN.md','q5_actual_process.py']
assert all(sha((OLD/n).read_bytes())==pin for n,pin in table.items())
def nodes(p):return{n.name:ast.dump(n,include_attributes=False)for n in ast.parse(p.read_bytes()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
a=nodes(OLD/name);b=nodes(NEW/name);assert a.keys()==b.keys()
assert all(v==b[k]for k,v in a.items()if k!='_EncryptedInput')
parent382=json.loads(Path('/tmp/qwen-toy5-actual-checkpoint-runtime-handoff-20261104/sources.json').read_bytes())
assert all(files[n]==pin for n,pin in parent382.items()if n.startswith(('parent/','runtime/jaxite_word/','common/')))
for n in files:
    if n.endswith('.py'):ast.parse((NEW/n).read_bytes(),filename=n)
OUT.mkdir();put(OUT/'sources.json',raw(files));put(OUT/'parent387-sources.json',mp);put(OUT/'build_capture_source.py',Path(__file__).read_bytes())
with tarfile.open(OUT/'sources.tar.gz','w:gz')as archive:
    for n in files:
        data=(NEW/n).read_bytes();info=tarfile.TarInfo(n);info.size=len(data);info.mode=0o644;info.mtime=0
        archive.addfile(info,io.BytesIO(data))
with tarfile.open(OUT/'sources.tar.gz','r:gz')as archive:
    assert {m.name:sha(archive.extractfile(m).read())for m in archive.getmembers()}==files
review=dict(schema='qwen-toy5-original-table-snapshot-static-review-v1',passed=True,source_only=True,source_count=389,
 parent387_current_unchanged=True,parent250_core34_common12_exact=True,only_class_delta='_EncryptedInput',
 all_python_ast_parse=True,candidate_imports=False,tests=False,no_runtime_barrier_or_numerical_proof=True)
put(OUT/'source-review.json',raw(review))
handoff=dict(schema='qwen-toy5-owner-table-snapshot-handoff-v1',root=str(NEW),source_count=389,
 source_map_sha256=sha((OUT/'sources.json').read_bytes()),source_archive_sha256=sha((OUT/'sources.tar.gz').read_bytes()),
 source_review_sha256=sha((OUT/'source-review.json').read_bytes()),parent387_map_sha256=sha(mp),
 code_hashes={n:files[n]for n in ('q5_ciphertext_owner.py','q5_actual_process.py')},
 overlays=history['overlays'],additions=history['additions'],source_only=True,actual_runtime_issuer=False,actual_execution=False)
put(OUT/'handoff.json',raw(handoff));print(json.dumps(handoff,sort_keys=True))
