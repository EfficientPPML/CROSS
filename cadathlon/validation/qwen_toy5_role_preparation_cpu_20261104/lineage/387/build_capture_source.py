"""Narrow source-only retained-edge successor; no candidate imports."""
from pathlib import Path
import ast,hashlib,json,difflib,tarfile,io
OLD=Path('/tmp/qwen-toy5-actual-checkpoint-runtime-v2-20261104')
NEW=Path('/tmp/qwen-toy5-actual-checkpoint-runtime-v3-20261104')
OUT=Path('/tmp/qwen-toy5-actual-checkpoint-runtime-v3-handoff-20261104')
def sha(v):return hashlib.sha256(v).hexdigest()
def raw(v):return(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def put(p,b):
    with p.open('xb')as f:f.write(b)
mp=Path('/tmp/qwen-toy5-actual-checkpoint-runtime-v2-handoff-20261104/sources.json').read_bytes()
assert sha(mp)=='ddefc197d5323a064203d15a08c81c9a4c757dede8c77396e6586eed495346b0'
table=json.loads(mp);assert len(table)==385;NEW.mkdir()
for n,pin in table.items():
    b=(OLD/n).read_bytes();assert sha(b)==pin
    p=NEW/n;p.parent.mkdir(parents=True,exist_ok=True);put(p,b)
def once(t,a,b):assert t.count(a)==1,a[:100];return t.replace(a,b)
name='q5_actual_process.py';before=(OLD/name).read_text();text=before
text=once(text,"            owned=[],barriers=[],ntt_owners=[],ntt_context=None,ntt_captured=False,\n", "            owned=[],barriers=[],ntt_owners=[],ntt_edges=[],returned_bindings=None,ntt_context=None,ntt_captured=False,\n")
text=once(text,"        if r['ntt_captured']and getattr(ctx,'_polynomial_ntt_ctx',None)is not r['ntt_context']:\n            raise ValueError('original setup context NTT binding changed')\n", "        self._check_retained(r)\n")
marker="    def _retain_returned(self,r):\n"
helper='''    def _check_retained(self,r):
        ctx=r['fixed'][2]
        if r['ntt_captured']and getattr(ctx,'_polynomial_ntt_ctx',None)is not r['ntt_context']:
            raise ValueError('original setup context NTT binding changed')
        saved=r['returned_bindings']
        if saved is not None:
            value,payload,moduli,ntt=saved
            if r['value']is not value or value.polynomial is not payload or value.moduli_array is not moduli or value.ntt_ctx is not ntt:
                raise ValueError('original returned setup arrays/NTT binding changed')
        for ntt,ff in r['ntt_edges']:
            if getattr(ntt,'ff_ctx',None)is not ff:
                raise ValueError('original returned/attached finite-field owner changed')

'''
text=once(text,marker,helper+marker)
text=once(text,"        ctx=r['fixed'][2];value=r['value'];attached=getattr(ctx,'_polynomial_ntt_ctx',None)\n", "        self._check_retained(r)\n        ctx=r['fixed'][2];value=r['value'];attached=getattr(ctx,'_polynomial_ntt_ctx',None)\n")
text=once(text,"        if value is not None:\n            visit(value.polynomial);visit(value.moduli_array)\n", "        if value is not None:\n            if r['returned_bindings']is None:r['returned_bindings']=(value,value.polynomial,value.moduli_array,value.ntt_ctx)\n            visit(value.polynomial);visit(value.moduli_array)\n")
text=once(text,"            if ntt is None:continue\n            for owner in(ntt,getattr(ntt,'ff_ctx',None)):\n", "            if ntt is None:continue\n            ff=getattr(ntt,'ff_ctx',None)\n            if all(ntt is not original for original,_ in r['ntt_edges']):r['ntt_edges'].append((ntt,ff))\n            for owner in(ntt,ff):\n")
text=once(text,"        for value,wait in barriers:\n            final.attempt('original encrypted input/NTT array barrier',wait)\n", "        for value,wait in barriers:\n            final.attempt('original encrypted input/NTT array barrier',wait)\n            final.attempt('post-barrier original setup array/NTT/field bindings',lambda:self._check_retained(r))\n")
ast.parse(text);(NEW/name).write_text(text)
(NEW/'provenance'/'owner_edge_correction.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),text.splitlines(True),fromfile='held385/'+name,tofile='successor/'+name)))
(NEW/'SOURCE_DESIGN.md').write_text((OLD/'SOURCE_DESIGN.md').read_text()+'''
### Retained-edge refinement after held385

Held385/mapddefc197 preserves the first source-only owner correction. Root's final detail arrived after that freeze: retain and compare original returned NTT/ff edges, not only the attached context pointer. This successor adds original returned payload/moduli/NTT and each NTT-to-ff identity capture before any following callback. Checks before rescanning and after every original array barrier reject replacement while attempt-all cleanup still barriers all already captured originals. A partial encryption exception with only an attached NTT remains failed; no returned value or successful load is fabricated. Only _EncryptedInput and this disclosure change relative to385; all its CT/idempotence fixes remain exact. No imports/tests or issuer implementation occurred.
''')
history=dict(schema='qwen-toy5-owner-edge-source-successor-v1',parent_source_count=385,parent_source_map_sha256=sha(mp),
 parent_source_archive_sha256='794dd6e32f5d42051fb44b9df4feff79bec233c5b393ea2695083a807a9c38e7',
 overlays=['q5_actual_process.py','SOURCE_DESIGN.md'],additions=['owner_edge_correction_history.json','provenance/owner_edge_correction.patch'],
 scope='original returned array/NTT and NTT-to-ff edge retention/guard only; actual runtime still denies',
 candidate_imports=False,tests=False,body_reads=False,device_or_capacity_probes=False)
put(NEW/'owner_edge_correction_history.json',raw(history))
files={p.relative_to(NEW).as_posix():sha(p.read_bytes())for p in sorted(NEW.rglob('*'))if p.is_file()}
assert len(files)==387
assert sorted(n for n,pin in table.items()if files[n]!=pin)==['SOURCE_DESIGN.md','q5_actual_process.py']
assert all(sha((OLD/n).read_bytes())==pin for n,pin in table.items())
def definitions(p):return{n.name:ast.dump(n,include_attributes=False)for n in ast.parse(p.read_bytes()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
a=definitions(OLD/name);b=definitions(NEW/name);assert a.keys()==b.keys()
assert all(v==b[k]for k,v in a.items()if k!='_EncryptedInput')
for n in files:
    if n.endswith('.py'):ast.parse((NEW/n).read_bytes(),filename=n)
OUT.mkdir();put(OUT/'sources.json',raw(files));put(OUT/'parent385-sources.json',mp);put(OUT/'build_capture_source.py',Path(__file__).read_bytes())
with tarfile.open(OUT/'sources.tar.gz','w:gz')as archive:
    for n in files:
        data=(NEW/n).read_bytes();info=tarfile.TarInfo(n);info.size=len(data);info.mode=0o644;info.mtime=0
        archive.addfile(info,io.BytesIO(data))
with tarfile.open(OUT/'sources.tar.gz','r:gz')as archive:
    assert {m.name:sha(archive.extractfile(m).read())for m in archive.getmembers()}==files
review=dict(schema='qwen-toy5-owner-edge-static-review-v1',passed=True,source_only=True,source_count=387,
 exact_parent_files=383,parent385_current_unchanged=True,parent250_core34_common12_exact=True,
 only_class_delta='_EncryptedInput',all_python_ast_parse=True,candidate_imports=False,tests=False,
 no_runtime_barrier_or_numerical_proof=True)
put(OUT/'source-review.json',raw(review))
handoff=dict(schema='qwen-toy5-owner-edge-handoff-v1',root=str(NEW),source_count=387,
 source_map_sha256=sha((OUT/'sources.json').read_bytes()),source_archive_sha256=sha((OUT/'sources.tar.gz').read_bytes()),
 source_review_sha256=sha((OUT/'source-review.json').read_bytes()),parent385_map_sha256=sha(mp),
 code_hashes={n:files[n]for n in ('q5_ciphertext_owner.py','q5_actual_process.py')},
 overlays=history['overlays'],additions=history['additions'],source_only=True,actual_runtime_issuer=False,actual_execution=False)
put(OUT/'handoff.json',raw(handoff));print(json.dumps(handoff,sort_keys=True))
