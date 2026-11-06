"""Author an isolated source successor; stdlib only, no candidate import."""
from pathlib import Path
import hashlib,json,difflib,ast,shutil
OLD=Path('/tmp/qwen-toy5-actual-checkpoint-runtime-20261104')
NEW=Path('/tmp/qwen-toy5-actual-checkpoint-runtime-v2-20261104')
MP=Path('/tmp/qwen-toy5-actual-checkpoint-runtime-handoff-20261104/sources.json')
def sha(v):return hashlib.sha256(v).hexdigest()
def raw(v):return(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
mb=MP.read_bytes();assert sha(mb)=='b787490da5ad804dacdeca7c19cbb39a99868ab80ea762a607e1a6457cfb3180'
table=json.loads(mb);assert len(table)==382
NEW.mkdir()
for name,pin in table.items():
    b=(OLD/name).read_bytes();assert sha(b)==pin
    p=NEW/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
def replace(text,old,new,count=1):
    assert text.count(old)==count,(old[:80],text.count(old),count)
    return text.replace(old,new)

name='q5_ciphertext_owner.py';before=(OLD/name).read_text();text=before
text=replace(text,"            value=backend.polynomial.Polynomial.from_array(array,shapes,parameters);r['value']=value;self.value=value\n", "            value=backend.polynomial.Polynomial.from_array(array,shapes,parameters);r['value']=value;self.value=value\n            # Own both returned arrays before any metadata/security callback.\n            r['arrays'].append(value.polynomial);r['arrays'].append(value.moduli_array)\n")
text=replace(text,"            r['arrays'].append(value.polynomial);value.polynomial.block_until_ready();guard()\n", "            value.polynomial.block_until_ready();guard()\n")
block="        if self in _CLOSED:\n            if primary is not None:raise primary\n            return\n"
text=replace(text,block+block,block)
old="    def close(self,primary=None):\n        r=_record(self)\n        if r['closed']:\n"
new="    def close(self,primary=None):\n        if self in _CLOSED:\n            if primary is not None:raise primary\n            return\n        r=_record(self)\n        if r['closed']:\n"
text=replace(text,old,new)
ast.parse(text);(NEW/name).write_text(text)
(NEW/'provenance'/'owner_correction_ciphertext.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),text.splitlines(True),fromfile='held382/'+name,tofile='successor/'+name)))

name='q5_actual_process.py';before=(OLD/name).read_text();text=before
text=replace(text,"            owned=[],value=None,state=None,entered=False,closed=False,failure=None)\n", "            owned=[],barriers=[],ntt_owners=[],ntt_context=None,ntt_captured=False,\n            value=None,state=None,entered=False,closed=False,failure=None)\n")
text=replace(text,"        if r['closed']:raise ValueError('input encryption consumer closed')\n", "        if r['closed']:raise ValueError('input encryption consumer closed')\n        if r['ntt_captured']and getattr(ctx,'_polynomial_ntt_ctx',None)is not r['ntt_context']:\n            raise ValueError('original setup context NTT binding changed')\n")
marker="    def load(self):\n        r=self.check(loaded=False)\n"
helper='''    def _retain_returned(self,r):
        # Original encrypt may attach its NTT context and then raise before
        # returning a ciphertext. Retain that context as well as returned NTT/ff
        # objects; no new table/context is constructed or injected here.
        ctx=r['fixed'][2];value=r['value'];attached=getattr(ctx,'_polynomial_ntt_ctx',None)
        if r['ntt_captured']and attached is not r['ntt_context']:
            raise ValueError('original attached setup NTT context changed')
        r['ntt_context']=attached;r['ntt_captured']=True
        seen=set()
        def visit(value):
            if id(value)in seen:return
            seen.add(id(value));wait=getattr(value,'block_until_ready',None)
            if callable(wait):
                if all(value is not original for original,_ in r['barriers']):
                    r['owned'].append(value);r['barriers'].append((value,wait))
            elif type(value)is dict:
                for child in value.values():visit(child)
            elif type(value)in(tuple,list):
                for child in value:visit(child)
        if value is not None:
            visit(value.polynomial);visit(value.moduli_array)
        for ntt in((getattr(value,'ntt_ctx',None)if value is not None else None),attached):
            if ntt is None:continue
            for owner in(ntt,getattr(ntt,'ff_ctx',None)):
                if owner is not None and all(owner is not old for old in r['ntt_owners']):
                    r['ntt_owners'].append(owner);r['owned'].append(owner)
        for owner in tuple(r['ntt_owners']):visit(vars(owner))

'''
text=replace(text,marker,helper+marker)
text=replace(text,"            r['owned'].append(value.polynomial);value.polynomial.block_until_ready();b.secure(ctx,[value])\n", "            self._retain_returned(r)\n            value.polynomial.block_until_ready();b.secure(ctx,[value])\n")
text=replace(text,"        except BaseException as error:r['failure']=protect_body(error);_QUARANTINE.append((self,r,error));raise\n", "        except BaseException as error:\n            r['failure']=protect_body(error)\n            try:self._retain_returned(r)\n            except BaseException as secondary:error.add_note('setup returned/attached array retention: '+type(secondary).__name__)\n            _QUARANTINE.append((self,r,error));raise\n")
text=replace(text,"        owned=tuple(r['owned']);final=Finalizer({},primary)\n        final.attempt('original encryption binding',lambda:self.check(loaded=False))\n        for value in owned:\n            wait=getattr(value,'block_until_ready',None)\n            if callable(wait):final.attempt('original encrypted input barrier',wait)\n", "        final=Finalizer({},primary)\n        final.attempt('original encryption binding',lambda:self.check(loaded=False))\n        final.attempt('retain returned/attached setup arrays',lambda:self._retain_returned(r))\n        owned=tuple(r['owned']);barriers=tuple(r['barriers'])\n        for value,wait in barriers:\n            final.attempt('original encrypted input/NTT array barrier',wait)\n")
ast.parse(text);(NEW/name).write_text(text)
(NEW/'provenance'/'owner_correction_setup.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),text.splitlines(True),fromfile='held382/'+name,tofile='successor/'+name)))

design=(OLD/'SOURCE_DESIGN.md').read_text()
design+='''
## Separate owner-correction successor after held382 review

Held382/mapb787490d remains unchanged. Root review ea93e0dc found that InputOwner's second clean close consulted the removed registry before its weak tombstone, and that CiphertextOwner retained the returned payload but omitted the separately allocated moduli_array. This successor fixes tombstone order, removes one redundant CiphertextOwner tombstone branch, and captures both returned arrays immediately before any validation/security callback so original close barriers also run on failure.

Root's later setup-owner addendum aad79f9d identified the same omission in _EncryptedInput and the additional returned/context-attached NTT/finite-field table lifetime. The setup owner now retains returned payload/moduli arrays and reachable original returned or ctx._polynomial_ntt_ctx NTT/ff objects and table arrays. It captures original array wait callables, including on an encryption exception after context attachment, and attempts all barriers before key-scope exit. It preserves attached-context identity after capture. It does not inject a new setup NTT context, modify encrypt_slots, alter the original codec/primitive calls, or claim ownership of inaccessible constructor locals. Failed/uncertain records remain process-quarantined; clean records and borrowed mirrors still retire.

The unchanged NTTCiphertextContextBase initializer eagerly evaluates BarrettContext(moduli) as parameters.get's default even when finite_field_context is supplied. This extra temporary construction and possible asynchronous work are not eliminated by the explicit client field owner. They remain part of the unresolved resource/native-constructor and failed-process retirement obligation. Reachable returned/attached arrays are explicitly retained; invisible constructor locals are not claimed observed or synchronized.

All250 parent files,34 core files, six primitive definitions, default decoder and original numerical/profile metadata remain exact. No candidate import, collection, test, fixture/body/key/context/backend/device/capacity operation or queue change was performed. RuntimeAdmission and actual entry still deny. The separate issuer plan453599dc is not implemented by this cleanup successor.
'''
(NEW/'SOURCE_DESIGN.md').write_text(design)
resource=json.loads((OLD/'RESOURCE_STATUS.json').read_bytes())
resource['required_independent_review']+=['eager additional BarrettContext allocation in unchanged NTT initializer default expression',
 'reachable returned/context-attached setup NTT and finite-field arrays versus inaccessible constructor locals']
(NEW/'RESOURCE_STATUS.json').write_bytes(raw(resource))
history=dict(schema='qwen-toy5-owner-correction-source-v1',parent_source_count=382,parent_source_manifest_sha256=sha(mb),
 parent_source_archive_sha256='6cf407cc9c5b8b310acc282ab55d8c0a4ac9d59d8d89e845649af7ec74ab89b8',
 root_review_sha256='ea93e0dc06ae9a0465d4c988de93eadb429d6fbe226456401f8b2b2dedf4a490',
 setup_owner_addendum_sha256='aad79f9dde09778b6de52254b76c9670fde57b906f69bfd354405085a450f3f5',
 overlays=['q5_ciphertext_owner.py','q5_actual_process.py','SOURCE_DESIGN.md','RESOURCE_STATUS.json'],
 additions=['owner_correction_history.json','provenance/owner_correction_ciphertext.patch','provenance/owner_correction_setup.patch'],
 runtime_issuer_implemented=False,actual_entry_enabled=False,candidate_imports=False,tests=False,body_reads=False,device_or_capacity_probes=False,
 unchanged_scope=['all250 parent files','core34/default decoder','six primitive definitions','original profile/numerical metadata','original encrypt_slots and NTT constructors'],
 limits=['No actual runtime authority','No test execution','No complete peak/resource bound','No ownership or barrier claim for inaccessible constructor locals'])
(NEW/'owner_correction_history.json').write_bytes(raw(history))
for p in NEW.rglob('*.py'):ast.parse(p.read_bytes(),filename=str(p))
assert all(sha((OLD/n).read_bytes())==pin for n,pin in table.items())
print(json.dumps(dict(root=str(NEW),source_count=sum(p.is_file()for p in NEW.rglob('*')),source_only=True,
 corrected_ciphertext_sha256=sha((NEW/'q5_ciphertext_owner.py').read_bytes()),corrected_setup_sha256=sha((NEW/'q5_actual_process.py').read_bytes())),sort_keys=True))
