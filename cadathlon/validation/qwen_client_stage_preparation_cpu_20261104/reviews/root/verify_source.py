"""Authenticate the held client source addition; executes no selected modules."""
from pathlib import Path
import ast,datetime,hashlib,json
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_bytes())
source=Path('/tmp/qwen-client-stage-glue-20261104')
manifest=Path('/tmp/qwen-client-stage-glue-review-sources-v1-20261104.json')
parent_manifest=Path('/tmp/qwen-stage-integration-v2-sources-20261104.json')
assert sha(manifest)=='c2dfd572ffd3a8a9790825a5efb7e8ce737fbf0229db91fe079b2df227c5153d'
assert sha(parent_manifest)=='225cc51ab57696ad69b6f973f2402d8e2851db27b26fc35e66b5c610032e9f5e'
expected=load(manifest);parent=load(parent_manifest)
assert len(expected)==370 and len(parent)==366
assert set(expected)-set(parent)=={'client_stage.py','CLIENT_STAGE_DESIGN.md','client_stage_origin.json','client_parent366_manifest.json'}
assert all(expected[n]==digest for n,digest in parent.items())
assert {p.relative_to(source).as_posix():sha(p)for p in source.rglob('*')if p.is_file()}==expected
assert (source/'client_parent366_manifest.json').read_bytes()==parent_manifest.read_bytes()
assert expected['client_stage.py']=='e822bb2d77c8310875458c4a7821e9e9d48df977508757dbd6736de7909945dd'
origin=load(source/'client_stage_origin.json')
assert origin['overlays']=={} and origin['parent_all_exact']is True and origin['actual_execution_authorized']is False
assert origin['parent_manifest_sha256']==sha(parent_manifest)
assert origin['parent_review_sha256']=='2d451b65f7750ac710d13c9b4e1612b27bb64ca488aca2f3bf9968f9494afc5d'
assert origin['selected_default_decoder_sha256']==expected['runtime/jaxite_word/ckks_ctx.py']=='bc978cb7412b55416f7ed634a689ae616a263b0f47fd9eef996ba244f2505db6'
assert origin['shared_cleanup_origin_sha256']==expected['cleanup_origin.py']=='4543b26bf1168e29bc21375c2f80a9aa99648205a9fd9aea6d5a8059c6d18175'
assert origin['shared_finalizer_sha256']==expected['qwen_finalizer.py']=='46e6566a275a84b4ae4596e066c592a4aa3a60e41b3afb4ec30bb96b72551240'
tree=ast.parse((source/'client_stage.py').read_bytes())
functions={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
for name in ('create_client_input','create_client_output'):
    body=functions[name].body
    assert len(body)==1 and isinstance(body[0],ast.Raise) and isinstance(body[0].exc,ast.Call) and body[0].exc.func.id=='PermissionError'
codec_calls=[n for n in ast.walk(functions['consume_output_for_recording'])if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='codec']
assert len(codec_calls)==1
kwargs={k.arg:k.value for k in codec_calls[0].keywords}
assert ast.literal_eval(kwargs['slots_to_decode'])==65536 and ast.literal_eval(kwargs['validate_approximation'])is True
peer=Path('/tmp/qwen-client-stage-security-review-20261104/review.json')
assert sha(peer)=='f6ec0b6224a1d0bd2252b765f369398e5999335605c98505a66a8298e44f664e'
out=Path(__file__).with_name('source-review.json');assert not out.exists()
result=dict(schema='qwen-client-stage-root-source-review-v1',passed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_root=str(source),source_count=370,parent_sources_unchanged=366,source_manifest_sha256=sha(manifest),runtime_sha256=expected['client_stage.py'],security_review_sha256=sha(peer),script_sha256=sha(Path(__file__)),observations=[
    'Full client FilesAdmission and independently expected identities precede supplied body/key/output callbacks; actual factories raise immediately.',
    'Original embedding placement supplies immutable one/two full-slot vectors; input uses captured encrypt_slots at fresh request scale and captured publisher methods.',
    'The exact ResponseFiles constructor and complete readback precede private_loader; the returned owner must match public/request/digest and all38 records.',
    'Composite consumer scope owns original context/manifest/codec/partial arrays inside the key scope; final barriers and binding checks run before nested loader disposal.',
    'Defaultbc978 codec remains selected, actual scale/full65536/approximationTrue arguments are explicit, and unchanged ResponseFiles copies complete finite float64 scratch output before the next call.',
    'Late callback, handle, readiness and file cleanup failures preserve original ownership and shared error selection; tests are still required for this integration.'
],scope='Held source readiness only. No tests, selected imports, real codec/security/backend/context, model/key/token/CT-body access, actual-role grant or numerical acceptance. Focused new recording tests authorized in a separate successor; all370 held bytes immutable.')
out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(passed=True,path=str(out),sha256=sha(out),sources=370,parent_exact=366)))
