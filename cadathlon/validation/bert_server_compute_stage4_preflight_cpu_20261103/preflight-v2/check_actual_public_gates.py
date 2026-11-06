from pathlib import Path
import hashlib,json,os,sys,time
ROOT=Path('/tmp/bert-server-compute-stage4-successor-v2-20261103')
sys.path[:0]=[str(ROOT),str(ROOT/'runtime/stage'),str(ROOT/'joined'),str(ROOT/'joined/checkout/cadathlon/common'),str(ROOT/'runtime')]
opened=[]
def audit(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        name=os.fsdecode(args[0]);opened.append(name)
        if ('client_private' in Path(name).parts or Path(name).suffix in ('.npy','.npz','.safetensors') or Path(name).name=='preprocessed_input'):
            raise AssertionError('public-only gate must not read payload/private path')
sys.addaudithook(audit)
import key_gate,model_gate,stage3_gate
start=time.monotonic();profile=(ROOT/'runtime/stage/server_inputs/p60_profile_and_refresh.json').read_bytes();plan=(ROOT/'runtime/stage/server_inputs/twelve_layer_plan.json').read_bytes()
keys=key_gate.authenticate(Path('/tmp/bert-full-key-generation-component-20261103/actual-v1'),'022cbdf687273e765668184f32cb7cd501d371c4220916e1c8f02574cf3a7f08',Path('/dev/shm/cross-full-bert-keys-143b3f5c1573427d841760a3c9d5d33d'),profile_bytes=profile,plan_bytes=plan)
model=model_gate.authenticate(Path('/tmp/bert-public-model-stage2-v2-component-20261103/actual-v2'),'5b9589b51828d066706c4c46a6404d8c52fcd67a2c9aa9f3f57c73df8caeefe0',Path('/tmp/cross-bert-public-model-b51668856dab42a3846ba1b7c7a71d53'),keys=keys)
result=stage3_gate.authenticate(Path('/tmp/bert-input-encryption-stage3-component-v3-20261103/actual-v3'),'add5e6a3143be3931ad29c018cdee40bdf059497b8bfa987114f39baeb88631e',Path('/tmp/cross-bert-input-encryption-7b255c2357c94060b73e474a1f8ef5fa'),keys=keys,model=model)
report=dict(passed=True,scope='Public completed stage1/2/3 gate authentication only; payload/private opens forbidden by independent audit hook; no handler/context/model/HE execution.',
    context_id=result['context_id'],model_id=result['model_id'],public_batch_sha256=result['public_batch_sha256'],input_digest=result['input_digest'],stage3_report_sha256=result['stage3_report_sha256'],
    tile_count=len(result['tiles']),payload_bytes_attested=sum(x['payload_bytes'] for x in result['tiles']),no_jax_or_torch_imported=not any(n.split('.')[0] in ('jax','jaxlib','torch','transformers','jaxite_word') for n in sys.modules),
    elapsed_s=time.monotonic()-start,gate_source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('key_gate.py','model_gate.py','stage3_gate.py')},script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
Path('/tmp/stage4_actual_gates_public_20261103.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
