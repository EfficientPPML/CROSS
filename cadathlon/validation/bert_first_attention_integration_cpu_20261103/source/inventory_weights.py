"""Read four pinned public weight arrays; no model evaluation or backend."""
from pathlib import Path
from types import SimpleNamespace
from numbers import Integral,Real
from typing import Optional,Tuple
import ast,hashlib,json,math,struct,sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'inputs/common'))
from encrypted_chained_streaming_linear import linear_rotation_plan

def main():
    snapshot=Path(sys.argv[1])
    manifest=json.loads((HERE/'inputs/verified_model_manifest.json').read_text())
    path=snapshot/'model.safetensors'
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    assert snapshot.name==manifest['revision']
    assert digest==manifest['files']['model.safetensors']['sha256']
    # Execute only the unchanged pure public factor-selection function, keeping
    # JAX/finite-field imports out of this public matrix-inventory experiment.
    tree=ast.parse((HERE/'inputs/bsgs.py').read_text())
    fn=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='compute_bsgs_params')
    namespace=dict(Integral=Integral,Real=Real,Optional=Optional,Tuple=Tuple,math=math)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'inputs/bsgs.py','exec'),namespace)
    names=[f'bert.encoder.layer.0.attention.self.{kind}.weight' for kind in ('query','key','value')]
    names+=['bert.encoder.layer.0.attention.output.dense.weight']
    result={}
    with path.open('rb') as stream:
        length=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(length))
        for name in names:
            entry=header[name];assert entry['dtype']=='F32' and entry['shape']==[768,768]
            first,last=entry['data_offsets'];stream.seek(8+length+first);raw=stream.read(last-first)
            weight=np.frombuffer(raw,dtype='<f4').reshape(768,768)
            assert np.isfinite(weight).all()
            active=tuple(sorted({d%65536 for d in range(-767,768) if np.any(np.diag(weight,k=d))}))
            n1,n2=namespace['compute_bsgs_params'](65536,len(active),2.)
            schedule=linear_rotation_plan(SimpleNamespace(active=active,n1=n1,dimension=65536),strategy='chained')
            result[name]=dict(shape=[768,768],tensor_sha256=hashlib.sha256(raw).hexdigest(),
                active_diagonals=len(active),active_indices=list(active),n1=n1,n2=n2,
                giant_groups=len(schedule.giant_indices),rotation_calls=schedule.nonzero_rotation_calls,
                rotation_keys=list(schedule.required_keys))
    report=dict(passed=True,scope=__doc__,checkpoint_sha256=digest,revision=manifest['revision'],tensors=result,
        no_backend=not any(x.split('.')[0] in ('jax','jaxlib','torch','transformers') for x in sys.modules))
    assert report['no_backend']
    (HERE/'matrix_inventory.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({name:{k:v for k,v in item.items() if k not in ('active_indices',)} for name,item in result.items()},indent=2))

if __name__=='__main__':main()
