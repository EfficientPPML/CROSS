from pathlib import Path
import importlib.util,hashlib,io,json,tempfile
import numpy as np
SRC=Path('/tmp/bert-full-model-file-stages-20261103/server_public_model.py');HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('public_model_header_review',SRC);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
source=SRC.read_bytes();(HERE/'header_reviewed_source.py').write_bytes(source)
called=[];original_empty=np.empty
with tempfile.TemporaryDirectory(prefix='public-model-header-review-') as name:
    root=Path(name);records={};shapes=m.tensor_shapes()
    for i,(tensor,shape) in enumerate(shapes.items()):
        filename=f'tensor_{i:03d}.npy';raw=b''
        if i==0:
            buf=io.BytesIO();np.lib.format.write_array_header_1_0(buf,dict(descr='<f4',fortran_order=False,shape=(2**60,)));raw=buf.getvalue()
        (root/filename).write_bytes(raw)
        records[tensor]=dict(file=filename,shape=list(shape),dtype='<f4',tensor_sha256=m.TENSOR_SHA256[tensor],file_sha256=hashlib.sha256(raw).hexdigest())
    body=m._base('a'*64)|dict(tensors=records);manifest=body|dict(model_id=m.digest(body))
    (root/'model_manifest.json').write_bytes(m.canonical(manifest))
    def blocked(shape,*args,**kwargs):
        called.append(str(shape));raise AssertionError('blocked allocation requested by unvalidated NPY header')
    np.empty=blocked
    try:m.load(root,context_id='a'*64)
    except BaseException as exc:result=dict(exception=type(exc).__name__,message=str(exc),allocation_requests=called)
    finally:np.empty=original_empty
result.update(scope='No allocation performed; np.empty interception only.',source_sha256=hashlib.sha256(source).hexdigest(),source_unchanged=SRC.read_bytes()==source)
(HERE/'header_report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
