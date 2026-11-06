"""Fresh native environment guard probe; never constructs a CKKS context."""
from pathlib import Path
import hashlib,importlib,io,json,os,sys,tarfile,time
ROOT=Path('/tmp/bert-server-compute-stage4-successor-v2-20261103')
OUT=ROOT/'installed-guards-v1'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sources():
    original=json.loads((ROOT/'source-bound-preflight-v2/report.json').read_text())['source_hashes_before']
    expected={n:h for n,h in original.items() if not n.startswith('external/')}
    assert len(expected)==217
    raw={n:(ROOT/n).read_bytes() for n in expected}
    assert {n:sha(v) for n,v in raw.items()}==expected
    raw['external/installed_guard_probe.py']=Path(__file__).read_bytes()
    return raw

def main():
    start=time.monotonic();assert os.environ.get('JAX_PLATFORMS')=='cpu'
    OUT.mkdir(exist_ok=False);raw=sources();before={n:sha(b) for n,b in raw.items()}
    with tarfile.open(OUT/'sources.tar.gz','w:gz') as tar:
        for n,b in raw.items():m=tarfile.TarInfo(n);m.size=len(b);tar.addfile(m,io.BytesIO(b))
    forbidden_reads=[]
    def audit(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0]))
            if path.suffix in ('.npy','.npz','.safetensors') or 'client_private' in path.parts:
                forbidden_reads.append(str(path));raise AssertionError('installed probe forbids payload/private/model asset reads')
    sys.addaudithook(audit)
    sys.path[:0]=[str(ROOT),str(ROOT/'runtime/stage'),str(ROOT/'joined'),str(ROOT/'joined/checkout/cadathlon/common'),str(ROOT/'runtime')]
    import boundaries,file_stages,key_contexts,full_model_engine,client_keys,client_inputs
    full_model_engine._dependencies()  # Classes/functions only; no constructor.
    from jaxite_word import CKKSContext
    modules=[importlib.import_module(n) for n in ('key_gen','jaxite_word.key_gen')]
    entries=[(m,n,m.__name__+'.'+n) for m in modules for n in boundaries.KEY_METHODS]
    entries += [(CKKSContext,n,'CKKSContext.'+n) for n in boundaries.CODECS]
    entries += [(owner,name,owner.__name__+'.'+name) for owner,name in ((client_keys,'generate'),(client_inputs,'ClientEmbeddings'),
        (key_contexts,'load_client'),(key_contexts,'_read_private'),(file_stages,'client_key_generation'),
        (file_stages,'client_encode_encrypt_input'),(file_stages,'client_decrypt_decode'),(file_stages,'server_preprocess_model'),
        (file_stages,'_records'),(file_stages,'_snapshot'))]
    assert len(entries)==25
    originals={(id(owner),name):getattr(owner,name) for owner,name,label in entries}
    allowed=(key_contexts.load_server,file_stages.server_encrypted_compute)
    calls=[];denied=[]
    with boundaries.server_only(calls):
        assert (key_contexts.load_server,file_stages.server_encrypted_compute)==allowed
        for owner,name,label in entries:
            try:getattr(owner,name)(None)
            except RuntimeError as error:
                assert str(error)=='forbidden stage4 server action: '+label
                denied.append(label)
            else:raise AssertionError('installed entrypoint was not denied: '+label)
    assert len(calls)==25 and calls==denied
    assert all(getattr(owner,name)is originals[id(owner),name] for owner,name,label in entries)
    assert (key_contexts.load_server,file_stages.server_encrypted_compute)==allowed
    no_native=not any(n.split('.')[0] in ('torch','transformers') for n in sys.modules);assert no_native
    loaded={}
    for name,module in tuple(sys.modules.items()):
        file=getattr(module,'__file__',None)
        if not file:continue
        path=Path(file).resolve()
        if ROOT in path.parents and path.suffix=='.py':
            rel=path.relative_to(ROOT).as_posix();digest=sha(path.read_bytes())
            assert rel in before and digest==before[rel]
            loaded[name]={'source':rel,'sha256':digest}
    for name in ('file_stages','key_contexts','full_model_engine'):
        assert loaded[name]['source']=='runtime/stage/'+name+'.py'
    after={n:sha(b) for n,b in sources().items()};assert before==after and not forbidden_reads
    with tarfile.open(OUT/'sources.tar.gz') as tar:
        members=tar.getmembers();assert len(members)==len(before)==len({m.name for m in members})
        verified={m.name:sha(tar.extractfile(m).read()) for m in members}==before
    assert verified
    report=dict(passed=True,scope='Installed real-module entrypoint denial/restoration only; no context/key/model construction, codec execution, payload/model/private reads or encrypted handler.',
        source_count=len(before),candidate_source_count=217,source_hashes_before=before,source_hashes_after=after,source_files_unchanged=True,
        source_archive_sha256=sha((OUT/'sources.tar.gz').read_bytes()),source_archive_verified=verified,
        denied_count=len(denied),denied_entrypoints=denied,all25_restored=True,allowed_server_entrypoints_unchanged=True,
        distinct_keygen_aliases=modules[0] is not modules[1],no_torch_or_transformers_imported=no_native,
        loaded_local_modules=loaded,loaded_local_module_count=len(loaded),forbidden_payload_reads=forbidden_reads,
        cpu_affinity=sorted(os.sched_getaffinity(0)),elapsed_s=time.monotonic()-start)
    write(OUT/'report.json',report);print(json.dumps({k:v for k,v in report.items() if not isinstance(v,(dict,list))},indent=2))
if __name__=='__main__':main()
