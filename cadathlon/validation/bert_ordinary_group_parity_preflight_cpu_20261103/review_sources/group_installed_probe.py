"""Installed boundary/import probe only; no context or payload construction."""
from pathlib import Path
import argparse,hashlib,importlib,io,json,os,sys,tarfile,time
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'runtime/stage'),str(HERE/'joined'),str(HERE/'joined/checkout/cadathlon/common'),str(HERE/'runtime')]
import group_protocol as protocol

def main(output):
    start=time.monotonic();protocol.environment();output.mkdir(exist_ok=False)
    raw=protocol.sources();before=protocol.hashes(raw)
    with tarfile.open(output/'sources.tar.gz','w:gz') as tar:
        for n,b in raw.items():m=tarfile.TarInfo(n);m.size=len(b);tar.addfile(m,io.BytesIO(b))
    forbidden=[]
    def audit(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0]))
            if path.suffix in ('.npy','.npz','.safetensors') or 'client_private' in path.parts:
                forbidden.append(str(path));raise AssertionError('probe forbids private/model/ciphertext/key payloads')
    sys.addaudithook(audit)
    import boundaries,parity_boundaries,file_stages,key_contexts,full_model_engine,client_keys,client_inputs,server_public_model
    import encrypted_packed_qwen,prepared_rotation,loader_prepared_rotation,grouped_rotation,loader_grouped_rotation
    full_model_engine._dependencies() # Imports definitions only; inert engine remains forbidden.
    from jaxite_word import CKKSContext
    modules=[importlib.import_module(n) for n in ('key_gen','jaxite_word.key_gen')]
    entries=[(m,n,m.__name__+'.'+n) for m in modules for n in boundaries.KEY_METHODS]
    entries += [(CKKSContext,n,'CKKSContext.'+n) for n in boundaries.CODECS]
    entries += [(o,n,o.__name__+'.'+n) for o,n in ((client_keys,'generate'),(client_inputs,'ClientEmbeddings'),
        (key_contexts,'load_client'),(key_contexts,'_read_private'),(file_stages,'client_key_generation'),
        (file_stages,'client_encode_encrypt_input'),(file_stages,'client_decrypt_decode'),(file_stages,'server_preprocess_model'),
        (file_stages,'_records'),(file_stages,'_snapshot'),(file_stages,'server_encrypted_compute'),
        (full_model_engine,'server_engine'),(server_public_model,'load'))]
    assert len(entries)==28
    original={(id(o),n):getattr(o,n) for o,n,label in entries}
    allowed=(key_contexts.load_server,key_contexts.load_server_prepared,key_contexts.load_server_grouped)
    calls=[];denied=[]
    with parity_boundaries.rotation_only(calls):
        assert (key_contexts.load_server,key_contexts.load_server_prepared,key_contexts.load_server_grouped)==allowed
        for o,n,label in entries:
            try:getattr(o,n)(None)
            except RuntimeError as error:
                assert str(error) in ('forbidden stage4 server action: '+label,'forbidden rotation parity action: '+label)
                denied.append(label)
            else:raise AssertionError('installed action not denied: '+label)
    assert calls==denied and len(calls)==28
    assert all(getattr(o,n)is original[id(o),n] for o,n,label in entries)
    assert (key_contexts.load_server,key_contexts.load_server_prepared,key_contexts.load_server_grouped)==allowed
    assert not any(n.split('.')[0] in ('torch','transformers') for n in sys.modules)
    loaded=protocol.imported_sources(before)
    after=protocol.hashes(protocol.sources());assert before==after and not forbidden
    with tarfile.open(output/'sources.tar.gz') as tar:
        members=tar.getmembers();assert len(members)==len(before)==len({m.name for m in members})
        assert all(m.isfile() for m in members)
        assert {m.name:hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in members}==before
    result=dict(passed=True,source_hashes_before=before,source_hashes_after=after,source_count=len(before),
        source_files_unchanged=True,source_archive_verified=True,source_archive_sha256=hashlib.sha256((output/'sources.tar.gz').read_bytes()).hexdigest(),
        denied_count=28,all28_restored=True,denied_entrypoints=denied,allowed_loaders_unchanged=True,
        loaded_local_modules=loaded,distinct_keygen_aliases=modules[0]is not modules[1],forbidden_payload_reads=forbidden,
        elapsed_s=time.monotonic()-start,cpu_affinity=sorted(os.sched_getaffinity(0)),
        scope='Real installed imports/entrypoint guards only; no context, keys, ciphertexts, model or codec execution.')
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if not isinstance(v,(dict,list))}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(**vars(p.parse_args()))
