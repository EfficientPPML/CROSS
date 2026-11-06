"""Client setup/reference and separate authenticated public toy-weight loading.

The server calls only load_public_weights. It never constructs the fixed clear
graph, embeddings or checkpoint oracle. Public weight bodies stay in the owned
execution workspace and are excluded from the evidence archive.
"""
from pathlib import Path
import hashlib,io,json,os,stat
import numpy as np
from component_contract import ROOT,numerical_plan,canonical

def sha(raw):return hashlib.sha256(raw).hexdigest()

def description(array):
    if not isinstance(array,np.ndarray)or array.dtype!=np.dtype('float64')or not np.isfinite(array).all():raise ValueError('finite float64 public array required')
    return dict(shape=list(array.shape),dtype='float64',sha256=sha(array.tobytes(order='C')))

def pinned_fixture():
    raw=(ROOT/'fixture_manifest.json').read_bytes()
    if sha(raw)!=FIXTURE_MANIFEST_SHA:raise ValueError('exact fixed fixture descriptors required')
    return json.loads(raw)

def prepare_client_reference():
    # NumPy-only literal graph and separately implemented dense polynomial
    #oracle. These are client operations after source authentication.
    from public_plan.public_graph import run
    from public_plan.dense_reference import references as dense_references
    result=run();expected,refreshes=dense_references(result['case']['config'],result['weights'],result['embeddings'],result['mask'],result['approximations']);phase=0
    references=[]
    for row,value in zip(result['trace'],result['values']):
        key=(row['kind'],str(row['label']))
        if row['kind']=='phase'and row['label'].startswith('bounded homomorphic refresh'):
            ref=refreshes[phase//2];phase+=1
        else:ref=expected[key]
        # Phase duplicate labels carry the same boundary objects. The frozen
        #public plan already independently compared those; pin their bytes.
        if sha(value.tobytes())!=row['expected_array_sha256']:raise ValueError('literal reference bytes drift')
        if np.max(np.abs(value-ref))>1e-10:raise ValueError('independent dense polynomial reference differs')
        references.append(np.asarray(ref,dtype=np.float64))
    fixture=pinned_fixture()
    if {k:description(v)for k,v in sorted(result['weights'].items())}!=fixture['weights']:raise ValueError('fixed seeded public weights differ')
    if description(result['values'][0])!=fixture['input']:raise ValueError('fixed original embeddings differ')
    return result,references

def export_public_weights(directory,weights):
    from stage.client_keys import _OwnedDirectory,_write,_CloseState,_stream
    fixture=pinned_fixture();state=_CloseState();owner=_OwnedDirectory(directory)
    try:
        if {k:description(v)for k,v in sorted(weights.items())}!=fixture['weights']:raise ValueError('fixed public tensor inventory differs')
        records={}
        for index,(name,value)in enumerate(sorted(weights.items())):
            file=f'weight_{index:03d}.npy';fd=os.open(file,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600,dir_fd=owner.fd)
            try:
                with _stream(fd,'wb',state)as stream:np.save(stream,value,allow_pickle=False);stream.flush();os.fsync(fd)
            finally:state.close(fd)
            records[name]=dict(file=file,**fixture['weights'][name])
        manifest=dict(schema='toy4-public-weights-v1',fixture_manifest_sha256=FIXTURE_MANIFEST_SHA,weights=records)
        _write(owner,'manifest.json',manifest,_close_state=state);owner.check();return manifest
    finally:owner.close(_close_state=state)

def load_public_weights(directory):
    from stage.client_keys import _read,_CloseState
    directory=Path(directory);state=_CloseState();fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        original=os.fstat(fd);fixture=pinned_fixture();raw=_read(fd,'manifest.json',65536,_close_state=state);manifest=json.loads(raw)
        records={name:dict(file=f'weight_{i:03d}.npy',**desc)for i,(name,desc)in enumerate(sorted(fixture['weights'].items()))}
        expected=dict(schema='toy4-public-weights-v1',fixture_manifest_sha256=FIXTURE_MANIFEST_SHA,weights=records)
        if raw!=canonical(expected)or set(os.listdir(fd))!={'manifest.json',*(row['file']for row in records.values())}:raise ValueError('public tensor manifest/inventory differs')
        arrays={}
        for name,row in records.items():
            raw=_read(fd,row['file'],int(np.prod(row['shape']))*8+4096,_close_state=state)
            stream=io.BytesIO(raw);version=np.lib.format.read_magic(stream)
            if version!=(1,0):raise ValueError('fixed public NPY version required')
            shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream,max_header_size=512)
            if list(shape)!=row['shape']or fortran or dtype!=np.dtype('float64'):raise ValueError('public tensor header differs')
            body=raw[stream.tell():]
            if len(body)!=int(np.prod(shape))*8 or sha(body)!=row['sha256']:raise ValueError('public tensor bytes differ')
            # Bytes-backed views cannot be made writable; no public array
            #mutation can alter downstream arithmetic after authentication.
            arrays[name]=np.frombuffer(body,dtype=np.float64).reshape(shape)
        now=directory.lstat();held=os.fstat(fd)
        if not stat.S_ISDIR(now.st_mode)or(original.st_dev,original.st_ino)!=(now.st_dev,now.st_ino)or(held.st_dev,held.st_ino)!=(original.st_dev,original.st_ino):raise ValueError('public tensor directory replaced')
        return arrays
    finally:state.close(fd)

FIXTURE_MANIFEST_SHA='bf1be3b402101ce8b1a07864dcd25f39b8661d0621f66d64b52996af045b1041'
