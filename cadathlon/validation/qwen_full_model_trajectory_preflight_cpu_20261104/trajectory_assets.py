"""Fixed original fixture/public parameter authentication; no model construction."""
from pathlib import Path
import hashlib,io,json,os,stat,sys,tarfile
from validation_cleanup import cleanup_signals
import numpy as np
P=Path(__file__).resolve().parent
REQUIREMENT='964cc8dcd8c3cdd441773422072a710a1c4af5d1375b56eb1cf677ffa6940c6b'
def digest(raw):return hashlib.sha256(raw).hexdigest()
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def requirement():
    raw=(P/'trajectory_requirement.json').read_bytes()
    if digest(raw)!=REQUIREMENT:raise ValueError('fixed trajectory requirement changed')
    return json.loads(raw)
def runtime_gate(req):
    path=P/'runtime_predecessor.json'
    if sha(path)!=req['runtime_predecessor_sha256']:raise ValueError('runtime predecessor changed')
    expected=json.loads(path.read_bytes())['source_hashes']
    if len(expected)!=87 or {n:sha(P/n) for n in expected}!=expected:raise ValueError('exact 87-source runtime changed')
    if sha(P/'inputs/candidate_manifest.json')!=req['candidate_sha256']:raise ValueError('candidate changed')
    return expected

def calibration_gate(req):
    """Completed original416 identity is checked before opening token bodies."""
    c=req['calibration'];root=Path(c['root']);p=root/c['report']
    if sha(p)!=c['report_sha256']:raise ValueError('completed calibration report changed')
    r=json.loads(p.read_bytes())
    if any(r.get(n)is not True for n in ('passed','complete','source_files_unchanged','assets_unchanged','all_workload_entries_accounted')) or r.get('errors'):raise ValueError('completed original calibration prerequisite failed')
    if r['candidate_manifest_sha256']!=req['candidate_sha256']:raise ValueError('calibration candidate changed')
    before=r['source_hashes_before']
    if len(before)!=55 or before!=r['source_hashes_after'] or r['source_archive_sha256']!=c['archive_sha256']:raise ValueError('calibration source linkage failed')
    if before[c['tokens']]!=c['tokens_sha256']:raise ValueError('original token source pin differs')
    archive=root/c['archive']
    if sha(archive)!=c['archive_sha256']:raise ValueError('original calibration archive changed')
    with tarfile.open(archive,'r:gz') as t:
        ms=t.getmembers()
        if len(ms)!=len(before) or len({m.name for m in ms})!=len(ms) or {m.name for m in ms}!=set(before) or any(not m.isfile() or m.size>16<<20 for m in ms) or sum(m.size for m in ms)>128<<20:raise ValueError('calibration archive inventory differs')
        if {m.name:digest(t.extractfile(m).read()) for m in ms}!=before:raise ValueError('calibration archive bytes differ')
    return dict(report_sha256=c['report_sha256'],archive_sha256=c['archive_sha256'],sources=55,passed=True)

def asset_gate(req):
    a=req['assets'];actual={n:sha(Path(a['root'])/n) for n in a['sha256']}
    if actual!=a['sha256']:raise ValueError('pinned checkpoint/config changed')
    return actual

def selected_ids(req):
    c=req['calibration'];p=Path(c['root'])/c['tokens'];raw=p.read_bytes()
    if digest(raw)!=c['tokens_sha256']:raise ValueError('original calibration tokens changed')
    rows=json.loads(raw)
    if type(rows)is not list:raise ValueError('original token inventory must be list')
    row=next((r for r in rows if type(r)is dict and type(r.get('sequence_length'))is int and r['sequence_length']==8),None)
    if row is None:raise ValueError('first original S8 prefix missing')
    ids=row.get('token_ids')
    if type(ids)is not list or len(ids)!=8 or any(type(i)is not int or not 0<=i<151936 for i in ids):raise ValueError('fixed original S8 token schema differs')
    # File order alone selects; no reference/output/domain value is inspected.
    return ((0,),tuple(ids)),dict(token_file_sha256=digest(raw),selected_record_sha256=digest(json.dumps(row,sort_keys=True,separators=(',',':')).encode()),selector='first_file_order_sequence_length8')

def release(callbacks,primary=None):
    errors=[]
    try:
        with cleanup_signals():
            for callback in callbacks:
                try:callback()
                except BaseException as error:errors.append(error)
    except BaseException as error:errors.append(error)
    if not errors:return
    if primary is not None:
        for error in errors:
            if error is not primary:primary.add_note('public mapping release: '+repr(error))
        return
    first=next((error for error in errors if not isinstance(error,Exception)),errors[0])
    for error in errors:
        if error is not first:first.add_note(repr(error))
    raise first

class PublicParameters:
    """Authenticated read-only mappings held through every consumer; payloads local."""
    def __init__(self,req):
        self.req=req;self.maps={};self.handles={};self.records={};self.gamma_values={};self.closed=False
        self.qkv=json.loads((P/'qkv-export-v1/parameter_manifest.json').read_bytes())['matrices']
        self.decoder=json.loads((P/'inputs/decoder_parameters.json').read_bytes())['matrices']
        self.gamma_rows=json.loads((P/'inputs/gamma_preparation_report.json').read_bytes())['arrays']
    def matrix(self,name):
        if self.closed:raise ValueError('parameter owner closed')
        rows=self.qkv if name in self.qkv else self.decoder;row=rows[name]
        directory=Path(self.req['public_parameters']['qkv_directory' if name in self.qkv else 'decoder_directory'])
        filename=row['file']
        if type(filename)is not str or Path(filename).name!=filename:raise ValueError('flat public matrix filename required')
        p=directory/filename;key=str(p)
        if key not in self.maps:
            fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK);stream=None;pending_map=None
            try:
                st=os.fstat(fd)
                if not stat.S_ISREG(st.st_mode) or st.st_size>2<<30:raise ValueError('bounded regular public NPY required')
                stream=os.fdopen(fd,'rb',closefd=False)
                if hashlib.file_digest(stream,'sha256').hexdigest()!=row['file_sha256']:raise ValueError('public NPY digest differs')
                stream.seek(0);version=np.lib.format.read_magic(stream)
                if version==(1,0):shape,order,dtype=np.lib.format.read_array_header_1_0(stream,max_header_size=16384)
                elif version==(2,0):shape,order,dtype=np.lib.format.read_array_header_2_0(stream,max_header_size=16384)
                else:raise ValueError('supported bounded NPY header required')
                offset=stream.tell()
                if shape!=tuple(row['shape']) or dtype!=np.dtype('<f8') or order or st.st_size!=offset+int(np.prod(shape))*8:raise ValueError('public NPY shape/type/extent differs')
                value=np.memmap(stream,dtype=dtype,mode='r',offset=offset,shape=shape,order='C');pending_map=value
                if digest(memoryview(value).cast('B'))!=row['sha256'] or not np.isfinite(value).all():
                    raise ValueError('public matrix payload differs')
                self.maps[key]=value;self.handles[key]=(stream,fd,p,(st.st_dev,st.st_ino));stream=None;fd=None;pending_map=None
            finally:
                callbacks=[]
                if pending_map is not None:callbacks.append(pending_map._mmap.close)
                if stream is not None:callbacks.append(stream.close)
                if fd is not None:callbacks.append(lambda fd=fd:os.close(fd))
                release(callbacks,sys.exception())
        value=np.asarray(self.maps[key]);self.records[name]=dict(file_sha256=row['file_sha256'],sha256=row['sha256'],shape=row['shape'],dtype='<f8')
        return value
    def gamma(self,site):
        if self.closed:raise ValueError('parameter owner closed')
        if not self.gamma_values:
            p=Path(self.req['public_parameters']['gamma_path']);raw=p.read_bytes()
            if digest(raw)!=self.req['public_parameters']['gamma_sha256']:raise ValueError('gamma archive differs')
            with np.load(io.BytesIO(raw),allow_pickle=False) as z:
                if set(z.files)!=set(self.gamma_rows):raise ValueError('gamma inventory differs')
                for name in z.files:
                    value=z[name];d=self.gamma_rows[name]
                    if value.dtype!=np.dtype('float64') or list(value.shape)!=d['shape'] or digest(value.tobytes())!=d['sha256'] or not np.isfinite(value).all():raise ValueError('gamma payload differs')
                    self.gamma_values[name]=np.frombuffer(value.tobytes(),dtype=np.float64)
        return self.gamma_values[site]
    def readback(self):
        for key,(_,fd,p,identity) in self.handles.items():
            st=p.lstat()
            if (st.st_dev,st.st_ino)!=identity:raise ValueError('public file path identity changed')
            os.lseek(fd,0,os.SEEK_SET)
            digester=hashlib.sha256()
            while chunk:=os.read(fd,1<<20):digester.update(chunk)
            h=digester.hexdigest()
            rows=[r for n,r in {**self.qkv,**self.decoder}.items() if r['file']==p.name]
            if not rows or any(r['file_sha256']!=h for r in rows):raise ValueError('mapped public file changed')
        if self.gamma_values and sha(self.req['public_parameters']['gamma_path'])!=self.req['public_parameters']['gamma_sha256']:raise ValueError('gamma changed')
    def close(self):
        errors=[]
        for value in self.maps.values():
            try:value._mmap.close()
            except BaseException as e:errors.append(e)
        for stream,fd,_,_ in self.handles.values():
            for callback in (stream.close,lambda fd=fd:os.close(fd)):
                try:callback()
                except BaseException as e:errors.append(e)
        self.closed=True;self.maps.clear();self.handles.clear();self.gamma_values.clear()
        if errors:
            first=next((e for e in errors if not isinstance(e,Exception)),errors[0])
            for e in errors:
                if e is not first:first.add_note(repr(e))
            raise first

class FrozenWeights:
    def __init__(self,parameters):self.p=parameters
    def __getitem__(self,name):
        if name=='model.norm.weight':return self.p.gamma('final.inverse_sqrt')
        import re
        match=re.fullmatch(r'model.layers.(\d+).(input_layernorm|post_attention_layernorm|self_attn.[qk]_norm).weight',name)
        if match:
            layer,kind=match.groups();kind={'input_layernorm':'input','post_attention_layernorm':'post','self_attn.q_norm':'q','self_attn.k_norm':'k'}[kind]
            return self.p.gamma(layer+'.'+kind+'.inverse_sqrt')
        return self.p.matrix(name)
