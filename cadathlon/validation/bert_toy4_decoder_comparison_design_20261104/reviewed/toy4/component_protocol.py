"""Bounded public source/role receipts for isolated toy4 diagnostics."""
from pathlib import Path
import hashlib,io,json,os,stat,sys,tarfile
from component_contract import ROOT,canonical,numerical_plan,validate_contract,digest
from component_transport import _CloseState,_stream
from cleanup_origin import mark,select,protect_body

STAGES=('setup','server','validate')
class Finalizer:
    """Attempt every owned cleanup/publication; preserve real body primary."""
    def __init__(self,primary):self.primary=protect_body(primary);self.errors=[]
    def attempt(self,label,action):
        try:return action()
        except BaseException as error:self.errors.append((label,mark(error)));return None
    def raise_failure(self):
        chosen=select(self.primary,self.errors)
        if chosen is not None:raise chosen

def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(path,maximum=16<<20):
    state=_CloseState();fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode)or info.st_size>maximum:raise ValueError('bounded regular public file required')
        chunks=[];remaining=maximum+1
        while remaining:
            part=os.read(fd,min(remaining,1<<20))
            if not part:break
            chunks.append(part);remaining-=len(part)
        raw=b''.join(chunks);after=os.fstat(fd)
        if len(raw)!=info.st_size or any(getattr(info,k)!=getattr(after,k)for k in('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')):raise ValueError('public file changed or exceeded bound')
        return raw
    finally:state.close(fd)
def parse(raw):
    def pairs(items):
        out={}
        for key,value in items:
            if key in out:raise ValueError('duplicate metadata key')
            out[key]=value
        return out
    def invalid(x):raise ValueError('nonfinite JSON')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)
def write(path,value):
    path=Path(path);raw=canonical(value);state=_CloseState();fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        before=os.fstat(fd);temp=path.name+'.tmp';item=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=fd)
        try:
            with _stream(item,'wb',state)as stream:stream.write(raw);stream.flush();os.fsync(item)
        finally:state.close(item)
        now=path.parent.lstat()
        if not stat.S_ISDIR(now.st_mode)or(before.st_dev,before.st_ino)!=(now.st_dev,now.st_ino):raise ValueError('publication parent replaced')
        os.replace(temp,path.name,src_dir_fd=fd,dst_dir_fd=fd);os.fsync(fd)
    finally:state.close(fd)
def identity(path):
    info=Path(path).lstat()
    if not stat.S_ISDIR(info.st_mode)or stat.S_IMODE(info.st_mode)!=0o700:raise ValueError('owned private-mode directory required')
    return[info.st_dev,info.st_ino]
def check_identity(path,wanted):
    if type(wanted)is not list or len(wanted)!=2 or any(type(v)is not int for v in wanted)or identity(path)!=wanted:raise ValueError('owned directory replaced')
def relative(name):
    if type(name)is not str or not name or Path(name).is_absolute()or'..'in Path(name).parts or Path(name).as_posix()!=name:raise ValueError('canonical relative source name required')
    return name
def sources():
    inventory=parse(read(ROOT/'source_inventory.json',1<<20))
    if type(inventory)is not list or len(inventory)!=len(set(inventory)):raise ValueError('exact source inventory required')
    return{relative(n):read(ROOT/n,8<<20)for n in inventory}
def hashes(raw):return{k:sha(v)for k,v in raw.items()}
def check_sources(expected):
    if hashes(sources())!=expected:raise ValueError('reviewed execution sources changed')
    pins=parse(read(ROOT/'public_plan/predecessor.json'))['source_hashes']
    if len(pins)!=199 or {n:sha(read(ROOT/'runtime'/relative(n),8<<20))for n in pins}!=pins:raise ValueError('original199 sources changed')
    numerical_plan()
def verify_archive(raw,expected):
    if len(raw)>16<<20:raise ValueError('source archive exceeds bound')
    found={};total=0
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz')as tar:
        for member in tar:
            name=relative(member.name);total+=member.size
            if not member.isfile()or name in found or member.size>8<<20 or total>16<<20 or name not in expected:raise ValueError('source archive inventory/size differs')
            body=tar.extractfile(member).read(8<<20)
            if len(body)!=member.size:raise ValueError('truncated source member')
            found[name]=sha(body)
    if found!=expected:raise ValueError('source archive bytes differ')
def imported_sources(expected):
    found={}
    for name,module in tuple(sys.modules.items()):
        filename=getattr(module,'__file__',None)
        if not filename:continue
        path=Path(filename).resolve()
        if ROOT in path.parents and path.suffix=='.py':
            relative_name=path.relative_to(ROOT).as_posix();pin=sha(read(path,8<<20))
            if expected.get(relative_name)!=pin:raise ValueError('import outside fixed source closure: '+name)
            found[name]=dict(path=relative_name,sha256=pin)
    return found
def require_ready():
    value=parse(read(ROOT/'execution_authorization.json'))
    if type(value)is not dict or set(value)!={'ready'}or value['ready']is not True:raise ValueError('actual diagnostic execution remains unauthorized')
def terminal_gate(path,expected_sha,*,stage,expected_sources,workspace,workspace_identity):
    raw=read(path)
    if sha(raw)!=expected_sha:raise ValueError('external completed parent receipt digest differs')
    receipt=parse(raw)
    if receipt.get('stage')!=stage or stage not in STAGES or any(receipt.get(k)is not True for k in('passed','complete','source_verified','worker_verified','cleanup_completed','terminal_scope_completed','termination_scope_completed')):raise ValueError('successful complete parent role receipt required')
    child=receipt.get('child',{})
    if any(child.get(k)is not True for k in('direct_child_reaped','group_empty_verified','cleanup_completed'))or type(child.get('direct_returncode'))is not int or child['direct_returncode']!=0:raise ValueError('direct wait and owned group absence required')
    if receipt.get('workspace')!=str(workspace)or receipt.get('workspace_identity')!=workspace_identity:raise ValueError('role workspace binding differs')
    check_identity(workspace,workspace_identity)
    if receipt.get('source_hashes')!=expected_sources:raise ValueError('parent source closure differs')
    worker_path=Path(path).parent/'worker/report.json';worker_raw=read(worker_path)
    if sha(worker_raw)!=receipt.get('worker_report_sha256'):raise ValueError('worker receipt changed')
    worker=parse(worker_raw)
    if worker.get('stage')!=stage or any(worker.get(k)is not True for k in('passed','complete','source_verified','cleanup_completed','terminal_scope_completed','termination_scope_completed')):raise ValueError('complete successful worker required')
    if worker.get('source_hashes_start')!=expected_sources or worker.get('source_hashes_end')!=expected_sources:raise ValueError('worker source binding differs')
    fresh=worker.get('fresh_source_guard',{})
    if fresh.get('passed')is not True or fresh.get('fresh')is not True or type(fresh.get('source_count'))is not int or fresh['source_count']!=len(expected_sources)or fresh.get('manifest_sha256')!=digest(expected_sources):raise ValueError('fresh authenticated worker source admission required')
    archive=read(Path(path).parent/'worker/sources.tar.gz')
    if sha(archive)!=worker.get('source_archive_sha256'):raise ValueError('worker source archive changed')
    verify_archive(archive,expected_sources)
    return worker
