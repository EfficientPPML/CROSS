"""Authenticate all nine public prerequisites before reading private fixtures."""
from pathlib import Path
from dataclasses import dataclass
from contextlib import contextmanager
import hashlib,io,json,os,stat,tarfile
import key_gate,input_gate
HERE=Path(__file__).resolve().parent
REPORT_SHA='ad1458f5183c6ccf2631c2a8771d0da92196f5f9c690016e2a0b12fa7b88f4d7'
SCHEDULE=tuple((size,count,seed) for size,count in ((1,100),(2,200),(3,408)) for seed in (42,43,44))
NAMES={1:'small',2:'medium',3:'large'}

def expected_files(size):
    name=NAMES[size]
    return {f'datasets/{name}/dataset_labels.txt',f'datasets/{name}/dataset_samples.jsonl',
        f'datasets/{name}/intermediate/test_input.txt',f'datasets/{name}/intermediate/test_labels.txt',
        f'io/{name}/intermediate/client_preprocessed_input','private_selection.json'}

def public(evidence,expected_sha256):
    evidence=Path(evidence);raw=key_gate.read(evidence/'report.json',4<<20)
    if key_gate.digest(raw)!=key_gate.sha_field(expected_sha256):raise ValueError('external ranked fixture report hash differs')
    r=key_gate.parse(raw)
    flags=('passed','private_workspace_retained','sources_unchanged','assets_unchanged','upstream_originals_unchanged',
           'source_archive_verified','cleanup_completed','terminal_scope_completed','model_construction_and_forward_guards')
    if (type(r)is not dict or r.get('schema')!='bert-ranked-tokenizer-fixtures-v1' or any(r.get(n)is not True for n in flags)
        or r.get('error') or r.get('cleanup_errors') or r.get('model_forward')is not False or r.get('encrypted_execution')is not False):
        raise ValueError('completed tokenizer-only ranked fixture prerequisite required')
    if type(r.get('ranked_cases_expected'))is not int or r['ranked_cases_expected']!=9 or type(r.get('selected_records_expected'))is not int or r['selected_records_expected']!=2124:
        raise ValueError('exact nine-case accounting required')
    pins=key_gate.parse((HERE/'producer_source_manifest.json').read_bytes())
    before=r.get('source_hashes_before')
    if before!=pins or before!=r.get('source_hashes_after') or len(before)!=27:raise ValueError('exact unchanged producer sources required')
    archive=key_gate.read(evidence/'sources.tar.gz',16<<20)
    if key_gate.digest(archive)!=key_gate.sha_field(r.get('source_archive_sha256')):raise ValueError('producer archive digest differs')
    with tarfile.open(fileobj=io.BytesIO(archive)) as t:
        members=t.getmembers()
        if (len(members)!=len(before) or {m.name for m in members}!=set(before) or len({m.name for m in members})!=len(members)
            or any(not m.isfile() or m.size>8<<20 for m in members) or sum(m.size for m in members)>16<<20):
            raise ValueError('exact bounded producer archive required')
        source_bytes={m.name:t.extractfile(m).read() for m in members}
        if {n:key_gate.digest(b) for n,b in source_bytes.items()}!=before:raise ValueError('producer archive source bytes differ')
    if r.get('asset_file_sha256_before')!=input_gate.ASSETS or r.get('asset_file_sha256_after')!=input_gate.ASSETS:
        raise ValueError('pinned dataset/model/tokenizer asset bindings differ')
    upstream={n:v['sha256'] for n,v in key_gate.parse(source_bytes['upstream_origin.json']).items()}
    if r.get('upstream_original_sha256_before')!=upstream or r.get('upstream_original_sha256_after')!=upstream:
        raise ValueError('original upstream source identities differ')
    cases=r.get('cases');root=Path(r.get('private_workspace',''))
    identity=r.get('private_workspace_identity')
    if (not root.is_absolute() or type(identity)is not dict or set(identity)!={'device','inode'}
        or any(type(x)is not int or x<0 for x in identity.values()) or type(cases)is not list or len(cases)!=9):
        raise ValueError('bound retained private workspace and nine cases required')
    for row,(size,count,seed) in zip(cases,SCHEDULE):
        if type(row)is not dict or any(type(row.get(n))is not int or row[n]!=v for n,v in dict(size=size,count=count,base_seed=seed,record_count=count,fixed_token_length=128).items()):
            raise ValueError('ranked case order/count/seed differs')
        if type(row.get('generator_seed'))is not int or not 0<=row['generator_seed']<2147483647:
            raise ValueError('exact public generator seed required')
        if any(row.get(n)is not True for n in ('selection_byte_parity','tokenization_byte_parity','labels_private','all_private_directories_0700','all_private_files_0600')):
            raise ValueError('ranked case producer validation incomplete')
        files=row.get('input_file_sha256');case=root/f'size{size}-seed{seed}'
        if type(files)is not dict or set(files)!=expected_files(size):raise ValueError('exact six-file case inventory required')
        for h in files.values():key_gate.sha_field(h)
        for n in ('private_target_order_sha256','private_row_order_sha256'):key_gate.sha_field(row.get(n))
        expected=case/f'io/{NAMES[size]}/intermediate/client_preprocessed_input'
        if row.get('preprocessed_input_path')!=str(expected):raise ValueError('ranked input path is not its fixed owned child')
        hist=row.get('valid_token_count_histogram')
        if (type(hist)is not dict or any(type(k)is not str or not k.isdecimal() or not 1<=int(k)<=128 or type(v)is not int or v<1 for k,v in hist.items()) or sum(hist.values())!=count):
            raise ValueError('ranked token-length histogram differs')
    return dict(report=r,report_sha256=expected_sha256,workspace=root,identity=identity)

_FAILURE_QUARANTINE=[]

class BorrowedFixtures:
    """Held root/nested descriptors, strict read-only authenticated bytes."""
    def __init__(self,authorized):self.authorized=authorized;self.handles=[];self.states={};self.raw={};self.closed=False;self.inventories={};self.close_errors=[];self.uncertain_file_descriptors=[];self.uncertain_directory_descriptors=[]
    def _open_dir(self,path,parent_fd=None):
        fd=os.open(path if parent_fd is None else path.name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent_fd);self.handles.append(fd)
        info=os.fstat(fd);visible=path.lstat()
        if (not stat.S_ISDIR(visible.st_mode) or (info.st_dev,info.st_ino)!=(visible.st_dev,visible.st_ino) or stat.S_IMODE(info.st_mode)!=0o700):
            raise ValueError('original private directory identity/mode differs')
        self.states[path]=('dir',fd,info.st_dev,info.st_ino);return fd
    @staticmethod
    def _signature(info):
        return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)
    @classmethod
    def _file_state(cls,info):
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode)!=0o600 or not 0<=info.st_size<=16<<20:
            raise ValueError('bounded private regular0600 file required')
        return cls._signature(info)
    def _read(self,path,parent_fd,expected=None):
        visible=self._file_state(os.stat(path.name,dir_fd=parent_fd,follow_symlinks=False))
        if expected is not None and visible!=expected:raise ValueError('borrowed private file changed before open')
        fd=os.open(path.name,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW,dir_fd=parent_fd)
        try:
            first=self._file_state(os.fstat(fd))
            if first!=visible or expected is not None and first!=expected:raise ValueError('private fixture replaced while opening')
            # One owned descriptor, bounded os.read: no dup/fdopen handoff or
            # unbounded digest reader can lose ownership or read a replacement.
            pieces=[];remaining=(16<<20)+1
            while remaining:
                piece=os.read(fd,min(1<<20,remaining))
                if not piece:break
                pieces.append(piece);remaining-=len(piece)
            raw=b''.join(pieces)
            last=self._file_state(os.fstat(fd));now=self._file_state(os.stat(path.name,dir_fd=parent_fd,follow_symlinks=False))
            if first!=last or first!=now or len(raw)!=first[2]:raise ValueError('private fixture changed during authenticated read')
            if expected is None:self.states[path]=('file',parent_fd,first)
            return raw
        finally:
            primary=__import__('sys').exception()
            try:os.close(fd)
            except BaseException as error:
                self.uncertain_file_descriptors.append(fd);self.close_errors.append(type(error).__name__);_FAILURE_QUARANTINE.append(self)
                if primary is None:raise
                primary.add_note('private fixture file descriptor cleanup: '+type(error).__name__)
    def __enter__(self):
        try:
            root=self.authorized['workspace'];fd=self._open_dir(root);info=os.fstat(fd)
            if {'device':info.st_dev,'inode':info.st_ino}!=self.authorized['identity']:raise ValueError('retained producer workspace replaced')
            expected_roots={f'size{s}-seed{seed}' for s,c,seed in SCHEDULE}
            if set(os.listdir(fd))!=expected_roots:raise ValueError('exact nine-case private root inventory required')
            self.inventories[fd]=expected_roots
            for row,(size,count,seed) in zip(self.authorized['report']['cases'],SCHEDULE):
                case=root/f'size{size}-seed{seed}';expected=expected_files(size)
                directories={Path('.')}
                for name in expected:
                    parent=Path(name).parent
                    while parent!=Path('.'):directories.add(parent);parent=parent.parent
                fds={}
                for rel in sorted(directories,key=lambda p:len(p.parts)):
                    fds[rel]=self._open_dir(case/rel,fd if rel==Path('.') else fds[rel.parent])
                for rel,handle in fds.items():
                    children={Path(n).relative_to(rel).parts[0] for n in expected if Path(n).is_relative_to(rel)}
                    if set(os.listdir(handle))!=children:raise ValueError('exact private case file/directory inventory required')
                    self.inventories[handle]=children
                values={}
                for name,h in row['input_file_sha256'].items():
                    raw=self._read(case/name,fds[Path(name).parent])
                    if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('original private case bytes differ')
                    values[name]=raw
                self.raw[(size,seed)]=values
            self.check();return self
        except BaseException as primary:
            self.close(primary);raise
    def check(self):
        for fd,names in self.inventories.items():
            if set(os.listdir(fd))!=names:raise ValueError('borrowed private inventory changed')
        for path,state in self.states.items():
            now=path.lstat()
            if state[0]=='dir':
                held=os.fstat(state[1])
                if not stat.S_ISDIR(now.st_mode) or stat.S_IMODE(now.st_mode)!=0o700 or (now.st_dev,now.st_ino)!=(state[2],state[3]) or (held.st_dev,held.st_ino)!=(state[2],state[3]):raise ValueError('borrowed private directory changed')
            else:
                if self._file_state(now)!=state[2]:raise ValueError('borrowed private file changed')
                raw=self._read(path,state[1],expected=state[2])
                if hashlib.sha256(raw).digest()!=hashlib.sha256(self.raw_for(path)).digest():raise ValueError('borrowed private file content changed')
        return True
    def raw_for(self,path):
        relative=path.relative_to(self.authorized['workspace']);case=relative.parts[0]
        for (size,seed),values in self.raw.items():
            if case==f'size{size}-seed{seed}':return values[Path(*relative.parts[1:]).as_posix()]
        raise ValueError('borrowed fixture file was not authenticated')
    def close(self,primary=None):
        failure=None;pending,self.handles=self.handles,[]
        for fd in reversed(pending):
            try:os.close(fd)
            except BaseException as error:
                # close may have closed the fd before reporting failure. Its
                # integer may already be reused; quarantine and never retry it.
                self.uncertain_directory_descriptors.append(fd);self.close_errors.append(type(error).__name__)
                if primary is not None:primary.add_note('borrowed fixture close: '+type(error).__name__)
                elif failure is None or not isinstance(error,Exception) and isinstance(failure,Exception):failure=error
        self.closed=not self.uncertain_directory_descriptors and not self.uncertain_file_descriptors
        if not self.closed:_FAILURE_QUARANTINE.append(self)
        if primary is None and failure is not None:raise failure
    def __exit__(self,kind,error,tb):self.close(error)
