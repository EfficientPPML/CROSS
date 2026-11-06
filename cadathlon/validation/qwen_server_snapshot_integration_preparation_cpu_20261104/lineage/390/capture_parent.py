"""Capture declared public387 parent before isolated source-only edits."""
import gzip,hashlib,io,json,pathlib,stat,tarfile
P=pathlib.Path('/tmp/qwen-server-actual-runtime-20261104')
N=pathlib.Path('/tmp/qwen-server-snapshot-runtime-20261104')
H=pathlib.Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(x):return(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n').encode()
raw={};states={}
for p in sorted(P.rglob('*')):
    a=p.lstat();assert not p.is_symlink()
    if stat.S_ISDIR(a.st_mode):continue
    assert stat.S_ISREG(a.st_mode)
    n=p.relative_to(P).as_posix();assert not n.endswith(('.pyc','.so')) and '__pycache__'not in p.parts
    raw[n]=p.read_bytes();states[n]=(a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)
assert len(raw)==387
mapping={n:sha(b)for n,b in raw.items()}
assert mapping['qwen_server_process.py']=='e8347c90d359cb0dd309ed3abe8979e636ca8efb6604bd82ccbb9a9af273bc7f'
assert mapping['qwen_server_backend.py']=='682e5130612abcb5741f266db7164f14389aac7df1b932646c9500a22fa8b740'
(H/'parent387-sources.json').write_bytes(canon(mapping))
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w',format=tarfile.PAX_FORMAT)as t:
    for n,b in raw.items():
        m=tarfile.TarInfo(n);m.size=len(b);m.mode=0o644;m.mtime=0;m.uid=m.gid=0;m.uname=m.gname=''
        t.addfile(m,io.BytesIO(b))
(H/'parent387-sources.tar.gz').write_bytes(gzip.compress(buf.getvalue(),mtime=0))
for n,b in raw.items():
    p=P/n;a=p.lstat();assert states[n]==(a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)and p.read_bytes()==b
N.mkdir(exist_ok=False)
for n,b in raw.items():p=N/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
receipt={'schema':'qwen-server387-parent-capture-v1','parent':str(P),'successor':str(N),'source_count':387,'bytes':sum(map(len,raw.values())),'map_sha256':sha(canon(mapping)),'archive_sha256':sha((H/'parent387-sources.tar.gz').read_bytes()),'parent_before_after_exact':True,'copy_exact_before_edits':True,'candidate_imports':False,'tests':False,'body_or_capacity_reads':False,'capture_script_sha256':sha(pathlib.Path(__file__).read_bytes())}
(H/'parent-capture.json').write_bytes(canon(receipt));print(json.dumps(receipt,sort_keys=True))
