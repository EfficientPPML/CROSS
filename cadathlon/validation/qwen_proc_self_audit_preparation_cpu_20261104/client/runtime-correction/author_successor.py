"""Copy declared public sources and apply only the reviewed kernel-path rule."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil

BASE=Path('/tmp/qwen-actual-client-role-runtime-20261104')
ROOT=Path('/tmp/qwen-actual-client-role-runtime-v2-20261104')
OUT=Path(__file__).parent
MAP=Path('/tmp/qwen-actual-client-role-runtime-sources-v1-20261104.json')
SECURITY=Path('/tmp/qwen-keygen-actual-runtime-v3-20261104/keygen_dependencies.py')
def h(raw):return hashlib.sha256(raw).hexdigest()
def dump(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
raw=MAP.read_bytes();assert h(raw)=='41d563410ec152d2ae390396284fc776da940276141552a867ae775a797f93d2'
expected=json.loads(raw);assert len(expected)==464
assert not ROOT.exists()
for name,digest in expected.items():
    data=(BASE/name).read_bytes();assert h(data)==digest
    target=ROOT/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
before=(BASE/'qwen_client_dependencies.py').read_text();after=before
security=SECURITY.read_bytes();assert h(security)=='9e6760172efb58342b4aee5d7a3ae1799930869d0ae479258d691ba9e7365ca4'
helper='''def _literal_proc_self(value,dir_fd=None):
    # Preserve os.open's original argument before Path normalization. Direct
    # builtin audit events can attest only their already-normalized argument.
    # The wrapper rejects bytes/PathLike, relative and dirfd aliases here.
    return type(value)is str and value in _PROC_SELF_KERNEL and dir_fd is None
'''
after=after.replace('_QUARANTINE=[]\n','_QUARANTINE=[]\n_PROC_SELF_KERNEL=frozenset((\'/proc/self/status\',\'/proc/self/cgroup\',\'/proc/self/mountinfo\'))\n',1)
after=after.replace('def object_bytes(',helper+'\n\ndef object_bytes(',1)
old="""    def access(self,p,flags):
        if type(flags)is not int or not p.is_absolute()or '..'in p.parts:self.deny('open spelling or flags')
"""
new="""    def access(self,p,flags,*,literal_kernel=False):
        if type(flags)is not int or type(literal_kernel)is not bool or not p.is_absolute()or '..'in p.parts:self.deny('open spelling or flags')
        if str(p)in _PROC_SELF_KERNEL:
            allowed=os.O_CLOEXEC|os.O_NOFOLLOW|os.O_NONBLOCK
            if not literal_kernel or p not in self.kernel or flags&~allowed:self.deny('nonliteral or writable proc-self read')
            if any(p==r or r in p.parents for r in self.protected):self.deny('protected body')
            if p.resolve()!=Path('/proc')/str(os.getpid())/p.name:self.deny('foreign proc-self resolution')
            return
"""
assert after.count(old)==1;after=after.replace(old,new,1)
old="""            selected=self.resolve(path,dir_fd);self.access(selected,flags)
            previous=getattr(self.local,'open',None);self.local.open=(path,selected,flags)
"""
new="""            selected=self.resolve(path,dir_fd);literal=_literal_proc_self(path,dir_fd)
            self.access(selected,flags,literal_kernel=literal)
            previous=getattr(self.local,'open',None);self.local.open=(path,selected,flags,literal)
"""
assert after.count(old)==1;after=after.replace(old,new,1)
old="""                p=pending[1]if pending is not None and args[0]==pending[0]else self.resolve(args[0])
                self.access(p,args[2])
"""
new="""                matched=pending is not None and args[0]==pending[0]
                p=pending[1]if matched else self.resolve(args[0])
                literal=pending[3]if matched else _literal_proc_self(args[0])
                self.access(p,args[2],literal_kernel=literal)
"""
assert after.count(old)==1;after=after.replace(old,new,1)
ast.parse(after)
(ROOT/'qwen_client_dependencies.py').write_text(after)
(OUT/'before_qwen_client_dependencies.py').write_text(before)
(OUT/'qwen_client_dependencies.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='held464-v1/qwen_client_dependencies.py',tofile='kernel-successor464-v2/qwen_client_dependencies.py')))
(OUT/'parent464-sources.json').write_bytes(raw)
(OUT/'shared_keygen_rule.json').write_bytes(dump({'path':str(SECURITY),'sha256':h(security),'scope':'Exact helper and special proc-self branch shared; ordinary client input/key/output audit remains unchanged.'}))
print(json.dumps({'root':str(ROOT),'changed_source':'qwen_client_dependencies.py','sha256':h(after.encode()),'sources':len(expected)},sort_keys=True))
