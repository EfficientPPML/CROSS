"""Fresh stdlib source bootstrap and reserved infrastructure terminal status.

Reference: reviewed checker141 entry. This readonly adaptation uses shared4543
marker semantics even before trusted helpers can be imported; NumPy is enabled
only after source, external grant, expected-store and resource admission.
"""
import argparse
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys

ROOT=Path(os.path.abspath(__file__)).parent
INFRASTRUCTURE_EXIT=70
FORBIDDEN=frozenset(('jax','jaxlib','jaxite_word','torch','transformers',
    'tensorflow','flax','torch_xla','safetensors','tokenizers','libtpu'))
EARLY_PROTECTED=('/home/jianming_gatech/cross_validation_assets', '/tmp/qwen-candidate2-fresh-validation-20261103', '/tmp/qwen-candidate3-calibration-20261103', '/tmp/qwen-candidate3-fresh56-validation-20261103', '/tmp/qwen-candidate3-published-regression-20261103', '/tmp/qwen-decoder-parameters-20261103/actual-v1/public_matrices', '/tmp/qwen-expanded-candidate-checker-actual-v1-20261104', '/tmp/qwen-expanded-candidate-checker-private-v1-20261104', '/tmp/qwen-expanded-prefix-observer-actual-v1-20261104', '/tmp/qwen-expanded-prefix-observer-actual-v2-20261104', '/tmp/qwen-expanded-prefix-observer-private-v1-20261104', '/tmp/qwen-expanded-prefix-observer-private-v2-20261104', '/tmp/qwen-expanded-prefix-observer-terminal-outer-root-20261104', '/tmp/qwen-expanded-prefix-observer-terminal-root-review-20261104', '/tmp/qwen-expanded-prefix-selector-actual-v1-20261104', '/tmp/qwen-expanded-prefix-selector-root-launch-20261104', '/tmp/qwen-expanded-prefix-selector-terminal-outer-root-20261104', '/tmp/qwen-expanded-prefix-selector-terminal-root-review-20261104', '/tmp/qwen-expanded-prefix-token-inputs-20261104-v1', '/tmp/qwen-full-model-trajectory-v2-20261104', '/tmp/qwen-native-calibration-20261103', '/tmp/qwen-qkv-attention-join-20261103/public_qkv_parameters', '/tmp/qwen-rmsnorm-owner-20261103', '/tmp/qwen-s1-final-rms-diagnosis-20261104')
_CLEANUP='_cross_toy4_cleanup_origin'
_BODY='_cross_toy4_body_origin'


def canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def parse(raw):
    def pairs(items):
        value={}
        for key,item in items:
            if key in value: raise ValueError('duplicate source/public JSON key')
            value[key]=item
        return value
    return json.loads(raw,object_pairs_hook=pairs,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite public JSON')))


def digest(value):
    return type(value)is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def path(value):
    if type(value)is not str or not 1<len(value)<=4096 or value.startswith('//') or '\\'in value or '\x00'in value:
        raise ValueError('canonical absolute public path required')
    p=Path(value)
    if not p.is_absolute() or str(p)!=value or '..'in p.parts or p==p.parent or p.resolve()!=p:
        raise ValueError('canonical resolved public path required')
    return p


def early_public(value):
    p=path(value)
    if p.suffix.lower()in ('.npy','.npz','.safetensors','.parquet'):
        raise ValueError('body format forbidden in bootstrap/public roles')
    for root in map(Path,EARLY_PROTECTED):
        if p==root or root in p.parents or p in root.parents:
            raise ValueError('public bootstrap role overlaps protected data')
    return p


def bounded(p,limit):
    primary=None
    fd=os.open(p,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW|os.O_CLOEXEC)
    try:
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0<before.st_size<=limit:
            raise ValueError('bounded regular bootstrap file required')
        blocks=[];size=0
        while size<=limit:
            block=os.read(fd,min(65536,limit+1-size))
            if not block: break
            blocks.append(block);size+=len(block)
        after=os.fstat(fd);named=os.stat(p,follow_symlinks=False)
        fields=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,stat.S_IMODE(s.st_mode))
        if fields(before)!=fields(after) or fields(before)!=fields(named) or not stat.S_ISREG(named.st_mode) or size!=before.st_size or size>limit:
            raise ValueError('bootstrap original file changed')
        return b''.join(blocks)
    except BaseException as error:
        primary=error
        if not (getattr(error,_CLEANUP,False)is True and getattr(error,_BODY,False)is not True):
            setattr(error,_BODY,True)
        raise
    finally:
        original=fd;fd=None
        try: os.close(original)
        except BaseException as error:
            if getattr(error,_BODY,False)is not True: setattr(error,_CLEANUP,True)
            if primary is None: raise
            if (getattr(primary,_BODY,False)is not True and getattr(primary,_CLEANUP,False)is True
                    and isinstance(primary,Exception) and not isinstance(error,Exception)):
                raise error from primary
            if error is not primary: primary.add_note('bootstrap close failure: '+type(error).__name__)


def snapshot(expected):
    if ROOT.resolve()!=ROOT: raise ValueError('source root alias differs')
    actual=set()
    def walk(directory,prefix=''):
        with os.scandir(directory)as items:
            for item in items:
                name=prefix+item.name
                if item.is_symlink(): raise ValueError('source symlink')
                if item.is_dir(follow_symlinks=False):
                    if not any(n.startswith(name+'/') for n in expected): raise ValueError('extra source directory')
                    walk(item.path,name+'/')
                elif item.is_file(follow_symlinks=False): actual.add(name)
                else: raise ValueError('nonregular source member')
    walk(ROOT)
    if actual!=set(expected): raise ValueError('exact selected source inventory differs')
    blobs={};total=0
    for name in sorted(expected):
        p=ROOT/name
        if p.resolve()!=p: raise ValueError('resolved source member differs')
        raw=bounded(p,min(8<<20,(32<<20)-total))
        total+=len(raw);blobs[name]=raw
        if total>32<<20 or hashlib.sha256(raw).hexdigest()!=expected[name]: raise ValueError('source bytes differ')
    return blobs


def origins(expected,*,initial=False):
    local={Path(n).stem for n in expected if n.endswith('.py')and '/'not in n}
    for name,module in tuple(sys.modules.items()):
        if name.split('.')[0]in FORBIDDEN or(initial and name.split('.')[0]=='numpy'):
            raise ValueError('preloaded/unselected backend or NumPy')
        if initial and name in local and name!='__main__': raise ValueError('preloaded local helper')
        if name=='__main__':
            # The selected CLI is a direct absolute script, not runpy/-m. Its
            # missing spec is an exact initial/postcondition, not permission
            # to ignore an unexpected outside-root spec origin.
            value=getattr(module,'__file__',None)
            if (not isinstance(value,(str,os.PathLike)) or Path(value)!=ROOT/'model_entry.py'
                    or Path(value).resolve()!=ROOT/'model_entry.py'
                    or getattr(module,'__spec__',None)is not None):
                raise ValueError('exact direct-CLI main file/spec origin required')
        values=(getattr(module,'__file__',None),getattr(getattr(module,'__spec__',None),'origin',None))
        found=[]
        for value in values:
            if value is None or isinstance(value,str)and value in('built-in','frozen'): continue
            if not isinstance(value,(str,os.PathLike)):
                if name in local or name=='__main__': raise ValueError('invalid critical module origin')
                continue
            raw_path=Path(value);p=raw_path.resolve()
            if not initial and name in local and raw_path!=ROOT/(name+'.py'):
                raise ValueError('canonical helper origin alias differs')
            if p==ROOT or ROOT in p.parents:
                expected_name='model_entry.py'if name=='__main__'else name+'.py'
                if expected_name not in expected or p!=ROOT/expected_name:
                    raise ValueError('unknown local module alias or origin')
            found.append(p)
        if not initial and name in local:
            if len(found)!=2 or any(p!=ROOT/(name+'.py') for p in found): raise ValueError('selected module origins differ')


def authenticate(manifest,expected_digest):
    if not sys.flags.isolated or not sys.dont_write_bytecode: raise ValueError('fresh Python -I -B required')
    if not digest(expected_digest): raise ValueError('external source digest required')
    raw=bounded(early_public(manifest),1<<20)
    if hashlib.sha256(raw).hexdigest()!=expected_digest: raise ValueError('external source map digest differs')
    expected=parse(raw)
    if type(expected)is not dict or not 29<=len(expected)<=512 or canonical(expected)!=raw:
        raise ValueError('canonical compact complete source map required')
    for name,h in expected.items():
        if type(name)is not str or not 0<len(name)<=512 or name.startswith('/')or '\\'in name or '\x00'in name or any(v in('','.','..')for v in name.split('/'))or not digest(h):
            raise ValueError('exact canonical relative source member required')
    required={'model_entry.py','model_output.py','model_validation_grant.py','model_validation_process.py',
        'model_admission.py','qwen_public_parameters.py','qwen_finalizer.py','cleanup.py','cleanup_origin.py',
        'model_source_inventory.json','model_readonly_policy.json','model_dependencies.json'}
    if not required<=set(expected): raise ValueError('complete model entry closure required')
    origins(expected,initial=True);blobs=snapshot(expected)
    inventory=parse(blobs['model_source_inventory.json'])
    if inventory!={'schema':'qwen-model-store-source-inventory-v1','files':sorted(expected)}:
        raise ValueError('selected source inventory differs')
    policy=parse(blobs['model_readonly_policy.json'])
    if sorted(policy['protected_paths']+policy['bootstrap_borrowed_roots'])!=list(EARLY_PROTECTED):
        raise ValueError('bootstrap/static path policy differs')
    return raw,expected,blobs


class FrozenLoader(importlib.abc.Loader):
    def __init__(self,name,data): self.name,self.data=name,data
    def create_module(self,spec): return None
    def exec_module(self,module):
        module.__file__=str(ROOT/(self.name+'.py'))
        exec(compile(self.data,module.__file__,'exec'),module.__dict__)


class FrozenImports(importlib.abc.MetaPathFinder):
    def __init__(self,blobs): self.blobs=blobs;self.allow_numpy=False
    def find_spec(self,fullname,path=None,target=None):
        first=fullname.split('.')[0]
        if first in FORBIDDEN or(first=='numpy'and not self.allow_numpy):
            raise ImportError('unselected import in readonly model phase')
        if '.'not in fullname and fullname+'.py'in self.blobs:
            return importlib.util.spec_from_loader(fullname,FrozenLoader(fullname,self.blobs[fullname+'.py']),origin=str(ROOT/(fullname+'.py')))
        return None


def terminal_status(call):
    try:
        value=call()
        if type(value)is not int or value not in(0,1): raise ValueError('exact normal terminal result required')
    except BaseException as error:
        try: print('Readonly model infrastructure failure: '+type(error).__name__,file=sys.stderr)
        except BaseException: pass
        return INFRASTRUCTURE_EXIT
    return value


def dispatch(args):
    raw,expected,blobs=authenticate(args.manifest,args.manifest_sha256)
    early_public(args.grant);path(args.output)
    if not digest(args.grant_sha256): raise ValueError('external grant SHA256 required')
    finder=FrozenImports(blobs);sys.meta_path.insert(0,finder)
    from model_validation_process import run
    return run(args,raw,expected,blobs,finder,snapshot,origins)


def main():
    parser=argparse.ArgumentParser(description='Independently granted readonly model-store validation')
    parser.add_argument('--role',required=True,choices=('validate_existing_model_stores',))
    for name in ('manifest','manifest-sha256','grant','grant-sha256','output'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    return terminal_status(lambda:dispatch(args))


if __name__=='__main__':
    raise SystemExit(main())
