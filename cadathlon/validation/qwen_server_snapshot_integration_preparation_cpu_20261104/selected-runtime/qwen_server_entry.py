"""Fresh server stdlib bootstrap; backend enablement follows public admission.

Reference: reviewed checker141 entry. This server adaptation uses shared4543
marker semantics even before trusted helpers can be imported; Core imports are enabled only after source, model, role, policy and grant admission.
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
FORBIDDEN=frozenset(('torch','transformers',
    'tensorflow','flax','torch_xla','safetensors','tokenizers'))
CORE_ROOTS=frozenset(('jax','jaxlib','jaxite_word','numpy','libtpu','jax_plugins'))
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
        raw=bounded(p,min(8<<20,(64<<20)-total))
        total+=len(raw);blobs[name]=raw
        if total>64<<20 or hashlib.sha256(raw).hexdigest()!=expected[name]: raise ValueError('source bytes differ')
    return blobs


ROOT_MODULES=frozenset(('qwen_admission_snapshot', 'qwen_server_entry', 'qwen_server_process', 'qwen_server_dependencies', 'qwen_server_authority_packet', 'qwen_server_resource', 'qwen_server_public_memory', 'qwen_server_admission', 'qwen_scientific_admission', 'scientific_source_pins', 'freeze_issuer', 'freeze_admission', 'qwen_server_backend', 'qwen_backend_identity', 'keygen_output', 'client_role_common', 'client_server_terminal', 'keygen_authority', 'keygen_boundary_pins', 'model_admission', 'public_source_pins', 'qwen_public_parse', 'qwen_file_contract', 'qwen_key_contract', 'qwen_key_adapter', 'qwen_key_files_admission', 'qwen_key_loader', 'qwen_key_readers', 'key_descriptor_io', 'cleanup', 'cleanup_origin', 'qwen_finalizer', 'qwen_public_parameters', 'reused_client_io', 'reused_rns_payload', 'qwen_serialization', 'qwen_layout', 'qwen_file_lifecycle', 'qwen_publisher', 'qwen_backend_composition', 'qwen_stage_composition', 'full_contract'))
COMMON_MODULES=frozenset(('encrypted_chained_streaming_linear', 'encrypted_compilation', 'encrypted_ntt', 'encrypted_packed_qwen', 'encrypted_packed_transformer', 'encrypted_polynomial', 'encrypted_primitives', 'encrypted_qwen_attention_bypass', 'encrypted_qwen_attention_join', 'encrypted_qwen_attention_routing', 'encrypted_qwen_causal_softmax', 'encrypted_qwen_causal_softmax_batched', 'encrypted_qwen_decoder_tail', 'encrypted_qwen_full_model', 'encrypted_qwen_head_routing', 'encrypted_qwen_rms_refresh', 'encrypted_qwen_rmsnorm', 'encrypted_qwen_silu', 'encrypted_refresh', 'encrypted_rotation', 'encrypted_scale_planning', 'encrypted_streaming_linear', 'encrypted_transformer', 'qwen_candidate_bridge', 'qwen_silu_contract', 'support'))


def module_source(name,expected):
    if name=='__main__':return 'qwen_server_entry.py'
    if name in ROOT_MODULES and name+'.py'in expected:return name+'.py'
    if name in COMMON_MODULES and 'runtime/common/'+name+'.py'in expected:return 'runtime/common/'+name+'.py'
    if name=='jaxite_word':return 'runtime/jaxite_word/__init__.py'
    if name.startswith('jaxite_word.'):
        relative='runtime/'+name.replace('.','/')+'.py'
        if relative in expected:return relative
    if '.'not in name and 'runtime/jaxite_word/'+name+'.py'in expected:return 'runtime/jaxite_word/'+name+'.py'
    return None


def origins(expected,*,initial=False,core_enabled=False):
    for name,module in tuple(sys.modules.items()):
        first=name.split('.')[0];selected=module_source(name,expected)
        if first in FORBIDDEN or ((initial or not core_enabled)and first in CORE_ROOTS):
            raise ValueError('preloaded/unselected backend')
        if initial and selected is not None and name!='__main__': raise ValueError('preloaded local/core helper')
        if name=='__main__':
            value=getattr(module,'__file__',None)
            if (not isinstance(value,(str,os.PathLike))or Path(value)!=ROOT/'qwen_server_entry.py'
                    or Path(value).resolve()!=ROOT/'qwen_server_entry.py'or getattr(module,'__spec__',None)is not None):
                raise ValueError('exact direct-CLI main file/spec origin required')
        values=(getattr(module,'__file__',None),getattr(getattr(module,'__spec__',None),'origin',None))
        found=[]
        for value in values:
            if value is None or isinstance(value,str)and value in('built-in','frozen'):continue
            if not isinstance(value,(str,os.PathLike)):
                if selected is not None:raise ValueError('invalid selected module origin')
                continue
            raw_path=Path(value);resolved=raw_path.resolve()
            if selected is not None and(raw_path!=ROOT/selected or resolved!=ROOT/selected):
                raise ValueError('selected module origin alias differs')
            if resolved==ROOT or ROOT in resolved.parents:
                if selected is None or selected not in expected or resolved!=ROOT/selected:
                    raise ValueError('unknown local/core alias or origin')
            found.append(resolved)
        if not initial and selected is not None and name!='__main__':
            if len(found)!=2 or any(x!=ROOT/selected for x in found):raise ValueError('both selected origins required')


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
    required={name+'.py'for name in ROOT_MODULES}|{'qwen_server_source_inventory.json','qwen_server_entry_status.json',
        'qwen_server_static_policy.json','qwen_server_dependencies.json','qwen_server_role_policy.json'}
    if not required<=set(expected):raise ValueError('complete server source closure required')
    origins(expected,initial=True);blobs=snapshot(expected)
    inventory=parse(blobs['qwen_server_source_inventory.json'])
    if inventory!={'schema':'qwen-server-source-inventory-v1','files':sorted(expected)}:
        raise ValueError('selected source inventory differs')
    policy=parse(blobs['qwen_server_static_policy.json'])
    if policy!={'schema':'qwen-server-static-policy-v1','protected_paths':list(EARLY_PROTECTED)}:
        raise ValueError('bootstrap/static path policy differs')
    return raw,expected,blobs


class FrozenLoader(importlib.abc.Loader):
    def __init__(self,relative,data):self.relative,self.data=relative,data
    def create_module(self,spec):return None
    def exec_module(self,module):
        module.__file__=str(ROOT/self.relative)
        exec(compile(self.data,module.__file__,'exec'),module.__dict__)


class FrozenImports(importlib.abc.MetaPathFinder):
    def __init__(self,blobs):self.blobs=blobs;self.core_enabled=False
    def find_spec(self,fullname,path=None,target=None):
        first=fullname.split('.')[0]
        relative=module_source(fullname,self.blobs)
        core=first in CORE_ROOTS or(relative is not None and relative.startswith('runtime/jaxite_word/'))
        if first in FORBIDDEN or(core and not self.core_enabled):raise ImportError('unselected import in server phase')
        if core:return None  # The following authenticated core-alias finder owns execution.
        if relative is not None:
            if target is not None:raise ImportError('selected server helper reload is forbidden')
            return importlib.util.spec_from_loader(fullname,FrozenLoader(relative,self.blobs[relative]),origin=str(ROOT/relative))
        local_stems={Path(n).stem for n in self.blobs if n.endswith('.py')and('/'not in n or n.startswith('runtime/common/'))}
        if first in local_stems:raise ImportError('local helper is outside the actual server import selection')
        # Historical copies and namespace packages below the source root are
        # never selected imports, even though their bytes are preserved evidence.
        if fullname.split('.')[0]in {n.split('/')[0]for n in self.blobs if '/'in n}:
            raise ImportError('historical source namespace is not selected')
        return None


def terminal_status(call):
    try:
        value=call()
        if type(value)is not int or value not in(0,1): raise ValueError('exact normal terminal result required')
    except BaseException as error:
        try: print('Server infrastructure failure: '+type(error).__name__,file=sys.stderr)
        except BaseException: pass
        return INFRASTRUCTURE_EXIT
    return value


def dispatch(args):
    raw,expected,blobs=authenticate(args.manifest,args.manifest_sha256)
    status=parse(blobs['qwen_server_entry_status.json'])
    if status.get('actual_entry_ready')is not True:
        raise PermissionError('actual server closure is source-only pending full preparation and grants')
    early_public(args.packet);early_public(args.archive)
    if not digest(args.packet_sha256)or not digest(args.archive_sha256):raise ValueError('independent public packet/archive pins required')
    finder=FrozenImports(blobs);sys.meta_path.insert(0,finder)
    from qwen_server_process import run
    return run(args,raw,expected,blobs,finder,snapshot,origins)


def run_server(args):
    if getattr(args,'role',None)!='full28_server':raise ValueError('exact full28 server entry role required')
    return dispatch(args)


def main():
    parser=argparse.ArgumentParser(description='Independently granted full28 Qwen server (currently source-only)')
    parser.add_argument('--role',required=True,choices=('full28_server',))
    for name in('manifest','manifest-sha256','archive','archive-sha256','packet','packet-sha256'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    return terminal_status(lambda:run_server(args))


if __name__=='__main__':
    raise SystemExit(main())
