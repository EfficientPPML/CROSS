"""Fixed ordinary-group parity contract; public sources/parameters only."""
from pathlib import Path
import hashlib,json,math,sys,os
from parity_protocol import environment
HERE=Path(__file__).resolve().parent
LEVELS=(49,25)
GROUPS=(('head',tuple(63*i for i in range(12))),('local_key',tuple(-16*i for i in range(16))),('edges',(0,63,63,-16,0)))
MAX_RESULTS=16
MAX_RESULT_BYTES=2<<30
MAX_DIGIT_BYTES=512<<20

def sources():
    paths={p.relative_to(HERE).as_posix():p for root in ('runtime','joined','base','server_inputs') for p in (HERE/root).rglob('*')
           if p.is_file() and p.suffix in ('.py','.json') and '__pycache__' not in p.parts}
    paths.update({p.name:p for p in HERE.glob('*.py')})
    paths.update({p.name:p for p in HERE.glob('*.json')})
    return {n:p.read_bytes() for n,p in paths.items()}

def hashes(raw):return {n:hashlib.sha256(b).hexdigest() for n,b in raw.items()}

def check_sources(*,allow_pending=False):
    origin=json.loads((HERE/'group_origin.json').read_bytes());deps=json.loads((HERE/'group_dependencies.json').read_bytes())
    if type(deps.get('ready'))is not bool or (not allow_pending and deps['ready']is not True):
        raise RuntimeError('ordinary-group genuine execution remains pending explicit review')
    overlays=deps['overlays']
    if set(overlays)!={'runtime/jaxite_word/herot.py','runtime/stage/key_contexts.py'}:
        raise ValueError('exact two parent runtime overlays required')
    for n,h in origin['parent_source_hashes'].items():
        expected=overlays.get(n,h)
        if hashlib.sha256((HERE/n).read_bytes()).hexdigest()!=expected:raise ValueError('parent/overlay source drift: '+n)
    for n,h in deps['new_runtime'].items():
        if hashlib.sha256((HERE/n).read_bytes()).hexdigest()!=h:raise ValueError('group runtime source drift: '+n)
    if deps['new_runtime']['grouped_rotation.py']!=origin['group_module_sha256'] or overlays['runtime/jaxite_word/herot.py']!=origin['raw_herot_sha256']:
        raise ValueError('reviewed raw/group proof bytes differ')
    return dict(ready=deps['ready'],parent_source_count=len(origin['parent_source_hashes']),overlays=overlays,
        new_runtime=deps['new_runtime'],group_public_archive_sha256=origin['group_archive_sha256'])

def plan(profile_bytes,plan_bytes):
    if os.environ.get('JAX_PLATFORMS')!='cpu':raise ValueError('public profile check requires explicit CPU before imports')
    import client_keys
    p,keys=client_keys.public_contract(profile_bytes,plan_bytes);client_keys._security_check(p)
    wanted={i%65536 for _,indices in GROUPS for i in indices if i%65536}
    if (p['degree']!=131072 or p['num_slots']!=65536 or p['dnum']!=8 or p['composite_degree']!=2
            or p['degree_layout']!=[256,512] or len(p['q_towers'])!=100 or len(p['p_towers'])!=14
            or not wanted<=set(keys)):
        raise ValueError('fixed secure profile/existing group key inventory differs')
    alpha=math.ceil(len(p['q_towers'])/p['dnum']);rows=[]
    for level in LEVELS:
        q=2*(level+1);parts=math.ceil(q/alpha);ct_bytes=2*p['degree']*q*4
        digits=parts*p['degree']*(q+len(p['p_towers']))*4
        for name,indices in GROUPS:
            nonzero=sum(i%65536!=0 for i in indices);result_bytes=nonzero*ct_bytes
            if len(indices)>MAX_RESULTS or result_bytes>MAX_RESULT_BYTES or digits>MAX_DIGIT_BYTES:raise ValueError('fixed group exceeds logical cap')
            rows.append(dict(level=level,group=name,indices=list(indices),normalized_indices=[i%65536 for i in indices],
                requests=len(indices),nonzero_completions=nonzero,zero_aliases=len(indices)-nonzero,
                input_payload_bytes=ct_bytes,digit_bytes=digits,result_bytes=result_bytes,decompositions=1,q_count=q,effective_parts=parts))
    return dict(parameters=p,rotation_inventory=list(keys),required_rotation_indices=sorted(wanted),rows=rows,
        exact_pair_comparisons=66,nonzero_pair_comparisons=58,zero_alias_comparisons=8,actual_rotation_calls=116,
        shared_decompositions=6,max_results=MAX_RESULTS,max_result_bytes=MAX_RESULT_BYTES,max_digit_bytes=MAX_DIGIT_BYTES,
        no_new_keys=True,no_decode=True,
        scope='Ordinary same-input completion parity only. Logical caps exclude controls, keys, compiler, conversion, allocator and retained original backing; no memory-fit, speed or numerical-accuracy claim.')

def imported_sources(expected):
    common=HERE/'joined/checkout/cadathlon/common';stage=HERE/'runtime/stage';core=HERE/'runtime/jaxite_word'
    names={p.stem:common/p.name for p in common.glob('*.py')}
    names.update({p.stem:stage/p.name for p in stage.glob('*.py')})
    names.update({p.stem:core/p.name for p in core.glob('*.py')})
    names.update({p.stem:p for p in HERE.glob('*.py') if not p.name.startswith('test_')})
    names['jaxite_word']=core/'__init__.py';result={}
    for name,module in tuple(sys.modules.items()):
        file=getattr(module,'__file__',None)
        if not file:continue
        path=Path(file).resolve();leaf=name.rsplit('.',1)[-1]
        wanted=names.get(name,names.get(leaf) if name.startswith('jaxite_word.') else None)
        if wanted is not None and path!=wanted:raise ValueError('mixed group runtime import: '+name)
        if HERE in path.parents and path.suffix=='.py':
            n=path.relative_to(HERE).as_posix();h=hashlib.sha256(path.read_bytes()).hexdigest()
            if expected.get(n)!=h:raise ValueError('loaded source outside exact closure: '+name)
            result[name]=dict(path=n,sha256=h)
    return result
