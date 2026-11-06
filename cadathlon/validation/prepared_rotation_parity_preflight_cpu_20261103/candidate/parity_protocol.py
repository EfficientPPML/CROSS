"""Fixed public schedule and source-only checks; no key or ciphertext reads."""
from pathlib import Path
import hashlib,json,math,os,sys
HERE=Path(__file__).resolve().parent
INDICES=(1,1,64,64,1,64)
LEVELS=(49,25)
MAX_ENTRIES=2
MAX_BYTES=2<<30

def environment():
    if os.environ.get('JAX_PLATFORMS')!='cpu':raise ValueError('explicit CPU process required')
    # No inherited alternate implementation, key-generation, fixture or mock
    #switch is accepted. The process helper's empty owned directory is inert.
    allowed={'CROSS_PUBLIC_KEY_DIRECTORY'}
    bad=[n for n in os.environ if (n.startswith(('CROSS_','BOOTSTRAP_')) and n not in allowed)
         or n.startswith(('PYTEST_','PYTHONINSPECT','PYTHONSTARTUP'))]
    if bad:raise ValueError('inherited runtime/test override forbidden: '+','.join(sorted(bad)))
    if os.environ.get('JAX_DISABLE_JIT','').lower() not in ('','0','false'):
        raise ValueError('disabled JIT is not the fixed actual runtime')
    if os.environ.get('XLA_FLAGS',''):raise ValueError('inherited XLA override forbidden')
    return {n:os.environ.get(n) for n in ('JAX_PLATFORMS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','JAX_ENABLE_X64')}

def sources():
    paths={p.relative_to(HERE).as_posix():p for root in ('runtime','joined') for p in (HERE/root).rglob('*')
           if p.is_file() and p.suffix in ('.py','.json') and '__pycache__' not in p.parts}
    paths.update({p.name:p for p in HERE.glob('*.py')})
    paths.update({p.name:p for p in HERE.glob('*.json')})
    return {n:p.read_bytes() for n,p in paths.items()}

def hashes(raw):return {n:hashlib.sha256(b).hexdigest() for n,b in raw.items()}

def check_sources():
    import dependencies
    deps=dependencies.check();origin=json.loads((HERE/'parity_origin.json').read_text())
    changed=set(origin['overlays'])|set(origin['changed_provenance'])
    for name,h in origin['base_source_hashes'].items():
        if name not in changed and hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=h:
            raise ValueError('unreviewed original source drift: '+name)
    for name,h in origin['overlays'].items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=h:raise ValueError('reviewed overlay changed: '+name)
    return dict(dependencies=deps,base_source_count=origin['base_source_count'],overlays=origin['overlays'],
                loader_proof_tar_sha256=origin['loader_proof_tar_sha256'])

def plan(profile_bytes,plan_bytes):
    import client_keys
    p,keys=client_keys.public_contract(profile_bytes,plan_bytes);client_keys._security_check(p)
    if (p['degree']!=131072 or p['num_slots']!=65536 or p['dnum']!=8 or p['composite_degree']!=2
            or p['degree_layout']!=[256,512] or len(p['q_towers'])!=100 or len(p['p_towers'])!=14
            or not {1,64}<=set(keys)):
        raise ValueError('fixed secure common profile/key inventory differs')
    rows=[];alpha=math.ceil(len(p['q_towers'])/p['dnum'])
    for level in LEVELS:
        q=(level+1)*2;digits=math.ceil(q/alpha)
        pair=2*digits*p['degree']*(q+len(p['p_towers']))*4+p['degree']*4
        rows.append(dict(level=level,q_count=q,effective_digits=digits,indices=list(INDICES),
                         prepared_operand_bytes=pair,max_two_entry_bytes=2*pair,
                         input_payload_bytes=2*p['degree']*q*4,expected_hits=4,expected_publications=2))
    if max(r['max_two_entry_bytes'] for r in rows)>MAX_BYTES:raise ValueError('fixed two-entry cap too small')
    return dict(parameters=p,rotation_inventory=list(keys),rows=rows,actual_rotation_calls=24,
                exact_pair_comparisons=12,max_entries=MAX_ENTRIES,max_bytes=MAX_BYTES,
                no_new_keys=True,no_decode=True,
                scope='Same-ciphertext representation parity only; cache hits and logical bytes are not a speed or numerical-accuracy claim.')

def imported_sources(expected):
    # Check every project module, including accidental imports from another
    #checkout. Generic third-party packages remain software-version evidence.
    common=HERE/'joined/checkout/cadathlon/common';stage=HERE/'runtime/stage';core=HERE/'runtime/jaxite_word'
    names={p.stem:common/p.name for p in common.glob('*.py')}
    names.update({p.stem:stage/p.name for p in stage.glob('*.py')})
    names.update({p.stem:core/p.name for p in core.glob('*.py')})
    names['jaxite_word']=core/'__init__.py'
    result={}
    for name,module in tuple(sys.modules.items()):
        file=getattr(module,'__file__',None)
        if not file:continue
        path=Path(file).resolve();leaf=name.rsplit('.',1)[-1]
        wanted=names.get(name,names.get(leaf) if name.startswith('jaxite_word.') else None)
        if wanted is not None and path!=wanted:raise ValueError('mixed actual runtime import: '+name)
        if HERE in path.parents and path.suffix=='.py':
            n=path.relative_to(HERE).as_posix();h=hashlib.sha256(path.read_bytes()).hexdigest()
            if expected.get(n)!=h:raise ValueError('loaded source outside exact closure: '+name)
            result[name]=dict(path=n,sha256=h)
    return result
