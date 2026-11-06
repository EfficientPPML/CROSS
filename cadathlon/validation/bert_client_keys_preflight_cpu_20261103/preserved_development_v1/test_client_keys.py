"""Metadata, tiny mocked-producer storage/lifecycle tests; never actual keygen."""
from pathlib import Path
import hashlib,json,os,stat,subprocess,sys
from types import SimpleNamespace
import numpy as np
import pytest
import client_keys as keys
PLAN=Path('/tmp/bert-twelve-layer-continuation-plan-20261103')
PROFILE=(PLAN/'p60_profile_and_refresh.json').read_bytes();RAW_PLAN=(PLAN/'report.json').read_bytes()


def test_backend_free_import():
    subprocess.run([sys.executable,'-c','import client_keys,sys; assert not any(n in sys.modules for n in ("jax","torch","jaxite_word"))'],cwd=Path(keys.__file__).parent,check=True)

def test_fixed_actual_public_profile_inventory_and_capacity():
    p,r=keys.public_contract(PROFILE,RAW_PLAN);spec=keys.inventory(p,r);resource=keys.resources(p,r)
    assert (p['degree'],p['max_level'],p['dnum'])==(131072,49,8)
    assert len(p['q_towers'])==100 and len(p['p_towers'])==14 and len(r)==130
    assert r==tuple(json.loads(RAW_PLAN)['union_keys']) and r[-1]==262143
    assert len(spec)==263 and spec['relin_a.npy'][0]==(8,114,131072)
    assert spec['public_key.npy'][0]==(2,100,131072)
    assert resource['public_array_bytes']==(131*2*8*114+2*100)*131072*4
    assert resource['private_array_bytes']==100*131072*4
    assert 117<(resource['public_required_bytes']+resource['private_required_bytes'])/2**30<119
    with pytest.raises(ValueError):keys.public_contract(PROFILE+b' ',RAW_PLAN)
    with pytest.raises(ValueError):keys.public_contract(PROFILE,RAW_PLAN+b' ')

def test_actual_security_predicates_without_backend_or_keygen(monkeypatch):
    import jax
    import jax._src.xla_bridge as xb
    def forbidden(*a,**kw):raise AssertionError('no backend/key generation authorized')
    monkeypatch.setattr(xb,'backends',forbidden)
    from jaxite_word import he_params
    for name in ('gen_pke_pair','gen_evaluation_key','gen_rotation_key','key_switch_gen'):monkeypatch.setattr(he_params.kg,name,forbidden)
    monkeypatch.setattr(keys,'_crypto',forbidden)
    p,_=keys.public_contract(PROFILE,RAW_PLAN)
    report=keys._security_check(p)
    assert report['target_bits']==128 and report['P_coverage'] and abs(report['actual_log2_qp']-3402.080255733439)<1e-8
    bad=dict(p);bad['p_towers']=[]
    with pytest.raises(ValueError):keys._security_check(bad)

@pytest.fixture
def mocked(tmp_path,monkeypatch):
    events=[];params=dict(degree=4,num_slots=2,q_towers=[17,97],p_towers=[193],composite_degree=2,
        degree_layout=[2,2],precision=32,max_level=0,scaling_factor=16.,dnum=1,batch=1,
        sigma=3.190000057220458984375,secret_distribution='uniform_ternary')
    rotations=(1,2,3);source_secret=[[1,2,3,4],[1,2,3,4]]
    monkeypatch.setattr(keys,'public_contract',lambda p,r:(params,rotations))
    def security(p):events.append('security');return {'mocked':True}
    monkeypatch.setattr(keys,'_security_check',security)
    monkeypatch.setattr(keys.shutil,'disk_usage',lambda p:SimpleNamespace(total=1<<45,free=1<<44))
    def switched():return dict(a=np.zeros((1,3,4),dtype=np.uint32),b=np.zeros((1,3,4),dtype=np.uint32),modulus={'Q':[17,97],'P':[193]},P=[193],shape=(1,3,4))
    class Fake:
        __file__=__file__
        def gen_pke_pair(self,q,p,n,**kw):
            events.append('pke');assert q==[17,97] and p==[] and n==4;return dict(secret_key=source_secret,public_key=np.ones((2,2,4),dtype=np.int64))
        def gen_evaluation_key(self,secret,**kw):
            events.append('relin');assert secret is source_secret and type(secret[0][0])is int;return switched()
        def gen_rotation_key(self,secret,q,p,index,**kw):
            events.append(('rotation',index));assert secret is source_secret;return {index:switched()}
    fake=Fake()
    def crypto():events.append('crypto');return fake
    monkeypatch.setattr(keys,'_crypto',crypto)
    return SimpleNamespace(public=tmp_path/'public_keys',private=tmp_path/'client_private',events=events,params=params,rotations=rotations,fake=fake)

def generate(m):return keys.generate(m.public,m.private,profile_bytes=PROFILE,plan_bytes=RAW_PLAN)

def test_mocked_complete_call_order_files_permissions_and_context(mocked):
    m=mocked;r=generate(m)
    assert m.events[:4]==['security','crypto','pke','relin']
    assert [e for e in m.events if isinstance(e,tuple)]==[('rotation',i) for i in m.rotations]
    assert m.events[-1]=='security'
    assert stat.S_IMODE(m.private.stat().st_mode)==0o700
    assert {p.name for p in m.private.iterdir()}=={'secret.npy','private_manifest.json'}
    assert all(stat.S_IMODE(p.stat().st_mode)==0o600 for p in m.private.iterdir())
    public=r['public_manifest'];body={k:v for k,v in public.items() if k!='context_id'}
    assert public['context_id']==keys.digest(body)
    assert r['private_manifest']['public_manifest_sha256']==hashlib.sha256(keys.canonical(public)).hexdigest()
    assert 'secret' not in json.dumps(public) and len(public['arrays'])==9
    assert keys.authenticate_public(m.public,profile_bytes=PROFILE,plan_bytes=RAW_PLAN)==public

@pytest.mark.parametrize('failure',('security','capacity','existing_public','existing_private'))
def test_fail_before_any_crypto(mocked,monkeypatch,failure):
    m=mocked
    if failure=='security':
        def reject(p):m.events.append('security');raise ValueError('insecure')
        monkeypatch.setattr(keys,'_security_check',reject)
    elif failure=='capacity':monkeypatch.setattr(keys.shutil,'disk_usage',lambda p:SimpleNamespace(total=10,free=0))
    else:
        path=m.public if failure=='existing_public' else m.private;path.mkdir();(path/'sentinel').write_text('retain')
    with pytest.raises((ValueError,OSError)):generate(m)
    assert 'crypto' not in m.events
    if failure.startswith('existing'):assert (path/'sentinel').read_text()=='retain'

@pytest.mark.parametrize('error',(RuntimeError('rotation failed'),KeyboardInterrupt('cancel')))
def test_primary_failure_removes_both_owned_trees(mocked,error):
    m=mocked
    def fail(*a,**kw):raise error
    m.fake.gen_rotation_key=fail
    with pytest.raises(type(error)) as caught:generate(m)
    assert caught.value is error and not m.public.exists() and not m.private.exists()

def test_second_directory_failure_removes_first_before_crypto(mocked,monkeypatch):
    original=keys._OwnedDirectory
    def create(path):
        if Path(path)==mocked.private:raise OSError('second create failed')
        return original(path)
    monkeypatch.setattr(keys,'_OwnedDirectory',create)
    with pytest.raises(OSError):generate(mocked)
    assert not mocked.public.exists() and 'crypto' not in mocked.events

def test_replacement_retained_and_other_owned_tree_removed(mocked):
    m=mocked;old=m.public.with_name('moved-original')
    def replace(*args,**kwargs):
        m.public.rename(old);m.public.mkdir();(m.public/'sentinel').write_text('replacement');raise KeyboardInterrupt('original')
    m.fake.gen_rotation_key=replace
    with pytest.raises(KeyboardInterrupt) as caught:generate(m)
    assert (m.public/'sentinel').read_text()=='replacement' and old.is_dir() and not m.private.exists()
    assert any('replacement' in n for n in caught.value.__notes__)

@pytest.mark.parametrize('primary',(False,True))
def test_all_handles_close_and_primary_cancellation_preserved(mocked,monkeypatch,primary):
    m=mocked;closed=[];original=keys._OwnedDirectory.close
    def close(owner):
        original(owner);closed.append(owner.path)
        if owner.path==m.public:raise KeyboardInterrupt('close failed')
    monkeypatch.setattr(keys._OwnedDirectory,'close',close)
    error=KeyboardInterrupt('primary')
    if primary:
        def fail(*a,**kw):raise error
        m.fake.gen_rotation_key=fail
    with pytest.raises(KeyboardInterrupt) as caught:generate(m)
    assert closed==[m.public,m.private]
    if primary:assert caught.value is error and any('close' in n for n in error.__notes__)

@pytest.mark.parametrize('kind',('dtype_float','negative','outside','wrong_shape','bool'))
def test_producer_array_validation_precedes_conversion(kind):
    a=np.ones((2,3),dtype=np.int64)
    if kind=='dtype_float':a=a.astype(float)
    elif kind=='negative':a[0,0]=-1
    elif kind=='outside':a[0,0]=17
    elif kind=='wrong_shape':a=a[:1]
    else:a=a.astype(bool)
    with pytest.raises(ValueError):keys._array(a,(2,3),(17,97))

def record(path,shape):return dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),dtype='uint32',shape=list(shape),bytes=path.stat().st_size)
@pytest.mark.parametrize('kind',('float','object','fortran','shape','residue','symlink','fifo','extra_bytes','hash','bool_shape'))
def test_stored_array_rejects_malformed_before_use(tmp_path,kind,monkeypatch):
    shape=(2,3);a=np.ones(shape,dtype=np.uint32);path=tmp_path/'key.npy'
    if kind=='float':a=a.astype(float)
    elif kind=='object':a=a.astype(object)
    elif kind=='fortran':a=np.asfortranarray(a)
    elif kind=='shape':a=a[:1]
    elif kind=='residue':a[0,0]=17
    with path.open('wb') as f:np.save(f,a,allow_pickle=True)
    desc=record(path,shape)
    if kind=='symlink':path.rename(tmp_path/'actual.npy');path.symlink_to(tmp_path/'actual.npy')
    elif kind=='fifo':path.unlink();os.mkfifo(path)
    elif kind=='extra_bytes':
        with path.open('ab') as f:f.write(b'extra')
        desc=record(path,shape)
    elif kind=='hash':desc['sha256']='0'*64
    elif kind=='bool_shape':desc['shape']=[True,3]
    if kind not in ('residue',):
        def forbidden(*a,**kw):raise AssertionError('malformed header must reject before mmap')
        monkeypatch.setattr(keys.np,'memmap',forbidden)
    fd=os.open(tmp_path,os.O_RDONLY|os.O_DIRECTORY)
    try:
        with pytest.raises((ValueError,OSError)):keys._authenticate_array(fd,'key.npy',desc,shape,(17,97))
    finally:os.close(fd)

@pytest.mark.parametrize('kind',('context','extra_file','secret_public','typed_rotation','changed_hash'))
def test_authentication_rejects_identity_and_inventory(mocked,kind):
    m=mocked;r=generate(m);manifest=r['public_manifest'];path=m.public/'public_manifest.json'
    if kind=='extra_file':(m.public/'unexpected').write_bytes(b'')
    elif kind=='secret_public':(m.public/'secret.npy').write_bytes(b'')
    elif kind=='context':manifest['context_id']='0'*64;path.write_bytes(keys.canonical(manifest))
    elif kind=='typed_rotation':manifest['rotations'][0]=True;path.write_bytes(keys.canonical(manifest))
    else:
        key=m.public/'relin_a.npy';raw=bytearray(key.read_bytes());raw[-1]^=1;key.write_bytes(raw)
    with pytest.raises(ValueError):keys.authenticate_public(m.public,profile_bytes=PROFILE,plan_bytes=RAW_PLAN)
