"""Authenticated public gamma-only read; no model construction or forward."""
from pathlib import Path
from contextlib import contextmanager
import hashlib,json,sys,tarfile,io,os,time
import numpy as np
P=Path(__file__).resolve().parent
MODEL=Path('/home/jianming_gatech/cross_validation_assets/qwen/c1899de289a04d12100db370d81485cdf75e47ca')
CHECKPOINT='f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b'
CONFIG='660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd'
# Exact config identity comes from the already pinned public source inventory.
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def sources():
    return {name:P/name for name in ('prepare_gamma_fixtures.py','test_gamma_fixtures.py','site_metadata.json','source_inventory.json','inputs/config.json','inputs/learned_norm_bounds.json')}
@contextmanager
def no_modules():
    import torch
    original=torch.nn.Module.__init__;primary=None
    def deny(*args,**kwargs):raise RuntimeError('model construction forbidden in gamma-only extraction')
    try:
        torch.nn.Module.__init__=deny;yield
    except BaseException as error:primary=error;raise
    finally:
        try:torch.nn.Module.__init__=original
        except BaseException as error:
            if primary is None:raise
            primary.add_note('module guard restoration: '+type(error).__name__)
def validate_gamma(site,value,meta):
    value=np.asarray(value)
    if value.dtype!=np.dtype('float64') or value.shape!=(meta['group_width'],) or not np.isfinite(value).all():raise ValueError('gamma shape/dtype differs: '+site)
    if hashlib.sha256(value.astype('<f8',copy=False).tobytes()).hexdigest()!=meta['gamma_f64le_sha256']:raise ValueError('gamma bytes differ: '+site)
    return np.array(value,copy=True)
def extract(checkpoint,metadata):
    import torch
    from safetensors import safe_open
    values={}
    with no_modules(),safe_open(checkpoint,framework='pt',device='cpu') as reader:
        for site,row in sorted(metadata.items()):
            tensor=reader.get_tensor(row['learned_weight_tensor'])
            values[site]=validate_gamma(site,tensor.to(torch.float64).numpy(),row)
    return values
def collect():
    out=P/'gamma-preparation-v1';out.mkdir(exist_ok=False);report=dict(passed=False,scope='Public113 gamma vectors only; no model/data/HE execution.',model_forward=False);start=time.monotonic();primary=None;paths=sources()
    def write(): (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    try:
        write();before={n:sha(p) for n,p in paths.items()};report['source_hashes_before']=before
        with tarfile.open(out/'sources.tar.gz','w:gz') as t:
            for name,path in paths.items():
                raw=path.read_bytes();m=tarfile.TarInfo(name);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
        assets={n:sha(MODEL/n) for n in ('config.json','model.safetensors')};report['asset_sha256_before']=assets
        expected_config=CONFIG
        if sha(P/'site_metadata.json')!='19155627d1af13c3f68facdbf27e7b700b52f38072fea754d287c2791da3e4b8':raise ValueError('fixed public site inventory differs')
        if assets!=dict(**{'config.json':expected_config,'model.safetensors':CHECKPOINT}):raise ValueError('pinned model assets differ before tensor reads')
        metadata=json.loads((P/'site_metadata.json').read_bytes())
        if len(metadata)!=113:raise ValueError('all113 sites required')
        values=extract(MODEL/'model.safetensors',metadata)
        np.savez(P/'client_gamma_fixtures.npz',**values)
        report.update(fixture_npz_sha256=sha(P/'client_gamma_fixtures.npz'),arrays={n:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.astype('<f8',copy=False).tobytes()).hexdigest()) for n,v in values.items()},operation_complete=True)
    except BaseException as error:primary=error;report['error']=dict(type=type(error).__name__,message=str(error))
    finally:
        try:
            report['source_hashes_after']={n:sha(p) for n,p in paths.items()};report['asset_sha256_after']={n:sha(MODEL/n) for n in ('config.json','model.safetensors')}
            with tarfile.open(out/'sources.tar.gz') as t:
                ms=t.getmembers();report['archive_verified']=(len(ms)==len({m.name for m in ms})==len(report['source_hashes_before']) and all(m.isfile() for m in ms) and {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in ms}==report['source_hashes_before'])
            report['source_archive_sha256']=sha(out/'sources.tar.gz')
            report['passed']=bool(primary is None and report.get('operation_complete') and report['archive_verified'] and report['source_hashes_before']==report['source_hashes_after'] and report['asset_sha256_before']==report['asset_sha256_after'])
            report['elapsed_s']=time.monotonic()-start;write()
        except BaseException as error:
            if primary is not None:primary.add_note('final readback/publication: '+repr(error))
            else:raise
    if primary is not None:raise primary
    if not report['passed']:raise RuntimeError('gamma fixture provenance failed')
    print(json.dumps(dict(passed=True,sites=len(report['arrays']),elapsed_s=report['elapsed_s'],fixture_npz_sha256=report['fixture_npz_sha256'])))
if __name__=='__main__':collect()
