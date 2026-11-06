from pathlib import Path
from fractions import Fraction
import argparse,hashlib,json,math,tarfile


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    r=json.loads((root/'run/report.json').read_text())
    assert r['passed'] is True and r['complete'] is True and r['source_files_unchanged'] and r['assets_unchanged']
    assert r['source_hashes_before']==r['source_hashes_after']
    assert sha(root/'run/sources.tar.gz')==r['source_archive_sha256']
    with tarfile.open(root/'run/sources.tar.gz') as t:
        ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==7
        assert all(m.isfile() and not m.name.startswith('/') and '..' not in Path(m.name).parts for m in ms)
        payload={m.name:t.extractfile(m).read() for m in ms}
    assert {n:hashlib.sha256(b).hexdigest() for n,b in payload.items()}==r['source_hashes_before']
    pins=json.loads(payload['model_pins.json']);assert r['asset_sha256_before']==r['asset_sha256_after']==pins['asset_sha256']
    assert r['sites_count']==len(r['sites'])==113 and r['tensors_read']==len(r['tensor_records'])==310
    assert r['no_backend_or_model_imported'] and not r['model_forward'] and not r['coefficient_selection']
    eps=Fraction(1e-6)
    for site in r['sites'].values():
        low=Fraction(*map(int,site['lower_fraction']));hi=Fraction(*map(int,site['upper_fraction']))
        assert low==eps and low>0 and hi>=low and Fraction(site['upper'])>=hi
        assert Fraction(site['condition_ratio_upper'])>=hi/low and site['positive_epsilon']
        for head in site.get('head_upper',[]):assert Fraction(head['upper'])>=Fraction(*map(int,head['fraction']))
    expected={f'{i}.{role}.inverse_sqrt' for i in range(28) for role in ('input','post','q','k')}|{'final.inverse_sqrt'}
    assert set(r['sites'])==expected
    prior=r['embedding_coordinate_upper']
    def variance(values):return sum((Fraction(v)**2 for v in values),Fraction())/len(values)+eps
    for i in range(28):
        row=r['layers'][str(i)];assert Fraction(*map(int,r['sites'][f'{i}.input.inverse_sqrt']['upper_fraction']))==variance(prior)
        assert Fraction(*map(int,r['sites'][f'{i}.post.inverse_sqrt']['upper_fraction']))==variance(row['post_coordinate_upper'])
        for a,b,c in zip(prior,row['attention_coordinate_upper'],row['post_coordinate_upper']):assert Fraction(c)>=Fraction(a)+Fraction(b)
        for a,b,c in zip(row['post_coordinate_upper'],row['mlp_coordinate_upper'],row['residual_coordinate_upper']):assert Fraction(c)>=Fraction(a)+Fraction(b)
        assert len(r['sites'][f'{i}.q.inverse_sqrt']['head_upper'])==16
        assert len(r['sites'][f'{i}.k.inverse_sqrt']['head_upper'])==8
        prior=row['residual_coordinate_upper']
    assert Fraction(*map(int,r['sites']['final.inverse_sqrt']['upper_fraction']))==variance(prior)
    terminal=json.loads((root/'terminal.json').read_text());assert terminal['session_id']==63482 and terminal['observed_exit_code']==0 and terminal['reaped']
    for n,expected in json.loads((root/'artifact_sha256.json').read_text()).items():assert sha(root/n)==expected
    return dict(passed=True,scope='Offline source/asset-pin/113-envelope arithmetic consistency; no rerun or tensor extraction.',source_count=7,sites=113,tensor_records=310)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path,nargs='?',default=Path(__file__).resolve().parent)
    print(json.dumps(verify(p.parse_args().root),indent=2))
