"""Offline exact-source and aggregate consistency verification; no model load."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,tarfile
ROOT=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def value(x):
    v=F(*map(int,x['fraction']))
    assert F(x['lower_binary64'])<=v<=F(x['upper_binary64'])
    return v

def main():
    for name,digest in json.loads((ROOT/'artifact_sha256.json').read_text()).items():assert sha(ROOT/name)==digest,name
    report=json.loads((ROOT/'public-v1/report.json').read_text());pre=json.loads((ROOT/'source-bound-v1/report.json').read_text())
    assert report['passed'] and pre['passed'] and report['complete']
    before=report['source_hashes_before']
    assert before==report['source_hashes_after']==pre['source_hashes_before']==pre['source_hashes_after'] and len(before)==8
    assert sha(ROOT/'public-v1/sources.tar.gz')==report['source_archive_sha256']
    with tarfile.open(ROOT/'public-v1/sources.tar.gz') as t:
        members=t.getmembers();assert len(members)==len({x.name for x in members})==len(before)
        assert all(m.isfile() and not m.name.startswith('/') and '..' not in Path(m.name).parts for m in members)
        data={m.name:t.extractfile(m).read() for m in members}
    assert {n:hashlib.sha256(b).hexdigest() for n,b in data.items()}==before
    pins=json.loads(data['inputs/model_pins.json']);outer=json.loads(data['inputs/outer_domains.json'])
    assert report['asset_sha256_before']==report['asset_sha256_after']==pins['asset_sha256'] and len(pins['asset_sha256'])==7
    assert report['tensors_read']==len(report['tensor_records'])==57
    for name,record in report['tensor_records'].items():assert record==outer['tensor_records'][name]
    e=report['embedding'];assert e['rows']==151936 and e['width']==1024
    lo=value(e['sum_squares_minimum']);hi=value(e['sum_squares_maximum']);eps=value(e['configured_binary64_epsilon']);eps32=value(e['binary32_epsilon'])
    lower=value(e['denominator_interval']['lower']);upper=value(e['denominator_interval']['upper'])
    assert lower==lo/1024+eps and upper==hi/1024+eps and 0<eps<lower<upper
    assert value(e['exact_stored_squares_with_binary32_epsilon_interval']['lower'])==lo/1024+eps32
    assert value(e['exact_stored_squares_with_binary32_epsilon_interval']['upper'])==hi/1024+eps32
    assert value(e['epsilon_binary32_minus_configured'])==eps32-eps
    prior=F(*map(int,outer['sites']['0.input.inverse_sqrt']['upper_fraction']))
    assert prior==value(e['prior_coordinate_upper'])>=upper
    assert value(e['condition_ratio'])==upper/lower and value(e['prior_condition_ratio'])==prior/eps
    assert value(e['upper_tightening_ratio'])==prior/upper and value(e['lower_improvement_ratio'])==lower/eps
    checks=e['independent_fraction_rows'];assert [x['row'] for x in checks]==[i*151935//63 for i in range(64)]
    assert all(lo<=value(x['sum_squares'])<=hi for x in checks)
    assert 0<=e['minimum_row']<151936 and 0<=e['maximum_row']<151936
    score=report['post_qk_norm_score_bounds'];assert [x['layer'] for x in score]==list(range(28))
    for row in score:
        products=[]
        for kind in ('same_position_gamma_product','rope_pair_gamma_product','global_gamma_product'):
            p=value(row[kind]['gamma_product']);b=value(row[kind]['exact_native_score_abs_upper']);products.append(p)
            assert b*b>=128*p*p
            assert value(row[kind]['K32_centered_abs_upper'])==2*F(31,32)*b
            assert value(row[kind]['conditional_uniform_norm_delta_1e6_score_abs_upper'])==b*(1+F(1,10**6))**2
        assert products==sorted(products)
        assert value(row['pair_bound_tightening_ratio'])==products[2]/products[1]
    assert not report['model_forward'] and not report['coefficient_selection'] and not report['model_inputs_or_tokens_read'] and not report['HE_or_backend_initialized']
    assert report['no_backend_or_model_imported'] and not e['native_float32_reduction_equivalence_claimed']
    print(json.dumps(dict(passed=True,source_files=8,assets=7,tensor_identities=57,reported_exact_rows=151936,independent_selected_rows=64,score_bound_layers=28,scope='Offline archive and rational aggregate consistency. All-row tensor arithmetic is source-bound run evidence; this verifier does not reread weights.')))
if __name__=='__main__':main()
