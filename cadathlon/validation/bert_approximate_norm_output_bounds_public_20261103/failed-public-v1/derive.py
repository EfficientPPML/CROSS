"""Conditional ideal-circuit normalization bounds for all24 fixed BERT-v2 sites.

Uses only pinned public polynomial coefficients and learned affine tensors.
Variance-domain membership is assumed. Encoding, CKKS and refresh error are
excluded; no model, tokenizer, input rows, ciphertexts or keygen are executed.
"""
from pathlib import Path
from fractions import Fraction as F
import hashlib,io,json,math,os,sys,tarfile,time
import numpy as np
from exact_bounds import polynomial_contract,affine_bound,ceil_float
HERE=Path(__file__).resolve().parent
MANIFEST_SHA='4e2277beea7189aa1d77b168c500c8a175f6c630afa520fcc6b38cb5a179cff9'
CHECKPOINT=Path('/home/jianming_gatech/cross_validation_assets/bert/f150c1d609d1e50dd5e2e5408661cfac8339277c/model.safetensors')
CHECKPOINT_SHA='e1f8085ef673c9eb8a1a93a84270742e8aa0ccbceadef01eef15e26c44b46034'
def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def fraction(x):x=F(x);return [str(x.numerator),str(x.denominator)]
def record(x):return dict(exact_fraction=fraction(x),upper_binary64=ceil_float(x))
def write(path,row):path.write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
def read_affines():
    assert digest(CHECKPOINT)==CHECKPOINT_SHA
    values={};pins={}
    with CHECKPOINT.open('rb') as stream:
        length=int.from_bytes(stream.read(8),'little');header=json.loads(stream.read(length));base=length+8
        for layer in range(12):
            for side,suffix in [('attention','attention.output.LayerNorm'),('mlp','output.LayerNorm')]:
                key=f'{layer}.{side}';vectors=[];names={}
                for leaf in ('weight','bias'):
                    name=f'bert.encoder.layer.{layer}.{suffix}.{leaf}';desc=header[name]
                    assert desc['dtype']=='F32' and desc['shape']==[768]
                    a,b=desc['data_offsets'];stream.seek(base+a);raw=stream.read(b-a)
                    vectors.append(np.frombuffer(raw,dtype='<f4').astype(np.float64))
                    names[name]=hashlib.sha256(raw).hexdigest()
                values[key]=vectors;pins[key]=names
    assert digest(CHECKPOINT)==CHECKPOINT_SHA
    return values,pins

def main():
    out=HERE/'public-v1';out.mkdir(exist_ok=False);start=time.monotonic()
    paths=[*HERE.glob('*.py'),HERE/'candidate_manifest.json'];before={p.name:digest(p) for p in paths}
    assert digest(HERE/'candidate_manifest.json')==MANIFEST_SHA
    manifest=json.loads((HERE/'candidate_manifest.json').read_text());affines,pins=read_affines();sites={}
    for name,rule in manifest['norms'].items():
        contract=polynomial_contract(rule);bound=affine_bound(contract,*affines[name]);c=rule['polynomial']
        assert hashlib.sha256(np.asarray(c['coefficients'],dtype='<f8').tobytes()).hexdigest()==c['coefficient_f64le_sha256']
        sites[name]=dict(method=rule['method'],degree=c['degree'],domain=c['domain'],coefficient_sha256=c['coefficient_f64le_sha256'],
            actual_evaluated_coefficients=[float(v).hex() for v in contract['evaluated_coefficients']],
            actual_evaluated_coefficient_sha256=hashlib.sha256(contract['evaluated_coefficients'].astype('<f8').tobytes()).hexdigest(),
            rounded_map_A=record(contract['A']),rounded_map_B=record(contract['B']),actual_mapped_endpoints=[fraction(contract['tlo']),fraction(contract['thi'])],
            extension_M=record(contract['M']),exact_P_chebyshev=[fraction(x) for x in contract['P']],exact_H_chebyshev=[fraction(x) for x in contract['H']],
            squared_relative_envelope=record(contract['squared_relative_envelope']),inverse_energy_factor=record(contract['normalization_factor']),
            variance_weight=record(bound['alpha']),normalized_coordinate_factor=record(bound['coordinate_factor']),
            output_absolute_bound=record(bound['output_bound']),guarded_output_bound=record(bound['guarded_bound']),refresh_bound=bound['refresh_bound'],
            maximum_coordinate=bound['maximum_coordinate'],public_affine_tensor_sha256=pins[name],
            scope='Ideal rounded public coefficients and exact real arithmetic only; v=alpha*sum(centered²)+epsilon in declared domain, epsilon>0. No zero-sum centering premise, sign certificate, encoding/CKKS/refresh error or propagated-domain guarantee.')
    assert len(sites)==24
    report=dict(passed=True,scope=__doc__,manifest_sha256=MANIFEST_SHA,checkpoint_sha256=CHECKPOINT_SHA,numpy_version=np.__version__,sites=sites,
        maximum_squared_relative_envelope=max(r['squared_relative_envelope']['upper_binary64'] for r in sites.values()),
        maximum_output_absolute_bound=max(r['output_absolute_bound']['upper_binary64'] for r in sites.values()),
        maximum_guarded_output_bound=max(r['guarded_output_bound']['upper_binary64'] for r in sites.values()),
        all_sites_fit_B128_with_9_over_8_headroom=all(r['refresh_bound']<=128 for r in sites.values()),
        changed_coefficients=False,changed_evaluation_strategies=False,model_evaluated=False,HE_executed=False,
        actual_floating_trajectory_or_CKKS_error_certified=False,
        proof='H(t)=((t-B)/A)*P(t)^2-1 in exact Chebyshev coefficients. On actual t=A*v+B, |t|<=M, |H|<=sum_n|h_n|T_n(M). Since v=alpha*sum(u²)+epsilon with alpha=Fraction(float(1/768)) and epsilon>0, |u_i P(v)|<=sqrt((1+delta)/alpha). Apply each learned gamma_i/beta_i with outward roots and exact rational products.',
        source_hashes_before=before,cpu_affinity=sorted(os.sched_getaffinity(0)))
    if any(n.split('.')[0] in ('jax','jaxlib','jaxite_word','torch','transformers') for n in sys.modules):raise RuntimeError('backend/model unexpectedly imported')
    after={p.name:digest(p) for p in paths};assert before==after
    with tarfile.open(out/'sources.tar.gz','w:gz') as t:
        for p in paths:b=p.read_bytes();m=tarfile.TarInfo(p.name);m.size=len(b);t.addfile(m,io.BytesIO(b))
    with tarfile.open(out/'sources.tar.gz') as t:assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers()}==before
    report.update(source_hashes_after=after,source_files_unchanged=True,source_archive_sha256=digest(out/'sources.tar.gz'),elapsed_s=time.monotonic()-start)
    write(out/'report.json',report)
    print(json.dumps({k:report[k] for k in ('passed','maximum_squared_relative_envelope','maximum_output_absolute_bound','maximum_guarded_output_bound','all_sites_fit_B128_with_9_over_8_headroom','elapsed_s')}))
if __name__=='__main__':main()
