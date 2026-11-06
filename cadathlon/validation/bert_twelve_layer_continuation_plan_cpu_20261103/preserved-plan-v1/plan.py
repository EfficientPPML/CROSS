"""Fixed-v2 twelve-layer continuation levels, public bounds and implementation gaps.

No model forward, CKKS context, key generation, encryption or refresh is run.
Public weight bytes are read only to obtain the actual linear key inventory.
"""
from pathlib import Path
from types import SimpleNamespace
from typing import Optional,Tuple
from numbers import Integral,Real
import ast,hashlib,io,json,math,os,sys,tarfile,time
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'checkout/cadathlon/common'))
from encrypted_polynomial import EncryptedPolynomialEvaluator,PolynomialApproximation
from encrypted_bert_attention_tile import BertAttentionTile
from encrypted_bert_v2_row_norm import V2RowLayerNorm
from encrypted_score_reduction import ScoreReductionLayout
from encrypted_chained_streaming_linear import linear_rotation_plan
MANIFEST_SHA='4e2277beea7189aa1d77b168c500c8a175f6c630afa520fcc6b38cb5a179cff9'
CHECKPOINT=Path('/home/jianming_gatech/cross_validation_assets/bert/f150c1d609d1e50dd5e2e5408661cfac8339277c/model.safetensors')
CHECKPOINT_SHA='e1f8085ef673c9eb8a1a93a84270742e8aa0ccbceadef01eef15e26c44b46034'
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def norm_plan(rule):
    p=rule['polynomial'];c=tuple(p['coefficients']);degree=p['degree']
    assert hashlib.sha256(np.asarray(c,dtype='<f8').tobytes()).hexdigest()==p['coefficient_f64le_sha256']
    assert degree==len(c)-1 and rule['method'] in ('power','balanced_chebyshev')
    strategy='power' if rule['method']=='power' else 'chebyshev'
    approximation=PolynomialApproximation('inverse_sqrt',tuple(p['domain']),c)
    depth=EncryptedPolynomialEvaluator.polynomial_depth(SimpleNamespace(strategy=strategy),approximation)
    formula=2+(degree-1).bit_length() if strategy=='power' else 2+2*(degree-1).bit_length()
    assert depth==formula
    # Diagnostic only for balanced sites: these huge monomial coefficients
    # must NOT become the evaluation path selected by this public plan.
    power=np.polynomial.chebyshev.cheb2poly(c);l1=float(np.abs(power).sum())
    if strategy=='power':assert len(power)<=33 and l1<=1e8
    return dict(method=rule['method'],degree=degree,domain=p['domain'],coefficient_sha256=p['coefficient_f64le_sha256'],
        inverse_depth=depth,row_norm_depth=depth+5,power_coefficient_l1_diagnostic=l1,
        selected_power_conditioning_guard_passes=(len(power)<=33 and l1<=1e8) if strategy=='power' else None,
        conversion_of_balanced_to_power_allowed=False,
        precision_obligation='The library conditioning guard is not a CKKS precision certificate; high-L1 selected power sites need actual-ring tests.')

def one_layer(layer,K,manifest,norms,gelu_bound,*,first_layer_fresh):
    s=manifest['softmax'][str(layer)]['squarings'];n=manifest['normalization_reciprocal_by_public_K'][str(K)]['iterations']
    rounds=[];events=[];reserve=2;R=25;round_depth=n+5
    if first_layer_fresh:
        assert layer==0
        attention_input=49;projection=48;scores=45;mean=43;t=42;initial=23
    else:
        attention_input=25;projection=24;scores=21;mean=19;t=18;initial=6
        events.append(dict(stage='input row tile',bound=128,output_level=R,per_query_tile=True))
        events.append(dict(stage='normalized_t before base exp',bound=2,input_level=t,output_level=R,per_query_tile=True))
    # Exact existing static6-update initial inverse: tail from already mapped
    #t costs power8(3)+sum1 + denominator2 +inverse11 +product1+mask1 =19.
    assert manifest['initial_reciprocal']['iterations']==6
    assert initial==(t-19 if first_layer_fresh else R-19)
    level=initial
    for index in range(s):
        before=level;refresh=level<round_depth+reserve
        if refresh:
            assert level>=2;level=R
            events.append(dict(stage=f'normalized round{index+1}',bound=2,input_level=before,output_level=R,per_query_tile=True))
        level-=round_depth;assert level>=reserve
        rounds.append(dict(round=index+1,iterations=n,depth=round_depth,input_level=before,refresh_before=refresh,output_level=level))
    # The complete attention needs three PV levels, one output dense, the
    #selected norm and two reserved levels. Every frozen schedule ends below
    #this capacity, so refresh and its real final mask are explicit.
    required=3+1+norms[f'{layer}.attention']['row_norm_depth']+reserve
    assert level<required and level>=2
    events.append(dict(stage='final probability continuation',bound=2,input_level=level,output_level=R,per_query_tile=True))
    probability=R-1;attended=min(probability,projection)-3;output_dense=attended-1
    attention_out=output_dense-norms[f'{layer}.attention']['row_norm_depth'];assert attention_out>=2
    events.append(dict(stage='attention norm output before FFN',bound=128,input_level=attention_out,output_level=R,per_query_tile=True))
    gelu_depth=2+2*(manifest['gelu'][str(layer)]['degree']-1).bit_length()
    middle=R-1;activated=middle-gelu_depth;assert activated>=2
    events.append(dict(stage='GELU output before output dense',bound=gelu_bound,input_level=activated,output_level=R,per_query_tile=True))
    mlp_out=R-1-norms[f'{layer}.mlp']['row_norm_depth'];assert mlp_out>=2
    return dict(layer=layer,K=K,real_query_tiles=(K+15)//16,first_layer_fresh=first_layer_fresh,
        centered_domain=manifest['softmax'][str(layer)]['centered_domain'],squarings=s,
        mapped_t_divisor=2**(s-1),t_refresh_public_bound=None if first_layer_fresh else 2,
        initial_static_inverse_updates=6,initial_tail_depth=19,
        levels=dict(attention_input=attention_input,QKV=projection,QK=scores,score_mean=mean,normalized_t=t,
            initial_probabilities=initial,final_probabilities_after_refresh_mask=probability,PV=attended,attention_dense=output_dense,
            attention_norm=attention_out,FFN_input_refresh=R,intermediate_dense=middle,GELU=activated,
            GELU_refresh=R,output_dense=R-1,MLP_norm=mlp_out),rounds=rounds,
        refresh_events=events,refresh_calls_per_query_tile=len(events),refresh_calls_all_real_tiles=len(events)*((K+15)//16),
        final_probability_mask_levels=1,numerical_readiness=False)

def linear_inventory():
    core=ast.parse((HERE/'checkout/jaxite_word/bsgs.py').read_text())
    fn=next(n for n in core.body if isinstance(n,ast.FunctionDef) and n.name=='compute_bsgs_params')
    tree=ast.parse((HERE/'checkout/cadathlon/common/encrypted_packed_qwen.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MatrixDiagonals')
    ns=dict(np=np,math=math,Integral=Integral,Real=Real,Optional=Optional,Tuple=Tuple)
    exec(compile(ast.Module(body=[fn,cls],type_ignores=[]),'public_matrix_inventory','exec'),ns)
    assert sha(CHECKPOINT)==CHECKPOINT_SHA
    rows={};keys=set()
    with CHECKPOINT.open('rb') as stream:
        length=int.from_bytes(stream.read(8),'little');header=json.loads(stream.read(length));base=8+length
        for layer in range(12):
            for suffix in ('attention.self.query','attention.self.key','attention.self.value','attention.output.dense','intermediate.dense','output.dense'):
                name=f'bert.encoder.layer.{layer}.{suffix}.weight';desc=header[name];assert desc['dtype']=='F32'
                start,end=desc['data_offsets'];stream.seek(base+start);raw=stream.read(end-start)
                a=np.frombuffer(raw,dtype='<f4').reshape(desc['shape']);source=ns['MatrixDiagonals'](a,65536,4096)
                p=linear_rotation_plan(source,strategy='chained');keys.update(p.required_keys)
                rows[name]=dict(shape=desc['shape'],raw_f32_sha256=hashlib.sha256(raw).hexdigest(),
                    active_diagonals=len(source.active),n1=source.n1,rotation_calls=p.nonzero_rotation_calls,keys=list(p.required_keys))
    assert sha(CHECKPOINT)==CHECKPOINT_SHA
    return rows,keys

def run():
    raw=(HERE/'candidate_manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==MANIFEST_SHA
    m=json.loads(raw);certificate=json.loads((HERE/'gelu_refresh_certificates.json').read_text())
    assert certificate['passed'] and certificate['manifest_sha256']==MANIFEST_SHA and not certificate['encrypted_execution']
    norms={name:norm_plan(rule) for name,rule in m['norms'].items()}
    assert len(norms)==24
    bounds=[]
    for i in range(12):
        rule=m['gelu'][str(i)];cert=certificate['sites'][str(i)]
        assert cert['coefficient_sha256']==rule['polynomial']['coefficient_f64le_sha256']
        assert cert['degree']==rule['degree'] and cert['domain']==rule['polynomial']['domain']
        bounds.append(cert['refresh_bound'])
    assert bounds==[16,64,32,32,32,32,16,64,64,64,64,16]
    schedules=[one_layer(i,K,m,norms,bounds[i],first_layer_fresh=i==0) for K in range(1,129) for i in range(12)]
    for K in range(1,129):
        group=[row for row in schedules if row['K']==K]
        assert [row['layer'] for row in group]==list(range(12))
        for previous,next_layer in zip(group,group[1:]):assert previous['levels']['MLP_norm']>=2 and next_layer['levels']['attention_input']==25
    public_profile=json.loads((HERE/'p60_profile_and_refresh.json').read_text());p=public_profile['profile']
    reference=next(v for v in json.loads((HERE/'actual_profile_audit.json').read_text())['profiles'] if v['name']=='depth49_p60_d8')
    assert p['q']==reference['q'] and p['p']==reference['p']
    Q,P=p['q'],p['p'];alpha=(len(Q)+7)//8
    assert len(Q)==100 and len(P)==14 and math.prod(P)>=max(math.prod(Q[i:i+alpha]) for i in range(0,len(Q),alpha))
    assert math.prod(Q+P).bit_length()<=3523
    matrices,linear_keys=linear_inventory()
    graph_keys=set(BertAttentionTile.required_rotation_indices())|set(V2RowLayerNorm.required_rotation_indices())|set(ScoreReductionLayout.rotation_indices())|linear_keys
    refresh_keys=set(public_profile['refresh_plan']['rotations']);union=graph_keys|refresh_keys
    native=json.loads((HERE/'native_learned_norm_bounds.json').read_text())
    native_bert={name:row for name,row in native['normalizations'].items() if name.startswith('bert:')}
    assert max(row['exact_real_abs_output_upper'] for row in native_bert.values())<128
    pooler=dict(last_layer_MLP_output=norms['11.mlp'],input_level=24-norms['11.mlp']['row_norm_depth'],
        dense_output_level=23-norms['11.mlp']['row_norm_depth'],frozen_tanh_domain=m['tanh']['domain'],
        base_degree=m['tanh']['base_degree'],doublings=m['tanh']['doublings'],reciprocal_updates=m['tanh_reciprocal']['iterations'],
        note='Separate implementation required: fuse input/64 with fixed base-domain mapping; base power depth6, then six11-level static reciprocal/product rounds. Refreshes and final classifier must preserve actual scales; no final logits tested here.')
    return dict(passed=True,scope=__doc__,manifest_sha256=MANIFEST_SHA,encrypted_execution=False,model_forward=False,
        changed_coefficients=False,changed_evaluation_strategies=False,normalizations=norms,gelu_refresh_bounds=bounds,
        schedules=schedules,all1536_layer_keycount_paths_have_continuation_capacity=True,
        minimum_MLP_output_level=min(row['levels']['MLP_norm'] for row in schedules),
        K128_refresh_calls=sum(row['refresh_calls_all_real_tiles'] for row in schedules if row['K']==128),
        K1_refresh_calls=sum(row['refresh_calls_all_real_tiles'] for row in schedules if row['K']==1),
        public_linear_inventory=matrices,linear_keys=sorted(linear_keys),graph_keys=sorted(graph_keys),refresh_keys=sorted(refresh_keys),union_keys=sorted(union),
        raw_rotation_pair_bytes=reference['rotation_pair_bytes'],raw_union_rotation_bytes=len(union)*reference['rotation_pair_bytes'],
        resource_scope='Raw reusable key storage only; no peak-device/host-memory or performance claim. All layers reuse same ring/keys; runtime must stream/discard tiles and controls.',
        actual_QP_bit_length=math.prod(Q+P).bit_length(),P_coverage=True,canonical_scale=reference['canonical_scale'],
        pooler=pooler,native_BERT_output_bound_max=max(row['exact_real_abs_output_upper'] for row in native_bert.values()),
        native_bound_transfer_to_approximate_or_CKKS_proved=False,
        obligations=['Implement a new initial-softmax prefix/tail split after normalized_t, retaining fixed coefficients/static6 inverse. Layer-specific t=centered/2**(squarings-1).',
            'Generalize immutable contracts to all12 layers/24norm sites, keeping selected power versus balanced strategies. High-L1 selected power norms need common-ring precision evidence.',
            'Generalize balanced GELU to exact48/64/96/128/192coefficients and the certified per-layer16/32/64 refresh bounds; never convert high-degree balanced sites to monomials.',
            'Generalize norm recurrence: power inverse depth6/7; balanced inverse14/16; full row norms11/12/19/21. Current wrapper only supports layer0.',
            'For continuation consumeT=ceil(K/16) low-level outputs, refresh realtiles, derive empty key/value padding tiles by genuine ciphertext subtraction at refreshed level, and preserve original valid positions. Do not relabel them freshL49 schema inputs.',
            'B128 is justified here only for exact-real native learned normalization. Approximate inverse multiplier/domain and CKKS/refresh perturbation certificates remain separate.',
            'B2 normalizedt/probabilities and GELUbounds are conditional ideal contracts, not encrypted error bounds. No clipping, decoding or clientrefresh may replace the server arithmetic.',
            'Validate actual full65536independent-slot p60 refresh at all requiredbounds and repeated composition; sparseperiod128/p56 proofs do not transfer.',
            'Retainedresiduals must stay immutable throughrefresh/callbacks; actual noncanonical scales must followprimitive order, with real masks and no metadata reassignment.',
            'Implement pooler/frozenrationaltanh/classifier, strict full-model three-process/file-stage transport, real embedding-only client boundary, and final ranked100/200/408 gates.',
            'Public train3668/heldout408 approximation success is distinct from ciphertext execution, runtime/calibration and arbitrary-input domain certification.'])

if __name__=='__main__':
    start=time.monotonic();paths=list((HERE/'checkout').rglob('*.py'))+[HERE/n for n in ('plan.py','frozen_approximate.py','candidate_manifest.json','actual_profile_audit.json','p60_profile_and_refresh.json','native_learned_norm_bounds.json','gelu_refresh_certificates.json')]
    before={str(p.relative_to(HERE)):sha(p) for p in paths};result=run()
    if any(n.split('.')[0] in ('jax','jaxlib','jaxite_word','torch','transformers') for n in sys.modules):raise RuntimeError('backend/model unexpectedly imported')
    with tarfile.open(HERE/'sources.tar.gz','w:gz') as archive:
        for name in before:
            data=(HERE/name).read_bytes();member=tarfile.TarInfo(name);member.size=len(data);archive.addfile(member,io.BytesIO(data))
    after={str(p.relative_to(HERE)):sha(p) for p in paths};assert before==after
    with tarfile.open(HERE/'sources.tar.gz') as archive:
        assert {member.name:hashlib.sha256(archive.extractfile(member).read()).hexdigest() for member in archive.getmembers()}==before
    result.update(source_hashes_before=before,source_hashes_after=after,source_files_unchanged=True,source_archive_sha256=sha(HERE/'sources.tar.gz'),
        no_backend_or_model_imported=True,checkpoint_sha256=CHECKPOINT_SHA,elapsed_s=time.monotonic()-start,cpu_affinity=sorted(os.sched_getaffinity(0)))
    save(HERE/'report.json',result)
    print(json.dumps({name:result[name] for name in ('passed','minimum_MLP_output_level','K128_refresh_calls','K1_refresh_calls','raw_union_rotation_bytes','elapsed_s')}))
