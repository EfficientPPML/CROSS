"""Fixed-v2 first-block level outline; public-only and no cryptographic execution."""
from pathlib import Path
import json,hashlib,math,tarfile,io
HERE=Path(__file__).resolve().parent
MANIFEST=Path('/tmp/bert-mrpc-train-calibration-20261103/candidate-v2/candidate_manifest.json')
PROFILE=Path('/tmp/bert-normalized-squaring-integration-plan-20261103/report.json')
LEARNED=Path('/tmp/learned-normalization-bounds-20261103/result-v2/report.json')
TRAIN=Path('/tmp/bert-mrpc-train-calibration-20261103/approximate-train3668-v2/report.json')
VALID=Path('/tmp/bert-mrpc-train-calibration-20261103/approximate-validation408-v2/report.json')

def inverse_depth(updates,mode):
    if mode=='integer-affine-static':return 2*updates-1
    if mode=='prospective-parallel-error':return 1 if updates==1 else updates+1
    raise ValueError(mode)

def solve(profile,manifest,K,mode):
    D=profile['depth'];R=profile['refresh']['output_level'];reserve=2
    updates=manifest['normalization_reciprocal_by_public_K'][str(K)]['iterations']
    inv=inverse_depth(updates,mode)
    # Q/K projection1, product1, forward score route9. Score mean2,
    # centered/domain map1, exp powers3+final mask1, mean-exp2,
    # initial inverse, probability product1 and public1/K1.
    initial=D-22-inverse_depth(manifest['initial_reciprocal']['iterations'],mode)
    states={initial:(0,[])}
    for round_index in range(manifest['softmax']['0']['squarings']):
        next_states={}
        for level,(cost,events) in states.items():
            for rp in (False,True):
                if rp and level<reserve:continue
                start=R if rp else level
                if start<3:continue
                square=start-2;denom=start-3
                for rd in (False,True):
                    if rd and denom<reserve:continue
                    den_after=R if rd else denom
                    result=min(square,den_after-inv)-1
                    if result<reserve:continue
                    count=cost+int(rp)+int(rd)
                    row=dict(round=round_index,input=level,start_after_optional_refresh=start,square=square,
                        denominator=denom,denominator_after_optional_refresh=den_after,inverse=den_after-inv,
                        output=result,refresh_probability=rp,refresh_denominator=rd,
                        probability_ideal_bound='conditional near1; separate encrypted margin required',
                        denominator_domain=manifest['normalization_reciprocal_by_public_K'][str(K)]['domain'])
                    if result not in next_states or count<next_states[result][0]:next_states[result]=(count,events+[row])
        if not next_states:return dict(K=K,feasible=False,failed_round=round_index,updates_per_round=[updates]*5,inverse_levels=inv,initial_probability_level=initial)
        states=next_states
    level,(count,events)=min(states.items(),key=lambda item:(item[1][0],-item[0]))
    continuation=[];extra=0
    if level<24:
        continuation.append(dict(op='refresh_final_probability',input=level,output=R,ideal_bound='conditional near1'));level=R;extra+=1
    for name,depth in [('inverse_score_route',9),('probability_times_saved_V',1),('output_linear_and_residual',1),('attention_LayerNorm',11)]:
        continuation.append(dict(op=name,input=level,output=level-depth));level-=depth
    if level<2:return dict(K=K,feasible=False,reason='continuation lacks refresh reserve')
    # Full FFN16 + MLP norm11 + next-block reserve2 needs29 levels.
    if level<29:
        continuation.append(dict(op='refresh_attention_norm',input=level,output=R,
            bound='128 is suggested by exact-native learned bound76.24955; actual public polynomial/noise envelope still required'))
        level=R;extra+=1
    continuation.append(dict(op='FFN64_and_residual',input=level,output=level-16));level-=16
    if level<13:
        if level<2:return dict(K=K,feasible=False,reason='FFN lacks refresh reserve')
        continuation.append(dict(op='refresh_FFN_residual',input=level,output=R,bound='unproved; cannot assume16 or128'))
        level=R;extra+=1
    continuation.append(dict(op='MLP_LayerNorm',input=level,output=level-11));level-=11
    return dict(K=K,feasible=level>=2,updates_per_round=[updates]*5,inverse_levels=inv,
        initial_probability_level=initial,rounds=events,normalization_refreshes=count,continuation=continuation,
        total_refreshes_per_query_tile=count+extra,output_level=level,
        scope='Level feasibility only. Gates, common-ring ports, bounds, correlation/error and actual scale planning are not established.')

def main():
    sources=[Path(__file__),MANIFEST,PROFILE,LEARNED,TRAIN,VALID]
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    m=json.loads(MANIFEST.read_text());profiles=json.loads(PROFILE.read_text())['profiles'];learned=json.loads(LEARNED.read_text())
    assert before[str(MANIFEST)]=='4e2277beea7189aa1d77b168c500c8a175f6c630afa520fcc6b38cb5a179cff9'
    rows=[]
    for p in profiles:
        q,pprimes=p['q'],p['p'];alpha=math.ceil(len(q)/p['dnum'])
        assert math.prod(q+pprimes).bit_length()<=3523 and math.prod(pprimes)>=max(math.prod(q[i:i+alpha]) for i in range(0,len(q),alpha))
        modes={}
        for mode in ('integer-affine-static','prospective-parallel-error'):
            schedules=[solve(p,m,K,mode) for K in range(1,129)]
            feasible=[x for x in schedules if x['feasible']]
            modes[mode]=dict(all_K_level_feasible=len(feasible)==128,failed_K=[x['K'] for x in schedules if not x['feasible']],
                K15=schedules[14],K128=schedules[-1],all_K=schedules,
                status='New public fixed-v2 level planning only; prospective mode is not implemented or measured.')
        rows.append(dict(name=p['name'],depth=p['depth'],scale_bits=p['scale_bits'],dnum=p['dnum'],Q_count=p['Q_count'],P_count=p['P_count'],
            log2_QP=p['log2_QP'],refresh_output=p['refresh']['output_level'],raw_direct_rotation_store_GiB=p['resources']['raw_rotation_store_GiB'],
            direct_graph_plus_refresh_keys=p['resources']['graph_plus_bootstrap_count'],top_ciphertext_bytes=p['resources']['top_ciphertext_bytes'],modes=modes))
    report=dict(scope=__doc__,passed=True,first_block_contracts={
        'score_centered_domain':m['softmax']['0']['centered_domain'],'normalized_squarings':5,'max_K_inverse_updates_each_round':13,
        'GELU':{'domain':m['gelu']['0']['domain'],'degree':64,'nonlinear_levels':14,'FFN_levels':16,'coefficient_sha256':m['gelu']['0']['polynomial']['coefficient_f64le_sha256']},
        'attention_norm':{'domain':m['norms']['0.attention']['domain'],'degree':12,'strategy':'power','row_levels':11},
        'MLP_norm':{'domain':m['norms']['0.mlp']['domain'],'degree':16,'strategy':'power','row_levels':11}},
        profiles=rows,exact_native_norm_bounds={k:v for k,v in learned['normalizations'].items() if 'bert:bert.encoder.layer.0.' in k},
        security_scope='Reuses actual ordered Q/P and secure preflight from the independently archived public profile report; rechecks integer QP bit length and P partition coverage. No new keygen/context/device call.',
        caveats=['The old per-stage inverse counts8/9/12/13/13 cannot replace frozen v2 counts13/13/13/13/13 forK128.',
            'C1 inverse values can reach2K=256 on the declared interval. Prior C32 inverse proof does not establish this frozen unscaled C1 circuit.',
            'Exact-real reciprocal reassociation may reduce depth to n+1, but requires its own scale/correlation and encrypted-error proof.',
            'Raw direct keys exclude all transient tables/compiler/formatted-key and simultaneous-ciphertext workspace; this is not a peak-memory claim.',
            'Frozen finite training/dev coverage does not certify future inputs or CKKS-shifted nonlinear domains.'],
        source_hashes_before=before)
    with tarfile.open(HERE/'sources.tar.gz','w:gz') as tar:
        for name in before:
            raw=Path(name).read_bytes();info=tarfile.TarInfo(name.lstrip('/'));info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
    with tarfile.open(HERE/'sources.tar.gz') as tar:
        assert len(tar.getmembers())==len(before)
        assert all(hashlib.sha256(tar.extractfile(x).read()).hexdigest()==before['/'+x.name] for x in tar.getmembers())
    report['source_hashes_after']={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in before}
    assert before==report['source_hashes_after'];report['source_files_unchanged']=True;report['source_archive_verified']=True
    report['source_archive_sha256']=hashlib.sha256((HERE/'sources.tar.gz').read_bytes()).hexdigest()
    (HERE/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for p in rows:
        print(p['name'],{mode:{'all_K':v['all_K_level_feasible'],'K128_refreshes':v['K128'].get('total_refreshes_per_query_tile'),'output':v['K128'].get('output_level')} for mode,v in p['modes'].items()})
if __name__=='__main__':main()
