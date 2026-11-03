"""Public-only output31/32 and probability-refresh schedule follow-up."""
from pathlib import Path
import datetime,hashlib,io,json,math,os,tarfile,time
import baseline_plan as base
LAB=Path(__file__).resolve().parent

def main():
    started=time.perf_counter()
    paths=[Path(__file__),LAB/'baseline_plan.py',LAB/'source_origin.json',*sorted((LAB/'inputs').glob('*')),
      *sorted((LAB/'checkout/jaxite_word').glob('*.py')),*sorted((LAB/'checkout/cadathlon/common').glob('*.py'))]
    paths=[p for p in paths if p.is_file()]
    before={str(p.relative_to(LAB)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with tarfile.open(LAB/'sources.tar.gz','w:gz') as archive:
        for p in paths:
            data=p.read_bytes();info=tarfile.TarInfo(str(p.relative_to(LAB)));info.size=len(data);archive.addfile(info,io.BytesIO(data))
    schedule=base.PublicSchedule(json.loads((LAB/'inputs/baseline_manifest.json').read_text())['power_coefficients_in_t_2x'])
    direct=base.key_inventory();dense=set(direct['groups']['dense_QKV_attention_output_FFN'])
    chained_dense={1,64,61760,64064}
    chained_graph=set().union(*(set(v) for k,v in direct['groups'].items() if k!='dense_QKV_attention_output_FFN'))|chained_dense
    specs=[('comparison53_p60_d14',53,60,14),('depth55_p60_d28',55,60,28),('depth56_p58_d20',56,58,20),
        ('depth56_p58_d28',56,58,28),('depth56_p56_d14',56,56,14),('rejected57_p58_d29',57,58,29)]
    profiles=[]
    for spec in specs:
        p=base.profile(*spec,direct)
        if not p['accepted']:profiles.append(p);continue
        reduced=chained_graph|set(p['refresh']['bootstrap_rotations']);pair=p['resources']['rotation_pair_bytes']
        p['prospective_chained_resources']={'graph_rotation_count':len(chained_graph),'dense_linear_keys':sorted(chained_dense),
            'total_graph_bootstrap_keys':len(reduced),'graph_bootstrap_keys':sorted(reduced),'raw_key_GiB':len(reduced)*pair/2**30,
            'raw_key_bytes':len(reduced)*pair,'scope':'Prospective ideal-only dense chained/Horner key contract; no CKKS noise or same-ring memory proof. Generic sparse matrices require actual gaps.'}
        rows=[base.solve(p,K,schedule,allow_denominator_refresh=False) for K in range(1,129)]
        p['probability_only_schedules']=rows
        feasible=[r for r in rows if r['feasible']]
        p['summary']={'feasible_K':[r['K'] for r in feasible],'infeasible_K':[r['K'] for r in rows if not r['feasible']],
                      'K15':rows[14],'K128':rows[127]}
        for row in feasible:
            assert all(event['op']!='refresh_denominator' for rnd in row['normalized_rounds'] for event in rnd['events'])
            assert all(rnd['output_level']>=2 for rnd in row['normalized_rounds'])
        profiles.append(p)
    observations=[]
    for K in (1,15,32,128):
        for j in range(5):
            c=schedule.stage(K,j);ratio=c['ideal_ratio_upper'];mass_lower=1-1e-12;eps=.001
            lower_p=mass_lower/(1+(K-1)*ratio)
            observations.append({'K':K,'stage':j,'existing_a':c['a'],'existing_b':c['b'],
                'ideal_ratio_upper':ratio,'ideal_positive_component_lower':lower_p,
                'illustrative_refresh_abs_error':eps,'epsilon_below_ideal_component_lower':eps<lower_p,
                'mass_only_square_lower':max(0,mass_lower-K*eps)**2/K,'mass_only_square_upper':(1+math.sqrt(K)*eps)**2,
                'inside_broad_half_over_K_to2_before_square_reduction_error':max(0,mass_lower-K*eps)**2/K>=.5/K and (1+math.sqrt(K)*eps)**2<=2,
                'broad_domain_is_not_the_existing_tighter_stage_contract':True})
    after={name:hashlib.sha256((LAB/name).read_bytes()).hexdigest() for name in before}
    report={'scope':__doc__,'passed':before==after and not base.FORBIDDEN,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_hashes_before':before,'source_hashes_after':after,'source_files_unchanged':before==after,
        'source_archive_sha256':hashlib.sha256((LAB/'sources.tar.gz').read_bytes()).hexdigest(),
        'forbidden_calls':base.FORBIDDEN,'profiles':profiles,'direct_dense_key_count':len(dense),
        'direct_graph_count':direct['graph_rotation_count'],'prospective_chained_dense_keys':sorted(chained_dense),
        'prospective_chained_graph_count':len(chained_graph),'probability_perturbation_conditions':observations,
        'scope_limits':['The earlier four-profile/denominator-refresh report remains unchanged; this scheduler allows probability refresh only before whole rounds.',
            'Output31 permits mask1+square1+anchor1+13-update inverse25+correlated product1=29levels, leaving2. The static inverse remains unrefreshed.',
            'No denominator refresh preserves the ideal same-squares relation. Refreshed probability positivity, mass and ratio assumptions may still fail.',
            'Per-slot1e-3 is an illustrative uniform-error assumption, not a proved guarantee. A component threshold does not certify future ciphertexts.',
            'Broad[0.5/K,2] mass-only examples are a separate possible certificate; tighter stage domains/counts are not changed or declared noise-robust. Actual masked-square and reduction error still require bounds.',
            'New-chain bootstrap error, bounds16/64 and same-ring row components require numerical proofs. Security and available levels alone are insufficient.',
            'Prospective four-key dense linears have ideal tests only. Chained rotations change keyswitch noise and live-set behavior; no promotion or HBM fit is implied.',
            'The fixed first-layer domains fail fullMRPC native coverage, and newer training-only candidates require other degrees/round counts. Costs here remain conditional on the unchanged layer0 five-round graph.',
            'Raw key bytes exclude keygen temporaries, formatted controls, NTT/FFT tables, compiler, allocator and concurrent ciphertexts. No TPU peak or performance measurement.'],
        'elapsed_s':time.perf_counter()-started,'cpu_affinity':sorted(os.sched_getaffinity(0))}
    (LAB/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':report['passed'],'source_count':len(before),'profiles':[
        {k:p.get(k) for k in ('name','accepted','Q_count','P_count','actual_partitions','log2_QP','error')}|({'output':p['refresh']['output_level'],'direct_GiB':p['resources']['raw_rotation_store_GiB'],'prospective_keys':p['prospective_chained_resources']['total_graph_bootstrap_keys'],'prospective_GiB':p['prospective_chained_resources']['raw_key_GiB'],'feasible_K_count':len(p['summary']['feasible_K']),'K128_refreshes':p['summary']['K128'].get('normalization_refreshes'),'K128_output':p['summary']['K128'].get('normalization_output_level')} if p['accepted'] else {}) for p in profiles]},indent=2))

if __name__=='__main__':main()
