"""Standalone readback/recomputation; no original /tmp paths or HE packages needed."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,tarfile

def verify(directory):
    directory=Path(directory);report=json.loads((directory/'report.json').read_text())
    before=report['source_hashes_before']
    if not report['passed'] or before!=report['source_hashes_after']:raise ValueError('unsuccessful or changed public plan')
    archive_path=directory/'sources.tar.gz'
    if hashlib.sha256(archive_path.read_bytes()).hexdigest()!=report['source_archive_sha256']:raise ValueError('source archive hash differs')
    with tarfile.open(archive_path) as archive:
        members=archive.getmembers()
        if len(members)!=len(before) or any(not m.isfile() for m in members):raise ValueError('archive closure differs')
        source={'/'+m.name:archive.extractfile(m).read() for m in members}
    if {k:hashlib.sha256(v).hexdigest() for k,v in source.items()}!=before:raise ValueError('archive source bytes differ')
    plan_path=directory/'plan.py';plan_name=next(k for k in before if k.endswith('/bert-first-block-v2-integration-plan-20261103/plan.py'))
    if hashlib.sha256(plan_path.read_bytes()).hexdigest()!=before[plan_name]:raise ValueError('standalone plan source differs')
    spec=importlib.util.spec_from_file_location('standalone_first_block_plan',plan_path);plan=importlib.util.module_from_spec(spec);spec.loader.exec_module(plan)
    manifest=json.loads(source[str(plan.MANIFEST)]);profiles=json.loads(source[str(plan.PROFILE)])['profiles']
    training=json.loads(source[str(plan.TRAIN)]);validation=json.loads(source[str(plan.VALID)])
    if not training['passed'] or not validation['passed'] or training['candidate_manifest_sha256']!=validation['candidate_manifest_sha256']:raise ValueError('candidate audit linkage differs')
    expected_contracts={'score_centered_domain':manifest['softmax']['0']['centered_domain'],
        'normalized_squarings':manifest['softmax']['0']['squarings'],
        'max_K_inverse_updates_each_round':manifest['normalization_reciprocal_by_public_K']['128']['iterations'],
        'GELU':{'domain':manifest['gelu']['0']['domain'],'degree':manifest['gelu']['0']['degree'],
            'nonlinear_levels':2+2*(manifest['gelu']['0']['degree']-1).bit_length(),
            'FFN_levels':4+2*(manifest['gelu']['0']['degree']-1).bit_length(),
            'coefficient_sha256':manifest['gelu']['0']['polynomial']['coefficient_f64le_sha256']}}
    for name,label in [('attention_norm','0.attention'),('MLP_norm','0.mlp')]:
        rule=manifest['norms'][label]
        expected_contracts[name]={'domain':rule['domain'],'degree':rule['degree'],'strategy':rule['method'],
            'row_levels':7+(rule['degree']-1).bit_length()}
    if report['first_block_contracts']!=expected_contracts:raise ValueError('fixed polynomial contracts differ')
    learned=json.loads(source[str(plan.LEARNED)])
    if report['exact_native_norm_bounds']!={k:v for k,v in learned['normalizations'].items() if 'bert:bert.encoder.layer.0.' in k}:
        raise ValueError('learned normalization bounds differ')
    checks=0
    for recorded,profile in zip(report['profiles'],profiles,strict=True):
        for name in ('name','depth','scale_bits','dnum','Q_count','P_count','log2_QP'):
            if recorded[name]!=profile[name]:raise ValueError('profile identity differs')
        for name,value in [('refresh_output',profile['refresh']['output_level']),
                           ('raw_direct_rotation_store_GiB',profile['resources']['raw_rotation_store_GiB']),
                           ('direct_graph_plus_refresh_keys',profile['resources']['graph_plus_bootstrap_count']),
                           ('top_ciphertext_bytes',profile['resources']['top_ciphertext_bytes'])]:
            if recorded[name]!=value:raise ValueError('profile resource metadata differs')
        for mode,data in recorded['modes'].items():
            expected=[plan.solve(profile,manifest,K,mode) for K in range(1,129)]
            if expected!=data['all_K']:raise ValueError('recomputed fixed-v2 schedule differs')
            if data['K15']!=expected[14] or data['K128']!=expected[-1]:raise ValueError('summary schedules differ')
            if data['failed_K']!=[x['K'] for x in expected if not x['feasible']]:raise ValueError('failed-key-count summary differs')
            checks+=len(expected)
    return dict(passed=True,exact_archived_sources=len(source),recomputed_schedules=checks,
        no_original_paths_required=True,source_archive_sha256=report['source_archive_sha256'])

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',nargs='?',type=Path,default=Path(__file__).resolve().parent)
    print(json.dumps(verify(parser.parse_args().directory),indent=2))
