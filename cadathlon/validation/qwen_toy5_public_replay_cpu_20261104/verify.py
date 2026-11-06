"""Portable public bytes/metadata verification only; never imports replay code."""
import argparse,ast,hashlib,json,math,tarfile
from pathlib import Path
P=Path(__file__).resolve().parent
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def read(name):return (P/name).read_bytes()
def doc(name):return json.loads(read(name))
def main(current=False):
    expected=doc('artifact_hashes.json');whitelist=doc('whitelist.json')
    actual={p.relative_to(P).as_posix()for p in P.rglob('*')if p.is_file()}
    assert not any(p.is_symlink()for p in P.rglob('*'))
    assert actual==set(whitelist['files'])==set(expected)|{'artifact_hashes.json'}
    assert {n:sha(read(n))for n in expected}==expected
    origins=doc('copy_origins.json')
    for n,row in origins.items():assert len(read(n))==row['bytes']and sha(read(n))==row['sha256']
    sources=doc('sources.json');assert len(sources)==27
    assert sha(read('sources.json'))=='ecd56b237be287c8357788880a29f47426e9dc387487cd61c03b389f30783cba'
    assert sha(read('sources.tar.gz'))=='c2e21cf998dd2cd0f163b19bda0ac2751d563b383ddffc3db6774649d3a57a02'
    with tarfile.open(P/'sources.tar.gz')as t:
        members=t.getmembers();assert len(members)==len({m.name for m in members})==27
        assert all(m.isfile()and not Path(m.name).is_absolute()and '..'not in Path(m.name).parts for m in members)
        bodies={m.name:t.extractfile(m).read()for m in members}
    assert {n:sha(raw)for n,raw in bodies.items()}==sources
    for n in origins:
        if n.startswith('source-readable/'):assert read(n)==bodies[n.removeprefix('source-readable/')]
    parent=json.loads(bodies['parent199_manifest.json'])['source_hashes'];refs=json.loads(bodies['source_refs.json'])['selected'];assert len(parent)==199 and len(refs)==14
    for n,row in refs.items():assert sha(bodies[n])==row['sha256']==parent[row['parent199_member']]
    selections=json.loads(bodies['ast_selection.json'])['definitions'];selected=0
    for n,symbols in selections.items():
        tree=ast.parse(bodies[n])
        for symbol,digest in symbols.items():
            nodes=tree.body
            for part in symbol.split('.'):
                found=[node for node in nodes if isinstance(node,(ast.ClassDef,ast.FunctionDef))and node.name==part];assert len(found)==1
                node=found[0];nodes=node.body
            assert sha(ast.dump(node,include_attributes=False).encode())==digest;selected+=1
    assert selected==41
    report=doc('report.json');d=report['diagnostic'];terminal=doc('root-review/terminal-review.json');grant=doc('root-review/grant.json');launch=doc('root-review/launch.json')
    assert sha(read('report.json'))=='1207f2cd2328ddfc5a7e06f622579d956968382ffd581d59f769ef8909168cdd'
    assert sha(read('root-review/terminal-review.json'))=='680be72aad00bcd854bb31b3431345871a1954a4cabd4ca6aaa8ad06181d318f'
    assert report['passed']is True and report['HE']is report['pretrained_model']is report['key_generation']is False
    assert report['source_before']==report['source_after']==sources
    assert report['source_manifest_sha256']==sha(read('sources.json'))
    assert report['grant_sha256']==sha(read('root-review/grant.json'))==launch['grant_sha256']
    assert grant['source_manifest_sha256']==report['source_manifest_sha256'] and grant['entry_sha256']==sources['entry.py']
    assert report['backend_modules']==report['audit_denials']==[]and report['affinity']==[108,109,110,111]
    runtime=json.loads(bodies['runtime_pins.json']);assert len(runtime['sources'])==4
    assert report['runtime_sources_before']==report['runtime_sources_after']==runtime['sources']
    assert report['runtime_origins']=={n:{'file':p,'spec':p}for n,p in runtime['required_origins'].items()}
    assert d['complete']is d['dense_complete']is d['native_fixture_matches']is d['literal_matches_independent_dense_at_1e_10']is True
    assert d['physical_slots']==65536 and d['replicas']==64 and d['packed_period']==1024
    compared=[r for r in d['trace']if r['independent_dense_max_error']is not None]
    assert len(compared)==d['compared_semantic_records']==140 and not d['unpaired_dense_records']
    assert max(r['independent_dense_max_error']for r in compared)==d['maximum_semantic_absolute_error']==terminal['maximum_semantic_absolute_error']
    assert max(r['periodicity_max_error']for r in d['trace'])==d['maximum_periodicity_absolute_error']==0
    assert len(d['literal_domain_checks'])==len(d['dense_domain_checks'])==22
    for rows in (d['literal_domain_checks'],d['dense_domain_checks']):
        assert all(r['passed']is True and r['domain'][0]<=r['minimum']<=r['maximum']<=r['domain'][1]for r in rows)
    margin=min(min(r['minimum']-r['domain'][0],r['domain'][1]-r['maximum'])for r in d['literal_domain_checks'])
    assert margin==terminal['minimum_ideal_domain_margin']
    R=len(d['refresh_events']);assert R==len(d['refresh_magnitude_checks'])==19
    assert all(r['passed']is True and r['magnitude']<=r['public_bound']for r in d['refresh_magnitude_checks'])
    assert len(d['minimal_checkpoints'])==9+2*R==47 and d['schedule']['formula_holds']is True
    assert d['schedule']['observed_count_used_to_construct_schedule']is False
    assert [r['index']for r in d['minimal_checkpoints']]==sorted({r['index']for r in d['minimal_checkpoints']})
    assert all(d['trace'][r['index']]==r for r in d['minimal_checkpoints'])
    assert d['final_metadata']=={'level':5,'nsd':1,'scale':6.89940115708969e16}
    m=d['fixed_original_numeric_metrics'];assert d['fixed_original_numeric_gate']is True and m==terminal['clear_metrics']
    assert m['nrmse']<=.10 and m['max_abs_error']<=.20 and m['nrmse']>.01
    assert terminal['session']==92362 and terminal['terminal_tool']=='fac5a0'and terminal['exit_code']==0 and terminal['normal_zero_reaped']is True
    assert terminal['pid']==terminal['pgid']==report['pid']==report['pgid']==86804 and terminal['pid_group_absent']is True
    assert terminal['report_sha256']==sha(read('report.json'))and terminal['source_archive_sha256']==sha(read('sources.tar.gz'))
    assert terminal['independent_live_observation']is False and doc('root-review/live-observation.json')['observed_live']is False
    assert terminal['late_live_observation_sha256']==sha(read('root-review/live-observation.json'))
    assert terminal['actual_HE_pass']is terminal['future_HE_execution_authorized']is terminal['first_encrypted_corruption_identified']is False
    negative=doc('negative/summary.json');assert negative['case_passed']is False and negative['full_sweep_terminal_admitted']is False
    assert doc('negative/stdout.json')['error_type']=='ApproximationErrorTooHigh'
    if current:
        root=Path(doc('source-handoff.json')['source_root'])
        assert {n:sha((root/n).read_bytes())for n in sources}==sources
        assert {p:sha(Path(p).read_bytes())for p in runtime['sources']}==runtime['sources']
        for n,row in origins.items():assert sha(Path(row['path']).read_bytes())==row['sha256']
    print(json.dumps({'passed':True,'indexed_files':len(expected),'closed_files':len(actual),'sources':27,'semantic_pairs':140,'domains':22,'refreshes':19,'checkpoints':47,'current_public_checks':current,'HE_pass':False,'replay_rerun':False},sort_keys=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--current',action='store_true');main(p.parse_args().current)
