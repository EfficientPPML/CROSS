"""Read-only consistency audit; does not rerun encrypted arithmetic."""
import ast,hashlib,json,math,tarfile
from datetime import datetime,timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
def digest(data):return hashlib.sha256(data).hexdigest()
report=json.loads((HERE/'run/report.json').read_text())
preflight=json.loads((HERE/'preflight/report.json').read_text())
assert report['passed'] and report['source_files_unchanged'] and not report['cleanup_errors']
before=report['source_hashes_before'];after=report['source_hashes_after']
assert before==after==preflight['source_hashes_before']==preflight['source_hashes_after']
assert len(before)==95 and preflight['preflight_only']
archive_hashes={}
for group in ('preflight','run'):
    with tarfile.open(HERE/group/'sources.tar.gz') as archive:
        members=archive.getmembers()
        assert len(members)==len({m.name for m in members})==95
        assert all(m.isfile() for m in members)
        blobs={m.name:archive.extractfile(m).read() for m in members}
    assert set(blobs)==set(before)
    assert all(digest(data)==before[name] for name,data in blobs.items())
    archive_hashes[group]=digest((HERE/group/'sources.tar.gz').read_bytes())
for name in ('check_first_round.py','public_schedule.py','baseline_manifest.json'):
    assert digest((HERE/name).read_bytes())==before[name]
origin=json.loads((HERE/'origin/origin_report.json').read_text())
assert origin['exact_runtime_files_compared']==92
for name,row in origin['runtime_sources'].items():
    assert row['proof_snapshot_sha256']==row['origin_snapshot_sha256']==before['checkout/'+name]
for name,value in origin['input_patch_sha256'].items():
    assert digest((HERE/'origin'/name).read_bytes())==value
ring=report['ring'];q,p=ring['q_towers'],ring['p_towers']
assert len(q)==50 and len(p)==7 and ring['degree']==65536 and ring['dnum']==8
assert math.prod(q+p).bit_length()==1614<=1747==ring['security_bit_length_ceiling']
assert math.isclose(math.log2(math.prod(q+p)),ring['actual_log2_QP'],rel_tol=1e-14)
assert math.prod(p)>=max(math.prod(q[i:i+7]) for i in range(0,len(q),7))
assert all(value%(2*ring['degree'])==1 for value in q+p)
policy=ast.parse(blobs['checkout/jaxite_word/he_params.py'].decode())
table=next(n.value for n in policy.body if isinstance(n,ast.Assign)
    and any(isinstance(t,ast.Name) and t.id=='_OPENFHE_CEILINGS_RAW' for t in n.targets))
ceilings=next(ast.literal_eval(v) for k,v in zip(table.keys,table.values)
    if isinstance(k,ast.Tuple) and isinstance(k.elts[0],ast.Attribute)
    and k.elts[0].attr=='CLASSICAL' and ast.literal_eval(k.elts[1])==128)
assert ceilings[65536]==1747
assert report['keyless_server'] and set(report['key_generation_guarded_modules'])=={'key_gen','jaxite_word.key_gen'}
contract=report['public_contract'];assert contract['K']==128 and contract['s']==4. and contract['iterations']==8
assert contract['distinct_input_count']==len(set(contract['input_points']))==258
assert min(contract['input_points'])==contract['a'] and max(contract['input_points'])==contract['b']
assert contract['inactive_dummy'] in contract['input_points']
assert len(report['checks'])==8 and report['fixed_absolute_error_gate']==1e-6
level=23;scale=ring['scale']
for index,check in enumerate(report['checks']):
    if index:
        scale=scale*ring['scale']/math.prod(q[2*(level-2):2*(level-1)])
        level-=3
    assert check['update']==index+1 and check['passed'] and check['metadata_ok']
    assert check['level']==level and check['nsd']==1 and check['all_slots_checked']==32768
    assert math.isclose(check['scale'],scale,rel_tol=1e-12)
    assert check['max_abs_error_against_exact_float_circuit']<1e-6
    for actual in check['execution_rings']:
        assert actual['q_towers']==q and actual['p_towers']==p and actual['degree']==65536
        assert actual['security_bits']==128 and actual['security_model']=='classical'
assert level==2 and report['final_true_inverse']['passed']
assert report['final_true_inverse']['max_abs_error']<1e-6
logged=[json.loads(line) for line in (HERE/'run/run1.log').read_text().splitlines() if line.startswith('{')]
assert logged==report['phases']
phases=[p['name'] for p in logged]
assert phases.index('Newton update8 complete')<phases.index('entire eight-update server circuit complete; client validation begins')<phases.index('client validation finished')
assert not Path('/proc/819762').exists() and not Path('/proc/819763').exists()
summary={'verification_passed':True,'verified_utc':datetime.now(timezone.utc).isoformat(),
 'encrypted_run_passed':True,'observed_exit_code':0,'terminal_observation':'exec session25112 returned exit_code0; wrapper819762/child819763 independently absent',
 'source_count':95,'runtime_origin_count':92,'source_start_end_archives_and_origin_match':True,
 'source_archive_sha256':archive_hashes,'report_sha256':digest((HERE/'run/report.json').read_bytes()),
 'source_origin':origin['candidate_scope'],'ring':ring,
 'distinct_public_inputs':258,'all_slots_checked':32768,'input_interval':[contract['a'],contract['b']],
 'updates':8,'remaining_level':2,'fixed_absolute_error_gate':1e-6,
 'worst_update_error':max(c['max_abs_error_against_exact_float_circuit'] for c in report['checks']),
 'final_true_inverse_max_abs_error':report['final_true_inverse']['max_abs_error'],
 'exact_float_circuit_error':report['final_true_inverse']['exact_float_circuit_max_abs_error'],
 'elapsed_s':report['elapsed_s'],'peak_host_rss_kib':report['peak_host_rss_kib'],
 'cleanup_errors':report['cleanup_errors'],'limitations':report['limitations'],
 'review_scope':'Archive/report/source/security consistency and process absence; encrypted numerical errors were measured by the preserved driver, not rerun by this audit.'}
(HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
