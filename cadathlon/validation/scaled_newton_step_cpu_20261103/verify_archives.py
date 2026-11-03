"""Read-only independent audit of two recorded CKKS component runs.

No encryption is rerun. This checks exact source bytes, recorded pass/fail
arithmetic, full Q/P products and coverage, and current process absence.
"""
import ast,difflib,hashlib,json,math,os,tarfile
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath

HERE=Path(__file__).resolve().parent
def sha(data):return hashlib.sha256(data).hexdigest()

def security_ceiling(source,degree):
    tree=ast.parse(source)
    value=next(n.value for n in tree.body if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='_OPENFHE_CEILINGS_RAW' for t in n.targets))
    for key,table in zip(value.keys,value.values):
        if (isinstance(key,ast.Tuple) and isinstance(key.elts[0],ast.Attribute)
                and key.elts[0].attr=='CLASSICAL' and ast.literal_eval(key.elts[1])==128):
            return ast.literal_eval(table)[degree]
    raise AssertionError('archived128-bit classical ceiling missing')

results={};drivers={};runtime_sets={}
for name,driver_name in (('low_precision','check_scaled_newton_step_cpu.py'),
                         ('high_precision','check_scaled_newton_step_cpu_high_precision.py')):
    folder=HERE/name;report=json.loads((folder/'report.json').read_text())
    before,after=report['source_hashes_start'],report['source_hashes_end']
    assert before==after and report['source_files_unchanged']
    with tarfile.open(folder/'sources.tar.gz') as archive:
        members=archive.getmembers()
        assert len({m.name for m in members})==len(members)==len(before)==98
        assert all(m.isfile() and not PurePosixPath(m.name).is_absolute()
                   and '..' not in PurePosixPath(m.name).parts for m in members)
        blobs={m.name:archive.extractfile(m).read() for m in members}
    expected={}
    for path,digest in before.items():
        p=Path(path)
        member=('external/'+driver_name if p.name==driver_name
                else str(p.relative_to('/home/jianming_gatech/CROSS')))
        assert member in blobs and sha(blobs[member])==digest,(name,member)
        expected[member]=digest
    assert set(expected)==set(blobs)
    driver=(folder/driver_name).read_bytes()
    assert driver==blobs['external/'+driver_name]
    drivers[name]=driver.decode()
    runtime_sets[name]={k:v for k,v in expected.items() if not k.startswith('external/')}
    q,p=report['ring']['q_towers'],report['ring']['p_towers']
    degree,dnum=report['ring']['degree'],report['ring']['dnum']
    ceiling=security_ceiling(blobs['jaxite_word/he_params.py'].decode(),degree)
    product=math.prod(q+p);actual_log=math.log2(product)
    assert product.bit_length()<=ceiling
    assert len(set(q+p))==len(q+p) and all(x%(2*degree)==1 for x in q+p)
    alpha=(len(q)+dnum-1)//dnum
    largest_partition=max(math.prod(q[i:i+alpha]) for i in range(0,len(q),alpha))
    assert math.prod(p)>=largest_partition
    assert math.isclose(actual_log,report['ring']['actual_log2_qp'],rel_tol=1e-14)
    assert report['ring']['security_bits']==128 and report['keyless_server']
    assert set(report['key_generation_guarded_modules'])=={'key_gen','jaxite_word.key_gen'}
    assert report['public_test_cases']==105 and report['fixed_max_abs_error_gate']==1e-6
    assert set(report['checks'])=={'product','correction','newton_update'}
    scale=float(q[-2])*q[-1]
    for check_name,level in (('product',2),('correction',1),('newton_update',0)):
        check=report['checks'][check_name]
        assert check['level']==level and check['metadata_ok']
        assert check['all_slots_checked']==degree//2
        expected_scale=scale if level else scale*scale/math.prod(q[2:4])
        assert math.isclose(check['scale'],expected_scale,rel_tol=1e-12)
        assert check['passed']==(math.isfinite(check['max_abs_error']) and check['max_abs_error']<1e-6)
        for ring in check['execution_rings']:
            assert ring['q_towers']==q and ring['p_towers']==p and ring['degree']==degree
            assert ring['security_bits']==128 and ring['security_model']=='classical'
    assert report['passed']==all(c['passed'] for c in report['checks'].values())
    logged={}
    for line in next(folder.glob('*.log')).read_text().splitlines():
        try:record=json.loads(line)
        except json.JSONDecodeError:continue
        for key in report['checks']:
            if key in record:
                assert key not in logged
                logged[key]=record[key]
    assert logged==report['checks']
    assert not report['cleanup_errors']
    assert not any(report[k] for k in ('full_model_evaluated','full_reciprocal_evaluated','refresh_evaluated'))
    results[name]={'recorded_passed':report['passed'],'verified_source_count':len(expected),
        'source_archive_sha256':sha((folder/'sources.tar.gz').read_bytes()),
        'report_sha256':sha((folder/'report.json').read_bytes()),'driver_sha256':sha(driver),
        'log_sha256':sha(next(folder.glob('*.log')).read_bytes()),
        'actual_log2_QP':actual_log,'QP_bit_length':product.bit_length(),'security_bit_length_ceiling':ceiling,
        'P_partition_coverage':True,'source_start_end_and_archive_bytes_match':True,
        'logged_checks_match_report':True,
        'recorded_cleanup_errors':report['cleanup_errors'],'elapsed_s':report['elapsed_s'],
        'errors':{k:v['max_abs_error'] for k,v in report['checks'].items()}}

assert runtime_sets['low_precision']==runtime_sets['high_precision']
def extract(text,start,end):return text[text.index(start):text.index(end)]
fixture=extract(drivers['low_precision'],'  # Public test points','  keys=key_gen.gen_pke_pair')
assert fixture==extract(drivers['high_precision'],'  # Public test points','  keys=key_gen.gen_pke_pair')
circuit=extract(drivers['low_precision'],'  product=arithmetic.multiply','  updated.polynomial.block_until_ready()')
assert circuit==extract(drivers['high_precision'],'  product=arithmetic.multiply','  updated.polynomial.block_until_ready()')
diff=''.join(difflib.unified_diff(drivers['low_precision'].splitlines(True),drivers['high_precision'].splitlines(True),
                               fromfile='low_precision_driver',tofile='high_precision_driver'))
(HERE/'driver_profile_difference.patch').write_text(diff)
names={'check_scaled_newton_step_cpu.py','check_scaled_newton_step_cpu_high_precision.py'}
alive=[]
for directory in Path('/proc').iterdir():
    if not directory.name.isdigit():continue
    try:
        argv=(directory/'cmdline').read_bytes().split(b'\0')
        if any(Path(arg.decode(errors='replace')).name in names for arg in argv if arg):
            alive.append({'pid':int(directory.name),'argv':[a.decode(errors='replace') for a in argv if a]})
    except (OSError,ValueError):pass
assert not alive,alive
verification={'scope':__doc__,'verification_passed':True,'verified_utc':datetime.now(timezone.utc).isoformat(),
 'runs':results,'same97_runtime_sources':True,'same105_public_fixture_generator':True,
 'fixture_generator_sha256':sha(fixture.encode()),'same_fixed_server_circuit':True,
 'server_circuit_sha256':sha(circuit.encode()),'matching_processes_at_verification':alive,
 'terminal_status':{'low_precision':{'exit_code':1,'source':'Observed by parent orchestrator before archive review; independently confirmed process absent.'},
   'high_precision':{'exit_code':None,'source':'Success report complete; independently confirmed process absent and exec session57270 closed. Numeric exit code unavailable; not inferred.'}},
 'limitations':['Audit verifies stored bytes and report consistency; it does not recreate ciphertexts or independently measure errors.',
 'Only one Newton update is covered, with105 distinct public cases repeated over4096/8192slots. No full reciprocal, refresh, softmax or model result.',
 'Two different secure ring/precision profiles and fresh keys were used; no controlled performance comparison.']}
(HERE/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print(json.dumps(verification,indent=2))
