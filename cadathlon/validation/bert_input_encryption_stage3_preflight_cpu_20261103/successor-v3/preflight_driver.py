"""Freeze and exercise the isolated stage3 preparation; never launch stage3."""
from pathlib import Path
import argparse,hashlib,importlib.util,io,json,os,subprocess,sys,tarfile,time,xml.etree.ElementTree as ET
LAB=Path('/tmp/bert-input-encryption-stage3-component-v3-20261103')
sys.path.insert(0,str(LAB))
spec=importlib.util.spec_from_file_location('stage3_preflight_collector',LAB/'collect.py');collector=importlib.util.module_from_spec(spec);spec.loader.exec_module(collector)
def sha(b):return hashlib.sha256(b).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def execute(out):
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();primary=None
    report=dict(passed=False,scope='Mocked stage3 lifecycle/binding tests and actual installed entrypoint denial probe only; no stage3 encryption, model/context construction, or key generation.',pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)))
    before={};raw={}
    try:
        write(out/'report.json',report)
        raw=collector.sources()|{'preflight_driver.py':Path(__file__).read_bytes()};before={n:sha(b) for n,b in raw.items()}
        copied=out/'snapshot';copied.mkdir()
        with tarfile.open(out/'sources.tar.gz','w:gz') as tar:
            for n,b in raw.items():
                p=copied/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
                m=tarfile.TarInfo(n);m.size=len(b);tar.addfile(m,io.BytesIO(b))
        origin=json.loads(raw['source_origin.json']);previous={'runtime/'+n:h for n,h in origin['first_handler_runtime_source_hashes'].items()}|origin['additional_source']
        revision=origin['cleanup_revision'];prior=revision['previous_candidate_source_hashes']
        changed=set(revision['changed_files']);added=set(revision['added_files'])
        if len(prior)!=78 or set(before)-{'preflight_driver.py'}!=set(prior)|added:raise AssertionError('exact previous and added source inventory required')
        if changed!={'cleanup.py','runtime/stage/key_contexts.py','worker.py','collect.py','test_worker.py','test_collector.py','source_origin.json'} or added!={'terminal.py','test_terminal_main.py'}:raise AssertionError('unexpected stage3 revision')
        if any(before[n]!=h for n,h in prior.items() if n not in changed):raise AssertionError('unchanged previous candidate drift')
        if revision['changed_and_added_hashes']!={n:before[n] for n in (changed|added)-{'source_origin.json'}}:raise AssertionError('reviewed candidate revision differs')
        inherited_overlay='runtime/stage/key_contexts.py'
        if len(previous)!=59 or any(before[n]!=h for n,h in previous.items() if n!=inherited_overlay):raise AssertionError('unchanged first-handler runtime drift')
        if before[inherited_overlay]!='7eb793deeccae8cd5a1e09f368f2465d39bfed21b379aefca46474bf6532c205' or before['cleanup.py']!='1798455cdd88d4bbac5e1769e6b3226676ccb9901e7be6faf383e7ad063c1ef6':raise AssertionError('reviewed cleanup overlay differs')
        report.update(source_hashes_start=before,source_archive_sha256=sha((out/'sources.tar.gz').read_bytes()),preserved_prior_runtime_sources=58,preserved_prior_runtime_sources_unchanged=True,reviewed_runtime_overlay={inherited_overlay:before[inherited_overlay]},reviewed_cleanup_proofs=revision['reviewed_proofs'])
        env=os.environ|dict(JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        command=['/home/jianming_gatech/miniconda3/envs/jaxite/bin/python','-m','pytest','-q','test_worker.py','test_collector.py','test_boundaries.py','test_request_readback.py','test_model_gate.py','test_input_gate.py','test_gate_revision.py','test_terminal_main.py','--junitxml='+str(out/'tests.xml')]
        with (out/'tests.log').open('w') as log:r=subprocess.run(command,cwd=copied,env=env,stdout=log,stderr=subprocess.STDOUT)
        report.update(test_command=command,test_returncode=r.returncode)
        if r.returncode:raise AssertionError('mocked stage3 suite failed')
        tests=ET.parse(out/'tests.xml').getroot();suites=list(tests.iter('testsuite'))
        report['mock_tests']=sum(int(x.attrib.get('tests',0)) for x in suites)
        if report['mock_tests']!=135 or any(int(x.attrib.get(n,0)) for x in suites for n in ('failures','errors','skipped')):raise AssertionError('complete135 mocked checks required')
        command=['/tmp/cross-pretrained-validation-env/bin/python',str(copied/'test_import_guards.py'),'--output',str(out/'real_import_guard.json')]
        with (out/'real_import_guard.log').open('w') as log:r=subprocess.run(command,cwd=copied,env=env,stdout=log,stderr=subprocess.STDOUT)
        report.update(import_probe_command=command,import_probe_returncode=r.returncode)
        if r.returncode:raise AssertionError('real installed guard probe failed')
        probe=json.loads((out/'real_import_guard.json').read_text())
        if probe.get('passed')is not True or probe.get('denial_count')!=21 or probe.get('restored_count')!=21:raise AssertionError('real entrypoint count incomplete')
        report.update(real_entrypoint_denials=21,operation_complete=True)
    except BaseException as error:primary=error;report['error']=dict(type=type(error).__name__,message=str(error))
    finally:
        try:
            current=collector.sources()|{'preflight_driver.py':Path(__file__).read_bytes()};after={n:sha(b) for n,b in current.items()}
            copied_hashes={n:sha((out/'snapshot'/n).read_bytes()) for n in before}
            with tarfile.open(out/'sources.tar.gz') as tar:
                members=tar.getmembers();archived={m.name:sha(tar.extractfile(m).read()) for m in members}
                exact=len(members)==len(set(m.name for m in members))==len(before) and all(m.isfile() for m in members)
            report.update(source_hashes_end=after,snapshot_hashes_end=copied_hashes,archive_member_hashes=archived,source_files_unchanged=bool(before) and before==after==copied_hashes==archived and exact,
                elapsed_s=time.monotonic()-start)
            report['passed']=bool(report.get('operation_complete') and report['source_files_unchanged'] and primary is None)
            write(out/'report.json',report)
        except BaseException as error:
            if primary is None:raise
            primary.add_note('preflight terminal publication: '+repr(error))
        if primary is not None:raise primary
    print(json.dumps(dict(passed=report['passed'],mock_tests=report['mock_tests'],real_entrypoint_denials=21,sources=len(before),source_archive_sha256=report['source_archive_sha256'],elapsed_s=report['elapsed_s'])))
    return 0 if report['passed'] else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();raise SystemExit(execute(a.output.absolute()))
