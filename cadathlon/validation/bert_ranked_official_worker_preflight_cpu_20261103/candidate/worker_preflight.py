"""Source-bound synthetic and installed-entrypoint preparation only."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time,xml.etree.ElementTree as ET
import ranked_collect as collect
from cleanup import termination_signals
from ranked_terminal import finish
HERE=Path(__file__).resolve().parent
TESTS=('test_ranked_acceptance.py','test_ranked_case_gate.py','test_ranked_handlers.py','test_ranked_private_reference.py',
    'test_ranked_private_io.py','test_ranked_stage_gate.py','test_ranked_workers.py','test_ranked_setup.py',
    'test_ranked_collect.py','test_ranked_terminal.py','test_ranked_cli.py','test_ranked_timing.py')
EXPECTED_TESTS=294


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);start=time.monotonic();expected={};primary=None
    report=dict(passed=False,scope='fresh official ranked worker preparation; synthetic handlers and installed forbidden-entrypoint probes only',
        actual_ranked_execution=False,private_fixture_reference_body_reads=False,model_forward=False,key_generation=False,he=False,
        wrapper_pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)))
    try:
        collect.write(output/'report.json',report)
        raw=collect.sources();expected=collect.hashes(raw);collect.archive(output/'sources.tar.gz',raw)
        if not collect.verify_archive(output/'sources.tar.gz',expected):raise ValueError('initial source archive differs')
        report.update(source_hashes_before=expected,source_archive_sha256=collect.key_gate.digest((output/'sources.tar.gz').read_bytes()))
        command=[sys.executable,'-m','pytest','-q',*TESTS,'--junitxml='+str(output/'tests.xml')]
        with (output/'tests.log').open('w') as log:
            p=subprocess.run(command,cwd=HERE,stdout=log,stderr=subprocess.STDOUT)
        report['tests_returncode']=p.returncode;report['test_command']=command
        suites=list(ET.parse(output/'tests.xml').getroot().iter('testsuite'))
        counts={n:sum(int(s.attrib.get(n,0)) for s in suites) for n in ('tests','failures','errors','skipped')}
        report['tests']=counts
        if p.returncode or counts!={'tests':EXPECTED_TESTS,'failures':0,'errors':0,'skipped':0}:raise ValueError('complete fixed synthetic suite did not pass')
        report['installed_probes']=[]
        for stage in range(1,6):
            with (output/f'installed{stage}.json').open('w') as stdout,(output/f'installed{stage}.log').open('w') as stderr:
                proc=subprocess.run([sys.executable,str(HERE/'installed_probe.py'),'--stage',str(stage)],cwd=HERE,stdout=stdout,stderr=stderr)
            if proc.returncode:raise ValueError('installed guard probe failed')
            proof=json.loads((output/f'installed{stage}.json').read_bytes())
            if proof.get('passed')is not True or proof.get('all_restored')is not True or proof['denied_count']!={1:15,2:24,3:21,4:25,5:21}[stage]:raise ValueError('installed guard scope differs')
            for row in proof['loaded_local_modules'].values():
                if expected.get(row['path'])!=row['sha256']:raise ValueError('installed local module differs from exact closure')
            report['installed_probes'].append(dict(stage=stage,denied_count=proof['denied_count'],returncode=proc.returncode,process_reaped=True))
        report['passed']=True
    except BaseException as error:
        primary=error;report.update(passed=False,error={'type':type(error).__name__,'message':str(error)})
    finally:
        try:
            after=collect.hashes(collect.sources());report.update(source_hashes_after=after,sources_unchanged=bool(expected) and expected==after,
                source_archive_verified=bool(expected) and collect.verify_archive(output/'sources.tar.gz',expected),elapsed_s=time.monotonic()-start)
            report['passed']=bool(report['passed'] and report['sources_unchanged'] and report['source_archive_verified']);collect.write(output/'report.json',report)
        except BaseException as error:
            if primary is None:raise
            primary.add_note('preflight finalization: '+type(error).__name__)
        if primary is not None:raise primary
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=finish(a.output/'report.json',lambda:run(a.output),collect.write)
    print(json.dumps({'passed':r['passed'],'tests':r['tests'],'sources':len(r['source_hashes_before']),'installed_denials':sum(x['denied_count'] for x in r['installed_probes']),'elapsed_s':r['elapsed_s']}))
    raise SystemExit(0 if r['passed'] else 1)
