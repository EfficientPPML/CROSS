"""One bounded preparation stage; never dispatches the genuine worker."""
from pathlib import Path
from contextlib import redirect_stdout
import argparse,json,os,sys,xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent
TESTS=('test_group_preflight.py','test_group_actual.py','test_group_arithmetic.py','test_group_protocol.py','test_group_worker.py','test_group_collector.py',
       'test_grouped_loader.py','test_key_contexts.py','test_prepared_rotation.py','test_prepared_loader.py','test_parity_boundaries.py')

def main(kind,output):
    os.chdir(HERE)
    if os.environ.get('JAX_PLATFORMS')!='cpu':raise ValueError('explicit CPU process required')
    if kind=='tests':
        os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
        import pytest
        with redirect_stdout(sys.stderr):
            code=pytest.main(['-q',*TESTS,'--junitxml='+str(output)])
        if code:raise RuntimeError('mock tests failed: '+str(code))
        suites=ET.parse(output).getroot();count=sum(int(s.attrib.get('tests',0)) for s in suites)
        if count!=214 or any(int(s.attrib.get('errors',0)) or int(s.attrib.get('failures',0)) for s in suites):
            raise ValueError('fixed complete preparation test count differs')
        result=dict(passed=True,tests_passed=count)
    elif kind=='installed':
        import group_installed_probe
        with redirect_stdout(sys.stderr):group_installed_probe.main(output)
        result=json.loads((output/'report.json').read_bytes())
        if result.get('passed')is not True or result.get('denied_count')!=28 or result.get('all28_restored')is not True:
            raise ValueError('installed guard proof differs')
        result={n:result[n] for n in ('passed','denied_count','all28_restored','source_archive_sha256')}
    elif kind=='public':
        import group_public_gates
        with redirect_stdout(sys.stderr):result=group_public_gates.group_collect.run(output,execute=False,**group_public_gates.args)
        if result.get('passed')is not True or result.get('execute')is not False or result.get('actual_execution_attempted')is not False or group_public_gates.forbidden:
            raise ValueError('public prerequisite scope differs')
        result=dict(passed=True,execute=False,actual_execution_attempted=False,forbidden_payload_reads=[])
    else:raise ValueError('unknown preparation stage')
    print(json.dumps(result,allow_nan=False));return 0

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=('tests','installed','public'));p.add_argument('output',type=Path)
    from cleanup import termination_signals
    with termination_signals():code=main(**vars(p.parse_args()))
    raise SystemExit(code)
