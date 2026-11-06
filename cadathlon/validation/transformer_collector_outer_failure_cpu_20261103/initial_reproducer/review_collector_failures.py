"""Read-only failure-path review of the frozen candidate collector."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path('/tmp/runtime-followup-integration-20261103/checkout')
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('reviewed_suite', ROOT/'cadathlon/validation/check_transformer_suite.py')
suite=importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)
implementation=ROOT/'cadathlon/agent_pset/p4_qwen/src/qwen_optimized_reference.py'

def arguments(name):
    return SimpleNamespace(model='qwen',implementation=implementation,output=HERE/name,
        cases=[1],warmups=1,repeats=3,case_timeout_s=None,cleanup_grace_s=30.)

results={}
args=arguments('initial-source-failure')
with patch.object(suite,'source_manifest',side_effect=FileNotFoundError('simulated unreadable initial source')):
    try:suite.evaluate(args)
    except FileNotFoundError as error:
        results['initial_source_failure']={'exception':str(error),
            'output_created':args.output.is_dir(),'report_exists':(args.output/'report.json').exists()}

args=arguments('cancel-final-publication-failure')
real_write=suite.write_json
def final_write(path,value):
    if 'elapsed_s' in value:raise OSError('simulated final publication failure')
    return real_write(path,value)
with patch.object(suite,'source_manifest',return_value={'source':'unchanged'}), \
     patch.object(suite,'run_case',side_effect=KeyboardInterrupt('primary cancellation')), \
     patch.object(suite,'write_json',side_effect=final_write):
    try:suite.evaluate(args)
    except BaseException as error:
        results['cancellation_masking']={'raised_type':type(error).__name__,'raised':str(error),
            'context_type':type(error.__context__).__name__,
            'primary_only_in_exception_context':isinstance(error.__context__,KeyboardInterrupt),
            'existing_report':json.loads((args.output/'report.json').read_text())}

assert results['initial_source_failure']['output_created'] and not results['initial_source_failure']['report_exists']
assert results['cancellation_masking']['raised_type']=='OSError'
assert results['cancellation_masking']['primary_only_in_exception_context']
results['scope']='Controlled metadata-only failure injection; no child launched, no JAX/TPU/crypto, no candidate edits.'
(HERE/'collector_failure_review.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
