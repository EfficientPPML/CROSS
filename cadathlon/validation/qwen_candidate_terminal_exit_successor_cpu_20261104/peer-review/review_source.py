"""Public-source-only reachability receipt. Never reads candidate outcomes/bodies."""
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path('/tmp/qwen-expanded-candidate-checker-ready-v2-20261104')
MANIFEST=Path('/tmp/qwen-expanded-candidate-checker-ready-sources-v2-20261104.json')
OUT=Path(__file__).parent/'review.json'
EXPECTED='ccd5834b83b9dfec2d3ea988fb395a61ec4b05ce3f2f54621f29200cbbb8e06a'
raw=MANIFEST.read_bytes()
assert hashlib.sha256(raw).hexdigest()==EXPECTED
mapping=json.loads(raw)
names=('entry.py','checker_process.py','freeze_io.py','cleanup_origin.py','checker_runtime.py',
       'native_capture.py','native_observer.py','input_gate.py','backend_environment.py')
blobs={name:(ROOT/name).read_bytes() for name in names}
before={name:hashlib.sha256(data).hexdigest() for name,data in blobs.items()}
assert len(mapping)==139 and all(mapping[n]==h for n,h in before.items())
spans={}
for name,data in blobs.items():
    tree=ast.parse(data)
    spans[name]={node.name:{'line':node.lineno,'last_line':node.end_lineno}
        for node in ast.walk(tree) if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in
        ('main','supervise','accept_candidate_worker','signals','collect','dispatch','execute_candidate',
         'close','write','reopen','_restore','__exit__','model_assets','import_scope')}
explicit=[]
for name,data in blobs.items():
    for node in ast.walk(ast.parse(data)):
        if isinstance(node,ast.Call):
            call=node.func
            if isinstance(call,ast.Name) and call.id=='SystemExit':
                explicit.append({'source':name,'line':node.lineno,'expression':ast.unparse(node)})
assert explicit==[{'source':'entry.py','line':216,'expression':'SystemExit(main())'}]
after={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
assert before==after
report={
 'schema':'qwen-candidate-terminal-reachability-source-review-v1',
 'created_utc':datetime.now(timezone.utc).isoformat(),
 'scope':'Selected frozen139 local lifecycle source only; no actual candidate outcome, private journal, calibration, model, validation, backend execution, or dynamic probe.',
 'source_root':str(ROOT),'source_manifest_sha256':EXPECTED,'source_count':139,
 'reviewed_source_sha256':before,'reviewed_sources_unchanged':before==after,'function_spans':spans,
 'selected_explicit_system_exit_calls':explicit,
 'findings':[
  'supervise serializes report before final publication.close, then propagates a close error; a normal scientific failure and an ordinary escaped OSError both become OS exit1 through the original entry boundary.',
  'accept_candidate_worker checks code0 for passed and code1 for failed. It does not establish normal return after publication; parent collect uses the same supervise boundary.',
  'Selected signal handlers raise KeyboardInterrupt for TERM/INT/HUP or deferred cancellation, never SystemExit. They are restored before final publication.',
  'After final report serialization, reviewed publication close uses Output.close -> finish -> os.close. Ordinary OS failures are OSError subclasses; no selected local helper creates SystemExit0.',
  'Model/capture/asset cleanup completes before scientific report publication. Selected local cleanup implementations contain no explicit SystemExit call.',
  'No concrete unmodified ordinary cleanup/signal route to late SystemExit0 was identified. Synthetic injected SystemExit0 tests cover the exception contract; they do not demonstrate such a naturally reachable current path.'
 ],
 'limits':[
  'This is scoped static source review, not an exhaustive guarantee about native-library finalizers or arbitrary injected callbacks.',
  'stderr absence is not a positive normal-return certificate. Observed traceback may reject a run; silence does not repair the failed-run exit1 ambiguity.',
  'A new consumer cannot repair the old producer terminal contract. Original139 failed attempts remain inadmissible for reconstruction/revision under the root decision.',
  'An original139 PASS with exit0 may be assessed within the ordinary reviewed-code scope; this receipt does not certify any actual result.'
 ],
 'successor_design':{
  'scientific_normal_statuses':[0,1],'infrastructure_status':70,
  'normal_status_requires_exact_int':True,'all_escaped_BaseException_in_dispatch_maps_to_70':True,
  'argparse_help_and_usage_outside_dispatch_boundary':True,
  'intentional_final_SystemExit_outside_boundary':True,
  'diagnostics_best_effort_must_not_change_infrastructure_status':True,
  'frozen_sources_modified':False
 }
}
OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'receipt':str(OUT),'sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),'sources':len(before),'current_equal':True}))
