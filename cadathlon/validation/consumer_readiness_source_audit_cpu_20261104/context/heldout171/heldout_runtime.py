"""Authorized heldout or separate row0 model lifecycle; checker math unchanged.

The model/asset/backend sequence is copied from frozen checker execute_candidate.
Only public authorization, workload reader, adapter and accounting differ.
No backend is imported until the independent external grant is reauthenticated.
"""
import numpy as np
from checker_runtime import canonical,sha,StructuralFailure,ARCHITECTURE,NativeCaptureWithInputs
from heldout_adapters import HeldoutAdapters
import heldout_dispatch as dispatch

def runtime_policy():
    return dict(schema='qwen-heldout-runtime-policy-v1',cores=[108,109,110,111],
        torch_threads=4,torch_interop_threads=1,heldout_entries=1088,heldout_native_forwards=1120,
        row0_entries=1,row0_native_forwards=2,coefficient_selection=False,candidate_revision=False)

def execute_heldout(blobs,authorization,reader,private_output,progress):
    # Reauthenticate the external freeze/outer/grant before body or model access.
    import heldout_authorization
    binding=heldout_authorization.require_authorized(authorization,blobs)
    import backend_environment
    backend_environment.check()
    from freeze_io import parse,read
    from input_gate import model_assets
    from model_runtime import backend_check
    from candidate_domains import plan
    from cleanup_origin import cleanup_origin,protect_body,choose
    import os,sys,importlib.metadata,gc
    manifest=parse(binding.manifest_bytes)
    if sha(binding.manifest_bytes)!=binding.manifest_sha256:raise StructuralFailure('Admitted manifest bytes differ')
    if binding.frozen.ordinal!=1:raise StructuralFailure('Only the reviewed first candidate is executable')
    mode=binding.mode
    if mode not in ('heldout','row0'):raise StructuralFailure('Exact frozen workload mode required')
    plan(manifest,32) # Complete manifest observation contract before body reads.
    policy=parse(blobs['heldout_runtime_policy.json'])
    expected_policy=runtime_policy()
    if canonical(policy)!=canonical(expected_policy) or canonical(binding.runtime_policy)!=canonical(policy):
        raise StructuralFailure('Exact preparation runtime policy required')
    if sorted(os.sched_getaffinity(0))!=policy['cores']:raise StructuralFailure('Assigned heldout CPU affinity required')
    from heldout_input_gate import inspect_validation
    validation=inspect_validation(blobs,binding.frozen)
    if canonical(validation)!=canonical(binding.validation_descriptor):raise StructuralFailure('Admitted validation descriptor differs')
    if mode=='row0'and reader is not None:raise StructuralFailure('Row0 must not receive a validation reader')
    dependencies=parse(blobs['dependencies.json'])
    installed={name:sys.version if name=='python'else importlib.metadata.version(name)for name in dependencies}
    if installed!=dependencies:raise StructuralFailure('Pinned installed dependencies differ')
    if any(os.environ.get(n)!='1'for n in('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_HUB_DISABLE_TELEMETRY'))or any(os.environ.get(n)!='4'for n in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):
        raise StructuralFailure('Exact offline four-thread environment required')
    before=backend_check(blobs,read)
    rows=None
    if mode=='heldout':
        rows=reader(validation['path'],validation['sha256'],limit=validation['bytes'])
        dispatch.validation_inputs(rows,binding.frozen)
    model=weights=adapters=capture_owner=None;primary=None;body=None;report=None
    try:
        with model_assets(blobs)as model_root:
            with backend_environment.import_scope(blobs):
                import torch
                from transformers import Qwen3ForCausalLM
            torch.set_num_threads(4);torch.set_num_interop_threads(1)
            model,loading=Qwen3ForCausalLM.from_pretrained(model_root,local_files_only=True,use_safetensors=True,torch_dtype=torch.float32,attn_implementation='eager',output_loading_info=True)
            if any(loading.get(n)for n in('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise StructuralFailure('Exact model loading required')
            model.eval();config=model.config.to_dict()
            if {n:config[n]for n in ARCHITECTURE}!=ARCHITECTURE or model.training or model.config._attn_implementation!='eager':raise StructuralFailure('Pinned eager architecture required')
            if any(p.device.type!='cpu'or p.dtype!=torch.float32 for p in model.parameters()):raise StructuralFailure('CPU float32 native parameters required')
            if model.lm_head.weight.data_ptr()!=model.model.embed_tokens.weight.data_ptr():raise StructuralFailure('Pinned tied vocabulary weight required')
            weights={name:value.detach().numpy().astype(np.float64)for name,value in model.state_dict().items()if name!='lm_head.weight'}
            backend_check(blobs,read,loaded=True);progress(dict(stage='model_loaded',model_loaded=True));calls=dict(started=0,completed=0)
            def native(prefix):
                calls['started']+=1;progress(dict(stage='native_forward_started',native_forward_started=calls['started']))
                with torch.no_grad():
                    result=model(input_ids=torch.tensor([prefix],dtype=torch.long),attention_mask=torch.ones((1,len(prefix)),dtype=torch.long),position_ids=torch.arange(len(prefix))[None,:],use_cache=False).logits[:,-1].detach().numpy().astype(np.float64)
                calls['completed']+=1;progress(dict(stage='native_forward_completed',native_forward_completed=calls['completed']));return result
            with NativeCaptureWithInputs(model)as capture_owner:
                adapters=HeldoutAdapters(frozen=binding.frozen,mode=mode,native_forward=native,capture_owner=capture_owner,config=config,weights=weights,manifest=manifest,manifest_sha256=binding.manifest_sha256,private_output=private_output,progress=progress)
                report=adapters.run(rows)
            if capture_owner.cleanup_completed is not True:raise StructuralFailure('Native capture restoration incomplete')
            measured=report['phase_counts']['native']
            expected_calls=measured+len(report['native_invariance_by_length'])
            accounted=(type(measured)is int and calls['started']==calls['completed']==expected_calls
                and 0<=expected_calls<=(1120 if mode=='heldout'else 2))
            report['native_forward_accounting']=dict(expected=expected_calls,started=calls['started'],completed=calls['completed'],passed=accounted)
            if report['passed']and not accounted:
                raise StructuralFailure('Native forward accounting differs from complete comparison ledger')
            after=backend_check(blobs,read,loaded=True)
            if after!=before:raise StructuralFailure('Backend source changed during candidate work')
            report.update(model_loaded=True,model_forward=calls['started']>0,native_forward_started=calls['started'],native_forward_completed=calls['completed'],native_capture_cleanup_completed=capture_owner.cleanup_completed,
                backend_sources_before=before,backend_sources_after=after,backend_environment=backend_environment.check(),runtime_policy=policy,installed_dependencies=installed,
                candidate_ordinal=binding.frozen.ordinal,freeze_receipt_sha256=binding.frozen.receipt_sha256,
                workload_mode=mode,validation_input_sha256=binding.frozen.validation_input_sha256,
                selection_permanently_closed=True,revision_eligible_ledger=False,
                native_dtype='float32',exact_candidate_dtype='float64',scope='Frozen candidate heldout1088'if mode=='heldout'else'Separate frozen candidate token0 row0')
        report['model_asset_postchecks_completed']=True
    except BaseException as error:
        primary=error;body=None if cleanup_origin(error)else protect_body(error);raise
    finally:
        model=weights=adapters=capture_owner=None
        try:gc.collect()
        except BaseException as error:
            selected=choose(primary,error,'heldout/row0 model cleanup',body_primary=body)
            if selected is not primary:raise selected
    report['model_cleanup_completed']=True
    return report
