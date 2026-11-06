"""Client-only fixed-pair native all-layer attention-input/output fixtures, never server arithmetic."""
from pathlib import Path
import argparse,ast,hashlib,importlib.metadata,io,json,os,sys,tarfile,time
import numpy as np
from lifecycle import CleanupTracker
HERE=Path(__file__).resolve().parent
PRIOR=Path('/tmp/bert-first-attention-owner-20261103')
PRIOR_NPZ_SHA='184397324e50f8c33cd6ea62f47457b3b0992da85814c001927b412ca6d06ce7'
PAIRS=(
 ('The committee approved the proposal.','The committee approved the proposal.'),
 ('The committee reviews the proposal and discusses the evidence. '*20,
  'Members discuss the proposal before reaching a decision. '*20),
)
GATE=1e-12

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,row):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');temporary.replace(path)
def array_record(a):
    a=np.asarray(a)
    if a.dtype.kind not in 'fi' or not np.isfinite(a).all():raise ValueError('finite real client fixture required')
    return dict(shape=list(a.shape),dtype=a.dtype.str,sha256=hashlib.sha256(a.tobytes()).hexdigest())
def positions_from_mask(mask,expected):
    a=np.asarray(mask)
    if a.shape!=(128,) or a.dtype.kind not in 'biu' or not np.isin(a,[0,1]).all():raise ValueError('fixed128 binary public mask required')
    pos=np.flatnonzero(a).astype(np.int64)
    if len(pos)!=expected or not len(pos) or pos[0]!=0:raise ValueError('fixed key count/CLS position differs')
    return pos

def capture_native(model,forward,sequence):
    """Capture complete ordinary native outputs; remove every hook on failure."""
    if type(sequence)is not int or not 1<=sequence<=128 or len(model.bert.encoder.layer)!=12:raise ValueError('pinned native shape required')
    captured={};handles=[]
    def hook(key):
        def observe(module,inputs,output):
            if key in captured:raise ValueError('native label duplicated')
            if not isinstance(output,tuple) or not output:raise ValueError('native layer tuple output required')
            a=np.array(output[0].detach().cpu().numpy(),copy=True)
            if a.shape!=(1,sequence,768) or a.dtype!=np.float64 or not np.isfinite(a).all():raise ValueError('native float64 output shape/values differ')
            captured[key]=a[0]
        return observe
    try:
        for layer,module in enumerate(model.bert.encoder.layer):
            handles.append(module.attention.register_forward_hook(hook(('attention',layer))))
            handles.append(module.register_forward_hook(hook(('block',layer))))
        forward()
        if set(captured)!={(kind,i) for kind in ('attention','block') for i in range(12)}:raise ValueError('native capture incomplete')
        return captured
    finally:
        tracker=CleanupTracker(sys.exception())
        for handle in handles:tracker.attempt('remove native hook',handle.remove)
        tracker.finish()
        if tracker.errors and tracker.primary is None:raise RuntimeError('native hook cleanup failed: '+'; '.join(tracker.errors))

def parity(left,right):
    left,right=np.asarray(left),np.asarray(right)
    if left.shape!=right.shape or left.dtype!=np.float64 or right.dtype!=np.float64 or not np.isfinite(left).all() or not np.isfinite(right).all():raise ValueError('native parity arrays differ')
    with np.errstate(over='ignore',invalid='ignore'):error=float(np.max(np.abs(left-right)))
    if not np.isfinite(error):raise ValueError('native parity error nonfinite')
    if error>GATE:raise ValueError('native compaction gate failed: '+str(error))
    return error

def sources():
    files={'prepare_all_attention_fixtures.py':Path(__file__),'test_prepare_all_attention_fixtures.py':HERE/'test_prepare_all_attention_fixtures.py',
        'lifecycle.py':HERE/'lifecycle.py','verified_model_manifest.json':HERE/'verified_model_manifest.json',
        'prior_prepare_public_fixtures.py':PRIOR/'prepare_public_fixtures.py'}
    return {name:p.read_bytes() for name,p in files.items()}

def run(snapshot,output):
    output.mkdir(exist_ok=False);start=time.monotonic();primary=None;before={};assets={}
    report=dict(passed=False,scope=__doc__,fixed_pairs=[list(x) for x in PAIRS],HE_executed=False,labels_used=False,
        native_dtype='float64',threads=4,cpu_affinity=sorted(os.sched_getaffinity(0)),cases=[],arrays={},
        fixed_native_compaction_gate=GATE,polynomial_domain_selection_performed=False,
        limitations=['Native all-layer outputs are trusted component fixtures, not an encrypted transformer or arbitrary-input client path.',
                    'Inputs are the two previously fixed public pairs; no rows are chosen from polynomial-domain success or label accuracy.',
                    'Model arrays and client NPZ remain outside repository evidence.'])
    try:
        write(output/'fixture_preparation.json',report)
        raw=sources();before={n:hashlib.sha256(b).hexdigest() for n,b in raw.items()}
        with tarfile.open(output/'sources.tar.gz','w:gz') as t:
            for n,b in raw.items():m=tarfile.TarInfo(n);m.size=len(b);t.addfile(m,io.BytesIO(b))
        report['source_hashes_before']=before;report['source_archive_sha256']=digest(output/'sources.tar.gz')
        tree=ast.parse(raw['prior_prepare_public_fixtures.py'])
        original=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PAIRS' for t in n.targets))
        # literal_eval cannot evaluate the documented *20 string repetition;
        #compare the exact source expression to this helper's declaration.
        own=ast.parse(raw['prepare_all_attention_fixtures.py']);current=next(n.value for n in own.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PAIRS' for t in n.targets))
        if ast.dump(original)!=ast.dump(current):raise ValueError('fixed public pairs changed from prior fixture')
        pinned=json.loads(raw['verified_model_manifest.json']);assert snapshot.name==pinned['revision']
        assets={name:digest(snapshot/name) for name in pinned['files']}
        if assets!={n:v['sha256'] for n,v in pinned['files'].items()}:raise ValueError('pinned model/tokenizer assets differ')
        if digest(PRIOR/'client_fixtures.npz')!=PRIOR_NPZ_SHA:raise ValueError('prior layer0 fixture changed')
        report.update(asset_sha256=assets,revision=pinned['revision'],prior_fixture_sha256=PRIOR_NPZ_SHA,
                      versions={name:importlib.metadata.version(name) for name in ('numpy','torch','transformers','tokenizers','safetensors')})
        import torch
        from transformers import AutoTokenizer,BertForSequenceClassification
        torch.set_num_threads(4)
        model,loading=BertForSequenceClassification.from_pretrained(snapshot,local_files_only=True,use_safetensors=True,torch_dtype=torch.float64,attn_implementation='eager',output_loading_info=True)
        if loading['missing_keys'] or loading.get('mismatched_keys') or loading.get('error_msgs') or not set(loading['unexpected_keys'])<={'bert.embeddings.position_ids'}:raise ValueError('native model loading contract differs')
        model.eval();tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
        cfg=model.config
        if (cfg.hidden_size,cfg.num_attention_heads,cfg.intermediate_size,cfg.num_hidden_layers,cfg.layer_norm_eps)!=(768,12,3072,12,1e-12):raise ValueError('pinned architecture differs')
        arrays={}
        with np.load(PRIOR/'client_fixtures.npz',allow_pickle=False) as old,torch.no_grad():
            for index,pair in enumerate(PAIRS):
                token=tokenizer(*pair,max_length=128,truncation=True,padding='max_length',return_tensors='pt')
                K=15 if index==0 else 128;positions=positions_from_mask(token['attention_mask'][0].numpy(),K)
                embeddings=model.bert.embeddings(input_ids=token['input_ids'],token_type_ids=token['token_type_ids'])
                compact=embeddings[:,positions,:]
                full=capture_native(model,lambda:model(**token),128)
                compact_result=capture_native(model,lambda:model.bert.encoder(compact,attention_mask=torch.zeros((1,1,1,K),dtype=torch.float64),return_dict=True),K)
                checks=[]
                for layer in range(12):
                    attention_error=parity(full[('attention',layer)][positions],compact_result[('attention',layer)])
                    block_error=parity(full[('block',layer)][positions],compact_result[('block',layer)])
                    arrays[f'layer{layer}_K{K}_native']=compact_result[('attention',layer)]
                    arrays[f'layer{layer}_K{K}_input']=np.array(compact[0].cpu().numpy() if layer==0 else compact_result[('block',layer-1)],copy=True)
                    checks.append(dict(layer=layer,native_attention_compaction_max_abs=attention_error,native_block_compaction_max_abs=block_error))
                layer0_error=parity(arrays[f'layer0_K{K}_native'],old[f'K{K}_native'])
                if not np.array_equal(positions,old[f'K{K}_original_positions']):raise ValueError('prior original positions differ')
                arrays[f'K{K}_positions']=positions
                report['cases'].append(dict(K=K,original_positions=positions.tolist(),layers=checks,prior_layer0_max_abs=layer0_error,
                    tokenization_sha256=hashlib.sha256(json.dumps({k:v.tolist() for k,v in token.items()},sort_keys=True).encode()).hexdigest()))
                write(output/'fixture_preparation.json',report)
        path=HERE/'client_all_attention_fixtures.npz'
        if path.exists():raise ValueError('client fixture already exists; refusing overwrite')
        with path.open('xb') as stream:np.savez(stream,**arrays)
        report.update(fixture_npz_sha256=digest(path),fixture_path=str(path),arrays={name:array_record(a) for name,a in arrays.items()},passed=True)
    except BaseException as error:primary=error;report.update(passed=False,error={'type':type(error).__name__,'message':str(error)})
    finally:
        tracker=CleanupTracker(primary)
        after=tracker.attempt('source readback',lambda:{n:hashlib.sha256(b).hexdigest() for n,b in sources().items()})
        final_assets=tracker.attempt('asset readback',lambda:{n:digest(snapshot/n) for n in assets})
        def archive_verify():
            with tarfile.open(output/'sources.tar.gz') as t:
                rows=t.getmembers()
                return len(rows)==len(before)==len({m.name for m in rows}) and {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in rows}==before
        verified=tracker.attempt('archive readback',archive_verify)
        report.update(source_hashes_after=after,source_files_unchanged=bool(before) and before==after,source_archive_verified=verified is True,
            asset_hashes_after=final_assets,asset_files_unchanged=bool(assets) and assets==final_assets,cleanup_errors=tracker.errors,elapsed_s=time.monotonic()-start)
        report['passed']=bool(report.get('passed') and report['source_files_unchanged'] and report['asset_files_unchanged'] and verified and not tracker.errors)
        try:write(output/'fixture_preparation.json',report)
        except BaseException as error:
            preserved=primary if primary is not None else tracker.cancellation
            if preserved is None:raise
            preserved.add_note('Final fixture report write failed: '+repr(error))
        tracker.finish()
    if primary is not None:raise primary
    print(json.dumps({k:report[k] for k in ('passed','elapsed_s','fixture_npz_sha256','source_archive_sha256')}))
    return 0 if report['passed'] else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();raise SystemExit(run(a.snapshot,a.output))
