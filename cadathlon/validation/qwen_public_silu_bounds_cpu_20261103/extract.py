"""Public checkpoint gate bounds and tokenizer-only new-input preflight."""
from pathlib import Path
from fractions import Fraction
import argparse,datetime,gc,hashlib,importlib.metadata,io,json,math,os,sys,tarfile,time
import numpy as np
from exact_dyadic import rows
HERE=Path(__file__).resolve().parent
SNAPSHOT=Path('/home/jianming_gatech/cross_validation_assets/qwen/c1899de289a04d12100db370d81485cdf75e47ca')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def tensor_bytes(tensor,torch):
    if tensor.dtype==torch.bfloat16:return tensor.contiguous().view(torch.uint16).numpy().astype('<u2',copy=False).tobytes()
    if tensor.dtype==torch.float32:return tensor.contiguous().numpy().astype('<f4',copy=False).tobytes()
    raise ValueError('public extraction accepts only BF16/F32 checkpoint tensors')


def public_tensor(tensors,checkpoint,header,base,name,shape,torch):
    source=tensors.get_tensor(name)
    if tuple(source.shape)!=shape:raise ValueError('public tensor shape mismatch '+name)
    raw=tensor_bytes(source,torch);descriptor=header[name]
    if descriptor['shape']!=list(shape) or descriptor['dtype'] not in ('BF16','F32'):raise ValueError('header dtype/shape mismatch')
    begin,end=descriptor['data_offsets']
    with checkpoint.open('rb') as f:f.seek(base+begin);direct=f.read(end-begin)
    if raw!=direct:raise ValueError('source tensor differs from pinned raw checkpoint bytes')
    promoted=source.to(torch.float32)
    if not torch.equal(promoted.to(source.dtype),source) or not torch.isfinite(promoted).all():raise ValueError('public promotion not finite/exact')
    array=promoted.numpy().copy()
    return array,{'name':name,'shape':list(shape),'source_dtype':str(source.dtype),'source_header_dtype':descriptor['dtype'],
                  'source_raw_sha256':hashlib.sha256(raw).hexdigest(),'source_raw_bytes':len(raw),
                  'promoted_dtype':'<f4','promoted_sha256':hashlib.sha256(array.astype('<f4').tobytes()).hexdigest(),
                  'source_roundtrip_exact':True,'source_matches_direct_checkpoint_bytes':True}


def run(output):
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();primary=None;before={};assets={}
    report={'passed':False,'complete':False,'model_evaluated':False,'coefficient_selection':False,'encrypted_execution':False,
            'scope':__doc__,'layers':{},'errors':[],'cpu_affinity':sorted(os.sched_getaffinity(0))}
    try:
        save(output/'report.json',report)
        paths={p.name:p for p in HERE.glob('*.py')}
        paths.update({p.name:p for p in HERE.glob('*.json')})
        paths.update({'inputs/'+p.name:p for p in (HERE/'inputs').glob('*.json')})
        before={n:sha(p) for n,p in paths.items()}
        with tarfile.open(output/'sources.tar.gz','w:gz') as t:
            for n,p in paths.items():b=p.read_bytes();i=tarfile.TarInfo(n);i.size=len(b);t.addfile(i,io.BytesIO(b))
        with tarfile.open(output/'sources.tar.gz') as t:
            ms=t.getmembers()
            assert len(ms)==len({m.name for m in ms})==len(before) and all(m.isfile() for m in ms)
            assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in ms}==before
        report.update(source_hashes_before=before,source_archive_sha256=sha(output/'sources.tar.gz'))
        pins=json.loads((HERE/'model_pins.json').read_text());assets={n:sha(SNAPSHOT/n) for n in pins['asset_sha256']}
        if assets!=pins['asset_sha256']:raise ValueError('pinned public assets changed')
        report['asset_sha256_before']=assets
        import torch
        from safetensors import safe_open
        from transformers import AutoTokenizer
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        report['versions']={n:importlib.metadata.version(n) for n in ('torch','numpy','safetensors','transformers','tokenizers')}
        tokenizer=AutoTokenizer.from_pretrained(SNAPSHOT,local_files_only=True)
        old=[]
        for name in ('old_calibration_inputs.json','old_validation_inputs.json'):old+=json.loads((HERE/'inputs'/name).read_text())
        old_sequences={tuple(r['token_ids']) for r in old};seen=set();fresh=[]
        for item in json.loads((HERE/'fresh_validation_proposal.json').read_text()):
            full=tokenizer(item['text'],add_special_tokens=False)['input_ids'];S=item['sequence_length'];ids=full[:S]
            if len(full)<max(32,S) or len(ids)!=S:raise ValueError('insufficient natural token length '+item['id'])
            if tuple(ids) in seen:raise ValueError('duplicate fresh token sequence '+item['id'])
            if tuple(ids) in old_sequences:raise ValueError('fresh token sequence overlaps previous inputs '+item['id'])
            seen.add(tuple(ids));fresh.append(dict(item,raw_token_length=len(full),token_ids=ids))
        if len(fresh)!=56 or len(seen)!=56:raise ValueError('fresh56 input inventory mismatch')
        for S in (8,16,32):
            if sum(r['kind']=='prefill' and r['sequence_length']==S for r in fresh)!=16:raise ValueError('fresh bucket mismatch')
        if sum(r['kind']=='decode' and r['sequence_length']==16 for r in fresh)!=8:raise ValueError('fresh decode inventory mismatch')
        save(output/'tokenized_fresh_inputs.json',fresh)
        report['fresh_validation']={'passed':True,'source_sha256':sha(HERE/'fresh_validation_proposal.json'),'tokenized_sha256':sha(output/'tokenized_fresh_inputs.json'),
            'records':56,'distinct_token_sequences':56,'old_records_checked':len(old),'old_overlap_count':0,
            'minimum_raw_token_length':min(r['raw_token_length'] for r in fresh),'maximum_raw_token_length':max(r['raw_token_length'] for r in fresh),
            'model_evaluated':False,'fixed_before_any_revised_selection':True}
        print(json.dumps({'stage':'tokenizer_only','passed':True,**report['fresh_validation']}),flush=True)
        config=json.loads((SNAPSHOT/'config.json').read_text())
        if config['hidden_size']!=1024 or config['intermediate_size']!=3072 or config['num_hidden_layers']!=28 or not (math.isfinite(config['rms_norm_eps']) and config['rms_norm_eps']>0):raise ValueError('Qwen architecture/epsilon mismatch')
        checkpoint=SNAPSHOT/'model.safetensors'
        with checkpoint.open('rb') as f:header_length=int.from_bytes(f.read(8),'little');header=json.loads(f.read(header_length))
        base=8+header_length
        with safe_open(checkpoint,framework='pt',device='cpu') as tensors:
            for layer in range(28):
                prefix=f'model.layers.{layer}.';gamma_name=prefix+'post_attention_layernorm.weight';weight_name=prefix+'mlp.gate_proj.weight'
                if prefix+'mlp.gate_proj.bias' in tensors.keys():raise ValueError('unexpected gate bias')
                gamma,gd=public_tensor(tensors,checkpoint,header,base,gamma_name,(1024,),torch)
                weight,wd=public_tensor(tensors,checkpoint,header,base,weight_name,(3072,1024),torch)
                bounds=[]
                for index,(square,upper) in enumerate(rows(weight,gamma)):
                    if Fraction.from_float(upper)**2<square:raise AssertionError('non-outward bound')
                    bounds.append({'row':index,'squared_bound_fraction':[str(square.numerator),str(square.denominator)],'native_exact_real_upper':upper})
                peak=max(bounds,key=lambda row:row['native_exact_real_upper']);cover=1
                while cover<peak['native_exact_real_upper']:cover*=2
                report['layers'][str(layer)]={'gamma':gd,'gate_weight':wd,'rows':bounds,'row_count':len(bounds),
                    'minimum_row_upper':min(row['native_exact_real_upper'] for row in bounds),'maximum_row_upper':peak['native_exact_real_upper'],
                    'maximum_row':peak['row'],'power_of_two_cover':cover,'domain_before_polynomial_multiplier':[-peak['native_exact_real_upper'],peak['native_exact_real_upper']]}
                print(json.dumps({'stage':'public_weight_bound','layer':layer,'rows':len(bounds),'max_upper':peak['native_exact_real_upper'],'power_two_cover':cover,'elapsed_s':time.perf_counter()-started}),flush=True)
                del gamma,weight,bounds
        report.update(complete=True,passed=True,rows_computed=sum(x['row_count'] for x in report['layers'].values()),
            epsilon=config['rms_norm_eps'],maximum_gate_bound=max(x['maximum_row_upper'] for x in report['layers'].values()),
            no_jax_imported=not any(n.split('.')[0] in ('jax','jaxlib','jaxite_word') for n in sys.modules),
            limitations=['Exact-real native RMS/gate algebra only; no model activation evaluation or candidate fitting.',
                'A polynomial multiplier requires a uniform inverse-sqrt certificate and valid same-input variance relation.',
                'No floating-execution, CKKS, bootstrap or full-model error guarantee.',
                'Fresh validation inputs are tokenizer-only frozen; no outputs were generated. Original candidate1 validation failure remains unchanged.',
                'Checkpoint assets and source/script hashes plus package versions are not a complete binary environment image.'])
        report['passed']=report['no_jax_imported'] and report['rows_computed']==28*3072
    except BaseException as exc:
        primary=exc;report['passed']=False;report['errors'].append({'type':type(exc).__name__,'message':str(exc)})
    finally:
        try:
            after={n:sha(p) for n,p in paths.items()} if before else {}
            report.update(source_hashes_after=after,source_files_unchanged=bool(before) and before==after)
            report['asset_sha256_after']={n:sha(SNAPSHOT/n) for n in assets};report['assets_unchanged']=bool(assets) and assets==report['asset_sha256_after']
            report['passed']=report['passed'] and report['source_files_unchanged'] and report['assets_unchanged']
        except BaseException as exc:
            report['passed']=False;report['errors'].append({'phase':'final_provenance','type':type(exc).__name__,'message':str(exc)})
            if primary is None:primary=exc
            else:primary.add_note('Final provenance: '+repr(exc))
        report['elapsed_s']=time.perf_counter()-started
        try:save(output/'report.json',report)
        except BaseException as exc:
            if primary is None:raise
            primary.add_note('Final publication: '+repr(exc))
    if primary is not None:raise primary
    return 0 if report['passed'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    raise SystemExit(run(parser.parse_args().output))
