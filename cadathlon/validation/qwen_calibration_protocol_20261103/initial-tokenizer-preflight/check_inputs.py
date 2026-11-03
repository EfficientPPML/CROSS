"""Tokenizer-only disjointness check; no model, logits, activations or JAX."""
from pathlib import Path
import hashlib,json,os,sys,time
HERE=Path(__file__).resolve().parent
SNAPSHOT='/home/jianming_gatech/cross_validation_assets/qwen/c1899de289a04d12100db370d81485cdf75e47ca'
VALIDATION=Path('/home/jianming_gatech/CROSS/cadathlon/validation/qwen_native_workload_audit_20261103/full-v1/tokenized_fixtures.json')

def main():
    start=time.perf_counter();paths=[HERE/n for n in ('author_corpus.py','corpus.json','protocol.json','check_inputs.py')]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();before={p.name:sha(p) for p in paths}
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(SNAPSHOT,local_files_only=True)
    corpus=json.loads((HERE/'corpus.json').read_text());validation=json.loads(VALIDATION.read_text())
    heldout={tuple(f['token_ids']) for f in validation};seen={S:set() for S in corpus['prefill_buckets']};rows=[]
    for item in corpus['calibration_prompts']:
        raw=tokenizer(item['text'],add_special_tokens=False)['input_ids'];assert len(raw)>=32,item['id']
        for S in corpus['prefill_buckets']:
            tokens=tuple(raw[:S]);assert len(tokens)==S and tokens not in seen[S] and tokens not in heldout,item['id']
            seen[S].add(tokens);rows.append({'id':item['id'],'sequence_length':S,'raw_token_count':len(raw),'token_ids':list(tokens)})
    assert [len(seen[S]) for S in (8,16,32)]==[96,96,96]
    assert len(corpus['decode_prompt_ids'])==len(set(corpus['decode_prompt_ids']))==16
    assert set(corpus['decode_prompt_ids'])<={i['id'] for i in corpus['calibration_prompts']}
    assert not any(n.split('.')[0] in ('jax','jaxlib','jaxite_word') for n in sys.modules)
    after={p.name:sha(p) for p in paths};assert before==after
    (HERE/'tokenized_calibration_inputs.json').write_text(json.dumps(rows,indent=2)+'\n')
    report={'passed':True,'scope':__doc__,'corpus_sha256':sha(HERE/'corpus.json'),'protocol_sha256':sha(HERE/'protocol.json'),
        'source_hashes_before':before,'source_hashes_after':after,'source_files_unchanged':True,
        'validation_input_only_sha256':sha(VALIDATION),'tokenized_calibration_inputs_sha256':sha(HERE/'tokenized_calibration_inputs.json'),
        'calibration_prefills':288,'unique_per_bucket':{str(S):len(seen[S]) for S in seen},'decode_seed_count':16,
        'minimum_raw_token_count':min(r['raw_token_count'] for r in rows),'maximum_raw_token_count':max(r['raw_token_count'] for r in rows),
        'validation_token_intersection':0,'no_model_instantiated':True,'no_jax_imported':True,'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'elapsed_s':time.perf_counter()-start,'status':'Prepared for root protocol review; no calibration model measurement has run.'}
    (HERE/'input_preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
