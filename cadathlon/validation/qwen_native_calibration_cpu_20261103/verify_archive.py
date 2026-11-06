"""Offline verification of the source-bound native/exact calibration evidence."""
from pathlib import Path
import hashlib,json,tarfile
P=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def verify():
    proof=json.loads((P/'verification.json').read_text())
    for name,expected in proof['artifact_sha256'].items():assert sha((P/name).read_bytes())==expected,name
    protocol=json.loads((P/'protocol.json').read_text());corpus=json.loads((P/'corpus.json').read_text());inputs=json.loads((P/'tokenized_calibration_inputs.json').read_text())
    for run,full in [('smoke-v1',False),('native-exact-v1',True)]:
        report=json.loads((P/run/'report.json').read_text())
        assert report['passed'] and report['source_files_unchanged'] and report['assets_unchanged']
        assert report['source_hashes_before']==report['source_hashes_after']
        assert report['asset_sha256_before']==report['asset_sha256_after']==json.loads((P/'model_pins.json').read_text())['asset_sha256']
        with tarfile.open(P/run/'sources.tar.gz') as archive:
            members=archive.getmembers();assert all(i.isfile() for i in members)
            hashes={i.name:sha(archive.extractfile(i).read()) for i in members};assert hashes==report['source_hashes_before']
            for name,expected in hashes.items():
                if not name.startswith('software/'):assert sha((P/name).read_bytes())==expected,name
        assert sha((P/run/'sources.tar.gz').read_bytes())==report['source_archive_sha256']
        assert report['protocol_sha256']==sha((P/'protocol.json').read_bytes())
        assert report['corpus_sha256']==sha((P/'corpus.json').read_bytes())
        assert report['no_jax_imported'] and not report['encrypted_execution'] and not report['approximate_forward_evaluated'] and not report['candidate_selection_performed']
        assert report['semantic_gates']=={'max_abs':.001,'nrmse':.00001,'exact_top1':True}
        records=list(report['cases'])
        for decode in report['decode']:
            initial=next(r['token_ids'] for r in inputs if r['id']==decode['id'] and r['sequence_length']==16)
            for kind,choices in [('teacher',decode['native_tokens']),('free',decode['oracle_free_tokens'])]:
                rr=decode[kind+'_records'];assert len(rr)==4
                for i,r in enumerate(rr):assert r['token_ids']==initial+choices[:i]
                records.extend(rr)
        assert len(records)==report['total_forward_contexts']==(416 if full else 11)
        assert report['complete']==full
        for r in records:
            m=r['exact_oracle_vs_native'];assert m['passed'] and m['max_abs_error']<=.001 and m['nrmse']<=.00001 and m['top1_agreement']
            assert m['all_final_logits_checked']==151936 and r['native_top1']==r['oracle_top1']
            assert r['sequence_length']==len(r['token_ids'])
            data=b''.join(int(i).to_bytes(8,'little',signed=True) for i in r['token_ids']);assert sha(data)==r['context_sha256']
        for r in report['cases']:
            S=r['sequence_length'];ident=r['id'].rsplit('_S',1)[0]
            assert r['token_ids']==next(x['token_ids'] for x in inputs if x['id']==ident and x['sequence_length']==S)
        if full:
            assert len(report['cases'])==288 and len(report['decode'])==16
            assert [d['id'] for d in report['decode']]==corpus['decode_prompt_ids']
            assert all(sum(r['sequence_length']==S for r in report['cases'])==96 for S in (8,16,32))
            assert report['teacher_forced_decisions']==report['teacher_forced_top1_agreements']==64
            assert report['free_running_sequences']==report['free_running_complete_sequence_agreements']==16
        positions=sum(r['sequence_length'] for r in records)
        pairs=sum(r['sequence_length']*(r['sequence_length']+1)//2 for r in records)
        required=0
        for origin in ('native','exact'):
            ranges=report['observed_'+origin+'_ranges']
            assert ranges['final.inverse_sqrt']['count']==len(records)
            for layer in range(28):
                for suffix,multiplier in [('input',1),('post',1),('q',16),('k',8)]:
                    row=ranges[f'{layer}.{suffix}.inverse_sqrt'];assert row['count']==positions*multiplier and row['minimum']>0;required+=1
                assert ranges[f'{layer}.silu']['count']==positions*3072
                assert ranges[f'{layer}.softmax.centered']['count']==pairs*16
            assert all(row['nonfinite']==0 for row in ranges.values())
        assert report['required_input_site_count']==169 and report['required_input_sites_covered'] and report['finite_trajectories'] and report['positive_inverse_sqrt_inputs']
        assert len(report['hook_invariance_checks'])==4 and all(x['native_logits_bitwise_invariant'] and x['exact_logits_bitwise_invariant'] for x in report['hook_invariance_checks'])
    print(json.dumps({'passed':True,'native_exact_contexts':416,'semantic_logits_per_context':151936,'no_approximate_or_encrypted_claim':True}))
if __name__=='__main__':verify()
