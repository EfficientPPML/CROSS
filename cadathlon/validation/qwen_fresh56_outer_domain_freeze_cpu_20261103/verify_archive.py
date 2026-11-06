from pathlib import Path
import argparse,hashlib,json,tarfile


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    r=json.loads((root/'run/report.json').read_text());assert r['passed'] and r['complete'] and r['source_files_unchanged'] and r['assets_unchanged']
    assert r['source_hashes_before']==r['source_hashes_after'] and sha(root/'run/sources.tar.gz')==r['source_archive_sha256']
    with tarfile.open(root/'run/sources.tar.gz') as t:
        ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==11
        assert all(m.isfile() and not m.name.startswith('/') and '..' not in Path(m.name).parts for m in ms)
        payload={m.name:t.extractfile(m).read() for m in ms}
    assert {n:hashlib.sha256(b).hexdigest() for n,b in payload.items()}==r['source_hashes_before']
    pins=json.loads(payload['model_pins.json']);assert r['asset_sha256_before']==r['asset_sha256_after']==pins['asset_sha256']
    for n,h in json.loads(payload['input_pins.json']).items():assert hashlib.sha256(payload[n]).hexdigest()==h
    fresh=json.loads((root/'run/tokenized_fresh_inputs.json').read_text());assert len(fresh)==56
    assert sha(root/'run/tokenized_fresh_inputs.json')==r['tokenized_sha256']
    corpus=json.loads(payload['inputs/calibration_corpus.json']);oldtext={x['text'] for x in corpus['calibration_prompts']}
    oldseq=set()
    for n in ['calibration_tokens.json','published_tokens.json','candidate2_fresh_tokens.json']:
        records=json.loads(payload['inputs/'+n]);oldseq.update(tuple(x['token_ids']) for x in records);oldtext.update(x['text'] for x in records if 'text'in x)
    assert len({x['text'] for x in fresh})==len({tuple(x['token_ids']) for x in fresh})==56
    assert all(x['text'] not in oldtext and tuple(x['token_ids']) not in oldseq and len(x['token_ids'])==x['sequence_length'] and x['raw_token_count']>=32 for x in fresh)
    assert [sum(x['kind']=='prefill' and x['sequence_length']==s for x in fresh) for s in (8,16,32)]==[16]*3
    assert sum(x['kind']=='decode' and x['sequence_length']==16 for x in fresh)==8
    assert not r['model_forward'] and not r['coefficient_selection'] and r['no_model_backend_imported']
    terminal=json.loads((root/'terminal.json').read_text());assert terminal['session_id']==86577 and terminal['observed_exit_code']==0 and terminal['reaped']
    for n,h in json.loads((root/'artifact_sha256.json').read_text()).items():assert sha(root/n)==h
    return dict(passed=True,scope='Offline source/token inventory and equality-disjointness; tokenizer/model not rerun.',sources=11,records=56)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path,nargs='?',default=Path(__file__).resolve().parent)
    print(json.dumps(verify(p.parse_args().root),indent=2))
