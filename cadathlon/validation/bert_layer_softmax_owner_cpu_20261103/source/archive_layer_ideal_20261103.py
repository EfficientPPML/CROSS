"""Whitelist archival of successful isolated layer ideal proofs; no array payloads."""
from pathlib import Path
import argparse,hashlib,io,json,os,shutil,tarfile,xml.etree.ElementTree as ET

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(kind,observed):
    if observed!=0:raise ValueError('archive success requires authoritative zero exit')
    name='bert-layer-softmax-20261103' if kind=='softmax' else 'bert-layer-attention-20261103'
    lab=Path('/tmp')/name;run=lab/'source-bound-v1';r=json.loads((run/'report.json').read_text())
    assert r['passed'] is True and r['returncode']==0 and r['child_reaped'] is True
    assert r['source_files_unchanged'] and r['executed_snapshot_unchanged'] and r['source_archive_verified']
    try:os.kill(r['child_pid'],0)
    except ProcessLookupError:pass
    else:raise ValueError('measured child is still present')
    before=r['source_hashes_before'];assert before==r['source_hashes_after']
    assert before=={n:sha(lab/n) for n in before}=={n:sha(run/'snapshot'/n) for n in before}
    with tarfile.open(run/'sources.tar.gz') as t:
        members=t.getmembers();assert len(members)==len(before)==len({m.name for m in members})
        assert all(m.isfile() for m in members)
        assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in members}==before
    assert sha(run/'sources.tar.gz')==r['source_archive_sha256']
    xml=ET.parse(run/'tests.xml').getroot();suites=[xml] if xml.tag=='testsuite' else list(xml.iter('testsuite'))
    counts={name:sum(int(s.get(name,0)) for s in suites) for name in ('tests','failures','errors','skipped')}
    assert counts['failures']==counts['errors']==0
    dst=Path('/home/jianming_gatech/CROSS/cadathlon/validation')/('bert_layer_softmax_owner_cpu_20261103' if kind=='softmax' else 'bert_layer_attention_owner_cpu_20261103')
    dst.mkdir(exist_ok=False);measured=dst/'measured';measured.mkdir();source=dst/'source';source.mkdir()
    filenames=['report.json','tests.log','tests.xml','sources.tar.gz']
    if kind=='softmax':filenames+=['public_plan.json']
    else:
        assert r['checkpoint_unchanged'] and r['fixture_unchanged']
        assert set(r['cases'])=={f'{layer}/{K}' for layer in range(12) for K in (15,128)}
        assert all(x['passed'] for x in r['cases'].values())
        filenames+=['fixture_pins.json']+[f'development-layer{layer}-K{K}.json' for layer in range(12) for K in (15,128)]
    for filename in filenames:shutil.copy2(run/filename,measured/filename)
    selected=(['checkout/cadathlon/common/encrypted_bert_layer_softmax.py','test_layer_softmax.py','test_layer_guards.py','test_public_plan.py','public_plan.py','check_layer_softmax.py'] if kind=='softmax' else
        ['encrypted_bert_layer_attention.py','layer_attention_contract.py','test_layer_attention_contract.py','test_layer_attention_ideal.py','test_layer_attention_guards.py','attention_public_fixtures.py','layer_attention_reference.py','check_layer_attention_ideal.py','prepare_all_attention_fixtures.py'])
    for n in selected:
        assert sha(lab/n)==before[n];shutil.copy2(lab/n,source/Path(n).name)
    verification=dict(passed=True,scope='Archive-time independent readback of exact measured bytes; no HE claim.',observed_collector_exit=observed,
        source_files=len(before),source_archive_sha256=r['source_archive_sha256'],all_source_current_snapshot_archive_hashes_equal=True,
        test_counts=counts,elapsed_s=r['elapsed_s'],measured_child_absent=True,arrays_keys_ciphertexts_archived=False,
        archived_sha256={str(p.relative_to(dst)):sha(p) for p in sorted(dst.rglob('*')) if p.is_file()})
    (dst/'verification.json').write_text(json.dumps(verification,indent=2,allow_nan=False)+'\n')
    (dst/'README.md').write_text((lab/'README.md').read_text()+f"\nMeasured validation: {counts['tests']} tests passed; exact {len(before)} source files remained unchanged before/after and match the source archive. Collector duration {r['elapsed_s']:.3f}s, a validation duration rather than an implementation performance measurement. The collector was authoritatively reaped with exit0.\n")
    print(json.dumps(dict(path=str(dst),**{k:verification[k] for k in ('test_counts','source_files','source_archive_sha256','elapsed_s')})))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=('softmax','attention'));p.add_argument('--observed-exit-code',type=int,required=True);a=p.parse_args();main(a.kind,a.observed_exit_code)
