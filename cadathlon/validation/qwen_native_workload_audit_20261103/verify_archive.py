"""Offline archive and public workload verification; no model/backend imports."""
from pathlib import Path
import hashlib,json,tarfile
ROOT=Path(__file__).resolve().parent

def sources(attempt):
    report=json.loads((ROOT/attempt/'report.json').read_text())
    before=report['source_hashes_before']
    assert len(before)==21 and before==report['source_hashes_after'] and report['source_files_unchanged']
    assert hashlib.sha256((ROOT/attempt/'sources.tar.gz').read_bytes()).hexdigest()==report['source_archive_sha256']
    with tarfile.open(ROOT/attempt/'sources.tar.gz') as archive:
        members=archive.getmembers();assert len(members)==len(before) and {m.name for m in members}==set(before)
        for member in members:
            assert member.isfile() and not member.issym() and not member.islnk()
            assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
            assert hashlib.sha256(archive.extractfile(member).read()).hexdigest()==before[member.name]
    terminal=json.loads((ROOT/attempt/'terminal.json').read_text())
    assert terminal['authoritatively_reaped'] and terminal['exit_code']==0
    assert report['passed'] and report['all_exact_semantic_checks_passed'] and report['no_jax_imported']
    assert not report['fixed_domain_coverage_passed']
    return report

smoke=sources('smoke-v1');full=sources('full-v1')
assert smoke['source_hashes_before']==full['source_hashes_before']
assert full['complete'] and len(full['cases'])==48 and len(full['decode'])==8
assert full['semantic_gates']=={'max_abs':.001,'nrmse':.00001,'exact_top1':True}
for S in (8,16,32):
    bucket=[r for r in full['cases'] if r['sequence_length']==S]
    assert len(bucket)==16 and len({tuple(r['token_ids']) for r in bucket})==16
    assert all(len(r['token_ids'])==S for r in bucket)
records=list(full['cases'])
for d in full['decode']:
    for kind,tokens in [('teacher_records',d['native_tokens']),('free_records',d['oracle_free_tokens'])]:
        rows=d[kind];assert len(rows)==4
        initial=rows[0]['token_ids'];assert len(initial)==16
        for step,row in enumerate(rows):
            assert row['token_ids']==initial+tokens[:step] and row['sequence_length']==16+step
        records.extend(rows)
assert len(records)==112
for row in records:
    check=row['exact_oracle_vs_native']
    assert check['passed'] and check['all_final_logits_checked']==151936
    assert check['max_abs_error']<=.001 and check['nrmse']<=.00001 and check['top1_agreement']
    assert row['domain_violations']
assert full['teacher_forced_top1_agreements']==32 and full['free_running_complete_sequence_agreements']==8
assert {(r['layer'],r['valid_keys']) for r in full['missing_softmax_contracts']}=={(layer,K) for layer in range(28) for K in range(5,33)}
print(json.dumps({'passed':True,'source_files':21,'prefill':48,'teacher_decisions':32,'free_sequences':8,'all_contexts':112,'fixed_domain_coverage':False,'approximate_or_HE_claim':False},sort_keys=True))
