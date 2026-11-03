"""Verify terminal ownership and archive only public reciprocal component evidence."""
from pathlib import Path
import argparse,hashlib,json,math,os,shutil,tarfile

def sha(data):return hashlib.sha256(data).hexdigest()
def alive(pid):
    try:os.kill(pid,0)
    except ProcessLookupError:return False
    return True

def main():
    p=argparse.ArgumentParser();p.add_argument('lab',type=Path);p.add_argument('destination',type=Path)
    p.add_argument('--observed-exit-code',type=int,required=True);p.add_argument('--wrapper-pid',type=int,required=True);a=p.parse_args()
    out=a.lab/'encrypted-v1';r=json.loads((out/'report.json').read_text())
    if alive(r['process_pid']) or alive(a.wrapper_pid):raise RuntimeError('owned processes must be terminal and reaped')
    before=r.get('source_hashes_before',{});current={n:sha((a.lab/n).read_bytes()) for n in before}
    with tarfile.open(out/'sources.tar.gz') as tar:
        members=tar.getmembers()
        if any(not m.isfile() for m in members) or len({m.name for m in members})!=len(members):raise ValueError('invalid source inventory')
        archived={m.name:sha(tar.extractfile(m).read()) for m in members}
    source_ok=len(before)==110 and before==current==archived==r.get('source_hashes_after')
    archive_hash=sha((out/'sources.tar.gz').read_bytes());archive_ok=archive_hash==r.get('source_archive_sha256')
    checks=r.get('checks',{});rows=checks.get('checkpoints',[]);rules=checks.get('declared_rules',[])
    names=['initial','tanh']+[f'K{i}' for i in range(1,129)]
    numeric=(len(rows)==13 and [q.get('update') for q in rows]==list(range(1,14))
        and len(rules)==130 and [q.get('name') for q in rules]==names
        and sum(q.get('slots_checked',0) for q in rules)==32768
        and all(q.get('passed') is True and q.get('finite_and_shape') is True and q.get('error_summaries_finite') is True for q in rows+rules))
    if numeric:
        for row in rows:
            numeric=numeric and row.get('slots_checked')==32768 and row.get('metadata',{}).get('passed') is True
            for key in ('y_original_newton_max_abs_error','e_independent_power_max_abs_error','invariant_defect_max_abs'):
                v=row.get(key);numeric=numeric and isinstance(v,(int,float)) and math.isfinite(v) and 0<=v<=1e-6
        for rule in rules:
            for key in ('y_original_newton_max_abs_error','true_inverse_max_abs_error','true_relative_residual_max_abs','tracked_residual_max_abs','invariant_defect_max_abs'):
                v=rule.get(key);numeric=numeric and isinstance(v,(int,float)) and math.isfinite(v) and 0<=v<=1e-6
    fixture=r.get('fixture',{});segments=fixture.get('segments',[])
    original_counts={q.get('name'):q.get('iterations') for q in segments}
    stopping=bool(len(original_counts)==130 and all(original_counts.get(q.get('name'))==q.get('declared_update') for q in rules))
    boundary=(r.get('keyless_server') is True and r.get('server_complete_before_client_validation') is True
        and r.get('input_payload_and_metadata_unchanged') is True
        and set(r.get('key_generation_guards',[]))=={'key_gen','jaxite_word.key_gen'}
        and set(r.get('class_codec_guards',[]))=={'encrypt','decrypt','encrypt_slots','decrypt_slots','encrypt_slots_batch','decrypt_slots_batch','decode'})
    counts=r.get('operation_counts',{});operations=counts.get('production_ciphertext_multiplications')==24 and counts.get('diagnostic_ciphertext_multiplications')==1
    profile=r.get('ring',{});q=profile.get('q_towers',[]);pmods=profile.get('p_towers',[])
    security=profile.get('degree')==65536 and len(q)==32 and len(pmods)==9 and profile.get('dnum')==4 and sum(map(math.log2,q+pmods))<=1747
    security=security and all(math.prod(pmods)>=math.prod(q[i:i+8]) for i in range(0,len(q),8))
    cleanup=r.get('cleanup_completed') is True and not r.get('cleanup_errors') and r.get('no_key_files_written') is True
    log=(a.lab/'encrypted-v1.log').read_text();external=f'Exit status: {a.observed_exit_code}' in log
    success=bool(a.observed_exit_code==0 and r.get('passed') and source_ok and archive_ok and numeric and stopping and boundary and operations and security and cleanup and external)
    audit=dict(proof_success=success,scope='Actual CPU reciprocal components only; no refresh/composed softmax/model/TPU claim.',
        observed_exit_code=a.observed_exit_code,child_pid=r['process_pid'],wrapper_pid=a.wrapper_pid,owned_pids_absent=True,
        source_files=len(before),exact_before_after_current_archive_match=source_ok,source_archive_sha256=archive_hash,
        archive_digest_matches_report=archive_ok,all_fixed_numerical_gates=numeric,original_stopping_iterations=stopping,
        public_keyless_boundary=boundary,production_diagnostic_counts=operations,actual_security_p_coverage=security,
        cleanup_completed=cleanup,gnu_time_matches_observed_exit=external,
        whitelist='Public aggregate reports, exact source archives, logs and this verifier only; no keys/ciphertexts/private arrays.')
    a.destination.mkdir(parents=True,exist_ok=False)
    for src,name in ((out/'report.json','report.json'),(out/'sources.tar.gz','sources.tar.gz'),(a.lab/'encrypted-v1.log','run.log'),(Path(__file__),'archive_evidence.py')):shutil.copy2(src,a.destination/name)
    (a.destination/'verification.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    hashes={str(p.relative_to(a.destination)):sha(p.read_bytes()) for p in sorted(a.destination.rglob('*')) if p.is_file()}
    (a.destination/'artifact_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(audit));return 0 if source_ok and archive_ok and external else 1
if __name__=='__main__':raise SystemExit(main())
