"""Archive only public evidence after an owned isolated CPU child has reaped."""
from pathlib import Path
import argparse,hashlib,json,math,os,shutil,tarfile

def sha(data):return hashlib.sha256(data).hexdigest()

def alive(pid):
    try:os.kill(pid,0)
    except ProcessLookupError:return False
    return True

def main():
    p=argparse.ArgumentParser();p.add_argument('lab',type=Path);p.add_argument('destination',type=Path)
    p.add_argument('--observed-exit-code',type=int,required=True);p.add_argument('--wrapper-pid',type=int,required=True)
    p.add_argument('--scope',required=True);args=p.parse_args()
    lab=args.lab;out=lab/'encrypted-v1';report=json.loads((out/'report.json').read_text())
    if alive(report['process_pid']) or alive(args.wrapper_pid):raise RuntimeError('owned process still alive; reap before archiving')
    before=report.get('source_hashes_before',{});after=report.get('source_hashes_after',{})
    current={name:sha((lab/name).read_bytes()) for name in before}
    with tarfile.open(out/'sources.tar.gz') as tar:
        members=tar.getmembers()
        if any(not m.isfile() for m in members):raise RuntimeError('unexpected source archive entry')
        archived={m.name:sha(tar.extractfile(m).read()) for m in members}
        if len(archived)!=len(members):raise RuntimeError('duplicate source archive entry')
    equal=bool(before) and before==after==current==archived
    archive_digest=sha((out/'sources.tar.gz').read_bytes())
    digest_ok=archive_digest==report.get('source_archive_sha256')
    key_parent=report.get('owned_public_key_parent')
    cleanup=bool(report.get('public_key_store_removed')) and not report.get('cleanup_errors') and (key_parent is None or not Path(key_parent).exists())
    text=(lab/'encrypted-v1.log').read_text()
    matching_external_exit=f'Exit status: {args.observed_exit_code}' in text
    checks=report.get('checks',{})
    ring=report.get('ring',{})
    names=set(checks)
    if names=={'query','intermediate','output'}:
        expected_levels={name:0 for name in names};expected_slots=8192
    else:
        expected_levels={name:(2 if name.endswith(':mean') else 1) for name in names};expected_slots=65536
    numeric_complete=(len(checks)==(3 if expected_slots==8192 else 4) and
        all(c.get('passed') is True and c.get('slots_checked')==expected_slots and c.get('metadata_ok') is True
            and c.get('nsd')==1 and c.get('level')==expected_levels[name]
            and isinstance(c.get('max_abs_error'),(int,float)) and 0<=c['max_abs_error']<=1e-6
            and math.isclose(c.get('scale',0),ring.get('scale',-1),rel_tol=1e-12)
            for name,c in checks.items()))
    boundary=(report.get('keyless_server') is True and report.get('input_payload_and_metadata_unchanged') is True
        and report.get('server_complete_before_client_validation') is True
        and set(report.get('key_generation_guards',[]))=={'key_gen','jaxite_word.key_gen'}
        and set(report.get('class_codec_guards',[]))=={'encrypt','decrypt','encrypt_slots','decrypt_slots','encrypt_slots_batch','decrypt_slots_batch','decode'})
    actual_qp=sum(math.log2(q) for q in ring.get('q_towers',[])+ring.get('p_towers',[]))
    profile=ring.get('degree') in (16384,131072) and ring.get('security_bits')==128 and math.isclose(actual_qp,ring.get('actual_log2_qp',-1),abs_tol=1e-10)
    if profile:
        profile=actual_qp<=(438 if ring['degree']==16384 else 3523)
        alpha=math.ceil(len(ring['q_towers'])/ring['dnum']);product=math.prod(ring['p_towers'])
        profile=profile and all(product>=math.prod(ring['q_towers'][i:i+alpha]) for i in range(0,len(ring['q_towers']),alpha))
    audit=dict(scope=args.scope,observed_process_exit_code=args.observed_exit_code,
        child_pid=report['process_pid'],wrapper_pid=args.wrapper_pid,all_owned_pids_absent=True,
        source_files=len(before),exact_current_before_after_archive_match=equal,
        source_archive_sha256=archive_digest,archive_digest_matches_report=digest_ok,
        owned_public_key_directory_removed=cleanup,gnu_time_exit_matches_observed=matching_external_exit,
        proof_report_passed=report.get('passed',False),numerical_checks=checks,
        all_fixed_slot_level_scale_gates_complete=numeric_complete,public_only_boundary_verified=boundary,actual_profile_security_and_p_coverage_verified=profile,
        proof_success=bool(args.observed_exit_code==0 and report.get('passed') and equal and digest_ok and cleanup and matching_external_exit and numeric_complete and boundary and profile),
        limits=['CPU component only; no TPU or whole-model result.','Only aggregate reports/logs/exact source bytes copied; no key, ciphertext, client vector or weight arrays.'])
    # Failed numerical outcomes are still preserved. Incomplete provenance is explicit.
    args.destination.mkdir(parents=True,exist_ok=False)
    for src,name in ((out/'report.json','report.json'),(out/'sources.tar.gz','sources.tar.gz'),(lab/'encrypted-v1.log','run.log')):
        shutil.copy2(src,args.destination/name)
    for name in ('README.md','candidate.patch','source_origin.json','MONITORING.md'):
        if (lab/name).is_file():shutil.copy2(lab/name,args.destination/name)
    shutil.copy2(Path(__file__),args.destination/'archive_evidence.py')
    (args.destination/'verification.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    hashes={str(path.relative_to(args.destination)):sha(path.read_bytes()) for path in sorted(args.destination.rglob('*')) if path.is_file()}
    (args.destination/'artifact_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps({k:audit[k] for k in ('proof_success','source_files','exact_current_before_after_archive_match','owned_public_key_directory_removed','gnu_time_exit_matches_observed')}))
    return 0 if equal and digest_ok and matching_external_exit else 1

if __name__=='__main__':raise SystemExit(main())
