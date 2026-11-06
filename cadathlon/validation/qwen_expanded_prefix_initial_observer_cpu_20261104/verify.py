"""Verify public observer proof only; never open paths from private descriptors."""
from pathlib import Path, PurePosixPath
import argparse, hashlib, io, json, os, tarfile
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
MANIFEST = '8131c5ac045f1cf274a493bbc14278e330c0a8e057079edc4f10f9a23dd9bf91'
REPORT = 'bc77c79e104af3f2f7c0da709bbf5c9319483ce7ad9a1dbb7cb3b8002510afb3'
WORKER = '58005996a1348ac96fc50aee270429cf29f27280675140ae294603943df3d68d'
TAR = 'd4ccebcca8691b78a00eb3af58cbb67e213e286038c112d7f85550484303791c'
REVIEW = 'edc9a06ef42108d8ccdefdf5a805c262c0a0f85c5d4ce900b3ec8b13a7feccad'
OUTER = '120e6df818f6ea2207fa8c7b40aeaa34dcc709376294de909ba3438256c90644'
SCRIPT = '3690b5710a8f019f529062303289d415fad99d85bb579ff426e66ec9633ff2e3'
GRANT = 'e0068125a3a271880a69231d5108fe9591ae10489e05f06d043646d8be192704'
BASELINE = '7f6478a6bc710f3d2a370198e075e6ba8b0e55c144d981bca0eed59eaf176333'
CALIBRATION = '80eeaefad423167ee0710797d7eb76d8cf3a422e6f3b2e6069932d682835963b'
sha = lambda b: hashlib.sha256(b).hexdigest()
load = lambda p: json.loads(p.read_bytes())

def exact_int(value, expected):
    assert type(value) is int and value == expected

def archive(path, expected):
    result = {}; total = 0
    with tarfile.open(fileobj=io.BytesIO(path.read_bytes()), mode='r:gz') as handle:
        for member in handle:
            name = PurePosixPath(member.name); total += member.size
            assert member.isfile() and not name.is_absolute() and '..' not in name.parts
            assert str(name) == member.name and member.name not in result
            assert 0 <= member.size <= 16 << 20 and total <= 32 << 20
            result[member.name] = handle.extractfile(member).read()
    assert len(result) == 61 and {n:sha(v) for n,v in result.items()} == expected
    return result

def terminal(row):
    for key in ('passed','complete','cleanup_completed','signal_restoration_completed',
                'source_files_unchanged','terminal_after_body_cleanup_and_signal_restoration_attempts',
                'acceptance_requires_independent_exit0_reap_and_exact_report_hash'):
        assert row[key] is True, key
    assert row['audit_denials'] == []

def no_process(pid, group=False):
    try:
        (os.killpg if group else os.kill)(pid, 0)
    except ProcessLookupError:
        return True
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--current', action='store_true', help='Also read public current/execution source files and process metadata.')
    args = parser.parse_args()
    assert __debug__, 'Run without -O so integrity checks remain active.'
    files = {}
    for p in HERE.rglob('*'):
        assert not p.is_symlink()
        if p.is_file() and p.name != 'files.sha256.json':
            files[p.relative_to(HERE).as_posix()] = sha(p.read_bytes())
    assert files == load(HERE/'files.sha256.json')
    copied = load(HERE/'public_whitelist.json')
    assert copied['private_bodies_opened_or_copied_by_archiver'] is False
    assert len(copied['files']) == 29
    for n, descriptor in copied['files'].items():
        body = (HERE/n).read_bytes()
        assert descriptor['public_only'] is True and sha(body) == descriptor['sha256'] and len(body) == descriptor['bytes']
    base = HERE/'actual'
    r = load(base/'report.json'); w = load(base/'worker/report.json'); m = load(base/'source_manifest.json')
    assert sha((base/'report.json').read_bytes()) == REPORT
    assert sha((base/'worker/report.json').read_bytes()) == WORKER
    assert sha((base/'source_manifest.json').read_bytes()) == MANIFEST
    assert sha((base/'sources.tar.gz').read_bytes()) == TAR == r['source_archive']['sha256']
    blobs = archive(base/'sources.tar.gz', m)
    for row in (r,w):
        terminal(row)
        assert row['source_hashes_start'] == row['source_hashes_end'] == m
        assert row['source_manifest_sha256'] == MANIFEST and row['coefficient_selection'] is False
    assert r['worker'] == w and r['worker_report_sha256'] == WORKER
    exact_int(r['child_returncode'],0)
    assert r['child_reaped'] is True and r['child_group_absent'] is True
    assert r['actual_observer_stage'] is True and r['model_forward'] is False
    for k in ('model_loaded','model_loading_exact','model_forward','model_execution_requested','initial_selection_admissible',
              'calibration_reauthenticated','exact_order','private_directory_handle_closed','private_outputs_retained','external_inputs_never_removed'):
        assert w[k] is True,k
    for k in ('coefficient_selection_performed','encrypted_execution','validation_inputs_read','tokenizer_execution',
              'contains_tokens_or_logits','baseline_ranges_in_initial_selection'):
        assert w[k] is False,k
    for k in ('semantic_passes','expected_entries','retained_entries'): exact_int(w[k],3200)
    for k in ('native_forward_started','native_forward_completed'): exact_int(w[k],3232)
    exact_int(w['baseline_complete'],64)
    exact_int(w['new_candidates_evaluated'],0); exact_int(w['new_protocol_candidates_used'],0)
    assert w['native_dtype']=='float32' and w['exact_dtype']=='float64'
    assert w['calibration_sha256']==CALIBRATION and w['baseline_manifest_sha256']==BASELINE
    assert w['backend_sources_before']==w['backend_sources_after'] and len(w['backend_sources_before'])==14
    assert w['backend_environment']=={'FORCE_TF_AVAILABLE':'0','USE_FLAX':'0','USE_TF':'0','USE_TORCH':'1','USE_TORCH_XLA':'0'}
    assert w['prefills_by_length']=={str(i):96 for i in range(1,33)}
    for name,count in (('prefill',3072),('teacher',64),('free',64)):
        row=w['by_kind'][name]
        for k in ('complete','expected','observed','records'): exact_int(row[k],count)
        for k in ('invalid','semantic_failures','unmeasured'): exact_int(row[k],0)
    for name in ('hook_invariance_by_length','exact_invariance_by_length'):
        assert set(w[name])=={str(i) for i in range(1,33)} and all(x is True for x in w[name].values())
    assert w['input_ranges']['initial_bounds_include_baseline'] is False
    assert sha(blobs['public/candidate3_manifest.json'])==BASELINE
    assert sha(blobs['public/protocol.json'])=='eea8e4b38b4c2719e63a0aa13028150564901305e9e5f1f9d0f2985c821cb70f'
    actual_outer=load(base/'outer_receipt.json')
    exact_int(actual_outer['session'],2619); exact_int(actual_outer['exit_code'],0)
    assert actual_outer['reaped'] is True and actual_outer['process_group_absent'] is True
    assert actual_outer['report_sha256']==REPORT and actual_outer['worker_report_sha256']==WORKER and actual_outer['source_archive_sha256']==TAR
    assert actual_outer['pids']==[r['pid'],w['pid']] and r['child_pid']==r['child_pgid']==w['pid']
    root=HERE/'root-reconstruction'; review=load(root/'review.json'); outer=load(root/'outer_receipt.json'); detail=load(root/'terminal_detail.json')
    assert sha((root/'review.json').read_bytes())==REVIEW and sha((root/'outer_receipt.json').read_bytes())==OUTER
    assert sha((root/'review.py.txt').read_bytes())==SCRIPT
    assert review['schema']=='qwen-initial3200-root-review-contract-v2' and review['passed'] is True
    for k,pin in (('parent_report_sha256',REPORT),('worker_report_sha256',WORKER),('source_archive_sha256',TAR),('source_manifest_sha256',MANIFEST),('reconstruction_script_sha256',SCRIPT),('calibration_sha256',CALIBRATION)):
        assert review[k]==pin
    for k in ('aggregate_ranges_independently_recomputed','baseline_excluded','per_context_coverage_independently_verified',
              'private_journals_immutable_at_review','source_inventory_independently_verified','native_model_loaded','outer_reaped','outer_group_absent','worker_reaped','worker_group_absent'):
        assert review[k] is True,k
    for k,n in (('exact_ordered_contexts',3200),('baseline_free_contexts',64),('private_files',9666),('native_forward_started',3232),('native_forward_completed',3232),('per_context_norm_input_sites',113),('per_context_silu_sites',28),('per_context_causal_score_sites',28),('outer_exit_code',0),('worker_exit_code',0)):
        exact_int(review[k],n)
    assert outer=={'schema':'qwen-initial3200-root-reconstruction-terminal-v1','exit_code':0,'reaped':True,'process_group_absent':True,'review_sha256':REVIEW,'script_sha256':SCRIPT}
    for k,v in outer.items():assert detail[k]==v
    exact_int(detail['session'],64028)
    assert detail['snapshot_descriptor']==review['snapshot_descriptor'] and detail['initial_report_sha256']==REPORT
    assert review['snapshot_descriptor']['sha256']=='8d6edbdd61a7bc4a18143dfbe67ab56dd9e26e68d5266e83c00d9fa21327f98d'
    exact_int(review['snapshot_descriptor']['bytes'],4611984)
    # Descriptor equality above is public attestation; this verifier never opens that path.
    assert not any(n.endswith('reconstructed-snapshot.json') for n in files)
    grant=load(HERE/'launch-review/launch.json'); launch=load(HERE/'launch-review/review.json')
    assert sha((HERE/'launch-review/launch.json').read_bytes())==GRANT==w['grant_sha256']
    assert sha((HERE/'launch-review/review.json').read_bytes())=='58c787c89acd29fda3a112c50f3f0f3ff9adea347b0c8a90a342b1d4e90e08a6'==grant['independent_review_sha256']
    assert launch['passed'] is True and grant['execute'] is True and grant['source_manifest_sha256']==MANIFEST
    assert grant['calibration']['sha256']==CALIBRATION and grant['baseline_manifest_sha256']==BASELINE
    assert grant['private_output']==w['private_output']['path']
    for name,report_pin,tar_pin,session in (
        ('preflight-v5','cc65c93236162f553acc136f7a97e29f80bdb222687e9c61ab3a4a4b29b2bc77','de0b7086cc103b13bd69faf98991e0936957ae6ea77eeadab832704ad552cee4',44184),
        ('import-smoke-v5','d39f76292878ca56197ac12a015dd388f0de37c86094260f3210503ad7f8e2e5','a5e1389590028c98b5b429bea302dc2f079f7a8b00c5ac1304b3a8938ed08ba9',55494)):
        p=HERE/'preparation'/name; pr=load(p/'report.json'); pw=load(p/'worker/report.json')
        assert sha((p/'report.json').read_bytes())==report_pin and sha((p/'sources.tar.gz').read_bytes())==tar_pin
        assert load(p/'source_manifest.json')==m and archive(p/'sources.tar.gz',m)==blobs
        for row in (pr,pw):terminal(row);assert row['source_hashes_start']==row['source_hashes_end']==m
        assert pr['worker']==pw and pr['worker_report_sha256']==sha((p/'worker/report.json').read_bytes())
        exact_int(pr['child_returncode'],0)
        t=load(p/'owner_terminal.json');exact_int(t['session_id'],session);exact_int(t['authoritative_exit_code'],0)
        if name=='preflight-v5':
            exact_int(pw['tests'],274);assert len(pw['installed_import_denials'])==8
            cases=ET.parse(p/'worker/tests.xml').findall('.//testcase');assert len(cases)==274
            assert not any(c.find(k)is not None for c in cases for k in ('failure','error','skipped'))
            assert grant['preflight_report_sha256']==report_pin
        else:
            assert pw['import_smoke_completed'] is True and pw['model_class_imported'] is True
            assert pw['model_constructed'] is False and pw['model_forward'] is False
    if args.current:
        current=Path('/tmp/qwen-expanded-prefix-observer-runtime-v5-20261104')
        execution=Path('/tmp/qwen-expanded-prefix-observer-actual-v2-20261104/execution_source')
        for path in (current,execution):
            found={p.relative_to(path).as_posix():sha(p.read_bytes()) for p in path.rglob('*') if p.is_file()}
            assert found==m
        assert sha(Path('/tmp/root_qwen_initial3200_terminal_review_20261104.py').read_bytes())==SCRIPT
        assert all(no_process(pid) for pid in actual_outer['pids']+[detail['pid']])
        assert no_process(r['child_pgid'],True) and no_process(detail['pgid'],True)
    print(json.dumps({'passed':True,'indexed_public_files':len(files),'sources':61,'contexts':3200,'baseline_trace_contexts':64,'native_forwards':3232,'private_files_root_reconstructed':9666,'current':args.current,'private_body_reads_by_verifier':False}))

if __name__=='__main__':main()
