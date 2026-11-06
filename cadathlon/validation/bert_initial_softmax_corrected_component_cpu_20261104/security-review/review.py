"""Public evidence/source readback only; never imports or executes producer code."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import tarfile

ROOT = Path('/tmp/bert-initial-softmax-corrected-component-20261103')
OUT = Path(__file__).resolve().parent
OWNER = Path('/tmp/bert-initial-softmax-corrected-terminal-20261104/owner-terminal.json')
REPORT_SHA = 'f4c8e201cec1b643781e42e3f8ad542cb3e48f39cdd43418e1c9ce88b56f8f3e'
ARCHIVE_SHA = 'edbe516b8af583027268b54fee4b504c38cf7ba075c5a213820f90c2a58d161a'

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def group_absent(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    return False

def main():
    raw = (ROOT/'run-v1/report.json').read_bytes()
    assert sha(raw) == REPORT_SHA
    report = json.loads(raw)
    owner_raw = OWNER.read_bytes()
    owner = json.loads(owner_raw)
    assert owner['report_sha256'] == REPORT_SHA
    assert owner['source_archive_sha256'] == ARCHIVE_SHA
    assert owner['session_id'] == 44272 and owner['terminal_tool_chunk'] == 'e5927b'
    assert owner['exit_code'] == 0 and owner['reaped'] is True
    sources = report['source_hashes_before']
    assert len(sources) == 113 and sources == report['source_hashes_after']
    current = {n: sha((ROOT/n).read_bytes()) for n in sources}
    assert current == sources
    actual_inventory = set(ROOT.glob('*.py')) | {ROOT/n for n in ('candidate_manifest.json','actual_profile_audit.json','source_origin.json')}
    actual_inventory.update((ROOT/'checkout').rglob('*.py'))
    assert {str(p.relative_to(ROOT)) for p in actual_inventory} == set(sources)
    archive = ROOT/'run-v1/sources.tar.gz'
    assert sha(archive.read_bytes()) == ARCHIVE_SHA
    with tarfile.open(archive) as handle:
        members = handle.getmembers()
        assert len(members) == 113 and len({m.name for m in members}) == 113
        assert all(m.isfile() for m in members)
        assert {m.name: sha(handle.extractfile(m).read()) for m in members} == sources

    ring = report['ring']
    q, p = ring['q_towers'], ring['p_towers']
    assert (ring['degree'],ring['slots'],ring['depth'],ring['scale_bits'],ring['dnum'],len(q),len(p)) == (131072,65536,49,60,8,100,14)
    profile_raw = (ROOT/'actual_profile_audit.json').read_bytes()
    assert sha(profile_raw) == '2b2457e50da73b9ccf87984c089afcb99586fde863a48e98b89a1b7ccf872a17'
    profile = next(x for x in json.loads(profile_raw)['profiles'] if x['name'] == 'depth49_p60_d8')
    assert (q,p,ring['scale']) == (profile['q'],profile['p'],profile['canonical_scale'])
    assert len(set(q+p)) == 114
    assert all(type(x) is int and 1 < x < 2**31 and x % (2*ring['degree']) == 1 for x in q+p)
    qp_bits = math.prod(q+p).bit_length()
    assert qp_bits == ring['actual_QP_bit_length'] == 3403 and qp_bits <= 3523
    alpha = (len(q)+ring['dnum']-1)//ring['dnum']
    parts = [math.prod(q[i:i+alpha]) for i in range(0,len(q),alpha)]
    p_product = math.prod(p)
    assert alpha == 13 and len(parts) == ring['actual_switch_partitions'] == 8
    assert all(p_product >= x for x in parts)
    assert ring['actual_qp_is_secure_checked'] is True and ring['P_coverage_passed'] is True
    assert len(report['checks']) == 14
    for check in report['checks'].values():
        assert len(check['execution_rings']) == 1
        live = check['execution_rings'][0]
        assert (live['q_towers'],live['p_towers']) == (q,p)
        assert (live['degree'],live['num_slots'],live['dnum'],live['max_level'],live['security_bits'],live['security_model']) == (131072,65536,8,49,128,'classical')

    component = (ROOT/'component.py').read_text()
    ordered = ["ring=preflight(actual_security=True)", "keys=key_gen.gen_pke_pair", "server.program_initialization", "ct=client.encrypt_slots(scores)", "codec_patches.append", "output,checkpoints=graph.evaluate_tile", "if len(retained)!=14", "report['server_complete_before_any_decode']=True", "for module,name,original in codec_patches:setattr(module,name,original)", "decoded=np.asarray(client.decrypt_slots(value))"]
    positions = [component.index(x) for x in ordered]
    assert positions == sorted(positions)
    codecs = ['encrypt','decrypt','encrypt_slots','decrypt_slots','encrypt_slots_batch','decrypt_slots_batch','decode']
    assert report['codec_guards'] == codecs
    key_functions = ['gen_pke_pair','gen_evaluation_key','gen_rotation_key','key_switch_gen']
    assert report['key_generation_guards'] == {'key_gen':key_functions,'jaxite_word.key_gen':key_functions}
    labels = [p['label'] for p in report['phases']]
    expected_phases = ['two independent inputs encrypted and reduced to level38; all codecs guarded','all128_distinct16 seven ciphertext checkpoints complete','partial9_per_row_edge_keys seven ciphertext checkpoints complete','all14 ciphertext checkpoints complete; final client validation begins','client all-slot validation complete']
    phase_positions = [labels.index(x) for x in expected_phases]
    assert phase_positions == sorted(phase_positions)
    for flag in ('passed','complete','cleanup_completed','owned_key_directory_removed','source_files_unchanged','source_archive_verified','keyless_server_initialized','input_payload_and_metadata_unchanged','server_complete_before_any_decode'):
        assert report[flag] is True
    assert report['errors'] == report['cleanup_errors'] == report['collector_cleanup_errors'] == []
    for flag in ('preflight_only','full_attention_evaluated','full_block_evaluated','full_model_evaluated','refresh_performed','calibrated_timing'):
        assert report[flag] is False
    assert report['environment']['JAX_PLATFORMS'] == 'cpu'
    assert report['cpu_affinity'] == [176,177,178,179]
    assert report['owned_public_key_directory'] == '/dev/shm/cross-initial-softmax-keys-3uzzefji'
    assert not Path(report['owned_public_key_directory']).exists()
    assert all(not Path('/proc',str(pid)).exists() for pid in (1222982,1222978))
    assert group_absent(1222978)
    assert sha((ROOT/'run-v1/report.json').read_bytes()) == REPORT_SHA
    assert {n:sha((ROOT/n).read_bytes()) for n in sources} == current
    selected = ['collect.py','component.py','public_contract.py','lifecycle.py','fixtures.py','test_component_phase.py','test_prelaunch.py','source_origin.json','actual_profile_audit.json','checkout/cadathlon/common/support.py','checkout/jaxite_word/__init__.py','checkout/jaxite_word/he_params.py','checkout/jaxite_word/util.py','checkout/jaxite_word/ckks_ctx.py','checkout/jaxite_word/key_gen.py']
    result = dict(schema='bert-initial-softmax-corrected-security-source-review-v1',passed=True,observed_utc=datetime.now(timezone.utc).isoformat(),
        scope='Public report/source readback and scalar integer calculations only; no producer/backend imports, tests, cryptographic operations, or ciphertext/key/model bodies.',
        review_script_sha256=sha(Path(__file__).read_bytes()),report_sha256=REPORT_SHA,source_archive_sha256=ARCHIVE_SHA,
        owner_terminal_sha256=sha(owner_raw),owner_terminal=dict(session_id=44272,tool_chunk='e5927b',exit_code=0,reaped=True,authority='Producer owner receipt; reviewer independently checks current process/group absence.'),
        source_count=113,current_before_after_archive_exact=True,reviewed_source_pins={n:sources[n] for n in selected},
        actual_ring=dict(degree=131072,slots=65536,physical_q=100,physical_p=14,dnum=8,physical_partition_width=13,physical_partition_count=8,qp_bit_length=qp_bits,pinned_classical128_table_ceiling=3523,log2_qp=sum(map(math.log2,q+p)),p_bit_length=p_product.bit_length(),partition_bit_lengths=[x.bit_length() for x in parts],minimum_p_to_partition_ratio=p_product/max(parts),all14_recorded_live_contexts_equal=True),
        ordering=dict(source_order_checked=True,guarded_codecs=codecs,keygen_aliases=['key_gen','jaxite_word.key_gen'],phase_indices=phase_positions,all14_synchronized_before_restore_and_first_client_decode=True),
        cleanup=dict(reported_errors=[],reported_complete=True,key_directory_currently_absent=True,worker_and_parent_currently_absent=True,group_currently_absent=True,signal_restoration='Collector attempts restoration before final report; collector_cleanup_errors is empty and authoritative outer exit is 0.'),
        source_review=['Actual qp_is_secure and assert_p_coverage precede key generation. Context program_initialization also validates physical P coverage/kernel modulus envelope.',
            'require_secure_execution reads actual context Q/P/degree, checks classical-128 full-QP table, default noise and uniform-ternary assumptions, and binds ciphertext moduli to a prefix of that ring before arithmetic and at all14 client checks.',
            'Both key_gen aliases are guarded before server initialization; the server receives evaluation/rotation keys with no secret key. Seven class codec APIs remain guarded through both complete evaluations and their synchronization.',
            'The package unifies flat/qualified CKKSContext aliases; the producer adds its own checkout/common to import search paths. Historical report does not contain a separate loaded-module origin telemetry inventory.',
            'Cleanup attempts provider/compiled/context caches, retained references, JAX caches, rotation store, owned key directory, and keygen restoration. This is review of the successful ordinary path, not a new universal failure-lifecycle proof.'],
        limitations=['Security result is the frozen implementation/table and actual recorded ring check; no independent estimator or new cryptographic experiment.',
            'One dedicated process with separate client/server objects, not an OS sandbox or three-process boundary.',
            'Only two fixed synthetic initial-softmax fixtures and14 checkpoints; no future normalized rounds, refresh, full attention/block/model, calibrated timing, or precision transfer. Numerical gate readback is owned separately by root/producer.'])
    (OUT/'review.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,source_count=113,review_sha256=sha((OUT/'review.json').read_bytes()),qp_bits=qp_bits,p_bits=p_product.bit_length(),maximum_partition_bits=max(x.bit_length() for x in parts))))

if __name__ == '__main__':
    main()
