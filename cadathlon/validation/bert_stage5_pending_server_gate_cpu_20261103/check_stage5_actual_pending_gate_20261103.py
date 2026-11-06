"""Read public completed predecessors and reject the actual incomplete stage4."""
from pathlib import Path
import hashlib, io, json, os, sys, tarfile, time

LAB = Path('/tmp/bert-client-decryption-stage5-ready-20261103')
OUT = Path('/tmp/bert-stage5-actual-pending-gate-20261103')
sys.path[:0] = [str(LAB), str(LAB/'runtime/stage'), str(LAB/'runtime')]
import collect

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def main():
    OUT.mkdir(exist_ok=False)
    raw = collect.sources()
    raw['proof/check_stage5_actual_pending_gate_20261103.py'] = Path(__file__).read_bytes()
    before = {n: sha(v) for n,v in raw.items()}
    with tarfile.open(OUT/'sources.tar.gz', 'w:gz') as archive:
        for name, data in raw.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    stage4 = Path('/tmp/bert-server-compute-stage4-successor-v2-20261103/actual-v2')
    incomplete_raw = (stage4/'report.json').read_bytes()
    incomplete = json.loads(incomplete_raw)
    assert incomplete['passed'] is False
    assert incomplete.get('actual_execution_attempted') is True
    forbidden_roots = (
        Path('/dev/shm/cross-full-bert-keys-143b3f5c1573427d841760a3c9d5d33d/client_private'),
        Path('/tmp/bert-size0-input-client-yscphz3z'),
        Path('/tmp/bert-size0-reference-client-voxvu9y6'),
    )
    forbidden_attempts = []
    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str,bytes,os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).absolute()
        if any(path == p or p in path.parents for p in forbidden_roots):
            forbidden_attempts.append(str(path))
            raise AssertionError('private path opened before completed stage4')
        if path.suffix in ('.npz', '.npy', '.safetensors'):
            forbidden_attempts.append(str(path))
            raise AssertionError('payload opened in public pending-gate proof')
    sys.addaudithook(audit)
    worker = collect.worker
    private_entrypoints = []
    def deny(*args, **kwargs):
        private_entrypoints.append(True)
        raise AssertionError('later client input/reference gate was reached')
    worker.input_gate.authenticate = deny
    worker.reference_gate.public_metadata = deny
    start = time.monotonic()
    failure = None
    try:
        worker.authenticate(
            key_evidence=Path('/tmp/bert-full-key-generation-component-20261103/actual-v1'),
            key_report_sha256='022cbdf687273e765668184f32cb7cd501d371c4220916e1c8f02574cf3a7f08',
            key_workspace=Path('/dev/shm/cross-full-bert-keys-143b3f5c1573427d841760a3c9d5d33d'),
            model_evidence=Path('/tmp/bert-public-model-stage2-v2-component-20261103/actual-v2'),
            model_report_sha256='5b9589b51828d066706c4c46a6404d8c52fcd67a2c9aa9f3f57c73df8caeefe0',
            model_workspace=Path('/tmp/cross-bert-public-model-b51668856dab42a3846ba1b7c7a71d53'),
            stage3_evidence=Path('/tmp/bert-input-encryption-stage3-component-v3-20261103/actual-v3'),
            stage3_report_sha256='add5e6a3143be3931ad29c018cdee40bdf059497b8bfa987114f39baeb88631e',
            stage4_evidence=stage4, stage4_report_sha256=sha(incomplete_raw),
            io_workspace=Path('/tmp/cross-bert-input-encryption-7b255c2357c94060b73e474a1f8ef5fa'),
            fixture_evidence=Path('/tmp/bert-size0-input-fixture-20261103/actual-v1'),
            fixture_report_sha256='2218282c9d4ef468a448d00d6e7afeebc9c3a9fa0e9f3834227afe299ea28e10',
            reference_evidence=Path('/tmp/bert-size0-client-reference-20261103/actual-v1'),
            reference_report_sha256='b1309f39d7554542213a05c89864628472a76a3644b60d6633bf8eb3e741e1d9',
            profile_bytes=raw['runtime/stage/server_inputs/p60_profile_and_refresh.json'],
            plan_bytes=raw['runtime/stage/server_inputs/twelve_layer_plan.json'],
        )
    except ValueError as error:
        failure = str(error)
    assert failure == 'completed actual server proof required before client reads', failure
    assert not forbidden_attempts and not private_entrypoints
    current = collect.sources()
    current['proof/check_stage5_actual_pending_gate_20261103.py'] = Path(__file__).read_bytes()
    after = {n:sha(v) for n,v in current.items()}
    assert after == before and len(before) == 90
    with tarfile.open(OUT/'sources.tar.gz') as archive:
        members = archive.getmembers()
        assert len(members) == len({m.name for m in members}) == 90
        assert all(m.isfile() for m in members)
        assert {m.name:sha(archive.extractfile(m).read()) for m in members} == before
    report = dict(passed=True, scope='Public metadata prerequisite rejection only; no actual client decode.',
        source_hashes_before=before, source_hashes_after=after, source_count=90,
        source_archive_sha256=sha((OUT/'sources.tar.gz').read_bytes()),
        incomplete_stage4_report_sha256=sha(incomplete_raw), expected_rejection=failure,
        forbidden_private_or_payload_read_attempts=forbidden_attempts,
        later_client_gate_calls=len(private_entrypoints), actual_client_decode=False,
        cpu_affinity=sorted(os.sched_getaffinity(0)), elapsed_s=time.monotonic()-start)
    (OUT/'incomplete-stage4-report.json').write_bytes(incomplete_raw)
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({n:report[n] for n in ('passed','source_count','expected_rejection','actual_client_decode','elapsed_s')}))

if __name__ == '__main__':
    main()
