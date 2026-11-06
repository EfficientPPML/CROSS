"""Explicit public-only archival; never copies public weight bodies or private data."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import subprocess

LAB = Path('/tmp/qwen-actual-matrix-key-plan-20261103')
REVIEW = Path(__file__).resolve().parent
DEST = Path('/home/jianming_gatech/CROSS/cadathlon/validation/qwen_actual_matrix_key_plan_cpu_20261103')


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    DEST.mkdir(exist_ok=False)
    names = (
        'actual-v1/report.json', 'actual-v1/sources.tar.gz', 'actual-v1/process.log',
        'actual-v1/worker/report.json', 'actual-v1/worker/plan.json',
        'preflight-v1/report.json', 'preflight-v1/sources.tar.gz', 'preflight-v1/process.log',
        'preflight-v1/tests.xml', 'preflight-v1/terminal.json',
    )
    for name in names:
        target = DEST / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(LAB / name, target)
        assert digest(target) == digest(LAB / name)
    shutil.copyfile(REVIEW / 'verify.py', DEST / 'verify.py')
    shutil.copyfile(__file__, DEST / 'archive_driver.py')
    result = subprocess.run(['ps', '-p', '2037017,2037043,2035058', '-o', 'pid='],
                            check=False, capture_output=True, text=True)
    assert not result.stdout.strip()
    groups = subprocess.run(['ps', '-eo', 'pgid='], check=True, capture_output=True, text=True)
    assert not ({2037043, 2035058} & {int(line.strip()) for line in groups.stdout.splitlines() if line.strip()})
    write(DEST / 'session_terminal.json', {
        'session_id': 39684, 'observed_exit_code': 0, 'tool_session_reaped': True,
        'attestation_source': 'Parent agent /root authoritative tool observation communicated to reviewer',
        'parent_observed_terminal_at': '2026-11-03 18:04 UTC',
        'reviewer_direct_tool_session_poll': False,
        'independent_absence_checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'absent_producer_pids': [2037017, 2037043, 2035058],
        'absent_worker_process_groups': [2037043, 2035058],
        'scope': 'Parent-attested outer session exit/reap; archived direct-child wait plus independently observed process/group absence.',
    })
    (DEST / 'README.md').write_text('''# Public actual-matrix Qwen rotation plan

The completed public analysis covers all 234 matrix maps: 84 Q/K/V projections,
112 output/MLP projections, and 38 vocabulary blocks (37 × 4096 rows plus 384).
The tied embedding/head alias adds no second server matrix map. Every measured
matrix has all of its geometric diagonals present. Actual matrix keys are
`[1,64,128,60544,61504,62528,63552,64192]`; combining them with the 162 inherited
routing, normalization and bootstrap keys gives **166 keys**, including the
unreduced conjugation index 262143.

At the declared N131072/depth49/p60/d8 Q100/P14 profile, one raw rotation-key
pair occupies 956,301,312 bytes. The 166-key sum is 158,746,017,792 bytes
(147.84375 GiB). This is raw key-array accounting, excluding relinearization,
public/secret keys, model/ciphertext arrays, controls, compiler state, transfers
and duplicate storage. It is not a memory-peak or capacity guarantee. Recorded
chained linear calls total 22,008 per row tile across the 196 layer maps, plus
6,226 for the final-row vocabulary maps; these counts exclude nonlinear and
other graph work and are not benchmark timing.

The exact 22-source closure, 21-test preflight and actual child reports are
preserved unchanged. Parent agent `/root` authoritatively observed session
39684 exit 0 and reaped it at 18:04 UTC; the reviewer did not poll that handle.
The archived child record independently records direct wait 0 and an empty
owned process group, and this archive records a fresh PID/group absence check.
Collector duration was 19.30467984000279 seconds; no calibrated timing is claimed.

Run `python verify.py` for offline checks of all artifacts, both source archives,
all 234 map identities/shapes and chained gap schedules, exact base/combined
key union and byte arithmetic. `--current /tmp/qwen-actual-matrix-key-plan-20261103`
also checks current and both execution copies. The optional `--public-bodies`
reads only the 197 separately exported, pinned public matrix files, rechecks
file/payload/block hashes, and independently enumerates every actual diagonal
support. Public matrix bodies are not included here.

This evidence establishes the key plan for the specified F4096/S1..32 graph.
It does not establish a generated key store, actual-context security preflight,
complete model implementation, numerical HE accuracy, benchmark eligibility or
runtime readiness. Nonlinear-domain and refresh-error obligations remain.
No model forward, key generation, codec, backend or encrypted operation ran.
The source archives contain public code/metadata only, and no weight, key,
ciphertext, token, label or hidden-state body is copied into this archive.
''')
    write(DEST / 'artifact_hashes.json', {
        p.relative_to(DEST).as_posix(): digest(p) for p in sorted(DEST.rglob('*'))
        if p.is_file() and p != DEST / 'artifact_hashes.json'
    })
    print(DEST)


if __name__ == '__main__':
    main()
