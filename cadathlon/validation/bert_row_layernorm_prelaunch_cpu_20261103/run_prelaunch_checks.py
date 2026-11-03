"""Capture and verify the complete isolated LayerNorm prelaunch test sources."""
from pathlib import Path
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import time

LAB=Path(__file__).resolve().parent
sys.path.insert(0,str(LAB/'prototype'))
import collect


def main():
    output=Path(sys.argv[1])
    output.mkdir(parents=True,exist_ok=False)
    raw=collect.sources()
    before=collect.hashes(raw)
    with tarfile.open(output/'sources.tar.gz','w:gz') as archive:
        for name,data in raw.items():
            path=Path(name)
            member='repository/'+str(path.relative_to(collect.ROOT)) if path.is_relative_to(collect.ROOT) else 'prototype/'+path.name
            info=tarfile.TarInfo(member)
            info.size=len(data)
            archive.addfile(info,io.BytesIO(data))
    tests=['prototype/test_norm_arithmetic.py','prototype/test_norm_transport.py',
           'prototype/test_norm_assets.py','prototype/test_norm_component.py']
    command=[sys.executable,'-m','pytest','-q','-p','no:cacheprovider',*tests,
             '--junitxml='+str(output/'junit.xml')]
    started=time.perf_counter()
    with (output/'pytest.log').open('w') as stream:
        process=subprocess.run(command,cwd=LAB,stdout=stream,stderr=subprocess.STDOUT,check=False)
    after=collect.hashes(collect.sources())
    with tarfile.open(output/'sources.tar.gz') as archive:
        members=archive.getmembers()
        actual={member.name:hashlib.sha256(archive.extractfile(member).read()).hexdigest() for member in members}
        expected={('repository/'+str(Path(name).relative_to(collect.ROOT)) if Path(name).is_relative_to(collect.ROOT) else 'prototype/'+Path(name).name):digest for name,digest in before.items()}
        assert len(actual)==len(members) and actual==expected
    report=dict(passed=process.returncode==0 and before==after,command=command,
        returncode=process.returncode,source_hashes_before=before,source_hashes_after=after,
        source_files_unchanged=before==after,source_archive_members_verified=True,
        source_archive_sha256=hashlib.sha256((output/'sources.tar.gz').read_bytes()).hexdigest(),
        elapsed_s=time.perf_counter()-started,cpu_affinity=sorted(os.sched_getaffinity(0)),
        scope='CPU ideal arithmetic, metadata, artifact and lifecycle tests; no ciphertext execution.')
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({name:report[name] for name in ('passed','returncode','source_files_unchanged','elapsed_s')}))
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
