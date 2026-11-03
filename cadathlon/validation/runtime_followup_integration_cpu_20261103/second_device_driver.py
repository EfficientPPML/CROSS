"""Verify delayed BConv materialization preserves its construction device."""
from pathlib import Path
import hashlib
import io
import json
import os
import sys
import tarfile
import time

LAB=Path(__file__).resolve().parent;ROOT=LAB/'checkout';OUT=LAB/'second_device'
OUT.mkdir(exist_ok=False)
sys.path[:0]=[str(ROOT),str(ROOT/'cadathlon/common'),str(ROOT/'cadathlon/tests')]
os.chdir(ROOT)
def sources():
    paths={Path(__file__),ROOT/'cadathlon/tests/test_lazy_bconv.py',ROOT/'cadathlon/tests/test_encrypted_tensor_only.py',ROOT/'cadathlon/tests/test_scoped_ntt.py'}
    for folder in ('jaxite_word','cadathlon/common'):
        paths.update((ROOT/folder).glob('*.py'))
    return {str(p.relative_to(LAB)):p.read_bytes() for p in sorted(paths)}
raw=sources();before={n:hashlib.sha256(v).hexdigest() for n,v in raw.items()}
with tarfile.open(OUT/'sources.tar.gz','w:gz') as archive:
    for name,data in raw.items():
        member=tarfile.TarInfo(name);member.size=len(data);archive.addfile(member,io.BytesIO(data))
import jax
import pytest
assert jax.default_backend()=='cpu' and len(jax.devices())==2
command=['-q','cadathlon/tests/test_lazy_bconv.py::test_delayed_read_keeps_original_construction_device',
         'cadathlon/tests/test_scoped_ntt.py::test_guards_fall_back[other_device]',
         '--junitxml='+str(OUT/'junit.xml')]
started=time.perf_counter();result=pytest.main(command)
after={n:hashlib.sha256(v).hexdigest() for n,v in sources().items()}
report=dict(scope=__doc__,passed=result==0 and before==after,returncode=int(result),
    elapsed_s=time.perf_counter()-started,command=command,source_hashes_before=before,
    source_hashes_after=after,sources_unchanged=before==after,logical_cpu_devices=2,
    environment={key:os.environ.get(key) for key in ('JAX_PLATFORMS','XLA_FLAGS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')})
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('passed','returncode','elapsed_s','sources_unchanged')},indent=2))
raise SystemExit(0 if report['passed'] else 1)
