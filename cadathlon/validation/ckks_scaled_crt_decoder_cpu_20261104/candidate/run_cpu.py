"""Dedicated CPU test driver. No key/context/ciphertext/model operation."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
output = Path(sys.argv[1])
output.mkdir(parents=False, exist_ok=False)
identity = {'pid': os.getpid(), 'pgid': os.getpgrp(), 'scope': 'clear CPU unit arithmetic'}
(output / 'started.json').write_text(json.dumps(identity, sort_keys=True) + '\n')
for name, value in {'JAX_PLATFORMS': 'cpu', 'JAX_PLATFORM_NAME': 'cpu',
                    'JAX_ENABLE_X64': 'True', 'PYTHONDONTWRITEBYTECODE': '1'}.items():
    if os.environ.get(name) != value:
        raise RuntimeError(f'{name} must be set before test import')
import pytest
start = time.monotonic()
code = pytest.main(['-q', '-p', 'no:cacheprovider', '--rootdir=' + str(ROOT),
                    str(ROOT / 'test_real_decode.py'), '--junitxml=' + str(output / 'tests.xml')])
module = sys.modules.get('test_real_decode')
records = [] if module is None else module.RECORDS
report = {**identity, 'passed': code == 0, 'pytest_exit_code': int(code),
          'elapsed_seconds': time.monotonic() - start,
          'records': records, 'real_crt': True, 'real_fft': True,
          'ciphertext_decryption': False, 'model_forward': False, 'key_generation': False,
          'encrypted_failure_recovered': False,
          'source_path': str(ROOT), 'recorded_case_scale': None if module is None else module.SCALE}
if module is not None:
    report.update(jax_backend=module.jax.default_backend(),
                  device_platforms=[d.platform for d in module.jax.devices()],
                  numpy_version=module.np.__version__, jax_version=module.jax.__version__,
                  import_guard_denials=module.IMPORT_GUARD_DENIALS,
                  device_access_denials=module.DEVICE_ACCESS_DENIALS)
    local_origins = {}
    for name, loaded in tuple(sys.modules.items()):
        location = getattr(loaded, '__file__', None)
        if location is None:
            continue
        path = Path(location).resolve()
        if path.is_relative_to(ROOT / 'checkout') and path.is_file():
            local_origins[name] = {'path': str(path.relative_to(ROOT)),
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    report['loaded_local_modules'] = local_origins
(output / 'report.json').write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
raise SystemExit(code)
