"""Run the regression suite and retain numerical evidence from its secure check."""
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
records_path = Path(sys.argv[1])
code = pytest.main(sys.argv[2:])
records = {name: getattr(sys.modules.get(name), 'RECORDS', [])
           for name in ('test_lazy_bconv', 'test_encrypted_tensor_only')}
records_path.write_text(json.dumps(records, indent=2, allow_nan=False)+'\n')
raise SystemExit(int(code))
