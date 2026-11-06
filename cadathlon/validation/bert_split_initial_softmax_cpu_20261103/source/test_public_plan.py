from pathlib import Path
import json
from public_plan import build
def test_record_public_metadata_inventory():
    result=build()
    assert result['passed'] and len(result['layers'])==12
    assert [result['layers'][str(i)]['normalized_divisor'] for i in range(12)]==[16,64,64,64,32,64,64,32,32,32,32,32]
    assert set(result['union_keys'])==set(result['graph_keys'])|set(result['bootstrap_keys'])
    (Path(__file__).parent/'public_plan.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
