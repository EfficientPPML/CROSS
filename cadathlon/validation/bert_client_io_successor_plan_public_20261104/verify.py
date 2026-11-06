"""Public code/metadata plan audit; no producer payload or private reads."""
from pathlib import Path
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def main():
    pins = json.loads((ROOT / 'source_pins.json').read_bytes())
    assert len(pins) == 16
    for name, pin in pins.items():
        raw = (ROOT / name).read_bytes()
        assert len(raw) == pin['bytes']
        assert hashlib.sha256(raw).hexdigest() == pin['sha256']
    stage4 = json.loads((ROOT / 'inputs/stage5/stage4_source_manifest.json').read_bytes())
    assert len(stage4) == 217
    ready_raw = (ROOT / 'inputs/gain4/source-bound-preflight-v1/report.json').read_bytes()
    assert hashlib.sha256(ready_raw).hexdigest() == '6fbf898fbcb8a884e2f2c7f1e2308dfd04171be990bfde82a20200d20f00e19c'
    ready = json.loads(ready_raw)
    producer = ready['source_hashes_start']
    assert len(producer) == 163 and producer == ready['source_hashes_end']
    assert ready['source_archive_sha256'] == '382933c982162e6d83c7d5cc366e59b7abfd674ef408efcc529c9ba13db912ce'
    assert ready['passed'] is True and ready['execute'] is False
    inventory = json.loads((ROOT / 'inputs/gain4/source_inventory.json').read_bytes())
    assert set(inventory) == set(producer)
    for name in ('component_protocol.py', 'component_worker.py', 'collect_component.py', 'component_contract.py'):
        assert hashlib.sha256((ROOT / 'inputs/gain4' / name).read_bytes()).hexdigest() == producer[name]
    sources = {}
    for name, fn in (('component_protocol.py', 'terminal_gate'),
                     ('component_worker.py', 'prior_gates'),
                     ('collect_component.py', 'preceding_config')):
        tree = ast.parse((ROOT / 'inputs/gain4' / name).read_bytes())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn)
        if fn == 'terminal_gate':
            assert 'expected_sources' in [a.arg for a in function.args.kwonlyargs]
        else:
            calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute) and n.func.attr == 'terminal_gate']
            assert len(calls) == (1 if fn == 'prior_gates' else 2)
            assert all(any(k.arg == 'expected_sources' and isinstance(k.value, ast.Name)
                           and k.value.id == 'expected' for k in call.keywords) for call in calls)
            sources[fn] = 'currently passes current consumer expected map; successor must pass fixed producer map'
    contract = ast.parse((ROOT / 'inputs/gain4/component_contract.py').read_bytes())
    definitions = {n.name: n for n in contract.body if isinstance(n, ast.FunctionDef)}
    assert {'make_contract', 'validate_contract', 'ciphertext_contract'} <= definitions.keys()
    result = dict(plan_verified=True, copied_public_inputs=16,
                  original_stage5_parent_sources=89, original_stage4_producer_sources=217,
                  gain4_producer_sources=163, producer_map_sha256=hashlib.sha256(
                      (json.dumps(producer, sort_keys=True, separators=(',', ':')) + '\n').encode()).hexdigest(),
                  distinct_source_call_sites=sources,
                  gain4_server_completion='pending; externally pinned terminal report required',
                  client_v2_final_preflight='pending independent final readback',
                  actual_execution_authorized=False,
                  scope='Portable source/metadata plan checks only; no new runtime, key, ciphertext, private body, model or HE execution')
    (ROOT / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
