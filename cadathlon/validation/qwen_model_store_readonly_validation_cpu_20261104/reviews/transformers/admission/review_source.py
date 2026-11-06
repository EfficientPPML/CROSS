"""Source-only peer review capture; do not execute the root admission script."""
import ast
import hashlib
import json
from pathlib import Path

TARGET=Path('/tmp/qwen-model-store-readonly-root-launch-20261104/verify_terminal_and_admit.py')
OUT=Path('/tmp/qwen-model-root-admission-transformers-review-20261104')
sha=lambda raw:hashlib.sha256(raw).hexdigest()
raw=TARGET.read_bytes()
assert sha(raw)=='1b6c64d0b7be92864d199712220c2480ef0cda9c90b96504890c63f882582e34'
tree=ast.parse(raw,filename=str(TARGET))
pins={}
for node in tree.body:
    if isinstance(node,ast.Assign)and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
        if node.targets[0].id in ('SOURCE_SHA','ARCHIVE_SHA','CONSUMER_SHA','REPORT_SHA'):
            pins[node.targets[0].id]=ast.literal_eval(node.value)
assert pins==dict(SOURCE_SHA='b73e669b0cc2bc34d7930f2a8a4c620be719fe663dfa6ae7b12df7cae22c2bc0',
    ARCHIVE_SHA='06a7ae13b6df2adb9808af69d7906eec1e86ed811c419da9c1d064479fe95a25',
    CONSUMER_SHA='d67a5df7aff7cd6560e88bd7731cf124ceefba18f4a2c83c443aa1ad1aa72ff8',
    REPORT_SHA='cc8aaab893268b96692d9998412b121b97f7cfdd5ea47c5f3a60949fe620203c')
consumer=Path('/tmp/qwen-keygen-authority-boundary-20261104')
map_path=Path('/tmp/qwen-keygen-authority-boundary-sources-v1-20261104.json')
map_raw=map_path.read_bytes();expected=json.loads(map_raw)
assert sha(map_raw)==pins['CONSUMER_SHA']and len(expected)==28
for name,digest in expected.items():assert sha((consumer/name).read_bytes())==digest
pure=(consumer/'model_admission.py').read_bytes();pure_tree=ast.parse(pure)
functions={n.name:n for n in pure_tree.body if isinstance(n,ast.FunctionDef)}
assert 'authenticate_model'in functions and 'require_admitted'in functions
key_tree=ast.parse((consumer/'keygen_authority.py').read_bytes())
denied=next(n for n in key_tree.body if isinstance(n,ast.FunctionDef)and n.name=='create_actual_keygen_process')
assert len(denied.body)==1 and isinstance(denied.body[0],ast.Raise)
assert denied.body[0].exc.func.id=='PermissionError'
text=raw.decode()
assert 'consumer_source_manifest_sha256=CONSUMER_SHA'in text
assert 'assert model.execution_authorized is False'in text
assert 'key_generation_authorized=False'in text and 'model_execution_authorized=False'in text
assert 'scientific_acceptance_claimed=False'in text and 'atomic_snapshot_claimed=False'in text
assert 'tool_session_id=27288'in text and "terminal_tool_chunk='724ea7'"in text
assert TARGET.read_bytes()==raw
report=dict(schema='qwen-root-model-admission-peer-source-review-v1',passed=True,
    source_only_ready=True,target=str(TARGET),target_sha256=sha(raw),bindings=pins,
    preparation_consumer_count=28,consumer_current_bytes_equal_reviewed_map=True,
    pure_model_admission_sha256=sha(pure),consumer_actual_entry_immediate_denial=True,
    findings=[
      'Known root-owned27288/724ea7 normal-zero/reap is explicitly external input, not inferred from report presence. Script separately checks recorded PID/group absence.',
      'Actual42 current/source/archive and original grant/live/report hashes are bound before admission. Public report is read through a held original output directory/file with inode/extent/hash readback.',
      'Report stores must equal exact prelaunchc302 expectation. Post-terminal traversal is lstat-only over three declared roots and198 named files; no parameter file is opened or hashed.',
      'Unchanged authenticate_model validates full positive report/checks/counts/scope/outer/root/source chain. require_admitted binds exact preparation consumerd67a; this is a real reviewed consumer, not a fabricated future actual source.',
      'Root proof and summary explicitly leave model-forward/key-generation/scientific authority false. Future real role sources need separately trusted exact consumer pins and their own current-store/role grant checks.',
      'Pure imports occur only after consumer bytes are authenticated, reject NumPy/backends, verify imported local origins, and recheck both source trees before exclusive root proof publication.'
    ],unresolved_concrete_blockers=[],
    limitations=[
      'Reviewer has not executed the proposed script, imported the pure gate, observed current borrowed metadata, read model bodies, or issued a launch/admission.',
      'Metadata comparison is not atomic and cannot promise future unchanged contents; later held readers/role gates remain required.',
      'Any written root proof requires the root verifier itself to complete normally; partial output presence is not an admission.',
      'The single post-terminal metadata pass and output/root proof publication are proposed code behavior, not observations from this source review.'
    ],root_script_executed=False,pure_gate_imported=False,tests_rerun=False,
    borrowed_metadata_stat=False,borrowed_body_read=False,launch_or_model_admission_issued=False,
    review_script_sha256=sha(Path(__file__).read_bytes()))
(OUT/'review.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(passed=True,source_only=True,review_sha256=sha((OUT/'review.json').read_bytes()))))
