"""Stdlib source/hash/AST readback only; never imports candidate modules."""
from pathlib import Path
import ast,hashlib,io,json,tarfile
ROOT=Path('/tmp/qwen-keygen-actual-runtime-20261104')
HANDOFF=Path('/tmp/qwen-keygen-actual-runtime-handoff-20261104')
OUT=Path(__file__).parent
sha=lambda b:hashlib.sha256(b).hexdigest()
canonical=lambda v:(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
raw=(HANDOFF/'sources.json').read_bytes();expected=json.loads(raw)
assert sha(raw)=='fa75863298e5b41ce7adbafc1a03030ce27728fcbe68ca0742137f5f6ed706e4'
assert canonical(expected)==raw and len(expected)==85
current={n:sha((ROOT/n).read_bytes())for n in expected};assert current==expected
archive=(HANDOFF/'sources.tar.gz').read_bytes();assert sha(archive)=='099569e055733246fee17c452aee138fd9fb0689b0ad71d71fa6be59f1050c4a'
with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz')as tar:
 members=tar.getmembers();assert len(members)==85 and all(m.isfile()for m in members)
 assert {m.name:sha(tar.extractfile(m).read())for m in members}==expected
runtime=('keygen_entry.py','keygen_output.py','keygen_dependencies.py','keygen_process.py')
text={n:(ROOT/n).read_text()for n in runtime}
for v in text.values():ast.parse(v)
assert "name not in ('report.json','failure.json')"in text['keygen_output.py']
assert "core.load(('key_gen','he_params','util','rns'))"in text['keygen_process.py']
assert text['keygen_process.py'].index('launch,roles,profile,dep,res,public_records,protected=admit')<text['keygen_process.py'].index('core=install_core(')<text['keygen_process.py'].index('result=generate(')
assert "type(value)is not int or value not in(0,1)"in text['keygen_entry.py']and 'return INFRASTRUCTURE_EXIT'in text['keygen_entry.py']
encoding=[]
for name in('keygen_entry_status.json','keygen_static_policy.json'):
 data=(ROOT/name).read_bytes();parsed=json.loads(data);compact=canonical(parsed)
 assert data!=compact
 encoding.append(dict(file=name,sha256=sha(data),observed_bytes=len(data),required_compact_bytes=len(compact),
  compact_sha256=sha(compact),current_encoding_matches_parser=False))
assert json.loads((ROOT/'keygen_entry_status.json').read_text())['actual_entry_ready']is False
end={n:sha((ROOT/n).read_bytes())for n in expected};assert end==current
review=dict(schema='qwen-keygen-runtime85-source-peer-review-v1',scope='Source/readback only; no candidate import, test, backend, metadata inventory, capacity acquisition, body read or key generation.',
 source_root=str(ROOT),source_manifest_sha256=sha(raw),source_archive_sha256=sha(archive),source_count=85,
 exact_before_after_archive=True,reviewed_runtime={n:expected[n]for n in runtime},
 outcome='One concrete static JSON encoding mismatch; current deny remains safe, runnable successor requires correction.',
 findings=[dict(kind='static-policy canonical encoding mismatch',status='requires declared successor before meaningful runnable preparation',
  source='keygen_process.parsed() used for entry_status and static_policy requires canonical(value)==raw',files=encoding,
  limitation='Inferred directly from source and stdlib byte comparison; candidate parser and runner were not executed.')],
 source_findings=dict(whole_public_model_and_two_named_client_admissions_before_rng=True,
  dependency_environment_capacity_before_imports_and_genuine_security=True,
  external_original_parent_check_before_store_creation=True,original_Python_secret_used_for_all_168_primitives=True,
  public_private_manifest_separation_and_complete_readback=True,original_owners_and_cleanup_primary_policy=True,
  final_report_needs_independent_normal_zero_reaped_outer=True,late_publication_close_errors_escape_to_reserved70=True,
  per_loaded_backend_file_and_spec_origins_bound_by_external_policy=True),
 caveats=['FrozenImports core_enabled only gates package roots; current exact source installs then enables before load and no current premature flat import path was found. Client adaptation additionally gates flat core names.',
 'Declared installed policy, actual import-only proof, actual role maps/positive resource review and launch authority remain absent.',
 'Root source archive verification remains independent; runtime authenticates current source map and externally pinned archive identity, not an archive body.',
 'Existing client39 minimum lifecycle pin remains historical until explicitly selected successor.'],
 shared_output_adoption='Client source copy declares permitted logits.npy/result.json names; keygen85 unchanged and no execution proof transferred.',
 current_capacity_observed=False,candidate_imports=False,tests_executed=False,actual_execution_authorized=False)
(OUT/'review.json').write_text(json.dumps(review,sort_keys=True,indent=2)+'\n')
print(json.dumps({'source_review_complete':True,'finding_count':1,'review_sha256':sha((OUT/'review.json').read_bytes())}))
