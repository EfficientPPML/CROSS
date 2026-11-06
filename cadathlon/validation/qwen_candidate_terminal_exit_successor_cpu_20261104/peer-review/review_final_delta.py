"""Final source/count delta readback, public source only; no test execution."""
import hashlib,json
from pathlib import Path
root=Path('/tmp/qwen-expanded-candidate-checker-terminal-exit-successor-20261104')
manifest=Path('/tmp/qwen-expanded-candidate-checker-terminal-exit-sources-v1-20261104.json')
old_manifest=Path('/tmp/qwen-expanded-candidate-checker-ready-sources-v2-20261104.json')
here=Path(__file__).parent
sha=lambda b:hashlib.sha256(b).hexdigest()
raw=manifest.read_bytes();pin='45cd92473d9ba558288651d59d4f089c7fb9ba37650802848cad24687beaa437';assert sha(raw)==pin
expected=json.loads(raw);old=json.loads(old_manifest.read_bytes())
current={p.relative_to(root).as_posix():sha(p.read_bytes()) for p in root.rglob('*') if p.is_file()}
assert expected==current and len(current)==141
previous=json.loads((here/'successor-review.json').read_bytes())
for n in previous['reviewed_sources']:
 if n!='TERMINAL_SUCCESSOR.md':assert current[n]==previous['reviewed_sources'][n]
changed=sorted(n for n in old if current[n]!=old[n]);assert changed==['entry.py','preflight_expectations.json']
assert sorted(set(current)-set(old))==['TERMINAL_SUCCESSOR.md','test_terminal_exit.py']
a=json.loads((Path('/tmp/qwen-expanded-candidate-checker-ready-v2-20261104')/'preflight_expectations.json').read_bytes())
b=json.loads((root/'preflight_expectations.json').read_bytes())
assert a['synthetic_tests']==738 and b['synthetic_tests']==781
assert {k:v for k,v in a.items()if k!='synthetic_tests'}=={k:v for k,v in b.items()if k!='synthetic_tests'}
report=dict(schema='qwen-candidate-terminal-successor-final-delta-review-v1',scope='Public source and metadata only; no test rerun or actual outcome/body read.',
 source_root=str(root),source_manifest=str(manifest),source_manifest_sha256=pin,source_count=141,current_equals_manifest=True,
 inherited_source_count=139,inherited_unchanged=137,parent_overlays={n:current[n]for n in changed},
 additions={n:current[n]for n in sorted(set(current)-set(old))},
 prior_review_sha256=sha((here/'successor-review.json').read_bytes()),
 prior_entry_and_test_review_remains_applicable=True,
 expectations_delta={'synthetic_tests_before':738,'added_focused_tests':43,'synthetic_tests_after':781,'installed_import_denials_unchanged':10,'exclusions_unchanged':True},
 document_delta='Updates exact137+2+2 accounting and records peer-reported focused old36fail/7controls and successor43pass; independent source review did not rerun them.',
 full_preflight_still_required=True,frozen_original139_modified=False)
p=here/'final-delta-review.json';p.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'receipt':str(p),'sha256':sha(p.read_bytes()),'source_manifest_sha256':pin,'current_equal':True}))
