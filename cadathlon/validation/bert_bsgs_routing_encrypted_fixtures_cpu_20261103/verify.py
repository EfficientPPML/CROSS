from pathlib import Path
import hashlib,json,tarfile
p=Path(__file__).resolve().parent
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
pins=json.loads((p/'artifact_sha256.json').read_bytes())
assert {x.relative_to(p).as_posix() for x in p.rglob('*') if x.is_file() and '__pycache__'not in x.parts}==set(pins)|{'artifact_sha256.json'}
for n,h in pins.items():assert sha(p/n)==h,n
r=json.loads((p/'report.json').read_bytes());w=json.loads((p/'worker/worker_report.json').read_bytes())
assert r['passed']is True and r['execute']is True and r['stage']=='fixture' and r['worker_exit_code']==0
assert all(r[k]is True for k in ('worker_reaped','worker_group_empty_verified','all_direct_children_reaped','all_owned_groups_empty','process_cleanup_completed','terminal_scope_completed','source_files_unchanged','source_archive_verified'))
assert r['source_hashes_start']==r['source_hashes_end']==w['source_hashes_start']==w['source_hashes_end'] and len(r['source_hashes_start'])==147
assert sha(p/'worker/worker_report.json')==r['worker_report_sha256'] and w['passed']is True and w['phase_complete']is True and w['loader_cleanup_completed']is True and w['completed_ciphertexts']==2 and w['forbidden_attempts']==[]
assert sha(p/'sources.tar.gz')==r['source_archive_sha256']
with tarfile.open(p/'sources.tar.gz') as z:
 m=z.getmembers();assert len(m)==len({x.name for x in m})==147 and all(x.isfile() for x in m)
 assert {x.name:hashlib.sha256(z.extractfile(x).read()).hexdigest() for x in m}==r['source_hashes_start']
manifest=json.loads((p/'input_manifest.json').read_bytes());assert manifest==r['readback']['manifest']==w['readback']['manifest']
assert sha(p/'input_manifest.json')==r['readback']['manifest_sha256'] and r['readback']['tile_count']==2
for i in range(2):
 t=json.loads((p/f'tile{i}_transport.json').read_bytes())
 assert sha(p/f'tile{i}_transport.json')==manifest['tiles'][i]['manifest_sha256'] and t['tile_index']==i and t['tile_count']==2 and t['level']==49 and t['role']=='request'
 assert t['request_id']==manifest['contract']['nonce'] and t['context_id']==manifest['contract']['context_id']
review=json.loads((p/'root_readback.json').read_bytes());assert review['passed']is True and review['two_actual_ciphertexts_authenticated']is True and review['private_context_process_retired']is True and review['ciphertext_payload_bytes']==209715200
terminal=json.loads((p/'terminal_observation.json').read_bytes());assert terminal['observed_exit_code']==0 and terminal['reaped']is True and terminal['report_sha256']==sha(p/'report.json')
print('PASS:147 exact sources, two actual encrypted public fixtures; no decode/routing numerical result')
