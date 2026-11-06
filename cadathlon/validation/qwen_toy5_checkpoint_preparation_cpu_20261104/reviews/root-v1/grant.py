from pathlib import Path
import datetime,hashlib,json
sha=lambda b:hashlib.sha256(b).hexdigest()
root=Path('/tmp/qwen-toy5-checkpoint-root-review-20261104')
proof=Path('/tmp/qwen-toy5-checkpoint-focused-proof-20261104/preparation-v1')
reviews={str(root/'preexecution-review.json'):'7648cfeab362fb30de97c81e4d0feeec703d6cd33076705940b0b6bb43e53db4',str(root/'pytest-ancestor-config-observation.json'):'ee7f409ed81ed8686d68015eb8f9261ed84213fb9bd63bc5d7b1c42c0ebae894','/tmp/qwen-toy5-checkpoint-security-source-review-20261104/review.json':'ae43cafaa0ef5475c3bfc13402dbadd3ef1217e3297f8b46a35a6ba455f2ef93','/tmp/qwen-toy5-harness-transformers-review-20261104/review.json':'ac64cb8e297b8c34a560d5387612dcc9039c1c433930b8aa45926041daa8b2a6'}
for p,h in reviews.items():assert sha(Path(p).read_bytes())==h
review=json.loads((root/'preexecution-review.json').read_bytes());assert review['passed']is True
for n,h in review['harness_pins'].items():assert sha((proof/n).read_bytes())==h
for n in review['prior_outputs_absent']:assert not(proof/n).exists() and not(proof/n).is_symlink()
configobs=json.loads((root/'pytest-ancestor-config-observation.json').read_bytes())
assert sha(Path(configobs['discovery_source']).read_bytes())==configobs['discovery_source_sha256']
assert not any(Path(n).exists()or Path(n).is_symlink()for n in configobs['absent_paths'])
proposal=json.loads((proof/'launch-proposal.json').read_bytes())
grant=dict(schema='q5-focused-root-run-grant-v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),authorized=True,owner='root',argv=proposal['argv'],source_count=261,selected_tests=48,scope=proposal['scope'],reviews=reviews,harness_pins=review['harness_pins'],healthy_job_deadline=False,automatic_retry=False,actual_role_authority=False,backend_imports_authorized=False,resource_capacity_evaluation_authorized=False,public_toy_numpy_authorized=True,encrypted_execution_authorized=False,script_sha256=sha(Path(__file__).read_bytes()))
out=root/'run-grant.json'
with out.open('x')as f:json.dump(grant,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps(dict(path=str(out),sha256=sha(out.read_bytes()),authorized=True,tests=48)))
