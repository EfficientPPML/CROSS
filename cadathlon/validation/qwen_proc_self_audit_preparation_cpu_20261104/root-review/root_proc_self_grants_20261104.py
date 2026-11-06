from pathlib import Path
import datetime,hashlib,json
root=Path('/tmp/qwen-proc-self-root-review-20261104')
sha=lambda b:hashlib.sha256(b).hexdigest()
cases=(('client24','/tmp/qwen-client-kernel-focused-proof-20261104/preparation-v2','4018e140594b9335ff8c9bb7aad56d5a4ca3186e5b9974a36da0e4588c64fa13'),('keygen32','/tmp/qwen-keygen-proc-self-focused-proof-20261104/preparation-v2','c00ac9a85897feb835f6c3edee5f55b36d9f880ce073da40c9b1cb4c759da4e9'))
for label,proof_path,pin in cases:
 proof=Path(proof_path);review=(root/(label+'-preexecution-review.json')).read_bytes();assert sha(review)==pin
 obj=json.loads(review);assert obj['passed']is True
 for name,digest in obj['harness_pins'].items():assert sha((proof/name).read_bytes())==digest
 for name in obj['prior_outputs_absent']:assert not(proof/name).exists()
 raw=(proof/'launch-proposal.json').read_bytes();proposal=json.loads(raw)
 grant=dict(schema='qwen-proc-self-focused-root-run-grant-v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),label=label,authorized=True,root_review_sha256=pin,launch_proposal_sha256=sha(raw),argv=proposal['argv'],scope=proposal['scope'],source_count=obj['source_count'],tests=len(obj['static_ordered_nodes']),healthy_job_deadline=False,automatic_retry=False,actual_role_authority=False,backend_imports_authorized=False,resource_capacity_evaluation_authorized=False,grant_script_sha256=sha(Path(__file__).read_bytes()))
 path=root/(label+'-run-grant.json')
 with path.open('xb')as f:f.write((json.dumps(grant,sort_keys=True,indent=2)+'\n').encode())
 print(json.dumps(dict(label=label,path=str(path),sha256=sha(path.read_bytes()),authorized=True)))
