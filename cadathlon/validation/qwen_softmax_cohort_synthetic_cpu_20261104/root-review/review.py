"""Independent public-byte, process-retirement and operation-count review."""
from pathlib import Path
import hashlib,json,os,tarfile,xml.etree.ElementTree as ET
from datetime import datetime,timezone
P=Path('/tmp/qwen-causal-softmax-cohort-public-proof-20261104/development-v1')
S=Path('/tmp/qwen-causal-softmax-cohort-successor-20261104')
O=Path('/tmp/qwen-cohort-root-review-20261104');O.mkdir(exist_ok=False)
sha=lambda raw:hashlib.sha256(raw).hexdigest()
load=lambda p:json.loads(p.read_bytes())
m=load(P/'sources-before.json');r=load(P/'report.json')
assert len(m)==197 and sha((P/'sources-before.json').read_bytes())=='69bbd4561761a400fc676dee63ab31b50499096d100a6752cee1567d6819b1fd'
assert sha((P/'report.json').read_bytes())=='7c94effe3dc5490d3499e66ecc7ce0bae8bb5ec72b4e6cb0a9c109a5016d5a61'
assert r['passed']is True and r['tests']==136 and r['bad_tests']==0 and r['child_returncode']==0
assert r['synthetic_only']is True and r['model_forward']is False and r['encrypted_execution']is False
for root in (S,P/'execution_source'):
    assert {str(p.relative_to(root)):sha(p.read_bytes())for p in root.rglob('*')if p.is_file()}==m
assert m==load(P/'sources-after.json')==load(P/'execution-sources-after.json')
assert sha((P/'sources.tar.gz').read_bytes())=='023cb45bfa7e966846c54e6e1244d53be0094699185a14ce1e7db55988e7d212'
with tarfile.open(P/'sources.tar.gz')as archive:
    members=archive.getmembers();assert len(members)==197 and all(x.isfile()for x in members)
    assert {x.name:sha(archive.extractfile(x).read())for x in members}==m
parent=load(S/'cohort_parent_sources.json');assert len(parent)==191 and all(m[n]==h for n,h in parent.items())
assert all(sha((Path('/tmp/qwen-expanded-full-model-runtime-v3-20261104')/n).read_bytes())==h for n,h in parent.items())
cases=list(ET.parse(P/'tests.xml').getroot().iter('testcase'))
assert len(cases)==136 and all(all(x.find(tag)is None for tag in ('failure','error','skipped'))for x in cases)
for pid in (3175372,3175385):
    assert not Path('/proc',str(pid)).exists()
    try:os.killpg(pid,0)
    except ProcessLookupError:pass
    else:raise AssertionError('owned group still present')
manifest=load(S/'inputs/candidate_manifest.json');certificate=load(S/'inputs/certificates.json')
groups=[]
for tile in (0,1):
    row={}
    for k in range(max(2,16*tile+1),16*(tile+1)+1):
        rule=manifest['reciprocal_by_keys'][str(k)]
        bound=certificate['softmax_K'][k-1]['existing_inverse_refresh_bound']
        row.setdefault((rule['iterations'],bound),[]).append(k)
    groups.append([dict(iterations=n,bound=b,keys=ks)for(n,b),ks in row.items()])
assert list(map(len,groups))==[7,3]
literal=cohort=rotations=products=0
for layer in range(28):
    s=manifest['softmax'][str(layer)]['squarings'];literal+=31*(3+2*s)
    for group in sum(groups,[]):
        cohort+=3+2*s;rotations+=20+10*s;products+=18+2*s*group['iterations']
assert(literal,cohort,rotations,products)==(11656,3760,20200,32196)
result=dict(passed=True,utc=datetime.now(timezone.utc).isoformat(),sources=197,tests=136,parent191_preserved=True,
    source_archive_and_execution_exact=True,owned_processes_and_groups_absent=True,shared_tool_wrapper_excluded=True,
    owner_terminal_observation='session31548 direct51ec66 exit0/reaped; separate owner receipt',
    portable_capsule_verification='0776c8 exit0,42 indexed files',cohorts=groups,
    operation_counts=dict(literal_refreshes=literal,cohort_refreshes=cohort,cohort_rotations=rotations,cohort_products=products),
    scope='Synthetic full-slot component only; static/public arithmetic-count recomputation. No runtime imports, tests rerun, body reads, encrypted accuracy, measured speed, or full-model integration.')
(O/'review.py').write_bytes(Path(__file__).read_bytes());(O/'review.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(passed=True,review=str(O/'review.json'),sha256=sha((O/'review.json').read_bytes()),sources=197,tests=136)))
