"""Verify the public ready proof without imports, model work or private reads."""
from pathlib import Path, PurePosixPath
import argparse, hashlib, io, json, os, tarfile
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent
ASSESS='candidate_proof/selector-assessments.json'
CANDIDATE='344954b3873a8847ececba320646f52612ca7277412c7de81a7313894e21799a'
BASELINE='7f6478a6bc710f3d2a370198e075e6ba8b0e55c144d981bca0eed59eaf176333'
NUMERIC=('checker_runtime.py','candidate_domains.py','strict_metrics.py',
         'test_checker_runtime.py','test_candidate_domains.py','test_strict_metrics.py')
DENIALS={'torch','transformers','jax','jaxlib','jaxite_word','tensorflow','tokenizers','safetensors','flax','torch_xla'}
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(name):return (ROOT/name).read_bytes()
def load(name):return json.loads(read(name))
def require(value,label):
    if not value:raise ValueError(label)
def integer(value,n):require(type(value)is int and value==n,'typed integer')

def contents(raw,expected):
    found={};total=0
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz')as stream:
        for member in stream:
            name=PurePosixPath(member.name);total+=member.size
            limit=64<<20 if member.name==ASSESS else 8<<20
            require(member.isfile()and member.name in expected and member.name not in found
                    and not name.is_absolute()and '..'not in name.parts and str(name)==member.name
                    and 0<member.size<=limit and total<=96<<20,'bounded exact archive member')
            found[member.name]=stream.extractfile(member).read()
    require({n:sha(v)for n,v in found.items()}==expected,'exact source archive')
    return found

def terminal(row):
    for name in ('passed','complete','cleanup_completed','signal_restoration_completed','source_files_unchanged',
                 'terminal_after_body_cleanup_and_signal_restoration_attempts',
                 'acceptance_requires_independent_exit0_reap_and_exact_report_hash'):
        require(row[name]is True,name)
    require(row['audit_denials']==[]and 'error_type'not in row,'clean terminal')

def proof(folder,count,tests,ready):
    prefix=folder+'/';parent=load(prefix+'report.json');worker=load(prefix+'worker/report.json')
    expected=load(prefix+'source_manifest.json');owner=load(prefix+'owner_terminal.json')
    integer(len(expected),count);blobs=contents(read(prefix+'sources.tar.gz'),expected)
    require(parent['source_archive']['sha256']==sha(read(prefix+'sources.tar.gz'))==owner['source_archive_sha256'],'archive linkage')
    require(parent['worker']==worker and parent['worker_report_sha256']==sha(read(prefix+'worker/report.json')),'worker binding')
    require(owner['report_sha256']==sha(read(prefix+'report.json')),'owner report identity')
    require(owner['source_manifest_sha256']==sha(read(prefix+'source_manifest.json')),'owner manifest')
    integer(owner['exit_code'],0);require(owner['reaped']is True and owner['producer_pids_absent']is True,'authoritative outer terminal')
    integer(parent['child_returncode'],0);require(parent['child_reaped']is True and parent['child_group_absent']is True,'direct wait/group absence')
    for row in (parent,worker):
        terminal(row);require(row['source_hashes_start']==row['source_hashes_end']==expected,'source before/after')
    integer(worker['tests'],tests)
    for name in ('failures','errors','skipped','pytest_returncode'):integer(worker[name],0)
    require(set(worker['installed_import_denials'])==DENIALS and len(worker['installed_import_denials'])==10,'ten import denials')
    tree=ET.fromstring(read(prefix+'worker/tests.xml'))
    integer(len(tree.findall('.//testcase')),tests)
    require(not any(tree.findall('.//'+n)for n in ('failure','error','skipped')),'XML success')
    require(parent['actual_candidate_stage']is False and parent['model_forward']is False,'no actual candidate work')
    require(json.loads(blobs['candidate_binding.json'])['ready']is ready,'declared readiness')
    require(json.loads(blobs['execution_authorization.json'])=={'execute':False},'source does not grant execution')
    integer(json.loads(blobs['preflight_expectations.json'])['synthetic_tests'],tests)
    return expected,blobs,owner,parent

def protected(m):
    return dict(nonlinear={n:r for n,r in m['nonlinear'].items()if not n.endswith('.inverse_sqrt')},
                **{n:m[n]for n in ('softmax','exp','initial_reciprocal','reciprocal_by_keys')})

def public_ready(blobs):
    origin=json.loads(blobs['ready_proof_origin.json'])['public_files'];require(len(origin)==17,'17 public proof copies')
    require({n for n in blobs if n.startswith('candidate_proof/')}==set(origin),'exact public copies')
    for name,row in origin.items():
        require(len(blobs[name])==row['bytes']and sha(blobs[name])==row['sha256'],'public origin identity')
    integer(len(blobs[ASSESS]),37283359)
    config=json.loads(blobs['candidate_binding.json'])
    for group in ('initial_proof','selector_proof'):
        for row in config[group].values():require(sha(blobs[row['blob']])==row['sha256'],'binding proof digest')
    raw=blobs[config['candidate']['blob']];require(sha(raw)==config['candidate']['sha256']==CANDIDATE,'candidate1 identity')
    candidate=json.loads(raw);integer(candidate['candidate_ordinal'],1);integer(candidate['candidate_index'],1)
    require(config['budget']==dict(max_candidates=3,ordinal=1,previous_attempts=[],manifest_sha256=CANDIDATE),'candidate budget')
    base=blobs['public/candidate3_manifest.json'];require(sha(base)==BASELINE,'protected original')
    require(protected(candidate)==protected(json.loads(base)),'unchanged protected circuits')
    initial={n:json.loads(blobs[config['initial_proof'][n]['blob']])for n in ('parent','worker','root','outer')}
    for n in ('parent','worker'):terminal(initial[n])
    integer(initial['worker']['native_forward_started'],3232);integer(initial['worker']['native_forward_completed'],3232)
    integer(initial['worker']['retained_entries'],3200);require(initial['worker']['model_loaded']is True,'model loaded')
    integer(initial['root']['private_files'],9666);require(initial['root']['passed']is True,'root reconstruction')
    require(initial['outer']['exit_code']==0 and initial['outer']['reaped']is True and initial['outer']['process_group_absent']is True,'root outer terminal')
    require(initial['outer']['review_sha256']==config['initial_proof']['root']['sha256'],'root outer identity')
    selector={n:json.loads(blobs[config['selector_proof'][n]['blob']])for n in ('report','outer','protected_checks')}
    terminal(selector['report']);require(selector['report']['publication_complete']is True,'complete selection publication')
    integer(selector['outer']['exit_code'],0)
    require(selector['outer']['reaped']is True and selector['outer']['process_group_absent']is True,'selector outer terminal')
    require(selector['outer']['report_sha256']==config['selector_proof']['report']['sha256']and selector['outer']['candidate_manifest_sha256']==CANDIDATE,'selector binding')
    require(selector['report']['outputs']['assessments.json']['sha256']==sha(blobs[ASSESS]),'complete assessments')
    integer(len(selector['protected_checks']),28)
    require(all(all(row[k]is True for k in ('radius_covered','observed_silu_covered','observed_centered_scores_covered'))for row in selector['protected_checks'].values()),'28 protected gates')
    review=json.loads(blobs['candidate_proof/selector-root-review.json'])
    require(review['passed']is True and review['candidate_manifest_sha256']==CANDIDATE,'reviewed selector')
    integer(review['polynomial_attempts_replayed'],648)

def absent(pid,group=False):
    try:(os.killpg if group else os.kill)(pid,0)
    except ProcessLookupError:return True
    return False

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--current',type=Path);args=parser.parse_args()
    pins=load('artifact_hashes.json');actual={}
    for path in ROOT.rglob('*'):
        require(not path.is_symlink(),'no archive symlinks')
        if path.is_file()and path.name!='artifact_hashes.json':actual[path.relative_to(ROOT).as_posix()]=sha(path.read_bytes())
    require(actual==pins,'exact public artifact inventory')
    p115,b115,_,_=proof('parent115',115,696,False)
    p137,b137,_,_=proof('ready-v1',137,727,True)
    expected,blobs,owner,parent=proof('ready-v2',139,738,True)
    for name in NUMERIC:require(expected[name]==p115[name]==p137[name],'unchanged six numerical files')
    for name in b137:
        if name.startswith('candidate_proof/'):require(blobs[name]==b137[name],'unchanged full public publication')
    overlay=json.loads(blobs['ready_successor_origin.json'])
    for name,pin in overlay['unchanged'].items():require(expected[name]==p137[name]==pin,'137 exact inheritance')
    for name,row in overlay['overlays'].items():require(p137[name]==row['parent_sha256']and expected[name]==row['successor_sha256'],'declared successor overlay')
    public_ready(blobs)
    for folder,failures in (('negative-v1',3),('successor-v1',0)):
        prefix='construction/'+folder+'/';r=load(prefix+'review.json')
        require(r['source_hashes_before']==r['source_hashes_after'],'construction before/after')
        require(sha(read(prefix+'sources.tar.gz'))==r['source_archive_sha256'],'construction archive')
        contents(read(prefix+'sources.tar.gz'),r['source_hashes_before'])
        require(r['source_hashes_before']['checker_process.py']==(p137 if failures else expected)['checker_process.py'],'construction runtime binding')
        integer(r['test_counts']['tests'],4);integer(r['test_counts']['failures'],failures)
        integer(r['child_exit_code'],1 if failures else 0)
        require(r['child_reaped']is True and r['child_group_absent']is True,'construction child retirement')
    review=load('reviews/transformer-bounds.json')
    for name,pin in review['sources'].items():require(p137[name]==pin,'scoped137 bounds review')
    review=load('reviews/cnn-ready.json')
    require(review['passed']is True and review['unresolved_concrete_findings']==[],'final CNN source review')
    for name,pin in review['reviewed_sources_after'].items():require(expected[name]==pin,'final CNN reviewed source')
    require(sha(read('reviews/cnn-review.py.txt'))==review['review_script_sha256'],'CNN review script')
    review=load('root-launch/review.json');grant=load('root-launch/launch.json')
    require(sha(read('root-launch/review.json'))=='52bcc2fbd07160da9d0789aa138cc761f13aec35af20b734acccfb12352d423b','root review pin')
    require(sha(read('root-launch/launch.json'))=='c3983fd975047bba16066106e85f9ae0768ed1e8c0fa30bef4da02b53253cd80','root grant pin')
    require(sha(read('root-launch/review.py'))==review['review_script_sha256'],'root review script')
    require(review['passed']is True and review['preflight_report_sha256']==owner['report_sha256']and review['source_archive_sha256']==owner['source_archive_sha256'],'root review linkage')
    core=load('root_handoff.json')['required_grant_fields']
    core.update(independent_review_sha256=sha(read('root-launch/review.json')),evidence_output='/tmp/qwen-expanded-candidate-checker-actual-v1-20261104',private_output='/tmp/qwen-expanded-candidate-checker-private-v1-20261104')
    require(json.dumps(core,sort_keys=True)==json.dumps(grant,sort_keys=True),'root exact grant scope and chosen output paths')
    if args.current:
        current={p.relative_to(args.current).as_posix():sha(p.read_bytes())for p in args.current.rglob('*')if p.is_file()}
        require(current==expected,'current139 exact')
        require(all(absent(pid)for pid in owner['pids'])and absent(parent['child_pgid'],True),'current producer/group absence')
    print('PASS: ready139,738 tests,10 denied imports; exact candidate1 public proofs. No model execution or private body read. Preserved137 proof and construction negatives.')

if __name__=='__main__':main()
