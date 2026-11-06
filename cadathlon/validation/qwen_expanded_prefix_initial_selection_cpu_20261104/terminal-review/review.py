"""Independently replay the frozen selector's public certificates and terminal proof.

All coefficient/history bytes were frozen before this review. Replays use the
same reviewed certificate implementation, without model/heldout evaluation.
"""
from pathlib import Path
import collections,hashlib,json,os,sys,tarfile
P=Path('/tmp/qwen-expanded-prefix-selector-actual-v1-20261104')
SOURCE=Path('/tmp/qwen-expanded-prefix-selector-admission-preflight-v2-20261104/execution_source')
OUT=Path('/tmp/qwen-expanded-prefix-selector-terminal-root-review-20261104')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def raw(path,pin):
    data=Path(path).read_bytes();assert sha(data)==pin;return data
def load(path,pin):return json.loads(raw(path,pin))
def canon(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def same(a,b):assert canon(a)==canon(b)
report_pin='7ba0ffae120c7b68b51d2ec1fc040e51e3fd12d989f1ab73316f231091917ad9'
candidate_pin='344954b3873a8847ececba320646f52612ca7277412c7de81a7313894e21799a'
source_pin='f3d0bf68c503c2baba2cc34b7806c152efae153bbc180a4ebb85aaeae2d833f0'
grant_pin='99d3cca99b00231110b77ead984a47624296a05e15b875cc03c99e93a76bc4b9'
r=load(P/'report.json',report_pin);sources=load(P/'source_manifest.json',source_pin)
assert len(sources)==36
assert all(r.get(k)is True for k in ('passed','complete','publication_complete','cleanup_completed','signal_restoration_completed','source_files_unchanged','actual_coefficient_selection','candidate_supported','all_norm_sites_supported','protected_rules_supported','private_inputs_never_removed','terminal_after_body_cleanup_and_signal_restoration_attempts','acceptance_requires_independent_exit0_reap_and_exact_report_hash'))
assert all(r.get(k)is False for k in ('model_execution','token_or_calibration_body_reads','validation_body_reads','encrypted_execution','synthetic_only'))
assert r['audit_denials']==[] and 'error_type'not in r and r['candidate_ordinal']==1
same(r['source_hashes_start'],sources);same(r['source_hashes_end'],sources)
assert r['source_manifest_sha256']==source_pin and r['grant_sha256']==grant_pin
same({p.relative_to(SOURCE).as_posix():sha(p.read_bytes())for p in SOURCE.rglob('*')if p.is_file()},sources)
assert not any(p.is_symlink()for p in SOURCE.rglob('*'))
archive_pin='821ba2035e4c0a1efbcdc9f3ee718c6ecadb2f013cd63ddaec3911d632494b6c'
raw(P/'sources.tar.gz',archive_pin);assert r['source_archive']['sha256']==archive_pin
with tarfile.open(P/'sources.tar.gz')as t:
    members=t.getmembers();assert len(members)==len({m.name for m in members})==36 and all(m.isfile()for m in members)
    same({m.name:sha(t.extractfile(m).read())for m in members},sources)
assert not Path('/proc',str(r['pid'])).exists()
try:os.killpg(r['pid'],0)
except ProcessLookupError:pass
else:raise AssertionError('selector process group alive')
grant=load('/tmp/qwen-expanded-prefix-selector-root-launch-20261104/launch.json',grant_pin)
assert grant['output_directory']==str(P)and grant['candidate_ordinal']==1
evidence=grant['initial_evidence'];public={n:raw(d['path'],d['sha256'])for n,d in evidence.items()}
assert all(len(public[n])==d['bytes']for n,d in evidence.items())
for name,key in (('parent','initial_parent_report_sha256'),('worker','initial_worker_report_sha256'),('root_review','initial_root_review_sha256'),('reconstruction_script','reconstruction_script_sha256'),('root_reconstruction_outer_receipt','root_reconstruction_outer_sha256')):assert r[key]==evidence[name]['sha256']
initial=json.loads(public['worker']);root=json.loads(public['root_review'])
assert r['initial_snapshot_sha256']==root['snapshot_descriptor']['sha256']
assert set(r['outputs'])=={'candidate_manifest.json','assessments.json','protected_checks.json'}
outputs={name:raw(P/name,desc['sha256'])for name,desc in r['outputs'].items()}
assert all(len(outputs[name])==desc['bytes']for name,desc in r['outputs'].items())
assert sha(outputs['candidate_manifest.json'])==candidate_pin
c,a,checks=(json.loads(outputs[name])for name in ('candidate_manifest.json','assessments.json','protected_checks.json'))
assert c['schema']=='qwen-expanded-initial-selection-v1'and c['candidate_ordinal']==c['candidate_index']==1
assert c['synthetic_only']is False and c['actual_coefficient_selection']is True
assert c['selection_grant_sha256']==grant_pin and c['selector_source_manifest_sha256']==source_pin
sys.path.insert(0,str(SOURCE))
import selector,certificate,select_rules
same(set(a)if False else sorted(a),sorted(selector.NORMS))
base=json.loads((SOURCE/'inputs/candidate3_manifest.json').read_bytes())
assert sha((SOURCE/'inputs/candidate3_manifest.json').read_bytes())==selector.BASELINE_SHA256
same(selector.protected(c),selector.protected(base))
union=initial['input_ranges']['native_exact_union'];attempts=0
for ordinal,name in enumerate(selector.NORMS,1):
    row=a[name];observation=union[name]
    base_domain=[max(1e-6,.8*observation['minimum']),1.2*observation['maximum']]
    domain=[max(1e-6,base_domain[0]/4),base_domain[1]*4]
    same(row['base_domain'],base_domain);same(row['domain'],domain);same(row['rule'],c['nonlinear'][name])
    history=row['attempts'];assert 1<=len(history)<=6
    expected_degrees=list(selector.LADDER[:len(history)])
    assert [h['degree']for h in history]==expected_degrees
    for index,h in enumerate(history):
        assert set(h)=={'degree','rule','certificate','binary64_diagnostic','selected'}
        same(h['rule'],select_rules.rule(domain,h['degree']))
        cert=certificate.certify(h['rule']);grid=select_rules.grid_check(h['rule'])
        same(h['certificate'],cert);same(h['binary64_diagnostic'],grid)
        passed=cert['certified']is True and grid['passed']is True
        assert h['selected']is passed
        if index<len(history)-1:assert not passed
        attempts+=1
    if row['rule']['method']=='polynomial':
        assert history[-1]['selected']is True
        for field in ('rule','certificate','binary64_diagnostic'):same(row[field],history[-1][field])
    else:
        assert row['rule']['method']=='newton'and len(history)==6 and all(h['selected']is False for h in history)
        rule,cert,grid=selector.newton_fallback(domain)
        same(row['rule'],rule);same(row['certificate'],cert);same(row['binary64_diagnostic'],grid)
        assert cert['verified']is True and grid['passed']is True and 1<=rule['iterations']<=64
    if ordinal%16==0:print(json.dumps(dict(norm_sites_recertified=ordinal,polynomial_attempts_replayed=attempts)),flush=True)
gates_raw=(SOURCE/'inputs/norm_multipliers.json').read_bytes()
assert sha(gates_raw)=='761a07e232e0bba981ba014eb6d202cce96a9471616743594fecdc6a57d67840'
same(checks,selector.protected_checks(base,union,a,json.loads(gates_raw)))
assert len(checks)==28 and all(all(row[k]is True for k in ('radius_covered','observed_silu_covered','observed_centered_scores_covered'))for row in checks.values())
same(c['depth_by_sequence'],{str(s):select_rules.depths(c,s)for s in range(1,33)})
for name,desc in r['outputs'].items():raw(P/name,desc['sha256'])
raw(P/'report.json',report_pin)
same({p.relative_to(SOURCE).as_posix():sha(p.read_bytes())for p in SOURCE.rglob('*')if p.is_file()},sources)
methods=collections.Counter(c['nonlinear'][n]['method']for n in selector.NORMS)
OUT.mkdir(mode=0o700)
review=dict(passed=True,scope='Independent terminal/source/initial proof linkage and exact replay of frozen certificate implementation; no candidate-model, heldout, row0 or HE evaluation',
    report_sha256=report_pin,candidate_manifest_sha256=candidate_pin,source_archive_sha256=archive_pin,source_manifest_sha256=source_pin,grant_sha256=grant_pin,
    outer_session=48044,outer_exit_code=0,outer_reaped=True,producer_pid_independently_absent=r['pid'],producer_group_absent=True,source_files=36,
    complete_norm_sites=113,polynomial_attempts_replayed=attempts,selected_methods=dict(methods),newton_iterations_newly_replayed=True,
    uniform_domains_independently_recomputed=True,protected_rules_exact=True,protected_checks_replayed=28,all32_depths_recomputed=True,
    abstract_depths={str(s):c['depth_by_sequence'][str(s)]['final_depth']for s in (1,8,32)},
    assessment_bytes=len(outputs['assessments.json']),outputs=r['outputs'],initial_snapshot_sha256=r['initial_snapshot_sha256'],
    review_script_sha256=sha(Path(__file__).read_bytes()),review_pid=os.getpid(),review_pgid=os.getpgrp(),candidate_calibration_performed=False)
(OUT/'review.json').write_text(json.dumps(review,indent=2)+'\n')
(OUT/'review.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(passed=True,review=str(OUT/'review.json'),review_sha256=sha((OUT/'review.json').read_bytes()),pid=os.getpid(),pgid=os.getpgrp(),polynomial_attempts_replayed=attempts)))
