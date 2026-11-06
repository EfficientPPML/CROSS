"""Source/public-metadata readback only; never import the reviewed runtime."""
from pathlib import Path
from fractions import Fraction
import ast, datetime, hashlib, json, tarfile, xml.etree.ElementTree as ET

ROOT = Path('/tmp/qwen-expanded-full-model-runtime-20261104')
CAPTURE = Path('/tmp/qwen-expanded-full-model-runtime-development-capture-20261104')
OUT = Path(__file__).resolve().parent
sha = lambda b: hashlib.sha256(b).hexdigest()
canonical = lambda v: json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
read = lambda n: (ROOT / n).read_bytes()
doc = lambda n: json.loads(read(n))

def constants(name):
    result = {}
    for node in ast.parse(read(name)).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try: result[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError): pass
    return result

def main():
    held_raw = (CAPTURE / 'source-map-during-run.json').read_bytes()
    held = json.loads(held_raw)
    assert sha(held_raw) == 'd58e4f70a414f6efe7110125637c6d8ed1a3e502b41b474d172f23ce99b3cca5'
    assert len(held) == 186 and {n: sha(read(n)) for n in held} == held
    assert json.loads((CAPTURE / 'source-map-after.json').read_bytes()) == held
    with tarfile.open(CAPTURE / 'sources-during-run.tar.gz') as archive:
        members = archive.getmembers()
        assert len(members) == len(held) == len({m.name for m in members})
        assert all(m.isfile() for m in members)
        assert {m.name: sha(archive.extractfile(m).read()) for m in members} == held
    predecessor = doc('predecessor87.json')['sources']
    assert len(predecessor) == 87
    assert all(sha(read('before87/' + n)) == h for n, h in predecessor.items())
    changes = [n for n, h in predecessor.items() if sha(read(n)) != h]
    assert len(changes) == 15

    common = 'checkout/cadathlon/common/'
    names = [common + n for n in (
        'qwen_candidate_bridge.py', 'encrypted_qwen_head_routing.py', 'encrypted_qwen_rmsnorm.py',
        'encrypted_qwen_causal_softmax.py', 'qwen_silu_contract.py', 'encrypted_qwen_attention_join.py',
        'encrypted_qwen_decoder_tail.py', 'encrypted_qwen_full_model.py')]
    names += ['derive_expanded.py', 'certificate.py', 'newton_certificate.py', 'reference_sources/public_norm_plan.py',
        'derived_bindings.json', 'full_contract.py', 'plan_full.py', 'full_synthetic.py', 'test_expanded_contract.py',
        'test_full_owner.py', 'test_full_plan.py', 'execution_authorization.json', 'inputs/candidate_manifest.json',
        'inputs/candidate_bridge.json', 'inputs/normalization_certificates.json', 'inputs/site_metadata.json',
        'inputs/join_plan.json', 'inputs/certificates.json', 'inputs/decoder_parameters.json',
        'inputs/config.json', 'inputs/actual_key_plan.json', 'qkv-export-v1/parameter_manifest.json']
    reviewed = {n: sha(read(n)) for n in names}
    old, new = doc('before87/inputs/candidate_manifest.json'), doc('inputs/candidate_manifest.json')
    bridge = constants(common + 'qwen_candidate_bridge.py')
    assert sha(read('inputs/candidate_manifest.json')) == bridge['MANIFEST'] == '344954b3873a8847ececba320646f52612ca7277412c7de81a7313894e21799a'
    protected = lambda m: dict(nonlinear={k:v for k,v in m['nonlinear'].items() if not k.endswith('inverse_sqrt')},
        **{k:m[k] for k in ('softmax', 'exp', 'initial_reciprocal', 'reciprocal_by_keys')})
    assert canonical(protected(old)) == canonical(protected(new))
    assert sha(canonical(protected(new))) == bridge['PROTECTED']
    for name in ('inputs/certificates.json', 'inputs/config.json', 'inputs/decoder_parameters.json',
                 'inputs/actual_key_plan.json', 'qkv-export-v1/parameter_manifest.json'):
        assert read(name) == read('before87/' + name)
    relation = doc('inputs/candidate_bridge.json')
    assert relation['historical_certificate_sha256'] == sha(read('inputs/certificates.json')) == bridge['CERTIFICATE']
    assert relation['normalization_certificate_transferred'] is False
    assert relation['historical_bytes_rewritten'] is False
    assert relation['calibration_pass_claimed'] is False and relation['execution_authorized'] is False
    for role, name in [('qkv','qkv-export-v1/parameter_manifest.json'), ('decoder','inputs/decoder_parameters.json')]:
        p = doc(name)
        assert sha(read(name)) == bridge['PARAMETERS'][role]
        assert p['candidate_sha256'] == bridge['OLD_MANIFEST']
        assert p['config_sha256'] == bridge['CONFIG'] and p['checkpoint_sha256'] == bridge['CHECKPOINT']
    bound_paths = {'bridge_sha256':'inputs/candidate_bridge.json', 'join_plan_sha256':'inputs/join_plan.json',
        'normalization_certificates_sha256':'inputs/normalization_certificates.json', 'site_metadata_sha256':'inputs/site_metadata.json'}
    assert doc('derived_bindings.json') == {k:sha(read(n)) for k,n in bound_paths.items()}
    assert constants(common+'encrypted_qwen_rmsnorm.py')['SITE_METADATA'] == sha(read('inputs/site_metadata.json'))
    assert constants(common+'encrypted_qwen_attention_join.py')['JOIN_PLAN'] == sha(read('inputs/join_plan.json'))

    cert = doc('inputs/normalization_certificates.json')
    meta, oldmeta = doc('inputs/site_metadata.json'), doc('before87/inputs/site_metadata.json')
    rows = {r['site']:r for r in doc('inputs/join_plan.json')['normalizations']}
    oldrows = {r['site']:r for r in doc('before87/inputs/join_plan.json')['normalizations']}
    plans = {r['site']:r for r in cert['normalizations']}
    assert len(meta) == len(rows) == len(plans) == len(cert['certificates']) == 113
    assert set(meta) == set(rows) == set(plans) == set(cert['certificates'])
    assert cert['candidate_manifest_sha256'] == bridge['MANIFEST']
    assert cert['changed_rule_count'] == sum(new['nonlinear'][n] != old['nonlinear'][n] for n in meta) == 90
    counts = {'polynomial':0, 'newton':0}
    for n, record in cert['certificates'].items():
        rule = new['nonlinear'][n]
        assert record['rule_sha256'] == sha(canonical(rule)) and record['method'] == rule['method']
        proof = record['certificate']
        if rule['method'] == 'polynomial':
            counts['polynomial'] += 1
            assert proof['certified'] is True
            rel = proof['relative']
            assert 0 <= Fraction(int(rel['numerator_hex'],16),int(rel['denominator_hex'],16)) <= Fraction(1,10**6)
        else:
            counts['newton'] += 1
            assert proof['verified'] is True
        assert meta[n]['refresh_events'] == plans[n]['refresh_events'] == oldmeta[n]['refresh_events']
        assert meta[n]['output_level'] == plans[n]['output_level'] == rows[n]['output_level']
        assert rows[n]['raw_input_bound'] == plans[n]['conditional_input_refresh_bound']
        for key in ('gamma_max_abs','gamma_sha256','output_bound'):
            assert rows[n][key] == oldrows[n][key]
        assert {k:v for k,v in meta[n].items() if k not in ('output_level','refresh_events')} == {k:v for k,v in oldmeta[n].items() if k not in ('output_level','refresh_events')}
    assert counts == {'polynomial':36, 'newton':77}
    assert sum(len(r['refresh_events']) for r in meta.values()) == cert['inverse_refresh_count'] == 260
    raw_changes = [dict(site=n, old=oldrows[n]['raw_input_bound'], new=rows[n]['raw_input_bound']) for n in rows if rows[n]['raw_input_bound'] != oldrows[n]['raw_input_bound']]
    prior = json.loads(Path('/tmp/qwen-expanded-full-contract-security-review-20261104/review.json').read_bytes())
    assert raw_changes == prior['raw_bound_changes'] == [dict(site='final.inverse_sqrt', old=2048, new=8192)]
    final = plans['final.inverse_sqrt']
    assert (final['degree'],final['output_level'],final['inverse_level_before_terminal_refresh'],final['refresh_events']) == (256,3,5,[])
    assert rows['final.inverse_sqrt']['output_bound'] == 1024
    join, oldjoin = doc('inputs/join_plan.json'), doc('before87/inputs/join_plan.json')
    assert join['protected_bridge'] == relation
    assert all(join[k] == oldjoin[k] for k in ('rope_table','rope_bounds','score_domain_outer_bounds','all_centered_domains_proved'))
    auth = doc('execution_authorization.json')
    assert auth['first_pass_freeze_sha256'] is None and auth['actual_execution_grant_sha256'] is None
    assert all(auth[k] is False for k in ('scientific_calibration_pass_verified','model_forward_enabled','heldout_enabled','row0_enabled','he_enabled'))
    terminal = json.loads((CAPTURE/'terminal.json').read_bytes())
    suite = ET.parse(CAPTURE/'development-focused-v1.xml').getroot()
    cases = list(suite.iter('testcase'))
    assert len(cases) == 171 and sum(c.find('failure') is not None for c in cases) == 70
    assert terminal['exit_code'] == 1 and terminal['passed'] == 101 and terminal['failed'] == 70
    assert all("KeyError: 'resources'" in ''.join(c.find('failure').itertext()) for c in cases if c.find('failure') is not None)
    report = dict(schema='qwen-expanded-full-successor-security-source-review-v1', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Read-only source/public metadata review. No runtime imports, certificate recomputation, tests, model/input/tensor bodies or HE.',
        no_new_concrete_blocker_in_reviewed_delta=True, source_hashes=reviewed,
        held_source_map_sha256=sha(held_raw), held_source_count=186, held_map_scope='Captured during development run; not a prelaunch/startup-bound source proof.',
        held_current_after_tar_equal=True, predecessor87_exact=True, predecessor_changes=changes,
        derived_bindings=doc('derived_bindings.json'), certificate_summary=counts, changed_norm_rules=90, inverse_refresh_events=260,
        final=dict(degree=256, output_level=3, vocabulary_level=2, raw_bound=8192, normalized_bound=1024, inverse_refresh_events=0),
        checked_relations=[
            'Exact old certificate and parameter manifests retained with old identities; protected canonical subtree equality verified.',
            'Softmax/SiLU call the protected-certificate bridge before using historical proof fields; QKV/tail owners call parameter bridge before descriptor reuse.',
            'Runtime candidate/site/join hashes bind expanded rules and metadata. Fresh normalization certificates are supporting public proof artifacts, not an additional runtime certificate argument.',
            'Full contract derives final/head levels and includes new fields in typed state verification; full owner consumes those levels with actual drop-divisor scale replay.',
            'All gamma/output/RoPE bounds remain conditional on unchanged gamma/width and the same 1e-6 relative target; no changed normalization inherits old coefficients/certificate.',
        ],
        development=dict(source='peer terminal receipt plus independently read XML', tests=171, passed=101, failed=70, exit_code=1,
            failure="Inherited join_ideal fixture KeyError('resources'); no successful complete owner suite claimed."),
        pending=['Select a new collector/verifier and exact closure; inherited collectors/verifiers/inventories remain historical, unselected and unsuitable as expanded-proof acceptance.',
            'Preserve failed development; repair synthetic key fixture in separate successor, then complete fresh source-bound preparation and independently compare pure derivation to pinned outputs.'],
        limitations=['No calibration PASS, first-PASS freeze, model/HE execution grant or wholegraph numerical result.',
            '8192 is a conditional public amplitude envelope. Secure P60 tests at B16/B128 do not establish precision at 8192 or the former 2048; no encrypted head-L2 guarantee.',
            'Existing score-domain limitations and actual refresh-scale/perturbation premises remain.'])
    (OUT/'review.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(dict(no_new_concrete_blocker=True, reviewed_files=len(reviewed), held_files=len(held), report_sha256=sha((OUT/'review.json').read_bytes()))))

if __name__ == '__main__': main()
