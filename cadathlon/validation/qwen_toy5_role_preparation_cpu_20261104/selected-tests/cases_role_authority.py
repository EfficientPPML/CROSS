"""Pure SYNTHETIC metadata; all final gates are original source definitions."""
from dataclasses import replace
import json
import case_fixtures as f

def test_fixed_q5_profile_and_203_descriptors():
    p=f.profile();p.check();assert len(p.rotations)==100 and len(p.arrays)==203
    assert p.parameters['dnum']==4 and len(p.parameters['q_towers'])==100 and len(p.parameters['p_towers'])==24
    assert 65535 in p.rotations and 262143 in p.rotations and p.rotations==tuple(sorted(set(p.rotations)))
    assert p.arrays[0].shape==(2,100,131072)and all(x.shape==(4,124,131072)for x in p.arrays[1:])
    assert [s.name for s in p.arrays[3:]]==[f'rot_{i}_{k}.npy'for i in p.rotations for k in('a','b')]
    f.event('fixed_inventory',rotations=100,public_arrays=203,private_arrays=1)

def test_profile_substitution_reaches_exact_contract_rejection():
    from q5_key_contract import ExpectedIds,KEYGEN,authenticate_public
    p=f.profile();raw,_=f.key_bytes(p);base=json.loads(raw)
    def admitted(doc):
        body={k:v for k,v in doc.items()if k!='context_id'};doc['context_id']=f.digest(f.compact(body));b=f.compact(doc)
        expected=ExpectedIds(doc['context_id'],f.digest(b),KEYGEN,f.h('model'),f.h('setup'),f.h('client'))
        return authenticate_public(b,p,expected,role='server')
    admitted(f.clone(base));f.event('positive_contract_reached')
    altered=f.clone(base);altered['parameters']['dnum']=8
    f.expect(ValueError,lambda:admitted(altered),'external context/canonical public manifest differs')
    altered=f.clone(base);altered['arrays']['public_key.npy']['shape'][0]=True
    f.expect(ValueError,lambda:admitted(altered),'bounded exact key array descriptor differs')
    altered=f.clone(base);altered['rotations']=list(range(166))
    f.expect(ValueError,lambda:admitted(altered),'external context/canonical public manifest differs')

def test_complete_synthetic_setup_chain_admits_distinct_role_maps():
    s=f.setup_bundle();assert len(set(s.roles.values()))==3;s.admission.check()
    for role in('setup','server','client'):
        a,g,pins=f.phase_bundle(role,s);assert a.check()['role']==role and a.execution_authorized is False
        assert g['source_manifest_sha256']==s.roles[role];f.event('synthetic_phase',role=role)

def test_internal_generated_manifests_never_supply_later_terminal_authority():
    from q5_key_producer import _generate_bound
    s=f.setup_bundle();generation=f.generation_fixture();internal=_generate_bound(generation.intent,generation.deps)
    assert internal.execution_authorized is False and internal.terminal_ready is False
    a,_,_=f.phase_bundle('server',s)
    f.expect(ValueError,lambda:replace(a,setup=internal).check(),'independent completed setup required before key/body entry')
    f.expect(ValueError,lambda:replace(s.admission,evidence=()).check(),'complete ordered setup terminal evidence required')
    f.event('no_io_or_runtime_issued')

def test_repinned_outer_and_root_fail_at_semantic_join():
    s=f.setup_bundle();s.admission.check();f.event('baseline_complete_setup')
    a=s.rebuild(outer_change=lambda v:v.update(normal_return=False))
    f.expect(ValueError,a.check,'independent normal0/reap/publication required')
    a=s.rebuild(outer_change=lambda v:v.update(exit_code=True))
    f.expect(ValueError,a.check,'independent normal0/reap/publication required')
    a=s.rebuild(review_change=lambda v:v.update(manifest_sha256=f.h('different-original-manifest')))
    f.expect(ValueError,a.check,'independent setup root proof join differs')

def test_phase_source_and_original_proof_identity_reauthenticate():
    from q5_role_authority import ExpectedPhasePins
    a,g,pins=f.phase_bundle('server');a.check();f.event('baseline_phase')
    wrong=f.clone(g);wrong['roles']['server']=f.h('wrong-server-role');raw=f.compact(wrong);p=dict(pins,grant_sha256=f.digest(raw))
    bad=replace(a,grant_bytes=raw,expected=ExpectedPhasePins(f.compact(p)))
    f.expect(ValueError,bad.check,'current role source differs')
    object.__setattr__(a,'dependency_policy_bytes',f.compact(dict(schema='SYNTHETIC-mutated-policy')))
    f.expect(ValueError,a.check,'independent phase policy/grant bytes differ')
