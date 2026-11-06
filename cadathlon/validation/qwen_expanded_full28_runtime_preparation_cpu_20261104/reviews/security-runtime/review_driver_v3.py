"""Narrow source-only driver delta receipt; no candidate import or test run."""
from pathlib import Path
import ast, datetime, hashlib, json
OLD=Path('/tmp/qwen-expanded-full-preparation-driver-v2-20261104')
NEW=Path('/tmp/qwen-expanded-full-preparation-driver-v3-20261104')
OUT=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
    old=(OLD/'prepare_expanded.py').read_bytes();new=(NEW/'prepare_expanded.py').read_bytes()
    assert sha(old)=='27e16065bbe556d7af42a9755cd56aa6bd6fd2edd590aeffcc8e16637b1b9394'
    assert sha(new)=='1a7c49dff76aba6a0a1c492f043ac1c5248217fd909b02e6df1330e659572c14'
    # Reverse only the four explicitly reviewed replacements and require the
    # complete preceding source bytes, not an approximate semantic match.
    text=new.decode()
    replacements=[
        ("        if path in self.sources:\n            if path.resolve()!=path:self.deny('authenticated_source_alias',path)\n            return", "        if path in self.sources:return"),
        ('synthetic_tests_started=False,synthetic_component_tests_completed=False', 'synthetic_parent_loop_executed=True'),
        ("        report['synthetic_tests_started']=True\n", ''),
        ("        report['synthetic_component_tests_completed']=True\n", ''),
    ]
    for before,after in replacements:
        assert text.count(before)==1
        text=text.replace(before,after)
    assert text.encode()==old
    provenance=json.loads((NEW/'helper_provenance.json').read_bytes())
    for name,row in provenance.items():
        assert (NEW/name).read_bytes()==(OLD/name).read_bytes()
        assert sha((NEW/name).read_bytes())==row['sha256']
    tree=ast.parse(new)
    constants={}
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            try:constants[node.targets[0].id]=ast.literal_eval(node.value)
            except (ValueError,TypeError):pass
    assert constants['EXPECTED_TESTS']==176 and constants['INFRASTRUCTURE_EXIT']==70
    inventory=json.loads(Path('/tmp/qwen-expanded-full-model-runtime-v3-20261104/successor_inventory.json').read_bytes())
    assert inventory['selected_tests']==list(constants['TESTS'])
    hashes={p.name:sha(p.read_bytes()) for p in NEW.iterdir() if p.is_file()}
    report=dict(schema='qwen-expanded-preparation-driver-security-delta-v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Source-only final two-change review relative to parent-reviewed27e160 driver. No imports, probes, synthetic tests, model/input/private bodies or HE.',
        predecessor_driver_sha256=sha(old),source_hashes=hashes,no_new_concrete_blocker_in_reviewed_delta=True,
        observations=[
            'Authenticated source read allowance now also requires resolved path equality at each audited open, rejecting a post-admission symlink/ancestor alias instead of accepting only its lexical name.',
            'Both synthetic phase flags start false. Started becomes true immediately before pytest.main; completed becomes true only after zero pytest status and exactly176 clean XML cases.',
            'Completed flags describe completed synthetic checks and may remain true if later public planning or cleanup fails; report.passed plus independent actual outer status still governs acceptance.',
            'Exact frozen helper bytes and all other predecessor driver bytes are unchanged; selected nine-module inventory still agrees with held successor metadata.',
            'No model execution mode was added. Final status70 remains distinct for raised infrastructure failures, including late publication/close failures.'
        ],
        boundaries=['Python audit/source checks are not OS isolation or a same-user race guarantee.',
            'Final combined source closure/copy has not yet been assembled. It must include these exact entry/helper bytes plus the held candidate tree and receive separate before/after/archive/outer verification.',
            'Parent owns broader admission, Output lifecycle, capacity and final execution review; this receipt does not claim a176-case run.'])
    (OUT/'review-driver-v3.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(dict(no_new_concrete_blocker=True,report_sha256=sha((OUT/'review-driver-v3.json').read_bytes()),driver_sha256=sha(new))))
if __name__=='__main__':main()
