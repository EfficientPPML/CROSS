"""Source-only client proposals: no backend import, payload read or execution."""
from pathlib import Path
import ast
import difflib
import hashlib
import json
import math
import tarfile

BASE = Path('/tmp/bert-client-clean-launch-source-proposals-20261104')
DECODER = Path('/tmp/ckks-scaled-crt-decoder-candidate-v2-20261104')
OUT = Path('/tmp/bert-scaled-crt-decoder-client-proposals-20261104')
PIN = '9675b300d507fd9c6a3d03656c9df325d8da68a9331b4c92def0512784d0dc26'
OLD = 'bc978cb7412b55416f7ed634a689ae616a263b0f47fd9eef996ba244f2505db6'
REL = 'runtime/jaxite_word/ckks_ctx.py'

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()

def literal(raw, name):
    tree = ast.parse(raw)
    return ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)))

def named(tree):
    return {n.name: ast.dump(n, include_attributes=False) for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))}

def main():
    OUT.mkdir(mode=0o700, exist_ok=False)
    (OUT/'prepare.py').write_bytes(Path(__file__).read_bytes())
    candidate = (DECODER/'checkout/jaxite_word/ckks_ctx.py').read_bytes()
    original = (DECODER/'before/ckks_ctx.py').read_bytes()
    assert sha(candidate) == PIN and sha(original) == OLD
    proof = DECODER/'source-bound-v1'
    measured = json.loads((proof/'sources-before.json').read_bytes())
    assert measured == json.loads((proof/'sources-after.json').read_bytes()) and len(measured) == 207
    report = json.loads((proof/'report.json').read_bytes())
    assert report['passed'] is True and report['source_archive_sha256'] == sha((proof/'sources.tar.gz').read_bytes())
    with tarfile.open(proof/'sources.tar.gz', 'r:gz') as archive:
        assert archive.extractfile('checkout/jaxite_word/ckks_ctx.py').read() == candidate
        assert archive.extractfile('before/ckks_ctx.py').read() == original
    old_tree, new_tree = ast.parse(original), ast.parse(candidate)
    old_defs, new_defs = named(old_tree), named(new_tree)
    assert set(new_defs)-set(old_defs) == {'_scale_centered_crt_coefficient'}
    assert {n for n in old_defs if old_defs[n] != new_defs[n]} == {'_fast_decode'}
    # Replacing the two calls by their literal predecessor expressions recovers
    # the entire original module AST, including imports/classes/cache handling.
    class Undo(ast.NodeTransformer):
        count = 0
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == '_scale_centered_crt_coefficient':
                assert len(node.args) == 3 and not node.keywords
                assert ast.dump(node.args[1]) == ast.dump(ast.Name(id='scale', ctx=ast.Load()))
                assert ast.dump(node.args[2]) == ast.dump(ast.Name(id='scale_inv', ctx=ast.Load()))
                self.count += 1
                return ast.BinOp(left=ast.Call(func=ast.Name(id='float', ctx=ast.Load()),
                    args=[node.args[0]], keywords=[]), op=ast.Mult(), right=node.args[2])
            return self.generic_visit(node)
    undo = Undo()
    restored = undo.visit(ast.parse(candidate))
    restored.body = [n for n in restored.body if not isinstance(n, ast.FunctionDef)
        or n.name != '_scale_centered_crt_coefficient']
    assert undo.count == 2 and ast.dump(restored) == ast.dump(old_tree)
    review = dict(schema='bert-scaled-crt-client-source-proposal-v1', source_only=True,
        actual_client=False, backend_imported=False, private_payload_read=False,
        full_preflight_executed=False, launch_authorized=False,
        decoder=dict(path=str(DECODER/'checkout/jaxite_word/ckks_ctx.py'), sha256=PIN,
            predecessor_sha256=OLD, proof_report_sha256=sha((proof/'report.json').read_bytes()),
            proof_source_archive_sha256=report['source_archive_sha256'],
            proof_source_manifest_sha256=report['source_manifest_sha256'],
            proof_scope=report['scope'], proof_tests=report['tests'], proof_source_count=207,
            exact_module_AST_delta='one added scaling helper plus two _fast_decode call substitutions'),
        clients={})
    for label, count, previous_overlay_count in [('original122', 122, 7), ('combined125', 125, 8)]:
        parent = BASE/label
        before_raw = (parent/'source_manifest.json').read_bytes()
        before = json.loads(before_raw)
        assert len(before) == count
        old_raw = {n: (parent/'source'/n).read_bytes() for n in before}
        assert {n: sha(b) for n,b in old_raw.items()} == before
        assert old_raw[REL] == original
        dest = OUT/label
        root = dest/'source'
        root.mkdir(parents=True, mode=0o700)
        for n,raw in old_raw.items():
            path = root/n
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        (root/REL).write_bytes(candidate)
        guard = old_raw['client_source_guard.py'].decode()
        old_overlays = literal(guard, 'OVERLAYS')
        assert len(old_overlays) == previous_overlay_count and REL not in old_overlays
        new_overlays = old_overlays | {REL: PIN}
        guard = '\n'.join('OVERLAYS='+repr(new_overlays) if line.startswith('OVERLAYS=') else line
            for line in guard.split('\n'))
        (root/'client_source_guard.py').write_text(guard)
        test = old_raw['test_client_source_guard.py'].decode()
        words = {7: 'seven', 8: 'eight', 9: 'nine'}
        test = test.replace('test_complete_static_closure_and_'+words[previous_overlay_count]+'_overlays',
            'test_complete_static_closure_and_'+words[previous_overlay_count+1]+'_overlays')
        test = test.replace("len(r['overlays'])=="+str(previous_overlay_count),
            "len(r['overlays'])=="+str(previous_overlay_count+1))
        anchor = " assert r['critical_resolution']['client_installed_probe']=='client_installed_probe.py'\n"
        assert test.count(anchor) == 1
        test = test.replace(anchor, anchor +
            " assert r['overlays']['runtime/jaxite_word/ckks_ctx.py']=='"+PIN+"'\n"+
            " assert r['critical_resolution']['ckks_ctx']==r['critical_resolution']['jaxite_word.ckks_ctx']=='runtime/jaxite_word/ckks_ctx.py'\n")
        (root/'test_client_source_guard.py').write_text(test)
        readme = old_raw['README_CLIENT.md'].decode()
        if label == 'original122':
            readme = readme.replace('Exact89 parent plus five named runtime IO overlays plus two top-level consumer reader overlays.',
                'Exact89 parent plus five named runtime IO overlays, two top-level consumer reader overlays and one reviewed decoder overlay.')
            readme = readme.replace('all prior producer identities, arithmetic and P2 gates are unchanged.',
                'all prior producer identities, encrypted arithmetic and P2 gates are unchanged; the decoder change is described below.')
            readme = readme.replace('exactly seven replacements (five runtime, two top-level gates) and82 unchanged parent files',
                'exactly eight replacements (six runtime, two top-level gates) and81 unchanged parent files')
        else:
            readme = readme.replace('The exact reviewed 89-source client is retained with five explicit overlays:',
                'The historical115 predecessor retained the reviewed89 client with five initial overlays:')
            readme = readme.replace('Against original89 there are8 permitted overlays and81 unchanged files;',
                'Against original89 there are9 permitted overlays and80 unchanged files;')
        readme += '''
## Source-only decoder proposal

This new isolated proposal copies the completed clean consumer closure and adds
the reviewed `ckks_ctx.py` decoder SHA '''+PIN+'''.
It changes only the host conversion of centered CRT coefficients: the existing
`float(value) * scale_inv` path remains exact when conversion succeeds, and an
overflowing conversion uses the exact binary64 scale ratio. FFT, approximation
validation, decryption, ciphertext/key formats, parameter profile, lifecycle and
P2 acceptance are unchanged. The local guard explicitly admits this decoder;
all old producer source maps and gates remain literal predecessor bytes.

Only four files differ from the clean122/125 parent: this README, the decoder,
the guard's overlay dictionary, and the existing source-guard test expectation
plus decoder/alias assertions. No new test case or selected entrypoint is added.
The historical client preflight does not attest these new bytes: this proposal
has had source/AST/hash inspection only, with no backend import, installed probe,
full preflight, payload/private read or client execution. A separately reviewed
complete client proof and launch grant remain required.

The authenticated final BERT L2 response has six fixed Q towers: their product
is181 bits and centered coefficients are at most180 bits. This final boundary
therefore uses the unchanged successful-float path. The proposal is compatible
decoder robustness preparation, not a claimed fix for an observed full-BERT
failure, evidence of numerical accuracy, or recovery of the separate toy5 run.
Keep the actual source copy free of bytecode/import shadows and launch only via
the existing isolated externally pinned CLI. Do not rewrite old producer maps
to the new consumer hash or touch any current server response/request ledger.
'''
        (root/'README_CLIENT.md').write_text(readme)
        after_raw = {n: (root/n).read_bytes() for n in before}
        after = {n: sha(b) for n,b in after_raw.items()}
        changed = [n for n in sorted(before) if before[n] != after[n]]
        assert changed == ['README_CLIENT.md', 'client_source_guard.py', REL, 'test_client_source_guard.py']
        # Bind unchanged bootstrap names, producer inventories/gates and codec
        # facade without importing any local or backend module.
        assert set(literal(after_raw['client_bootstrap.py'], 'EXPECTED_NAMES')) == set(after)
        assert set(json.loads(after_raw['client_89_sources.json'])) | set(literal(guard, 'NEW')) == set(after)
        parent89 = json.loads(after_raw['client_89_sources.json'])
        assert len(parent89) == 89
        assert all(after[n] == new_overlays.get(n,h) for n,h in parent89.items())
        before_guard = ast.parse(old_raw['client_source_guard.py'])
        after_guard = ast.parse(after_raw['client_source_guard.py'])
        for tree in (before_guard, after_guard):
            tree.body = [n for n in tree.body if not (isinstance(n, ast.Assign) and
                any(isinstance(t,ast.Name) and t.id=='OVERLAYS' for t in n.targets))]
        assert ast.dump(before_guard) == ast.dump(after_guard)
        assert all(not p.is_symlink() for p in root.rglob('*'))
        assert {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()} == set(after)
        for n,b in after_raw.items():
            if n.endswith('.py'):ast.parse(b,filename=n)
        protected = [n for n in before if n.endswith('_manifest.json') or n in
            ('client_89_sources.json','client_89_report.json','source_origin.json','dependencies.json',
             'key_gate.py','model_gate.py','stage3_gate.py','stage4_gate.py','combined_server_gate.py',
             'combined_server_sources.json','combined_server_preflight.json','combined_server_preflight_sources.tar.gz',
             'runtime/stage/key_contexts.py','runtime/stage/client_keys.py','runtime/stage/file_stages.py',
             'runtime/stage/request_lifecycle.py','client_metrics.py','boundaries.py')]
        assert all(before[n] == after[n] for n in protected)
        (dest/'parent_source_manifest.json').write_bytes(before_raw)
        (dest/'source_manifest.json').write_bytes(encoded(after))
        patch=''.join(''.join(difflib.unified_diff(old_raw[n].decode().splitlines(True),
            after_raw[n].decode().splitlines(True),fromfile='parent/'+n,tofile='proposal/'+n)) for n in changed)
        (dest/'candidate.patch').write_text(patch)
        with tarfile.open(dest/'sources.tar.gz','w:gz') as archive:
            for n in sorted(after):archive.add(root/n,arcname=n,recursive=False)
        assert {n:sha((parent/'source'/n).read_bytes()) for n in before} == before
        review['clients'][label] = dict(source_root=str(root),source_count=count,
            source_bytes=sum(map(len,after_raw.values())),parent_source_manifest_sha256=sha(before_raw),
            source_manifest_sha256=sha((dest/'source_manifest.json').read_bytes()),
            source_archive_sha256=sha((dest/'sources.tar.gz').read_bytes()),
            changed_files={n:dict(before=before[n],after=after[n]) for n in changed},
            unchanged_count=count-len(changed),parent89_unchanged_count=89-len(new_overlays),
            overlays=new_overlays,protected_unchanged={n:after[n] for n in sorted(protected)},
            module_resolution={'ckks_ctx':REL,'jaxite_word.ckks_ctx':REL},
            critical_resolution_guard_unchanged=True,default_or_separate_receipt_contract_unchanged=True,
            tests_executed=False,installed_probe_executed=False)
    profile=json.loads((BASE/'original122/source/runtime/stage/server_inputs/p60_profile_and_refresh.json').read_bytes())['profile']
    q=profile['q'][:6];product=math.prod(q)
    review['final_BERT_response']=dict(level=2,moduli=6,q=q,product=str(product),
        q_product_bits=product.bit_length(),max_centered_coefficient_bits=(product//2).bit_length(),
        payload_bytes=1*2*256*512*6*4,overflow_fallback_needed_for_valid_final_coefficients=False,
        observed_ciphertext_read=False)
    review['call_chain']=['file_stages.client_decrypt_decode','key_contexts.load_client/_crypto',
        'jaxite_word.CKKSContext == jaxite_word.ckks_ctx.CKKSContext',
        'CKKSContext.decrypt_slots','_fast_decrypt_decode','_fast_decode',
        '_scale_centered_crt_coefficient']
    (OUT/'report.json').write_bytes(encoded(review))
    print(json.dumps(dict(path=str(OUT),report_sha256=sha((OUT/'report.json').read_bytes()),
        clients={k:v['source_manifest_sha256'] for k,v in review['clients'].items()})))

if __name__ == '__main__':
    main()
