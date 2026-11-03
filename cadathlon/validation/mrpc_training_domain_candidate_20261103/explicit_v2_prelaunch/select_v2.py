"""Explicit balanced-inverse candidate; preserve v1 and its exact coefficients."""
from pathlib import Path
import copy,hashlib,json,sys
from audit import sha,write
HERE=Path(__file__).resolve().parent
def main():
    source=HERE/'candidate-v1/candidate_manifest.json';out=Path(sys.argv[1]);out.mkdir(exist_ok=False)
    protocol=json.loads((HERE/'selection_protocol.json').read_text())
    protocol.update(version=2,inverse_sqrt_method='Retain conditioned power degree<=32; explicitly use balanced Chebyshev from the same already-target-passing serialized polynomial for every other norm. Newton candidatev1 is preserved.',
        v1_manifest_sha256=sha(source),candidate_change='Only unsupported inverse-square-root strategies change; polynomial domains, coefficients and analytic targets are byte-preserved.')
    manifest=json.loads(source.read_text());changes={};depths={}
    for name,rule in manifest['norms'].items():
        before=copy.deepcopy(rule)
        if rule['method']=='newton':
            if not rule['polynomial_grid_target_met']:raise ValueError('balanced polynomial target was not met')
            rule['method']='balanced_chebyshev'
            changes[name]=dict(before=before['method'],after=rule['method'],same_coefficient_sha256=rule['polynomial']['coefficient_f64le_sha256'])
        degree=rule['degree']
        inverse_depth=((degree-1).bit_length()+2 if rule['method']=='power' else 2*(degree-1).bit_length()+2)
        depths[name]=dict(inverse_depth=inverse_depth,mean=1,variance_square=1,variance_reduction=1,normalize_product=1,affine=1,
            row_normalization_critical_depth=inverse_depth+5)
    manifest['selection']='Explicit v2 from training-only v1; conditioned power or balanced Chebyshev inverse-square-root. All coefficients/domains and other methods retained; fixed before approximate training.'
    write(out/'selection_protocol.json',protocol);write(out/'candidate_manifest.json',manifest)
    report=dict(scope=__doc__,passed=True,v1_manifest_sha256=sha(source),v2_manifest_sha256=sha(out/'candidate_manifest.json'),
        v2_protocol_sha256=sha(out/'selection_protocol.json'),changes=changes,normalization_depths=depths,
        unchanged_softmax_squarings={k:v['squarings'] for k,v in manifest['softmax'].items()},
        public_K_reciprocal_counts={k:v['iterations'] for k,v in manifest['normalization_reciprocal_by_public_K'].items()},
        source_sha256=sha(Path(__file__)),training_or_validation_approximate_execution=False,
        limitation='Public planning only. Current encrypted row-LN helper has no high-degree balanced strategy; no helper/profile change or encrypted correctness claim.')
    write(out/'report.json',report);print(json.dumps({'sha256':report['v2_manifest_sha256'],'changes':list(changes),'max_row_norm_depth':max(r['row_normalization_critical_depth'] for r in depths.values())},indent=2))
if __name__=='__main__':main()
