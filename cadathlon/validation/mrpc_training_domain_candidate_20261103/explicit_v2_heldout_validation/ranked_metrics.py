"""Fixed P2 ranked accuracy gates, separate from local-toy diagnostics."""
import random
import numpy as np

def accuracy_gate_counts(size,approximate_correct,native_correct):
    if type(size) is not int or size<1 or any(type(x) is not int or not 0<=x<=size for x in (approximate_correct,native_correct)):
        raise ValueError('valid integer counts required')
    # Integer comparisons implement75% and2 percentage points exactly.
    minimum=4*approximate_correct>=3*size
    drop=100*(native_correct-approximate_correct)<=2*size
    return dict(minimum_accuracy_pass=minimum,accuracy_drop_pass=drop,ranked_accuracy_pass=minimum and drop)

def evaluate(target_ids,labels,native_logits,approximate_logits,domain_covered):
    ids=np.asarray(target_ids);labels=np.asarray(labels);native=np.asarray(native_logits);actual=np.asarray(approximate_logits);covered=np.asarray(domain_covered)
    if ids.shape!=(408,) or ids.dtype.kind not in 'iu' or len(set(ids.tolist()))!=408:raise ValueError('exact408 distinct integer IDs required')
    if labels.shape!=(408,) or labels.dtype.kind not in 'iu' or not np.isin(labels,[0,1]).all():raise ValueError('exact408 binary labels required')
    if native.shape!=(408,2) or actual.shape!=(408,2) or native.dtype.kind not in 'fi' or actual.dtype.kind not in 'fi':raise ValueError('two real logits per exact record required')
    if not np.isfinite(native).all():raise ValueError('native baseline must be complete and finite')
    if covered.shape!=(408,) or covered.dtype.kind!='b':raise ValueError('explicit boolean domain coverage required')
    finite=np.isfinite(actual).all(axis=1);complete=finite&covered
    native_prediction=(native[:,1]>=native[:,0]).astype(np.int64)
    approximate_prediction=(actual[:,1]>=actual[:,0]).astype(np.int64)
    results=[]
    for base in (42,43,44):
        seed=int(np.random.default_rng(base).integers(0,0x7fffffff))
        for size in (100,200,408):
            selected=np.array(list(range(408)) if size==408 else sorted(random.Random(seed).sample(range(408),size)),dtype=np.int64)
            count=int(complete[selected].sum());native_correct=int((native_prediction[selected]==labels[selected]).sum())
            row=dict(base_seed=base,derived_input_seed=seed,size=size,positions=selected.tolist(),target_ids=ids[selected].tolist(),
                native_correct=native_correct,native_accuracy=native_correct/size,complete_outputs=count,
                domain_coverage_pass=bool(covered[selected].all()),finite_logits_pass=bool(finite[selected].all()),
                ranked_accuracy_pass=False,approximate_correct=None,approximate_accuracy=None,accuracy_drop_percentage_points=None)
            if count==size:
                correct=int((approximate_prediction[selected]==labels[selected]).sum())
                row.update(approximate_correct=correct,approximate_accuracy=correct/size,
                    accuracy_drop_percentage_points=100*(native_correct-correct)/size,**accuracy_gate_counts(size,correct,native_correct))
                row['diagnostics']=dict(prediction_agreements=int((approximate_prediction[selected]==native_prediction[selected]).sum()),
                    all_predictions_agree=bool(np.array_equal(approximate_prediction[selected],native_prediction[selected])),
                    local_toy_logit_tolerance_pass=bool(np.all(np.abs(actual[selected]-native[selected])<=.02+.01*np.abs(native[selected]))),
                    max_abs_logit_error=float(np.max(np.abs(actual[selected]-native[selected]))),
                    affects_ranked_accuracy_gate=False)
            else:
                row['incomplete_reason']='Some selected records lack finite, domain-covered outputs; no approximate accuracy is claimed.'
            results.append(row)
    return dict(scope=__doc__,ranked_requirements={'minimum_label_accuracy':.75,'maximum_drop_percentage_points':2},
        all_ranked_subsets_pass=all(r['ranked_accuracy_pass'] for r in results),subsets=results,
        prediction_agreement_and_logit_tolerance_are_diagnostics=True)
