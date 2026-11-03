"""Derive required subset metrics from immutable native audit; no model rerun."""
from pathlib import Path
import hashlib,json,random,sys,io,tarfile
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    source=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(exist_ok=False)
    before={str(p):sha(p) for p in (Path(__file__).resolve(),source.resolve())}
    data=json.loads(source.read_text());rows=data['per_example'];allset=data['subsets']['408']
    assert data['passed'] and data['completed_examples']==408 and len(rows)==408
    assert allset['examples_with_valid_query_domain_violation']==408
    assert [r['position'] for r in rows]==list(range(408))
    assert len({r['target_idx'] for r in rows})==408
    result=[]
    for base in (42,43,44):
        seed=int(np.random.default_rng(base).integers(0,0x7fffffff))
        for size in (100,200,408):
            indices=list(range(408)) if size==408 else sorted(random.Random(seed).sample(range(408),size))
            selected=[rows[i] for i in indices];correct=sum(r['correct'] for r in selected)
            record=dict(base_seed=base,derived_input_seed=seed,size=size,positions=indices,
                target_ids=[r['target_idx'] for r in selected],correct=correct,accuracy=correct/size,
                examples_with_valid_query_domain_violation=size,domain_coverage_passed=False,
                domain_inference='Every one of the408 official records was reported to have a valid-query domain violation; therefore every selected subset record also does.',
                per_operator_subset_counts_available=(base==42 or size==408))
            if base==42:
                reference=data['subsets'][str(size)]
                assert indices==reference['positions'] and correct==reference['correct']
            if base==42 or size==408:
                reference=data['subsets'][str(size)] if base==42 else allset
                record.update(valid_query_failing_checks=reference['valid_query_failing_operators'],padded_query_failing_checks=reference['padded_query_failing_operators'])
            result.append(record)
    with tarfile.open(out/'sources.tar.gz','w:gz') as archive:
        for name in before:
            raw=Path(name).read_bytes();info=tarfile.TarInfo(name.lstrip('/'));info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
    with tarfile.open(out/'sources.tar.gz') as archive:
        assert {m.name for m in archive.getmembers()}=={n.lstrip('/') for n in before}
        assert all(hashlib.sha256(archive.extractfile(m).read()).hexdigest()==before['/'+m.name] for m in archive.getmembers())
    after={name:sha(name) for name in before};assert before==after
    report=dict(scope=__doc__,passed=True,domain_coverage_passed=False,rows=result,
        source_hashes_before=before,source_hashes_after=after,source_files_unchanged=True,
        source_archive_sha256=sha(out/'sources.tar.gz'),numpy_version=np.__version__,
        no_model_or_jax_imported=not any(n in sys.modules for n in ('torch','transformers','jax')),
        limitation='The original report does not retain per-example operator-domain counts. Exact per-operator counts for43/44 subsets of100/200 cannot be reconstructed; they are deliberately omitted.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ('base_seed','size','correct','accuracy','domain_coverage_passed')} for r in result],indent=2))
if __name__=='__main__':main()
