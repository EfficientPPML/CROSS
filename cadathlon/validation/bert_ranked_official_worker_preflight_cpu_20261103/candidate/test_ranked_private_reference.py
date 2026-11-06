"""Private-reader tests use only newly owned synthetic records, never retained data."""
from dataclasses import replace
from pathlib import Path
import hashlib,json,os
import pytest
import ranked_case_gate as gate
import ranked_private_reference as reader

HERE=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()


@pytest.fixture
def synthetic(tmp_path):
    # Public reports/archives only; the real private workspaces are never stat'ed.
    original=gate.authenticate(HERE/'public_evidence/fixture',gate.FIXTURE_REPORT_SHA,
        HERE/'public_evidence/reference',gate.REFERENCE_REPORT_SHA)
    root=tmp_path/'private';root.mkdir(mode=0o700);info=root.stat();workspace=gate.Workspace(root,info.st_dev,info.st_ino)
    cases=[];descriptors=[]
    def save(name,raw):
        (root/name).write_bytes(raw);(root/name).chmod(0o600)
        return gate.ReferenceFile(name,len(raw),sha(raw))
    for c in original.cases:
        rows=[dict(target_idx=i*7,label=1,native_fp32_logits=[0.,1.],original_positions=[0],
            synthetic_padding='x'*200) for i in range(c.count)]
        raw=(''.join(json.dumps(r)+'\n' for r in rows)).encode()
        file=save(c.reference_file.name,raw);descriptors.append(file)
        cases.append(replace(c,reference_workspace=workspace,reference_file=file,official_correct=c.count,
            target_order_sha256=sha(json.dumps([i*7 for i in range(c.count)]).encode()),valid_token_count_histogram=(('1',c.count),)))
    descriptors.append(save('master_references.jsonl',b'synthetic master; never selected\n'))
    c=cases[0];assets={Path(p).name:h for p,h in c.assets}
    manifest=dict(schema='bert-ranked-private-references-v1',
        bindings=dict(fixture_report_sha256=c.fixture_report_sha256,candidate_sha256=c.candidate_sha256,
            source_archive_sha256=c.reference_source_archive_sha256,checkpoint_sha256=assets['model.safetensors'],
            config_sha256=assets['config.json'],official_dtype='float32',official_attention='sdpa',
            diagnostic_dtype='float64',diagnostic_attention='eager',final_acceptance_requires_public_report=True,
            evaluation_completion_flags=dict(official_evaluation_complete=True,exact_reference_parity_passed=False,approximation_validation_passed=False)),
        files={d.name:dict(bytes=d.bytes,sha256=d.sha256) for d in descriptors},
        cases=[dict(size=c.size,seed=c.seed,count=c.count,input_sha256=c.input_sha256,private_target_order_sha256=c.target_order_sha256) for c in cases])
    descriptors.append(save('private_manifest.json',(json.dumps(manifest)+'\n').encode()))
    return replace(original,cases=tuple(cases),reference_workspace=workspace,reference_files=tuple(descriptors))


@pytest.mark.parametrize('size,seed',[(s,seed) for s in (1,2,3) for seed in (42,43,44)])
def test_all_selected_counts_no_approximation_gate(synthetic,size,seed):
    result=reader.load(synthetic,size=size,seed=seed)
    assert len(result.rows)==synthetic.case(size,seed).count
    assert [r['target_idx'] for r in result.records()]==list(range(0,len(result.rows)*7,7))
    if size==3:assert synthetic.case(size,seed).reference_file.bytes>65536


@pytest.mark.parametrize('fault',['missing','extra','symlink','mode','root_mode','bytes','inode','manifest_hash'])
def test_private_publication_mutation_rejected(synthetic,fault):
    root=synthetic.reference_workspace.path;p=root/synthetic.case(3,44).reference_file.name
    if fault=='missing':p.unlink()
    elif fault=='extra':(root/'extra').write_bytes(b'keep')
    elif fault=='symlink':p.unlink();p.symlink_to(root/'master_references.jsonl')
    elif fault=='mode':p.chmod(0o644)
    elif fault=='root_mode':root.chmod(0o755)
    elif fault=='bytes':p.write_bytes(b'changed')
    elif fault=='inode':root.rename(root.with_name('old'));root.mkdir(mode=0o700)
    else:(root/'private_manifest.json').write_bytes(b'{}')
    with pytest.raises((ValueError,OSError)):reader.load(synthetic,size=3,seed=44)
