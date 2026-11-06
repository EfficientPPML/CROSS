"""Public-only ranked fixture/reference authentication; no private body reads.

The returned frozen descriptors authorize no execution. A later client must
authenticate the private bytes/identity and completed encrypted stage proofs.
Official native availability is independent of approximate diagnostic success.
"""
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import io
import math
import tarfile
import key_gate
import input_gate
import ranked_input_gate

HERE = Path(__file__).resolve().parent
FIXTURE_REPORT_SHA = 'ad1458f5183c6ccf2631c2a8771d0da92196f5f9c690016e2a0b12fa7b88f4d7'
REFERENCE_REPORT_SHA = '77f93c90e03609bdaf12441cc8a8d07e08b5e1a475c1480f283b4f592668ae74'
CANDIDATE_SHA = '4e2277beea7189aa1d77b168c500c8a175f6c630afa520fcc6b38cb5a179cff9'
REVISION = 'f150c1d609d1e50dd5e2e5408661cfac8339277c'
REFERENCE_FLAGS = (
    'official_reference_complete', 'private_reference_publication_complete',
    'official_references_retained', 'private_workspace_retained', 'error_absent',
    'source_files_unchanged', 'source_archive_verified', 'assets_unchanged',
    'original_fixture_unchanged', 'public_prerequisite_unchanged',
    'cleanup_completed', 'terminal_scope_completed', 'no_jax_or_he_modules_loaded')


@dataclass(frozen=True)
class Workspace:
    path: Path
    device: int
    inode: int


@dataclass(frozen=True)
class ReferenceFile:
    name: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class RankedCase:
    size: int
    seed: int
    count: int
    generator_seed: int
    input_sha256: str
    target_order_sha256: str
    row_order_sha256: str
    preprocessed_input: Path
    official_correct: int
    fixture_report_sha256: str
    reference_report_sha256: str
    fixture_source_archive_sha256: str
    reference_source_archive_sha256: str
    candidate_sha256: str
    model_revision: str
    assets: tuple
    fixture_workspace: Workspace
    reference_workspace: Workspace
    reference_file: ReferenceFile
    input_files: tuple
    valid_token_count_histogram: tuple


@dataclass(frozen=True)
class CaseSet:
    cases: tuple
    fixture_report_sha256: str
    reference_report_sha256: str
    fixture_source_archive_sha256: str
    reference_source_archive_sha256: str
    assets: tuple
    candidate_sha256: str
    model_revision: str
    fixture_workspace: Workspace
    reference_workspace: Workspace
    reference_files: tuple

    def case(self, size, seed):
        if type(size) is not int or type(seed) is not int:
            raise ValueError('exact integer size/seed required')
        for row in self.cases:
            if (row.size,row.seed) == (size,seed):
                return row
        raise ValueError('case outside predeclared ranked schedule')

    @property
    def private_manifest(self):
        return next(f for f in self.reference_files if f.name == 'private_manifest.json')


def _workspace(path, identity):
    if type(path) is not str:
        raise ValueError('public workspace path must be a string')
    p=Path(path)
    if (not p.is_absolute() or '..' in p.parts or p==Path('/') or
        type(identity) is not dict or set(identity)!={'device','inode'} or
        any(type(v) is not int or v<0 for v in identity.values())):
        raise ValueError('exact absolute private owner descriptor required')
    # Deliberately no lstat/open here: these are public authenticated descriptors.
    return Workspace(p,identity['device'],identity['inode'])


def _reference_archive(evidence, report):
    expected=key_gate.parse(key_gate.read(HERE/'reference_source_manifest.json',1<<20))
    if (type(expected) is not dict or len(expected)!=41 or
        report.get('source_hashes_before')!=expected or report.get('source_hashes_after')!=expected):
        raise ValueError('exact41 unchanged reference sources required')
    for name,digest in expected.items():
        p=PurePosixPath(name)
        if type(name) is not str or p.is_absolute() or '..' in p.parts:
            raise ValueError('relative reference source path required')
        key_gate.sha_field(digest)
    raw=key_gate.read(Path(evidence)/'sources.tar.gz',16<<20)
    if key_gate.digest(raw)!=key_gate.sha_field(report.get('source_archive_sha256')):
        raise ValueError('completed reference archive hash differs')
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        members=archive.getmembers()
        if (len(members)!=41 or len({m.name for m in members})!=41 or
            set(m.name for m in members)!=set(expected) or
            any(not m.isfile() or not 0<=m.size<=8<<20 for m in members) or
            sum(m.size for m in members)>32<<20):
            raise ValueError('bounded exact reference archive inventory required')
        hashes={m.name:key_gate.digest(archive.extractfile(m).read()) for m in members}
    if hashes!=expected:
        raise ValueError('reference source bytes differ')
    return key_gate.digest(raw)


def authenticate(fixture_evidence, fixture_report_sha, reference_evidence, reference_report_sha):
    # Both caller-provided external pins must match this reviewed publication
    # before any evidence is opened. Self-consistent replacement hashes fail.
    if (key_gate.sha_field(fixture_report_sha)!=FIXTURE_REPORT_SHA or
        key_gate.sha_field(reference_report_sha)!=REFERENCE_REPORT_SHA):
        raise ValueError('reviewed external ranked publication pins required')
    authorized=ranked_input_gate.public(fixture_evidence,fixture_report_sha)
    fixture=authorized['report']
    raw=key_gate.read(Path(reference_evidence)/'report.json',4<<20)
    if key_gate.digest(raw)!=reference_report_sha:
        raise ValueError('external reference report bytes differ')
    report=key_gate.parse(raw)
    if (type(report) is not dict or any(report.get(n) is not True for n in REFERENCE_FLAGS) or
        any(report.get(n) is not True for n in ('execute','actual_native_started','official_evaluation_complete','all_nine_private_inputs_authenticated')) or
        report.get('error') or report.get('cleanup_errors') or report.get('raw_arrays_archived') is not False or
        report.get('reference_acceptance_requires')!=list(REFERENCE_FLAGS)):
        raise ValueError('complete official reference publication/retention/cleanup required')
    if (report.get('fixture_report_sha256')!=fixture_report_sha or report.get('candidate_sha256')!=CANDIDATE_SHA or
        type(report.get('unique_records')) is not int or report['unique_records']!=408 or
        type(report.get('selected_records')) is not int or report['selected_records']!=2124):
        raise ValueError('complete original ranked reference identity/counts required')
    policy=report.get('native_policy')
    if (type(policy) is not dict or policy.get('official_dtype')!='float32' or
        policy.get('official_attention')!='sdpa' or policy.get('official_attention_classes')!=['BertSdpaSelfAttention'] or
        type(policy.get('native_forward_original_tokens')) is not int or policy['native_forward_original_tokens']!=128 or
        type(policy.get('encoder_layers')) is not int or policy['encoder_layers']!=12):
        raise ValueError('official original-token FP32 checkpoint policy required')
    if report.get('asset_hashes_before')!=input_gate.ASSETS or report.get('asset_hashes_after')!=input_gate.ASSETS:
        raise ValueError('reference dataset/model/tokenizer asset identity differs')
    reference_archive=_reference_archive(reference_evidence,report)
    fixture_archive=key_gate.sha_field(fixture['source_archive_sha256'])
    fw=_workspace(str(authorized['workspace']),authorized['identity'])
    rw=_workspace(report.get('private_workspace'),report.get('private_workspace_identity'))
    if fw.path==rw.path or fw.path in rw.path.parents or rw.path in fw.path.parents:
        raise ValueError('reference and original fixture roots must be disjoint')
    expected_files={'master_references.jsonl','private_manifest.json'}|{
        f'size{s}-seed{seed}-references.jsonl' for s,c,seed in ranked_input_gate.SCHEDULE}
    files=report.get('private_reference_files')
    if type(files) is not dict or set(files)!=expected_files:
        raise ValueError('exact11 retained private reference descriptors required')
    descriptors={}
    for name,row in files.items():
        if (type(row) is not dict or set(row)!={'bytes','sha256'} or
            type(row['bytes']) is not int or not 0<row['bytes']<=16<<20):
            raise ValueError('bounded private reference descriptor required')
        descriptors[name]=ReferenceFile(name,row['bytes'],key_gate.sha_field(row['sha256']))
    # All three full-split reference files must bind the same master bytes.
    master=descriptors['master_references.jsonl']
    for seed in (42,43,44):
        item=descriptors[f'size3-seed{seed}-references.jsonl']
        if (item.bytes,item.sha256)!=(master.bytes,master.sha256):
            raise ValueError('full408 reference bytes differ from canonical master')
    rows=report.get('cases')
    if type(rows) is not list or len(rows)!=9:
        raise ValueError('exact ordered nine reference cases required')
    assets=tuple(sorted(input_gate.ASSETS.items()));cases=[]
    for row,original,(size,count,seed) in zip(rows,fixture['cases'],ranked_input_gate.SCHEDULE):
        if (type(row) is not dict or any(type(row.get(n)) is not int or row[n]!=v for n,v in
            (('size',size),('seed',seed),('count',count))) or row.get('official_reference_complete') is not True):
            raise ValueError('reference case order/count/completion differs')
        name=f'io/{ranked_input_gate.NAMES[size]}/intermediate/client_preprocessed_input'
        input_sha=original['input_file_sha256'][name];order=original['private_target_order_sha256']
        if row.get('input_sha256')!=input_sha or row.get('private_target_order_sha256')!=order:
            raise ValueError('reference case input/target-order binding differs')
        correct=row.get('official_correct');accuracy=row.get('official_accuracy')
        if (type(correct) is not int or not 0<=correct<=count or type(accuracy) is not float or
            not math.isfinite(accuracy) or accuracy!=correct/count):
            raise ValueError('complete official aggregate count/accuracy required')
        cases.append(RankedCase(size,seed,count,original['generator_seed'],input_sha,order,
            original['private_row_order_sha256'],Path(original['preprocessed_input_path']),correct,
            fixture_report_sha,reference_report_sha,fixture_archive,reference_archive,CANDIDATE_SHA,REVISION,assets,fw,rw,
            descriptors[f'size{size}-seed{seed}-references.jsonl'],tuple(sorted(original['input_file_sha256'].items())),
            tuple(sorted((int(k),v) for k,v in original['valid_token_count_histogram'].items()))))
    return CaseSet(tuple(cases),fixture_report_sha,reference_report_sha,fixture_archive,reference_archive,
        assets,CANDIDATE_SHA,REVISION,fw,rw,tuple(descriptors[n] for n in sorted(descriptors)))
