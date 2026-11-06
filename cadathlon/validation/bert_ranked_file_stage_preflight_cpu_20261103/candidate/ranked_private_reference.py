"""Client-only bounded selected-reference reader. Never used by public/server gates.

This API must be called only after all-nine public authentication and explicit
client execution authorization. Development tests use newly created synthetic
0700/0600 fixtures; no retained reference body is opened during preparation.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib,json,os,stat,sys
from ranked_case_gate import CaseSet
from ranked_private_io import PrivateReader
from ranked_readback import label
import key_gate


@dataclass(frozen=True)
class ReferenceRow:
    target_idx:int
    truth:int
    official_logits:tuple
    original_positions:tuple


@dataclass(frozen=True)
class PrivateReferences:
    size:int
    seed:int
    input_sha256:str
    target_order_sha256:str
    file_sha256:str
    rows:tuple

    def records(self):
        return [dict(target_idx=r.target_idx,label=r.truth,native_fp32_logits=list(r.official_logits)) for r in self.rows]


def load(case_set,*,size,seed):
    if type(case_set)is not CaseSet:raise ValueError('authenticated all-nine descriptor set required')
    case=case_set.case(size,seed);root=case_set.reference_workspace
    if case.reference_workspace!=root:raise ValueError('reference workspace binding differs')
    descriptors={d.name:d for d in case_set.reference_files}
    if len(descriptors)!=11 or case.reference_file!=descriptors.get(case.reference_file.name):
        raise ValueError('complete private publication descriptor required')
    owner=PrivateReader()
    try:
        fd=owner._open_dir(root.path);info=os.fstat(fd)
        if (info.st_dev,info.st_ino)!=(root.device,root.inode):raise ValueError('retained reference owner identity differs')
        if set(os.listdir(fd))!=set(descriptors):raise ValueError('exact private reference publication inventory required')
        states={}
        for name,d in descriptors.items():
            if (type(name)is not str or Path(name).name!=name or type(d.bytes)is not int
                    or not 0<d.bytes<=16<<20):raise ValueError('bounded ordinary private reference descriptor required')
            state=owner._file_state(os.stat(name,dir_fd=fd,follow_symlinks=False))
            if state[2]!=d.bytes:raise ValueError('private reference descriptor size differs')
            states[name]=state
        def read(name):
            d=descriptors[name];raw=owner._read(root.path/name,fd,expected=states[name])
            if hashlib.sha256(raw).hexdigest()!=d.sha256:raise ValueError('authenticated private reference bytes differ')
            return raw
        manifest=key_gate.parse(read('private_manifest.json'))
        if (type(manifest)is not dict or set(manifest)!={'schema','bindings','files','cases'}
                or manifest['schema']!='bert-ranked-private-references-v1'):
            raise ValueError('private reference manifest schema differs')
        expected_files={name:dict(bytes=d.bytes,sha256=d.sha256) for name,d in descriptors.items() if name!='private_manifest.json'}
        if manifest['files']!=expected_files:raise ValueError('private reference manifest file bindings differ')
        expected_cases=[dict(size=c.size,seed=c.seed,count=c.count,input_sha256=c.input_sha256,
            private_target_order_sha256=c.target_order_sha256) for c in case_set.cases]
        if key_gate.canonical(manifest['cases'])!=key_gate.canonical(expected_cases):
            raise ValueError('all-nine private case order/source bindings differ')
        b=manifest['bindings'];assets={Path(p).name:h for p,h in case.assets}
        required=dict(fixture_report_sha256=case.fixture_report_sha256,candidate_sha256=case.candidate_sha256,
            source_archive_sha256=case.reference_source_archive_sha256,checkpoint_sha256=assets['model.safetensors'],
            config_sha256=assets['config.json'],official_dtype='float32',official_attention='sdpa',
            diagnostic_dtype='float64',diagnostic_attention='eager',final_acceptance_requires_public_report=True)
        if (type(b)is not dict or set(b)!=set(required)|{'evaluation_completion_flags'}
                or any(type(b.get(k))is not type(v) or b[k]!=v for k,v in required.items())):
            raise ValueError('private reference public producer bindings differ')
        flags=b['evaluation_completion_flags']
        if (type(flags)is not dict or set(flags)!={'official_evaluation_complete','exact_reference_parity_passed','approximation_validation_passed'}
                or any(type(x)is not bool for x in flags.values()) or flags['official_evaluation_complete']is not True):
            raise ValueError('private official evaluation completion differs')
        raw=read(case.reference_file.name)
        try:rows=[key_gate.parse(line) for line in raw.decode('utf8').splitlines()]
        except Exception:raise ValueError('private reference records failed decoding') from None
        if len(rows)!=case.count:raise ValueError('complete private reference count required')
        selected=[];ids=[];hist={};correct=0
        for row in rows:
            if (type(row)is not dict or type(row.get('target_idx'))is not int or row['target_idx']<0
                    or type(row.get('label'))is not int or row['label'] not in (0,1)):
                raise ValueError('private reference target/label schema differs')
            predicted=label(row.get('native_fp32_logits'));pos=row.get('original_positions')
            if (type(pos)is not list or not 1<=len(pos)<=128 or pos[0]!=0
                    or any(type(x)is not int or not 0<=x<128 for x in pos) or pos!=sorted(set(pos))):
                raise ValueError('private reference original positions differ')
            ids.append(row['target_idx']);correct+=predicted==row['label'];hist[str(len(pos))]=hist.get(str(len(pos)),0)+1
            selected.append(ReferenceRow(row['target_idx'],row['label'],tuple(float(x) for x in row['native_fp32_logits']),tuple(pos)))
        if (len(set(ids))!=case.count or hashlib.sha256(json.dumps(ids).encode()).hexdigest()!=case.target_order_sha256
                or correct!=case.official_correct or hist!=dict(case.valid_token_count_histogram)):
            raise ValueError('private reference order/official aggregate/public positions binding differs')
        # Every descriptor's metadata stays stable; selected bytes were read
        # from held descriptors and copied, never passed as mutable memmaps.
        if set(os.listdir(fd))!=set(descriptors):raise ValueError('private reference inventory changed')
        for name,expected in states.items():
            if owner._file_state(os.stat(name,dir_fd=fd,follow_symlinks=False))!=expected:
                raise ValueError('private reference metadata changed during read')
        visible=root.path.lstat()
        if (visible.st_dev,visible.st_ino)!=(root.device,root.inode) or stat.S_IMODE(visible.st_mode)!=0o700:
            raise ValueError('private reference owner changed during read')
        return PrivateReferences(size,seed,case.input_sha256,case.target_order_sha256,case.reference_file.sha256,tuple(selected))
    finally:owner.close(sys.exception())
