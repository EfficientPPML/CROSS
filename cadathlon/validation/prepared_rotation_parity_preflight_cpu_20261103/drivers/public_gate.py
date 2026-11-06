"""Public completed-prerequisite preflight only. Deny every payload read."""
from pathlib import Path
import os,sys
ROOT=Path('/tmp/bert-prepared-rotation-parity-20261103')
sys.path.insert(0,str(ROOT))
forbidden=[]
def audit(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0]))
        if p.suffix in ('.npy','.npz','.safetensors') or 'client_private' in p.parts:
            forbidden.append(str(p));raise AssertionError('public preflight attempted payload/private read')
sys.addaudithook(audit)
import parity_collect
args=dict(key_evidence=Path('/tmp/bert-full-key-generation-component-20261103/actual-v1'),
 key_report_sha256='022cbdf687273e765668184f32cb7cd501d371c4220916e1c8f02574cf3a7f08',
 key_workspace=Path('/dev/shm/cross-full-bert-keys-143b3f5c1573427d841760a3c9d5d33d'),
 model_evidence=Path('/tmp/bert-public-model-stage2-v2-component-20261103/actual-v2'),
 model_report_sha256='5b9589b51828d066706c4c46a6404d8c52fcd67a2c9aa9f3f57c73df8caeefe0',
 model_workspace=Path('/tmp/cross-bert-public-model-b51668856dab42a3846ba1b7c7a71d53'),
 stage3_evidence=Path('/tmp/bert-input-encryption-stage3-component-v3-20261103/actual-v3'),
 stage3_report_sha256='add5e6a3143be3931ad29c018cdee40bdf059497b8bfa987114f39baeb88631e',
 io_workspace=Path('/tmp/cross-bert-input-encryption-7b255c2357c94060b73e474a1f8ef5fa'))
if __name__=='__main__':
    result=parity_collect.run(Path(sys.argv[1]),execute=False,**args)
    assert result['passed'] and not result['actual_execution_attempted'] and not forbidden
