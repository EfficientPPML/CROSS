from audit_public import Audit
import pytest

@pytest.mark.parametrize('event,args',[('import',('torch',)),('open',('/outside/weights.npy','r',0)),('open',('/home/jianming_gatech/cross_validation_assets/public.json','r',0))])
def test_body_or_backend_blocked_without_read(event,args):
    a=Audit()
    with pytest.raises(RuntimeError):a(event,args)
    assert len(a.denied)==1

def test_public_source_and_file_descriptors_allowed():
    a=Audit();a('open',('inputs/candidate_manifest.json','r',0));a('open',(5,'r',0));a('import',('json',))
    assert a.denied==[]
