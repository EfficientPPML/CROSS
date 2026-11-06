"""Full immutable Qwen public-contract admission before any reader call."""
from dataclasses import dataclass
from qwen_file_contract import PublicContract,canonical,parse
from qwen_key_adapter import Admission,admit


@dataclass(frozen=True)
class FilesAdmission:
    public_contract: PublicContract
    keys: Admission

    def check(self):
        if type(self) is not FilesAdmission or type(self.public_contract) is not PublicContract or type(self.keys) is not Admission:
            raise TypeError('exact full public and key admissions required')
        self.public_contract.check()
        identity=parse(self.keys.identity_bytes)
        # Full structural/source/profile schema before the narrower external-ID
        # and manifest bindings; neither is a grant to open actual key bodies.
        self.public_contract.check_identity(identity)
        self.keys.check()
        if canonical(self.keys.public.profile.parameters)!=canonical(self.public_contract.profile()):
            raise ValueError('key and full file profile differ')
        if tuple(self.public_contract.json('inputs/actual_key_plan.json')['combined_keys'])!=self.keys.public.profile.rotations:
            raise ValueError('key and full owner literal166 inventory differ')
        return self.keys

    @property
    def execution_authorized(self):return False


def bind_files(public_contract,public,identity_bytes,private=None):
    if type(public_contract)is not PublicContract:
        raise TypeError('exact source-bound PublicContract required')
    public_contract.check_identity(parse(identity_bytes))
    result=FilesAdmission(public_contract,admit(public,identity_bytes,private))
    result.check()
    return result
