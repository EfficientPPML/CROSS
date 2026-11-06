"""Q5 key-manifest/role identities; data-only, never launch authority."""
from dataclasses import dataclass
from q5_key_contract import PublicProfile,PublicKeyBinding,PrivateKeyBinding,parse,canonical,hash_value

@dataclass(frozen=True)
class FilesAdmission:
    public:PublicKeyBinding
    identity_bytes:bytes
    private:PrivateKeyBinding|None=None
    @property
    def keys(self):return self
    def check(self):
        if type(self)is not FilesAdmission or type(self.public)is not PublicKeyBinding:
            raise TypeError('exact Q5 key files admission required')
        self.public.check();identity=parse(self.identity_bytes,1<<16)
        fields={'context_id','model_id','producer_source_manifest_sha256','consumer_source_manifest_sha256'}
        if type(identity)is not dict or set(identity)!=fields or canonical(identity)!=self.identity_bytes:
            raise ValueError('closed independently bound Q5 identity required')
        for value in identity.values():hash_value(value)
        self.public.expected.require_identity(identity,self.public.role)
        if self.public.role=='client':
            if type(self.private)is not PrivateKeyBinding or self.private.public is not self.public:
                raise ValueError('same original public/private key binding required')
            self.private.check()
        elif self.private is not None:raise ValueError('server cannot carry private key identity')
        return identity
    @property
    def execution_authorized(self):return False

def bind_files(public,identity_bytes,private=None):
    result=FilesAdmission(public,identity_bytes,private);result.check();return result
