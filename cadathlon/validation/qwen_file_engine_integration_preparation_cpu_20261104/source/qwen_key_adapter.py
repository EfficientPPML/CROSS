"""Public role binding only. Real loader entry is deliberately unavailable."""
from dataclasses import dataclass
from qwen_key_contract import (ExpectedIds, PublicKeyBinding, PrivateKeyBinding,
                               canonical, parse, PUBLIC_LIMIT)


@dataclass(frozen=True)
class Admission:
    public: PublicKeyBinding
    identity_bytes: bytes
    private: PrivateKeyBinding | None = None

    def check(self):
        if type(self) is not Admission or type(self.public) is not PublicKeyBinding:
            raise TypeError('exact public role admission required')
        self.public.check()
        identity = parse(self.identity_bytes, PUBLIC_LIMIT)
        if self.identity_bytes != canonical(identity):
            raise ValueError('canonical admitted public identity bytes required')
        self.public.expected.require_identity(identity, self.public.role)
        if self.public.role == 'client':
            if type(self.private) is not PrivateKeyBinding or self.private.public is not self.public:
                raise ValueError('client needs the same authenticated public/private binding')
            self.private.check()
        elif self.private is not None:
            raise ValueError('server role cannot carry private binding')
        return identity

    @property
    def execution_authorized(self):
        return False


def admit(public, identity_bytes, private=None):
    result = Admission(public, identity_bytes, private)
    result.check()
    return result


def open_client(admission, public_keys, client_private):
    # Deny before examining paths/objects, importing a reader or factory, or
    # treating the pure dataclass as an execution capability.
    raise PermissionError('Qwen client key-body loading awaits independent role authorization and successor proof')


def open_server(admission, public_keys):
    raise PermissionError('Qwen server key-body loading awaits independent role authorization and successor proof')
