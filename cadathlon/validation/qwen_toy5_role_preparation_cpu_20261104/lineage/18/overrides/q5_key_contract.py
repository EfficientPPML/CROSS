"""Pure public Q5 100-rotation/203-array key-manifest admission; no I/O or execution authority."""
from dataclasses import dataclass
import hashlib
import json
import math
import re

PROFILE = 'bee048004e3f62ed82e9c308841e9bed86dbc65b147374d4ec720db87492e916'
PLAN = '1207f2cd2328ddfc5a7e06f622579d956968382ffd581d59f769ef8909168cdd'
KEYGEN = '742e6af9e1ce9025416fa2ef68ae037bca7e772d48fbbd6ca387cc79820119c8'
CASE = 'df67a2c4e59030d923b019c9d57a2eedd96d11ad1e4eb31bbbddee69d460c018'
CHECKPOINT_PLAN = '5162049ffc45bf47917964ce76dcc04d8ef988737b598b7905348ee24f781acc'
PUBLIC_LIMIT = 1 << 20
PRIVATE_LIMIT = 1 << 16
HEX = re.compile(r'[0-9a-f]{64}\Z')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def parse(raw, maximum):
    if type(raw) is not bytes or not 0 < len(raw) <= maximum:
        raise ValueError('exact bounded nonempty metadata bytes required')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate metadata key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite metadata constant')
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    canonical(value)  # Also rejects an overflowing JSON exponent.
    return value


def hash_value(value):
    if type(value) is not str or not HEX.fullmatch(value):
        raise ValueError('exact lowercase SHA256 identity required')
    return value


def bindings():
    return dict(case_sha256=CASE,profile_sha256=PROFILE,
                original_polynomial_metadata_sha256=PLAN,checkpoint_plan_sha256=CHECKPOINT_PLAN)


@dataclass(frozen=True)
class ExpectedIds:
    """Externally supplied expectations, not self-derived manifest values."""
    context_id: str
    public_manifest_sha256: str
    keygen_source_sha256: str
    model_id: str
    producer_source_manifest_sha256: str
    consumer_source_manifest_sha256: str
    private_manifest_sha256: str | None = None

    def check(self, role):
        if type(self) is not ExpectedIds or type(role) is not str or role not in ('client', 'server'):
            raise TypeError('exact expected IDs and role required')
        for name in ('context_id', 'public_manifest_sha256', 'keygen_source_sha256',
                     'model_id', 'producer_source_manifest_sha256', 'consumer_source_manifest_sha256'):
            hash_value(getattr(self, name))
        if self.keygen_source_sha256 != KEYGEN:
            raise ValueError('externally expected reviewed keygen source differs')
        if role == 'client':
            hash_value(self.private_manifest_sha256)
        elif self.private_manifest_sha256 is not None:
            raise ValueError('server admission cannot carry private manifest identity')

    def require_identity(self, identity, role):
        """Bind IDs after the caller's full PublicContract schema check.

        This intentionally checks only these four independent expected IDs;
        it does not replace the public file schema/authenticated producer gate.
        """
        self.check(role)
        if type(identity) is not dict:
            raise TypeError('public file identity dictionary required')
        for name in ('context_id', 'model_id', 'producer_source_manifest_sha256',
                     'consumer_source_manifest_sha256'):
            if type(identity.get(name)) is not str or identity[name] != getattr(self, name):
                raise ValueError('externally expected public role identity differs')


@dataclass(frozen=True)
class ArraySpec:
    name: str
    shape: tuple
    moduli: tuple

    @property
    def raw_bytes(self):
        return math.prod(self.shape) * 4

    def validate(self, descriptor):
        if (type(descriptor) is not dict or set(descriptor) != {'sha256','dtype','shape','bytes'} or
                type(descriptor.get('dtype')) is not str or descriptor['dtype'] != 'uint32' or
                type(descriptor.get('shape')) is not list or
                any(type(v) is not int for v in descriptor['shape']) or
                tuple(descriptor['shape']) != self.shape or
                type(descriptor.get('bytes')) is not int or
                not self.raw_bytes < descriptor['bytes'] <= self.raw_bytes + 4096):
            raise ValueError('bounded exact key array descriptor differs')
        hash_value(descriptor.get('sha256'))


@dataclass(frozen=True)
class PublicProfile:
    profile_bytes: bytes
    plan_bytes: bytes

    @classmethod
    def from_bytes(cls, profile_bytes, plan_bytes):
        result = cls(profile_bytes, plan_bytes)
        result.check()
        return result

    def check(self):
        if (type(self) is not PublicProfile or type(self.profile_bytes) is not bytes or
                type(self.plan_bytes) is not bytes or sha(self.profile_bytes)!=PROFILE or sha(self.plan_bytes)!=PLAN):
            raise ValueError('exact original Q5 profile/replay metadata required')
        profile=parse(self.profile_bytes,PUBLIC_LIMIT);report=parse(self.plan_bytes,PUBLIC_LIMIT)
        p=profile['ring'];rotations=report['diagnostic']['rotations']['union']
        if (type(rotations)is not list or len(rotations)!=100 or any(type(x)is not int for x in rotations)
                or rotations!=sorted(set(rotations)) or 65535 not in rotations or 262143 not in rotations):
            raise ValueError('complete literal100 Q5 rotations required')
        if (p['degree'],p['num_slots'],p['max_level'],p['dnum'],len(p['q_towers']),len(p['p_towers']))!=(131072,65536,49,4,100,24):
            raise ValueError('original Q5 physical profile differs')
        q=p['q_towers'];aux=p['p_towers']
        if any(type(v)is not int or not 1<v<2**32 for v in q+aux):raise ValueError('exact uint32 Q/P towers required')
        if math.prod(q+aux).bit_length()>3523 or math.prod(aux)<max(math.prod(q[i:i+25])for i in range(0,100,25)):
            raise ValueError('original Q5 actual QP/P-coverage bound fails')
        if type(profile['scale'])is not float or profile['scale']!=float(q[-2])*q[-1]:
            raise ValueError('original canonical Q5 scale differs')
        return profile,tuple(rotations)

    @property
    def parameters(self):
        profile,_=self.check();p=profile['ring']
        return dict(degree=131072,num_slots=65536,q_towers=p['q_towers'],p_towers=p['p_towers'],
                    composite_degree=2,degree_layout=[256,512],precision=32,max_level=49,
                    scaling_factor=profile['scale'],dnum=4,batch=1,
                    sigma=3.190000057220458984375,secret_distribution='uniform_ternary')

    @property
    def rotations(self):
        return self.check()[1]

    @property
    def arrays(self):
        p = self.parameters
        q = tuple(p['q_towers']);qp = q + tuple(p['p_towers'])
        n = p['degree'];switched = (4,124,n)
        rows = [ArraySpec('public_key.npy', (2,100,n), q)]
        rows.extend(ArraySpec('relin_'+part+'.npy',switched,qp) for part in ('a','b'))
        rows.extend(ArraySpec(f'rot_{index}_{part}.npy',switched,qp)
                    for index in self.rotations for part in ('a','b'))
        return tuple(rows)

    @property
    def secret(self):
        return ArraySpec('secret.npy',(100,131072),tuple(self.parameters['q_towers']))

    @property
    def resources(self):
        rows = self.arrays
        return dict(public_array_count=len(rows), rotation_count=len(self.rotations),
                    public_array_bytes=sum(r.raw_bytes for r in rows),
                    private_array_bytes=self.secret.raw_bytes,
                    public_npy_bytes_upper=sum(r.raw_bytes+4096 for r in rows),
                    private_npy_bytes_upper=self.secret.raw_bytes+4096,
                    exclusions='generation/compiler/control/transient/device/model/ciphertext bytes; not peak memory')


def public_body(profile, arrays, keygen_source):
    if type(profile) is not PublicProfile:
        raise TypeError('exact public profile required')
    return dict(schema='cross-qwen-toy5-public-keys-v1', bindings=bindings(),
                parameters=profile.parameters, rotations=list(profile.rotations), arrays=arrays,
                representation='standard-ntt-residues/digit-modulus-degree',
                keygen_source_sha256=keygen_source, security_target=dict(bits=128,attack='classical'),
                reusable_rotation_keys=True)


@dataclass(frozen=True)
class PublicKeyBinding:
    profile: PublicProfile
    expected: ExpectedIds
    raw: bytes
    role: str

    def check(self):
        if (type(self) is not PublicKeyBinding or type(self.expected) is not ExpectedIds or
                type(self.profile) is not PublicProfile):
            raise TypeError('exact immutable public key binding required')
        self.expected.check(self.role)
        parsed = parse(self.raw, PUBLIC_LIMIT)
        if sha(self.raw) != self.expected.public_manifest_sha256:
            raise ValueError('external public manifest identity differs')
        if type(parsed) is not dict or type(parsed.get('arrays')) is not dict:
            raise ValueError('public key manifest dictionary required')
        specs = self.profile.arrays
        if set(parsed['arrays']) != {s.name for s in specs}:
            raise ValueError('complete203 public array inventory required')
        for spec in specs:
            spec.validate(parsed['arrays'][spec.name])
        body = public_body(self.profile, parsed['arrays'], self.expected.keygen_source_sha256)
        expected = body | dict(context_id=sha(canonical(body)))
        if self.raw != canonical(expected) or expected['context_id'] != self.expected.context_id:
            raise ValueError('external context/canonical public manifest differs')
        return parsed

    @property
    def manifest(self):
        return self.check()


def authenticate_public(raw, profile, expected, *, role):
    result = PublicKeyBinding(profile, expected, raw, role)
    result.check()
    return result


@dataclass(frozen=True)
class PrivateKeyBinding:
    public: PublicKeyBinding
    raw: bytes

    def check(self):
        if type(self) is not PrivateKeyBinding or type(self.public) is not PublicKeyBinding:
            raise TypeError('exact immutable private key binding required')
        public = self.public.check()
        if self.public.role != 'client':
            raise ValueError('private binding only belongs to client role')
        parsed = parse(self.raw, PRIVATE_LIMIT)
        if sha(self.raw) != self.public.expected.private_manifest_sha256:
            raise ValueError('external private manifest identity differs')
        if type(parsed) is not dict or 'secret' not in parsed:
            raise ValueError('private key manifest dictionary required')
        self.public.profile.secret.validate(parsed['secret'])
        expected = dict(schema='cross-qwen-toy5-private-keys-v1', context_id=public['context_id'],
                        public_manifest_sha256=sha(self.public.raw), bindings=bindings(), secret=parsed['secret'])
        if self.raw != canonical(expected):
            raise ValueError('private/public canonical identity differs')
        return parsed

    @property
    def manifest(self):
        return self.check()


def authenticate_private(raw, public):
    result = PrivateKeyBinding(public, raw)
    result.check()
    return result
