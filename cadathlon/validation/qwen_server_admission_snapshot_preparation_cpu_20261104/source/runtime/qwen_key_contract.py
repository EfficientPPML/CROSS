"""Pure public Qwen166 key-manifest admission; no I/O or execution authority."""
from dataclasses import dataclass
import hashlib
import json
import math
import re

PROFILE = '2b2457e50da73b9ccf87984c089afcb99586fde863a48e98b89a1b7ccf872a17'
PLAN = 'bac55debfe95752a16a9a2e849e66488f4afbb576959403a2af4698358dd32ca'
KEYGEN = '742e6af9e1ce9025416fa2ef68ae037bca7e772d48fbbd6ca387cc79820119c8'
CHECKPOINT = 'f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b'
CONFIG = '660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd'
CANDIDATE = '344954b3873a8847ececba320646f52612ca7277412c7de81a7313894e21799a'
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
    return dict(checkpoint_sha256=CHECKPOINT, config_sha256=CONFIG,
                candidate_sha256=CANDIDATE, profile_sha256=PROFILE,
                rotation_plan_sha256=PLAN)


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
                type(self.plan_bytes) is not bytes or sha(self.profile_bytes) != PROFILE or
                sha(self.plan_bytes) != PLAN):
            raise ValueError('exact208 profile and actual234-matrix plan bytes required')
        audit = parse(self.profile_bytes, PUBLIC_LIMIT)
        selected = [p for p in audit['profiles'] if p['name'] == 'depth49_p60_d8']
        if len(selected) != 1:
            raise ValueError('one fixed profile required')
        p = selected[0]
        plan = parse(self.plan_bytes, PUBLIC_LIMIT)
        rotations = plan['combined_keys']
        if (plan['passed'] is not True or type(rotations) is not list or len(rotations) != 166 or
                any(type(x) is not int for x in rotations) or rotations != sorted(set(rotations)) or
                262143 not in rotations or 65535 not in rotations or plan['matrix_map_count'] != 234):
            raise ValueError('complete literal166 key inventory required')
        if (p['degree'], p['slots'], p['depth'], p['dnum'], len(p['q']), len(p['p'])) != (131072,65536,49,8,100,14):
            raise ValueError('fixed physical profile differs')
        product = math.prod(p['q'] + p['p'])
        alpha = 13
        parts = [math.prod(p['q'][first:first+alpha]) for first in range(0,100,alpha)]
        if product.bit_length() > 3523 or math.prod(p['p']) < max(parts):
            raise ValueError('fixed actual QP/P coverage check failed')
        return p, tuple(rotations)

    @property
    def parameters(self):
        p, _ = self.check()
        return dict(degree=p['degree'], num_slots=p['slots'], q_towers=p['q'], p_towers=p['p'],
                    composite_degree=2, degree_layout=[256,512], precision=32, max_level=p['depth'],
                    scaling_factor=p['canonical_scale'], dnum=p['dnum'], batch=1,
                    sigma=3.190000057220458984375, secret_distribution='uniform_ternary')

    @property
    def rotations(self):
        return self.check()[1]

    @property
    def arrays(self):
        p = self.parameters
        q = tuple(p['q_towers']);qp = q + tuple(p['p_towers'])
        n = p['degree'];switched = (8,114,n)
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
    return dict(schema='cross-qwen-full-public-keys-v1', bindings=bindings(),
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
            raise ValueError('complete335 public array inventory required')
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
        expected = dict(schema='cross-qwen-full-private-keys-v1', context_id=public['context_id'],
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
