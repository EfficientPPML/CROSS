"""Opaque six-field serialization; NumPy only, never encrypts or decodes slots."""
from dataclasses import dataclass
import io,json
import numpy as np
from qwen_file_contract import canonical,sha,same,scale
from reused_rns_payload import _payload

@dataclass(frozen=True)
class Serialized:
    raw: bytes
    binding: bytes

    def check(self,binding):
        if type(self)is not Serialized or type(self.raw)is not bytes or self.binding!=canonical(binding):
            raise ValueError('exact opaque bytes and complete typed binding required')
        _payload(self.raw,dict(shape=tuple(binding['shape']),moduli=binding['moduli'],layout=binding['degree_layout'],scale=binding['scale']))
        return self.raw


def serialize_array(payload, binding, *, moduli, degree_layout, precision, actual_scale, nsd):
    """Snapshot exact canonical uint32 storage before packaging; never cast it."""
    if type(payload)is not np.ndarray or payload.dtype!=np.dtype('uint32') or list(payload.shape)!=binding['shape'] or not payload.flags.c_contiguous:
        raise ValueError('exact contiguous uint32 physical payload required')
    same(list(moduli),binding['moduli'],'actual Q prefix differs')
    same(list(degree_layout),binding['degree_layout'],'actual degree layout differs')
    same(precision,32,'actual precision differs');same(nsd,1,'actual NSD differs')
    same(scale(actual_scale),binding['scale'],'actual scale differs')
    snapshot=np.frombuffer(payload.tobytes(order='C'),dtype=np.uint32).reshape(payload.shape)
    for i,q in enumerate(binding['moduli']):
        if np.any(snapshot[...,i]>=q):raise ValueError('actual residue outside original Q prefix')
    stream=io.BytesIO()
    np.savez(stream,payload=snapshot,moduli=np.asarray(moduli,dtype=np.int64),
        degree_layout=np.asarray(degree_layout,dtype=np.int64),precision=np.asarray(precision,dtype=np.int64),
        scale=np.asarray(actual_scale,dtype=np.float64),nsd=np.asarray(nsd,dtype=np.int64))
    result=Serialized(stream.getvalue(),canonical(binding));result.check(binding);return result


def serialize_ciphertext(value,binding):
    """Concrete existing Polynomial facade; caller owns context/security gates.

    No backend import is needed. The borrowed original arrays remain strongly
    held through barriers, host snapshot and metadata recheck. Synthetic tests
    use recording objects; actual ciphertext access is not enabled by a factory.
    """
    polynomial=value.polynomial;moduli_array=value.moduli_array
    def metadata():
        value.validate()
        return canonical(dict(shape=list(value.shape),moduli=list(value.get_moduli()),
            degree_layout=list(value.degree_layout),precision=value.precision,scale=value._ckks_scale,nsd=value._ckks_nsd))
    before=metadata();polynomial.block_until_ready();moduli_array.block_until_ready()
    host=np.asarray(polynomial)
    result=serialize_array(host,binding,moduli=value.get_moduli(),degree_layout=value.degree_layout,
        precision=value.precision,actual_scale=value._ckks_scale,nsd=value._ckks_nsd)
    if value.polynomial is not polynomial or value.moduli_array is not moduli_array or metadata()!=before:
        raise ValueError('borrowed ciphertext changed during serialization')
    return result
