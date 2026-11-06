"""Client-only pinned BERT embedding and original-position row packing.

This module never loads or executes encoder, pooler, or classifier tensors. It
performs no encryption; the caller owns the private target-index association.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import stat
import numpy as np

REVISION = 'f150c1d609d1e50dd5e2e5408661cfac8339277c'
ASSETS = {
    'config.json': 'ec2b6cd6ce741dda6c972aea96f9d1a449cbb4b5349e688c5370a9b2f6872d30',
    'model.safetensors': 'e1f8085ef673c9eb8a1a93a84270742e8aa0ccbceadef01eef15e26c44b46034',
}
EMBEDDING_SHAPES = {
    'word_embeddings.weight': (28996, 768),
    'position_embeddings.weight': (512, 768),
    'token_type_embeddings.weight': (2, 768),
    'LayerNorm.weight': (768,),
    'LayerNorm.bias': (768,),
}
FIELDS = frozenset(('target_idx', 'input_ids', 'token_type_ids', 'attention_mask'))


def _open_regular(path, limit):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError('regular file within declared byte limit required')
        return os.fdopen(fd, 'rb')
    except BaseException:
        os.close(fd)
        raise


def file_sha256(path, limit=1 << 30):
    with _open_regular(path, limit) as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def _json(raw):
    def reject(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(raw, object_pairs_hook=_object, parse_constant=reject)


def _sequence(value, name, upper):
    if type(value) not in (tuple, list) or len(value) != 128:
        raise ValueError(name + ' must contain exactly 128 integers')
    if any(type(x) is not int or not 0 <= x < upper for x in value):
        raise ValueError(name + ' contains a non-integer or out-of-range value')
    return tuple(value)


@dataclass(frozen=True)
class ValidatedInput:
    target_idx: int
    input_ids: tuple
    token_type_ids: tuple
    attention_mask: tuple

    def __post_init__(self):
        if type(self.target_idx) is not int or self.target_idx < 0:
            raise ValueError('target_idx must be a nonnegative exact integer')
        for name, upper in (('input_ids', 28996), ('token_type_ids', 2), ('attention_mask', 2)):
            object.__setattr__(self, name, _sequence(getattr(self, name), name, upper))
        if self.input_ids[0] != 101 or self.attention_mask[0] != 1:
            raise ValueError('original CLS token101 at position0 must be valid')


def validate_record(record):
    if type(record) is not dict or set(record) != FIELDS:
        raise ValueError('preprocessed input must have the four exact schema fields')
    if any(type(record[name]) is not list for name in FIELDS - {'target_idx'}):
        raise ValueError('JSON token and mask fields must be lists')
    return ValidatedInput(**record)


def read_preprocessed_jsonl(path, *, byte_limit=16 << 20):
    """Validate every record and all client IDs before any embedding execution."""
    if type(byte_limit) is not int or byte_limit <= 0:
        raise ValueError('positive exact byte limit required')
    with _open_regular(path, byte_limit) as stream:
        raw = stream.read(byte_limit + 1)
    if len(raw) > byte_limit:
        raise ValueError('input exceeds byte limit')
    lines = raw.decode('utf-8', errors='strict').splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError('nonempty JSONL without blank records required')
    records = tuple(validate_record(_json(line)) for line in lines)
    ids = tuple(row.target_idx for row in records)
    if len(ids) != len(set(ids)):
        raise ValueError('target_idx values must be unique within this client batch')
    return records


@dataclass(frozen=True)
class PreparedInput:
    target_idx: int  # Private association: never included in public_metadata().
    tiles: tuple
    original_positions: tuple
    original_attention_mask: tuple

    def public_metadata(self):
        k = len(self.original_positions)
        return dict(schema='bert-client-row-packing-v1', input_tile_count=8,
                    original_positions=list(self.original_positions),
                    original_attention_mask=list(self.original_attention_mask),
                    compact_key_mask=[int(i < k) for i in range(128)],
                    key_count=k, query_tiles=(k + 15) // 16,
                    valid_rows=[max(0, min(16, k - 16*i)) for i in range(8)],
                    packing=dict(slots=65536, row_pitch=4096, rows_per_tile=16,
                                 active_features=768, row_order='increasing original positions',
                                 positions_applied='before compaction', feature_padding='zero',
                                 unused_rows='zero', input_dtype='float64'))


def pack_embeddings(record, original_embeddings):
    """Pack all eight fresh tiles; learned positions have already been applied."""
    if not isinstance(record, ValidatedInput):
        raise TypeError('validated input required')
    record = ValidatedInput(record.target_idx, record.input_ids, record.token_type_ids, record.attention_mask)
    values = np.asarray(original_embeddings)
    if values.shape != (128, 768) or values.dtype != np.float64 or not np.isfinite(values).all():
        raise ValueError('finite original-position float64[128,768] embeddings required')
    positions = tuple(i for i, valid in enumerate(record.attention_mask) if valid)
    rows = np.zeros((128, 4096), dtype='<f8')
    rows[:len(positions), :768] = values[list(positions)]
    # Independent immutable bytes buffers prevent caller aliases or other tiles
    #from changing a prepared input after its public metadata has been bound.
    tiles = tuple(np.frombuffer(rows[16*i:16*(i+1)].tobytes(), dtype='<f8') for i in range(8))
    return PreparedInput(record.target_idx, tiles, positions, record.attention_mask)


class ClientEmbeddings:
    """Native BertEmbeddings only, using the exact pinned F32-to-F64 boundary."""
    def __init__(self, snapshot):
        self.snapshot = Path(snapshot)
        before = {name: file_sha256(self.snapshot / name) for name in ASSETS}
        if before != ASSETS:
            raise ValueError('pinned checkpoint/config bytes differ')
        with _open_regular(self.snapshot / 'config.json', 1 << 20) as stream:
            raw = _json(stream.read())
        expected = dict(hidden_size=768, vocab_size=28996, max_position_embeddings=512,
                        type_vocab_size=2, layer_norm_eps=1e-12, pad_token_id=0)
        if any(type(raw.get(k)) is not type(v) or raw[k] != v for k, v in expected.items()):
            raise ValueError('pinned embedding configuration differs')
        import torch
        from safetensors import safe_open
        from transformers import BertConfig
        from transformers.models.bert.modeling_bert import BertEmbeddings
        config = BertConfig.from_dict(raw)
        # Instantiate only the embedding module; preserve the caller's RNG state.
        with torch.random.fork_rng(devices=[]):
            module = BertEmbeddings(config).to(device='cpu', dtype=torch.float64)
        state = {}; tensor_sha = {}
        with safe_open(self.snapshot / 'model.safetensors', framework='pt', device='cpu') as reader:
            for name, shape in EMBEDDING_SHAPES.items():
                value = reader.get_tensor('bert.embeddings.' + name)
                if value.dtype != torch.float32 or tuple(value.shape) != shape or not torch.isfinite(value).all():
                    raise ValueError('pinned embedding tensor shape/type differs')
                tensor_sha[name] = hashlib.sha256(value.numpy().astype('<f4', copy=False).tobytes()).hexdigest()
                state[name] = value.to(dtype=torch.float64)
        module.load_state_dict(state, strict=True)
        module.requires_grad_(False); module.eval()
        after = {name: file_sha256(self.snapshot / name) for name in ASSETS}
        if after != before:
            raise ValueError('embedding assets changed during loading')
        self._module = module
        self.asset_sha256 = dict(before)
        self.tensor_sha256 = tensor_sha

    def original_embeddings(self, record):
        if not isinstance(record, ValidatedInput):
            raise TypeError('validated input required')
        record = ValidatedInput(record.target_idx, record.input_ids, record.token_type_ids, record.attention_mask)
        import torch
        if self._module.training or any(p.dtype != torch.float64 or p.device.type != 'cpu' for p in self._module.parameters()):
            raise ValueError('client embedding module must remain CPU float64 eval')
        ids = torch.tensor([record.input_ids], dtype=torch.long, device='cpu')
        types = torch.tensor([record.token_type_ids], dtype=torch.long, device='cpu')
        # No compact position_ids are ever passed. Native learned positions0..127
        #are assigned before the public mask is used to select rows.
        with torch.no_grad():
            result = self._module(input_ids=ids, token_type_ids=types)
        values = result[0].detach().cpu().numpy().copy()
        if values.shape != (128, 768) or values.dtype != np.float64 or not np.isfinite(values).all():
            raise ValueError('native embedding output violates the fixed boundary')
        return values

    def prepare(self, record):
        return pack_embeddings(record, self.original_embeddings(record))

    def prepare_batch(self, records):
        records = tuple(records)
        validated = tuple(ValidatedInput(r.target_idx, r.input_ids, r.token_type_ids, r.attention_mask)
                          if isinstance(r, ValidatedInput) else validate_record(r) for r in records)
        if not validated or len({r.target_idx for r in validated}) != len(validated):
            raise ValueError('nonempty batch with unique private target IDs required')
        return tuple(self.prepare(record) for record in validated)
