"""Pinned public layer-attention schema; no codec, filesystem transport, or HE.

The schema declares the client's original-position/compaction obligation. It cannot
prove that encrypted slots obey that obligation. Output scales are supplied from
an independently checked public operation plan, never relabeled to canonical.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parent
INPUT_SHA256 = {
    'candidate_manifest.json': '4e2277beea7189aa1d77b168c500c8a175f6c630afa520fcc6b38cb5a179cff9',
    'p60_profile_and_refresh.json': 'fe2408637f9d6de19ac6aef9f6f749aaef9dd56ccbb37b5334293cd223ede035',
    'verified_model_manifest.json': 'b7a39739f0eb11092de7a78e9d0bc3ae18603338aa402e85dd4dd55086d24c79',
    'matrix_inventory.json': 'd087eb087bf521faaf01f62719c9f35e6afce2889f8057f688c320eebdc30292',
}
# All96 raw F32 tensor pins were extracted from the checkpoint SHA
# e1f8085ef673c9eb8a1a93a84270742e8aa0ccbceadef01eef15e26c44b46034.
TENSOR_SHA256 = {0: {'query': {'weight': '1a8914c482024a643b1503123f2c1bac81ce565d0ab7c98f06d8951d8b277f47',
               'bias': '85b53d121a7751dbd243a12ce73c6be48132471667f817fde6020e1baaaed9c4'},
     'key': {'weight': '854e6ccfbdd78eb9108127baaf3374ec8a69120994aae532b1985cab67103981',
             'bias': 'aee8e9d6fb2d9f305221ee344c00e6a026bac065717dfd5191d86bcf8b0cdc4f'},
     'value': {'weight': 'd21b08742d7d298ef2139215a704546011010b5ed40fd315ae430af9b50eb174',
               'bias': 'dab39da7473f97300c2380a0cff1028e91062f1da40efa95fc3c419cfa19cff9'},
     'output': {'weight': 'e00c6a53896c0ed3d2a4fe73edd1313952d23e80bfc88e40402b2b9fd2df3e70',
                'bias': 'b80a500392b9bff59848d76a2d89d77a43cb132747beb292261f7b07bad800f3'}},
 1: {'query': {'weight': 'c9f862a497b34b7f69052d756be7383e68a37d453af9a8d2638bd750cb092ee7',
               'bias': '05a82acd3c3e7f382a8ba269a06805d8e5e7fcf8a3bf71b89926fbf3d068c186'},
     'key': {'weight': '38f71be6a7ed0164b622e1b191657ac5d2ce2beb95f17f0281315e5f2fc85bac',
             'bias': 'f42de7a66bb0be8a92cce047d0972993867dd5320368d2b449bcc89d68a8caf3'},
     'value': {'weight': '2e667d23de4b6ee92f432a2b3e67def4f8a4eb324ea388273bf8ef5fdf9f9873',
               'bias': '13f5c02ab6b0f5196803ae0ea68922a0cc8056918916b5c9e0cc5f683ca2cb4c'},
     'output': {'weight': '2982eb9d01df73c738b011a29003ed53ed54c86a4726a9ab72b617209b6463ed',
                'bias': 'b8fee71f9fbb7d927b53ff114d3f3edb20b82bcdee207fbeb580af658e0664fb'}},
 2: {'query': {'weight': '4b88541805cbd00f1a80c4f54c37f92bb9d3b3c50e03de33b49b1cc34ce6c68b',
               'bias': '40c022afb06a11cde7aebc0fd78cb1236afa23a8f0e9ce4c6d6093ae71fca71f'},
     'key': {'weight': '73f70e2f5d3ee7d808a2f92f65116599f3f447f6717771cdbde2815e599e51f3',
             'bias': 'a8579b07eb47bcc0af039233e2dc5413759ffaa522a28873c32d5ca4bda9e02b'},
     'value': {'weight': 'c0485f3a0d31da4c43d9b08d068619dc71fdbba19a198df05ce95ad70e839bf5',
               'bias': 'c4e3343c0e9edd27c2e5ec9c9b93d2c09b0ee4acaeb8e7c6bd0765db933a8ee4'},
     'output': {'weight': '8485f025b291676573172cb0954e1385ffbaaa97bc17c1803ed94e42f4bd1550',
                'bias': 'd46eb89b63d5cf0574af3b039bbab877f729435b50a5410b7d389119a5b686f8'}},
 3: {'query': {'weight': 'e4c5edf3e1ffe4394ca84e221e3815377e30d6dc08df6b44ab9c9e87e68e85c1',
               'bias': '5b4ea89f51fcd032d824c5f320859c4328033216fb49ca4366f6ed05fc4a201a'},
     'key': {'weight': 'f1ca2672af6e718cb0cdf9ead92de1d0d90996fd132765ddebf036262af4b328',
             'bias': 'b4cbc7156af0c51b1c87851f71871d8441a59a2526f2da3f6a55a107b0c3e942'},
     'value': {'weight': '9650a2fce3491b190fae63817e4fffde0fbfd59c6df63b22721e4e7fc56f1f5e',
               'bias': 'be1bbc2faff886e35132779b66d5c07d1d5e4d0788a99ed1fadfce3b3c9cb561'},
     'output': {'weight': 'b3d8e903e298eec2dd6dd3bd090fe62cc871d144efd0b1b974b6992403a2e9e1',
                'bias': '250f33e14330280f866a509d8be873e41ffdc44a8fdaded9680e5743ec682e4f'}},
 4: {'query': {'weight': 'f1840ec37f5699d94d9fbe4854e86091b12c91cba6170077f503d7c1f2a45d69',
               'bias': '4b5c3cc214c36b953d0c9c92aa371dd1fbd3e7076a9a03e8adaab355585014de'},
     'key': {'weight': '98dc8b9244e8502db77d8e921d899993f46d4cdadf1dffe346541e5d3ea320ba',
             'bias': 'd40d4e221c78e0773799d3c704cbfdf271f61173b117e08bf4b51266e7778fe2'},
     'value': {'weight': '9ce8311b7390516a49657d650db18d52057d0c8f226b7f72c2c333ec3ed667cc',
               'bias': '15afa95cc2e4045ca1df6475e455ed54348f31fafaf28bdcd1813ce63061695b'},
     'output': {'weight': '61d0c4f6170fea5208c345732295294cae46001cf33e357f9e18172dede432df',
                'bias': '85e77fa755fbeb35787efbb1ddbe9c09eb7873a2dee3fc294e03f8c239adc69a'}},
 5: {'query': {'weight': 'b3605e8f7b40b67ecdb8c0fcfef41756282fc48c01273fb56f64f8f089682c89',
               'bias': '59d15b8e5ced60279910cf2ce286eeee13263ee58c898f9c2dfc5d43bde641f6'},
     'key': {'weight': 'defe0eaa581bc5ea993c8d6289e891ace2b27f83c48525a5cf62016a9cb66f10',
             'bias': 'dc5ff95b347ea57d3dae6bcae19c469c633789f28e777b77db4a6e98a00aa61f'},
     'value': {'weight': '0208a0751d7815192e0e3466d0fa7c8e765811604dad531073242279d3fbb3cf',
               'bias': '5654e69b38f0197fad4b106f4db0361afbbcf2ff34126bbb50a7d28fe8ddcb2d'},
     'output': {'weight': '82bbc3524ee47ca3167df906e64009d3c731fd7e777fef71b21a7b37bb94ba52',
                'bias': '4c13d388d808008927022d0eb925f7d7108ce5db63c1dec10d68666cf06b60e8'}},
 6: {'query': {'weight': '028ed3cc30f8f70f0e55ca4556995146fe110ff521e7690ed591f191adcecb62',
               'bias': '425aecc047ea4297a0693ca3b49c1a0ee85fb35fe8610cb75152d9c4758dad32'},
     'key': {'weight': '95fbd56520a76ab5b840836e327a24b4c48905c5d131024547bb6181008385c9',
             'bias': '3853b60a0bdbd37028fddc1a0cbff66e5a82cefd548ddc6121a05c0c064c57e8'},
     'value': {'weight': '3c258dacc1563cc344339a412049fe66ae8d92de2fa426b6aa2c8b81c442258d',
               'bias': 'b28932739a0e0ce87bb2335773af18340bac872deb7c1b0ae2f0651e1b43cce1'},
     'output': {'weight': 'b78d1d07f783859285f13eb4275ea46c02671e167badb57cadc097113fa883bb',
                'bias': '56af0d6cd545e01b8fc07da5de011d5bbeb245b38d27a53db3c683dfe0409303'}},
 7: {'query': {'weight': '6863090655477c7c073f8308c9d77b19c328a9b7da356a8d2a550fa6a2d622b5',
               'bias': '7d182baf26420584da6146153cfab4531ef2b7cba9fcc4a4e7e16052c17a3de3'},
     'key': {'weight': 'f707df3329e59aa3b40250d02d69112974c0c7aa8d2c341645d7c2281a42a9d1',
             'bias': 'a947516d702e2b6dff0ddc05b8def55bcc41d0f7a0e7f146908e9e88ee11fae8'},
     'value': {'weight': 'f54850961bf33cb3b804b6131fa4d3aeae985453cca8361d652992e1e5fb4184',
               'bias': '15882948edc5571e3941cd924808cb077f38b77cb065ec39167044ca230823e6'},
     'output': {'weight': 'd60e98bf59a48917213a3cc736d663eb95f5f8b7b5d7b615448a2d54a5de347c',
                'bias': '09eedba2b4599e861eaf8c70c370d13f5ab31518e0b6ca592eb5e19b478c19b5'}},
 8: {'query': {'weight': '0a6bba60973e576c01f4d61cc92424aaf86a0cfa81839bb76225bb2102f14497',
               'bias': '0d8a144876192a5f062289868876dc04460e84c76bd2178e73c6e24648793a2d'},
     'key': {'weight': '86d07569791fec467dbe5ba1a673623d2874ebcc59402b207b72a121ab7d1c99',
             'bias': 'a2f227587519aa581692d2fe4049c18bd5ef27e7179970aedd80c7e6c87bbaae'},
     'value': {'weight': 'c6e196efe9d996be35049d9af35931fc79975162e4ad32d53b95897bf5bbfd5d',
               'bias': '0f0eb333af3ca9ac6e0ace7af4840eb79f55eec2f2588c955fc319463dc95699'},
     'output': {'weight': 'bca5c017de2bfb5198bd10614a5131443a2e0efcc2368bc42f86e272fd7e9f4e',
                'bias': '0beb1640b98f79c39216e73e2ae0f6cf7f469ab1d295aadfcf67bdc8eda41989'}},
 9: {'query': {'weight': '99b4f78364476cffe12f359b14dbfda08d8d1de8dae9608458fc90747955c077',
               'bias': 'f4464531e5d31f13399086413768893b710cd6a38b43b2fb99f33c8fcdc1acd2'},
     'key': {'weight': '33187d38f26549bf0d75221517336b1b1c44e4369c4299906a17daba050af150',
             'bias': '05743bbf26b91802b4636394c31c3b243f1588453e65eb81775271f07a708551'},
     'value': {'weight': '8a5d8c7c2ec3b1d3bed6546102648e24e6c2067663202b379a6d51d36f769b78',
               'bias': '07552494558a382b9712c4c145b780050dc6184bf083b4e3bedd726b4bb1d3dc'},
     'output': {'weight': '49c5f59b00d61cc28f1864edde39542ae08baf31597bd803b513e52f90dfb712',
                'bias': '19b2847b05b5742a53dc20f57ec8eb2ff70923fcf1a14773821596a2b9b3e6ee'}},
 10: {'query': {'weight': '917515dc0349b552f972e377ac15c578329ba34468265e15ca354ebdd6645fda',
                'bias': 'a7a732ddb395515f5d7cfbf9c345ab7e78815f21e1e8500cad0a780263b70ceb'},
      'key': {'weight': '0ad0e4b440d37c1613bf2289fcf34faf99ca0c18ed3d81c52a7662ddf37154fe',
              'bias': '58f3d13df25c11b4ceb0780148c8b8b6bd05623f79faaf1f97365149b0227d5b'},
      'value': {'weight': '0f197113a7358a9397b1d50419c0dca0b4919fc79b4c359c6ecd728f63e5d9e3',
                'bias': 'd4c2db7ed619f3da6353eff1bcf5aba21e5badd7d8816682d09f0416dfc64cd7'},
      'output': {'weight': 'f7b051c69aed15190f741a18859a02d86bd367442ef3bbebf2645b9eedf7ea6d',
                 'bias': 'bd94d971f68d4acab3761afa34f36ff0200bb159bc4755588b6854d0725b8ce3'}},
 11: {'query': {'weight': '4e1b3ca86271a7e1e1fa8a31bd40e7a6fa2d785c143015e2bbff48ee60fb1511',
                'bias': '676c6ea772d73dfe1f48b72acb9ededb4d138dc753e316c7c4d7fa55a4d9d174'},
      'key': {'weight': '7f79ce6d2c0b25b00c9c2ae030d7cb77257effabe7949d8ee5fc8e548517d245',
              'bias': '8cfb7c00002e803e74f9888cd1d183ff1956fc1c25e0715a929aa330fb3e37c7'},
      'value': {'weight': 'd4923074ea6129b6d4290a85d33a3d912b692aecdcfd00d6b6654d4d8b930640',
                'bias': '610aa95016d249975fdf4c1a1d96347a47befd5352e5b77977b0feeb0294ee72'},
      'output': {'weight': '33205f2ccd48e5af2c3bd4a7acbc50824e1517ab3bf8cb089a8a100be1470fb7',
                 'bias': 'e33bf5de8d1e95175269dc3b4e8f7738c2cfa21f6eb3b26520b1791dcb25244b'}}}
PREFIX = {'query': 'attention.self.query', 'key': 'attention.self.key', 'value': 'attention.self.value', 'output': 'attention.output.dense'}


def canonical(value):
    """Canonical exact JSON types; numeric aliases cannot change typed contracts."""
    def check(x):
        if x is None or type(x) in (str, bool, int, float):
            return
        if type(x) is list:
            for v in x: check(v)
            return
        if type(x) is dict and all(type(k) is str for k in x):
            for v in x.values(): check(v)
            return
        raise ValueError('contract requires exact JSON scalar/container types')
    check(value)
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _pinned(name):
    raw = (ROOT / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != INPUT_SHA256[name]:
        raise ValueError('changed pinned input: ' + name)
    return json.loads(raw)


def parameters():
    """Return a new canonical public context dictionary, without initialization."""
    p = _pinned('p60_profile_and_refresh.json')['profile']
    return dict(degree=p['degree'], num_slots=p['slots'], q_towers=p['q'],
                p_towers=p['p'], composite_degree=2, degree_layout=[256, 512],
                precision=32, max_level=p['depth'], scaling_factor=p['canonical_scale'],
                dnum=p['dnum'], batch=1, sigma=3.190000057220458984375,
                secret_distribution='uniform_ternary')


def _hex(value, length, name):
    if type(value) is not str or re.fullmatch('[0-9a-f]{%d}' % length, value) is None:
        raise ValueError(name + ' must be lowercase hex')
    return value


def _layer(layer):
    if type(layer) is not int or not 0 <= layer < 12:
        raise ValueError('layer must be an exact Python integer in 0..11')
    return layer


def make_contract(original_positions, context_id, request_id, *, layer):
    layer = _layer(layer)
    if type(original_positions) is not tuple or not 1 <= len(original_positions) <= 128:
        raise ValueError('original_positions must be a nonempty tuple of at most 128 indices')
    if any(type(i) is not int or not 0 <= i < 128 for i in original_positions):
        raise ValueError('positions must be exact Python integers in 0..127')
    if any(a >= b for a, b in zip(original_positions, original_positions[1:])):
        raise ValueError('positions must be strictly increasing')
    _hex(context_id, 64, 'context_id'); _hex(request_id, 32, 'request_id')
    model = _pinned('verified_model_manifest.json')
    _pinned('candidate_manifest.json'); _pinned('matrix_inventory.json')
    k = len(original_positions); t = (k + 15) // 16
    return dict(schema='bert-layer-attention-contract-v1', layer=layer, context_id=context_id,
                request_id=request_id, key_count=k, query_tiles=t,
                original_positions=list(original_positions),
                output_valid_rows=[min(16, k - 16*i) for i in range(t)],
                input_tile_count=8 if layer == 0 else t,
                input_level=49 if layer == 0 else None, input_level_range=[49, 49] if layer == 0 else [2, 49],
                output_level=9 if layer < 6 else 8,
                profile=parameters(), source_input_sha256=dict(INPUT_SHA256),
                model=dict(model_id=model['model_id'], revision=model['revision'],
                           checkpoint_sha256=model['files']['model.safetensors']['sha256'],
                           tensors={role: {kind: dict(name=f'bert.encoder.layer.{layer}.' + PREFIX[role] + '.' + kind,
                                                       shape=[768, 768] if kind == 'weight' else [768],
                                                       dtype='<f4', sha256=sha)
                                           for kind, sha in pins.items()}
                                    for role, pins in TENSOR_SHA256[layer].items()}),
                packing=dict(slots=65536, row_pitch=4096, rows_per_tile=16,
                             active_features=768, score_heads=12, score_head_dim=64,
                             learned_position_order='apply-original-positions-before-compaction',
                             compact_rows='selected-original-positions-in-increasing-order',
                             tail_rows='zero-pad-compacted-input-to-128-rows' if layer == 0 else 'refresh-T-tiles-then-subtract-zero-to-eight',
                             feature_padding='zero', output_rows='compacted-order'),
                public_key_mask=[i < k for i in range(128)])


def validate_contract(value):
    if type(value) is not dict or type(value.get('original_positions')) is not list:
        raise ValueError('contract must be a canonical mapping')
    expected = make_contract(tuple(value['original_positions']), value.get('context_id'), value.get('request_id'), layer=value.get('layer'))
    if canonical(value) != canonical(expected):
        raise ValueError('contract differs from pinned schema, profile, model or packing')
    return expected


def validate_projection_arrays(model, layer):
    """Return immutable bytes-backed float64 arrays with exact pinned F32 values."""
    pins_by_role = TENSOR_SHA256[_layer(layer)]
    if type(model) is not dict or set(model) != set(pins_by_role):
        raise ValueError('projection role inventory mismatch')
    result = {}
    for role, pins in pins_by_role.items():
        item = model[role]
        if type(item) is not dict or set(item) != {'weight', 'bias'}:
            raise ValueError('projection tensor inventory mismatch')
        result[role] = {}
        for kind, sha in pins.items():
            a = np.asarray(item[kind])
            shape = (768, 768) if kind == 'weight' else (768,)
            if a.shape != shape or a.dtype.kind != 'f' or a.dtype.itemsize not in (4, 8) or not np.isfinite(a).all():
                raise ValueError('projection must be finite F32 or exact F64 promotion with pinned shape')
            b = a.astype('<f4')
            if not np.array_equal(a.astype('<f8'), b.astype('<f8')) or hashlib.sha256(b.tobytes(order='C')).hexdigest() != sha:
                raise ValueError('projection differs from pinned tensor bytes')
            result[role][kind] = np.frombuffer(b.astype('<f8').tobytes(), dtype='<f8').reshape(shape)
    return result


def _scale(value):
    if type(value) is not float or not math.isfinite(value) or value <= 0:
        raise ValueError('scale must be an actual finite positive binary64 float')
    return value


def _level(value):
    if type(value) is not int or not 0 <= value <= 49:
        raise ValueError('level must be an exact Python integer in 0..49')
    return value


def make_descriptor(contract, *, role, tile_index, payload_sha256, level=None, scale=None, input_digest=None):
    """Metadata only; the caller must authenticate actual serialized payload bytes."""
    c = validate_contract(contract); p = c['profile']
    if type(role) is not str or role not in ('attention_input', 'attention_output'):
        raise ValueError('unknown role')
    count = c['input_tile_count'] if role == 'attention_input' else c['query_tiles']
    if type(tile_index) is not int or not 0 <= tile_index < count:
        raise ValueError('tile index out of range')
    _hex(payload_sha256, 64, 'payload_sha256')
    if role == 'attention_input':
        if input_digest is not None:
            raise ValueError('input descriptors cannot carry a response binding')
        if c['layer'] == 0:
            if level is not None and _level(level) != 49:
                raise ValueError('fresh layer0 input must be level49')
            if scale is not None and _scale(scale) != p['scaling_factor']:
                raise ValueError('fresh layer0 input must have canonical scale')
            level = 49; scale = p['scaling_factor']
        else:
            if _level(level) < 2:
                raise ValueError('later inputs require at least level2 for refresh')
            _scale(scale)
    else:
        if level is not None and _level(level) != c['output_level']:
            raise ValueError('output level differs from selected layer schedule')
        level = c['output_level']; _scale(scale); _hex(input_digest, 64, 'input_digest')
    return dict(schema='bert-layer-attention-descriptor-v1', layer=c['layer'], context_id=c['context_id'],
                request_id=c['request_id'], contract_sha256=digest(c), role=role,
                tile_index=tile_index, tile_count=count,
                valid_rows=max(0, min(16, c['key_count'] - 16*tile_index)),
                level=level, moduli=p['q_towers'][:2*(level+1)], scale=scale, nsd=1,
                degree=p['degree'], degree_layout=p['degree_layout'], precision=p['precision'],
                batch=p['batch'], ciphertext_components=2, payload_sha256=payload_sha256,
                input_digest=input_digest)


def validate_descriptors(values, contract, *, role, expected_levels, expected_scales, input_digest=None):
    """Check every descriptor against a separately supplied actual metadata plan."""
    c = validate_contract(contract)
    count = c['input_tile_count'] if role == 'attention_input' else c['query_tiles'] if role == 'attention_output' else 0
    if not count or type(values) is not list or len(values) != count:
        raise ValueError('missing, duplicate or extra descriptor inventory')
    for plan in (expected_levels, expected_scales):
        if type(plan) not in (list, tuple) or len(plan) != count:
            raise ValueError('both inventories require actual public level and scale plans')
    for level in expected_levels: _level(level)
    for scale in expected_scales: _scale(scale)
    if role == 'attention_input' and input_digest is not None:
        raise ValueError('request inventory has response-only binding')
    result = []
    for i, value in enumerate(values):
        if type(value) is not dict:
            raise ValueError('descriptor must be a mapping')
        expected = make_descriptor(c, role=role, tile_index=i, payload_sha256=value.get('payload_sha256'),
                                   level=expected_levels[i], scale=expected_scales[i], input_digest=input_digest)
        if canonical(value) != canonical(expected):
            raise ValueError('descriptor metadata, order or request binding mismatch')
        result.append(expected)
    return result


def request_digest(values, contract, *, expected_levels, expected_scales):
    """Bind all required inputs, order, layer, context, nonce and actual metadata."""
    return digest(validate_descriptors(values, contract, role='attention_input',
                                     expected_levels=expected_levels, expected_scales=expected_scales))
