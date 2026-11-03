"""Experimental real-CKKS sessions for the public transformer operator graphs.

This development implementation is not a validated or calibrated baseline.
Graphs exceeding one secure leveled chain require an explicit experimental
packed-pair bounded-refresh opt-in. Declared polynomial domains are conditions,
and each public case still requires end-to-end encrypted accuracy validation. No plaintext execution fallback or client refresh exists.
"""
from __future__ import annotations

import gc
import math
import os
import sys
import time

import numpy as np

from encrypted_polynomial import EncryptedPolynomialEvaluator
from encrypted_transformer import (EncryptedTensorOps, PairLayout, bert_ciphertext_forward,
                                   plan_ciphertext_depth, public_toy_approximations, qwen_ciphertext_forward)


def reference_ring_profile(circuit_depth):
    """Exact measured-control profiles; not a general lower-dnum heuristic."""
    if type(circuit_depth) is not int or circuit_depth < 1:
        raise ValueError('circuit depth must be a positive integer')
    if circuit_depth > 52:
        return 49, 4
    # These two actual generated Q/P sets passed security/P coverage checks.
    # Public full-ring control construction also confirmed smaller live NTT
    # tables than dnum=8. Encrypted full-graph validation remains separate.
    return circuit_depth, {38: 2, 50: 5}.get(circuit_depth, 8)


def check_device_payload_capacity(device, *, degree, num_q, input_features, output_features):
    """Reject only a proven impossible resident layout, before client keygen."""
    stats = device.memory_stats() or {}
    limit = stats.get('bytes_limit')
    if isinstance(limit, bool) or not isinstance(limit, (int, np.integer)) or limit <= 0:
        return
    # The current layout holds one two-component uint32 ciphertext per feature.
    # Even a fully depleted CD2 output retains the two base Q limbs.
    for kind, required in (('input', input_features * 2 * degree * num_q * 4),
                           ('output', output_features * 2 * degree * 2 * 4)):
        if required > limit:
            raise ValueError(f'experimental unsharded layout needs at least {required} bytes of '
                             f'{kind} ciphertext payload; device limit is {limit} bytes. '
                             'This model requires a different packed, sharded or streamed layout.')


class ToyReferenceSession:
    def __init__(self, model, config, weights, *, batch, sequence_length, devices, packed=False, packed_pairs=False, bounded_refresh=False, ntt_subsets=False, preserve_dispatch_cache=False):
        from jaxite_word import CKKSContext, he_params, key_gen, util
        from jaxite_word.composite_prime_gen import composite_prime_gen
        import jax

        self.model, self.config, self.weights = model, dict(config), weights
        self.packed = bool(packed)
        self.packed_pairs = bool(packed_pairs)
        if type(bounded_refresh) is not bool:
            raise TypeError('bounded_refresh must be an explicit boolean')
        if bounded_refresh and not packed_pairs:
            raise ValueError('bounded refresh is supported only by the experimental packed-pair layout')
        self.bounded_refresh = bounded_refresh
        if type(ntt_subsets) is not bool:
            raise TypeError('ntt_subsets must be an explicit boolean')
        if type(preserve_dispatch_cache) is not bool:
            raise TypeError('preserve_dispatch_cache must be an explicit boolean')
        self.preserve_dispatch_cache = preserve_dispatch_cache
        self.ntt_subsets = None
        self.closed = False
        if packed and packed_pairs:
            raise ValueError('select one experimental packed layout')
        if packed:
            from encrypted_packed_qwen import validate_packed_config
            if model != 'qwen':
                raise ValueError('experimental packed graph supports Qwen only')
            validate_packed_config(config, batch=batch, sequence_length=sequence_length)
        self.approximations = public_toy_approximations(model)
        if packed_pairs:
            from encrypted_packed_transformer import plan_packed_transformer
            plan = plan_packed_transformer(model, config, weights, batch=batch, sequence=sequence_length,
                                           attention_mask=np.ones((batch, sequence_length), bool))
        else:
            plan = plan_ciphertext_depth(model, config, weights, batch=batch, sequence=sequence_length,
                                         attention_mask=np.ones((batch, sequence_length), bool),
                                         approximations=self.approximations)
        self.plan = plan
        if not devices:
            raise ValueError('an explicit JAX execution device is required')
        if os.environ.get('CROSS_SKIP_TOPLEVEL_ROTKEYS') == '1':
            raise ValueError('this keyless evaluator requires retained pregenerated rotation keys; unset CROSS_SKIP_TOPLEVEL_ROTKEYS')
        degree = 131072
        self.layout = PairLayout(batch, sequence_length, degree // 2)
        if packed:
            from encrypted_packed_qwen import SingleTokenFeatureLayout
            self.layout = SingleTokenFeatureLayout(degree // 2)
        elif packed_pairs:
            from encrypted_packed_transformer import PackedPairLayout
            self.layout = PackedPairLayout(batch, sequence_length, plan['features'], degree // 2)
        needs_refresh = plan['depth'] > 52
        if needs_refresh and not bounded_refresh:
            raise ValueError(f'circuit depth {plan["depth"]} requires bounded homomorphic refresh; '
                             'its range-aware integration is not validated yet')
        chain_depth, dnum = reference_ring_profile(plan['depth'])
        q = composite_prime_gen(2, 2 * (chain_depth + 1), 61, 56, 2 * degree, 31)
        p = util.generate_p_towers(q, dnum, degree, aux_bits=30)
        target = he_params.SecurityTarget(128, he_params.CostModel.CLASSICAL)
        if not he_params.qp_is_secure(q, p, degree, target):
            raise ValueError('generated transformer ring exceeds 128-bit security ceiling')
        he_params.assert_p_coverage(q, p, dnum)
        output_features = weights['classifier.weight' if model == 'bert' else 'model.embed_tokens.weight'].shape[0]
        self.output_features = output_features
        check_device_payload_capacity(devices[0], degree=degree, num_q=len(q),
            input_features=1 if packed or packed_pairs else config['hidden_size'],
            output_features=1 if packed or packed_pairs else output_features)
        rotations = set(plan['rotations'])
        if packed:
            from encrypted_packed_qwen import packed_rotation_indices
            rotations = set(packed_rotation_indices(config, weights, degree // 2))
        if needs_refresh:
            from encrypted_refresh import refresh_plan
            budget = min(4, max(2, (self.layout.period - 1).bit_length()))
            refresh_contract = refresh_plan(degree, self.layout.period, (budget, budget), max_level=chain_depth)
            rotations.update(refresh_contract['rotations'])
        rotations = tuple(sorted(rotations))
        scale = float(q[-2]) * q[-1]
        params = dict(degree=degree, num_slots=degree // 2, scaling_factor=scale, output_scale=scale,
                      q_towers=q, p_towers=p, composite_degree=2, degree_layout=(256, 512), p=56,
                      CKKS_M_FACTOR=1, max_bits_in_word=61, noise_scale_degree=1)
        started = time.perf_counter()
        print(f'[experimental-{model}] preparing secure N={degree}, QP={math.prod(q+p).bit_length()} bits, '
              f'chain depth={chain_depth}, circuit depth={plan["depth"]}, refresh={needs_refresh}', file=sys.stderr, flush=True)
        # Key generation belongs to the client. The separately initialized
        # evaluator receives only relinearization/rotation keys, not the secret.
        keys = key_gen.gen_pke_pair(q, [], degree)
        evaluation = key_gen.gen_evaluation_key(keys['secret_key'], q=q, P=p, dnum=dnum,
                                                noise_std=he_params.DEFAULT_SIGMA, noise_scale=1)
        from encrypted_refresh import ClientRotationKeyStore
        self.key_store = ClientRotationKeyStore.generate(keys['secret_key'], q, p,
            degree=degree, dnum=dnum, rotations=rotations,
            directory=os.environ.get('CROSS_PUBLIC_KEY_DIRECTORY'))
        print(f'[experimental-{model}] generated public/evaluation keys in {time.perf_counter()-started:.1f}s', file=sys.stderr, flush=True)
        self.client = CKKSContext(dict(params, **keys))
        with jax.default_device(devices[0]):
            evaluator_params = dict(params, evaluation_key=[
                jax.numpy.asarray(evaluation[name], dtype='uint32').transpose(0, 2, 1) for name in ('a', 'b')])
            self.ctx = CKKSContext(evaluator_params)
            self.ctx.program_initialization(list(rotations), dnum=dnum, r=256, c=512,
                                            batch=1, pregenerated_rotation_keys=self.key_store.keys)
            if ntt_subsets:
                from encrypted_ntt import enable_scoped_ntt_subsets
                self.ntt_subsets = enable_scoped_ntt_subsets(self.ctx)
        # Ciphertext wrappers need public NTT tables, not a second copy of the
        # full large-ring table in the client's codec. Sharing this immutable
        # public object does not expose the client's secret to the evaluator.
        self.client._polynomial_ntt_ctx = self.ctx._param_cache.ntt_q_max
        def report(phase):
            stats = devices[0].memory_stats() or {}
            resident = stats.get('bytes_in_use')
            memory = f', device resident {resident / 2**30:.2f} GiB' if resident is not None else ''
            print(f'[experimental-{model}] {phase}; elapsed {time.perf_counter()-started:.1f}s{memory}',
                  file=sys.stderr, flush=True)
        self.refresher = None
        if needs_refresh:
            from encrypted_refresh import BoundedSecureRefresh
            self.refresher = BoundedSecureRefresh(self.ctx, period=self.layout.period,
                level_budget=refresh_contract['level_budget'], low_memory=True,
                compiled=True, meta_precision=10,
                preserve_dispatch_cache=preserve_dispatch_cache)
        if packed_pairs:
            if needs_refresh:
                from encrypted_bounded_transformer import BoundedPackedTransformer
                self.ops = BoundedPackedTransformer(self.ctx, self.layout, self.refresher, low_memory=True,
                    preserve_dispatch_cache=preserve_dispatch_cache)
            else:
                from encrypted_packed_transformer import PackedTransformer
                self.ops = PackedTransformer(self.ctx, self.layout, low_memory=True,
                    preserve_dispatch_cache=preserve_dispatch_cache)
            self.evaluator = self.ops.arithmetic
            self.evaluator.progress = report
        elif packed:
            from encrypted_packed_qwen import PackedQwen1
            self.ops = PackedQwen1(self.ctx, low_memory=True,
                preserve_dispatch_cache=preserve_dispatch_cache)
            self.evaluator = self.ops.arithmetic
            self.evaluator.progress = report
        else:
            self.evaluator = EncryptedPolynomialEvaluator(self.ctx, strategy='power', low_memory=True,
                progress=report, refresher=self.refresher, compiled_multiply=True, compiled_linear=True,
                preserve_dispatch_cache=preserve_dispatch_cache)
            self.ops = EncryptedTensorOps(self.evaluator, self.layout)
        print(f'[experimental-{model}] setup completed in {time.perf_counter()-started:.1f}s', file=sys.stderr, flush=True)

    def security_contexts(self):
        return [self.client, self.ctx]

    def _check_open(self):
        if self.closed:
            raise RuntimeError('encrypted reference session is closed')

    def encrypt(self, embeddings):
        self._check_open()
        return [self.client.encrypt_slots(value) for value in self.layout.pack_client_embeddings(embeddings)]

    def evaluate(self, encrypted, *, attention_mask=None):
        self._check_open()
        # Only public weights/config/mask and ciphertexts enter the evaluator.
        if self.packed or self.packed_pairs:
            if len(encrypted) != 1:
                raise ValueError('packed evaluator requires one ciphertext')
            if self.packed_pairs:
                return [self.ops.evaluate(encrypted[0], self.model, self.config, self.weights,
                                          self.approximations, attention_mask)]
            return [self.ops.evaluate(encrypted[0], self.config, self.weights, self.approximations)]
        if self.model == 'bert':
            return bert_ciphertext_forward(self.ops, encrypted, self.config, self.weights,
                                           attention_mask, self.approximations)
        return qwen_ciphertext_forward(self.ops, encrypted, self.config, self.weights, self.approximations)

    def decrypt(self, ciphertexts):
        self._check_open()
        decoded = [self.client.decrypt_slots(value) for value in ciphertexts]
        features = (self.layout.unpack_client_features(decoded, self.output_features) if self.packed_pairs
                    else self.layout.unpack_client_features(decoded))
        return features[:, 0 if self.model == 'bert' else -1]

    def close(self):
        if getattr(self, 'closed', False):
            return
        from jaxite_word import ckks_ctx
        import jax
        contexts = {id(context): context for context in
                    (getattr(self, 'client', None), getattr(self, 'ctx', None))
                    if context is not None}
        key_store = getattr(self, 'key_store', None)
        ntt_subsets = getattr(self, 'ntt_subsets', None)
        self.ntt_subsets = None
        self.client = self.ctx = self.evaluator = self.ops = self.refresher = None
        self.key_store = None
        self.closed = True
        errors = []

        def attempt(label, operation, *args):
            try:
                operation(*args)
            except BaseException as error:
                error.add_note(f'While closing experimental transformer {label}')
                errors.append(error)

        if ntt_subsets is not None:
            attempt('public NTT subsets', ntt_subsets.close)
        for context in contexts.values():
            attempt('context', ckks_ctx.evict_context_caches, context)
        if contexts:
            del context
        contexts.clear()
        # Context-owned primitive executables can retain large public keys.
        # This process-wide cleanup is outside the measured request interval.
        attempt('JAX caches', jax.clear_caches)
        attempt('unreferenced contexts', gc.collect)
        if key_store is not None:
            attempt('public key store', key_store.close)
        if errors:
            for later in errors[1:]:
                errors[0].add_note(f'Additional cleanup failure: {later}')
            raise errors[0]


def create_toy_session(model, config, weights, *, batch, sequence_length, devices, packed=False, packed_pairs=False, bounded_refresh=False, ntt_subsets=False, preserve_dispatch_cache=False):
    # Preserve a partially initialized owner so setup failures can release any
    # already-created device contexts as well as normal close() paths.
    session = object.__new__(ToyReferenceSession)
    try:
        session.__init__(model, config, weights, batch=batch, sequence_length=sequence_length, devices=devices,
                         packed=packed, packed_pairs=packed_pairs, bounded_refresh=bounded_refresh, ntt_subsets=ntt_subsets,
                         preserve_dispatch_cache=preserve_dispatch_cache)
        return session
    except BaseException as error:
        try:
            session.close()
        except BaseException as cleanup_error:
            error.add_note(f"Experimental session cleanup also failed: {cleanup_error}")
        raise
