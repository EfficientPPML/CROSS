"""Contestant entrypoint. Preserve model semantics and the validated ring."""


def build_service(model, input_shape, *, batch, devices, keys=None):
    """Return a prepared encrypted-inference service for `batch` images.

    The returned object must provide:

        encrypt(images)  -> an opaque ciphertext bundle   (client side)
        execute(bundle)  -> an opaque ciphertext bundle   (server side)
        decrypt(bundle)  -> array of shape (batch, 10)    (client side)
        release()        -> free device memory

    For actual-ring checks, expose live CKKSContext objects as ctx/_ctx or
    through security_contexts(). Custom wrappers must return every client and
    evaluator context; a declared ring/profile alone is not sufficient.

    Optional: warmup_input() returns a dummy encrypted-zero bundle covering
    every physical execution shape the service uses. Stage 3 evaluates it
    before ranked requests and validates input/output security. Omit this
    hook to retain full-batch warmup. A smaller bundle is sufficient only if
    execute() uses the same fixed Mapping shapes for every subsequent group.

    `keys` is a CKKS key pair supplied by the suite bridge so that the client
    process and the server process share one key pair. Pass it straight to
    `Mapping(..., keys=keys)`. When it is None, generate your own. Ignoring it
    makes the suite-backed evaluation fail to decrypt.

    `execute` runs on encrypted data only: it may not decrypt, call the
    cleartext model, inspect labels, cache results by image identity or return
    precomputed answers.

    TODO: begin with baseline.py, then optimize the encrypted serving path.
    """
    raise NotImplementedError("Implement build_service in bench_opt.py")
