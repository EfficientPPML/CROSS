"""Contestant implementation: Qwen inference using real CROSS ciphertexts."""

# Remove the unimplemented marker when replacing the starter with real HE code.
IMPLEMENTATION_STATUS = 'unimplemented'



def create_session(config, weights, *, batch, sequence_length, devices):
    """Return an object with encrypt/evaluate/decrypt/close methods.

    config uses Hugging Face Qwen3Config keys; weights use checkpoint names.
    encrypt accepts (batch, sequence_length, hidden_size) embeddings.
    evaluate returns encrypted last-position logits for the full vocabulary.
    decrypt returns (batch, vocab_size) float64 logits.

    For actual-ring checks, expose live CKKSContext objects as ctx/_ctx or
    through security_contexts(). Custom wrappers must return every client and
    evaluator context; a declared ring/profile alone is not sufficient.

    TODO: implement operators, packing, depth/refresh planning and the decoder.
    Keep the server evaluator free of secret keys and plaintext activations.
    reference.py explains semantics, but is not an encrypted implementation.
    """
    raise NotImplementedError("Implement CROSS-backed create_session in qwen_he.py")


def create_operator_session(operation, config, weights, *, batch, sequence_length, devices):
    """A2 entrypoint; same encrypt/evaluate/decrypt/close lifecycle.

    operation: rmsnorm, qk_rope, gqa, gated_mlp or linear.
    See the handout for operand shapes and weight dictionary conventions.
    """
    raise NotImplementedError("Implement the encrypted operator milestone")


def create_block_session(config, weights, *, layer_index, batch, sequence_length, devices):
    """A3 entrypoint: encrypted (B,S,H) input -> encrypted (B,S,H) block output.

    Include both residuals; do not include final model norm or vocabulary head.
    weights use the full checkpoint's named tensors for the specified layer.
    """
    raise NotImplementedError("Implement the encrypted decoder-block milestone")
