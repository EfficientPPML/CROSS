"""Contestant TODOs: implement BERT inference using CROSS CKKS primitives.

No DESILO, plaintext evaluator, label lookup or client refresh is permitted.
The benchmark invokes the five file-based functions in separate processes.
See stages.py for each function's path dictionary and doc/problem.md for gates.
"""


def create_session(config, weights, *, batch, sequence_length, devices):
    """Local synthetic fixtures, with public NumPy weights and client embeddings.

    Return a session with encrypt(embeddings), evaluate(ciphertexts,
    attention_mask=mask), decrypt(ciphertexts), close(). encrypt/evaluate must
    return a CROSS Polynomial or nonempty nested lists/tuples of Polynomials.
    decrypt returns (batch, 2) logits. The attention mask is public.
    Reuse your real benchmark circuit; this interface is only a smoke driver.
    """
    raise NotImplementedError("TODO: implement the CROSS BERT session")


def client_key_generation(paths):
    """Persist fresh secret material in paths['client_private']; public and
    evaluation keys in paths['public_keys']. Record actual secure parameters.
    """
    raise NotImplementedError("TODO: generate and serialize CROSS keys")


def server_preprocess_model(paths):
    """Load the frozen checkpoint; encode public weights into 'server_data'.
    Include checkpoint, ring and packing hashes in cache identity. This stage
    must not need a secret key or any client inputs.
    """
    raise NotImplementedError("TODO: pack the BERT checkpoint for CROSS")


def client_encode_encrypt_input(paths):
    """Read 'preprocessed_input' JSONL. Apply the checkpoint's bert.embeddings
    in eval mode, then encrypt all (128,768) activations into 'upload'. Only
    attention masks, opaque sample positions and public metadata may be clear.
    """
    raise NotImplementedError("TODO: embed, encode and encrypt with CROSS")


def server_encrypted_compute(paths):
    """Read 'upload', 'public_keys', 'server_data'; write 'download'. Execute
    all 12 encoder layers, CLS pooler and classifier under CKKS. Synchronize
    before writing outputs/timings. Server paths omit client-private files.
    """
    raise NotImplementedError("TODO: evaluate encrypted BERT using CROSS")


def client_decrypt_decode(paths):
    """Decrypt 'download' with 'client_private'. Write 'decrypted_results'
    JSONL records {target_idx: int, he_logits: [float, float]}, one per input.
    Keep target IDs on the client; use preprocessed_input to restore identity.
    """
    raise NotImplementedError("TODO: decrypt and decode two logits per pair")
