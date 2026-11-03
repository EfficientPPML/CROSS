"""Float64 Qwen3 semantic oracle, deliberately plaintext and not an HE solver.

No padding, no cache, positions 0..S-1. For checkpoint validation also compare
to the pinned upstream model, whose normalization/softmax may use float32.
"""
import numpy as np


def rmsnorm(x, weight, eps):
    return x * (np.mean(x * x, axis=-1, keepdims=True) + eps) ** -0.5 * weight


def rope(x, theta):
    # x: B, heads, S, D. Qwen rotates the two halves, not adjacent pairs.
    d = x.shape[-1]
    inv_freq = theta ** (-np.arange(0, d, 2, dtype=np.float64) / d)
    angles = np.arange(x.shape[-2])[:, None] * inv_freq[None, :]
    angles = np.concatenate((angles, angles), axis=-1)[None, None, :, :]
    rotated = np.concatenate((-x[..., d // 2:], x[..., :d // 2]), axis=-1)
    return x * np.cos(angles) + rotated * np.sin(angles)


def forward(config, weights, embeddings, *, start_layer=0, output_hidden=False):
    c, w = config, weights
    x = np.asarray(embeddings, dtype=np.float64)
    b, s, h = x.shape
    nq, nkv, d = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    if h != c["hidden_size"] or nq % nkv or d % 2:
        raise ValueError("Invalid Qwen dimensions")
    eps = c["rms_norm_eps"]
    for layer in range(start_layer, start_layer + c["num_hidden_layers"]):
        prefix = f"model.layers.{layer}."
        a = prefix + "self_attn."
        z = rmsnorm(x, w[prefix + "input_layernorm.weight"], eps)
        q = (z @ w[a + "q_proj.weight"].T).reshape(b, s, nq, d).transpose(0, 2, 1, 3)
        k = (z @ w[a + "k_proj.weight"].T).reshape(b, s, nkv, d).transpose(0, 2, 1, 3)
        v = (z @ w[a + "v_proj.weight"].T).reshape(b, s, nkv, d).transpose(0, 2, 1, 3)
        q = rope(rmsnorm(q, w[a + "q_norm.weight"], eps), c["rope_theta"])
        k = rope(rmsnorm(k, w[a + "k_norm.weight"], eps), c["rope_theta"])
        k, v = np.repeat(k, nq // nkv, axis=1), np.repeat(v, nq // nkv, axis=1)
        scores = q @ k.swapaxes(-1, -2) / np.sqrt(d)
        mask = np.tril(np.ones((s, s), dtype=bool))
        scores = np.where(mask, scores, -np.inf)
        exps = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        probs = exps / exps.sum(axis=-1, keepdims=True)
        attended = (probs @ v).transpose(0, 2, 1, 3).reshape(b, s, nq * d)
        x = x + attended @ w[a + "o_proj.weight"].T
        z = rmsnorm(x, w[prefix + "post_attention_layernorm.weight"], eps)
        m = prefix + "mlp."
        gate = z @ w[m + "gate_proj.weight"].T
        # Numerically stable sigmoid without overflow on negative inputs.
        sigmoid = np.exp(-np.logaddexp(0, -gate))
        mlp = (gate * sigmoid) * (z @ w[m + "up_proj.weight"].T)
        x = x + mlp @ w[m + "down_proj.weight"].T
    if output_hidden:
        return x
    last = rmsnorm(x[:, -1, :], w["model.norm.weight"], eps)
    return last @ w["model.embed_tokens.weight"].T


def decoder_block(config, weights, embeddings, layer_index):
    return forward(dict(config, num_hidden_layers=1), weights, embeddings,
                   start_layer=layer_index, output_hidden=True)


def operator_forward(operation, config, weights, operands):
    """Plaintext reference for the five A2 interfaces in the handout."""
    eps = config["rms_norm_eps"]
    if operation == "rmsnorm":
        return rmsnorm(operands, weights["weight"], eps)
    if operation == "qk_rope":
        q, k = operands
        return (rope(rmsnorm(q, weights["q_norm.weight"], eps), config["rope_theta"]),
                rope(rmsnorm(k, weights["k_norm.weight"], eps), config["rope_theta"]))
    if operation == "gqa":
        q, k, v = operands
        b, nq, s, d = q.shape
        repeats = nq // k.shape[1]
        k, v = np.repeat(k, repeats, axis=1), np.repeat(v, repeats, axis=1)
        scores = q @ k.swapaxes(-1, -2) / np.sqrt(d)
        scores = np.where(np.tril(np.ones((s, s), dtype=bool)), scores, -np.inf)
        exps = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs = exps / exps.sum(axis=-1, keepdims=True)
        return (probs @ v).transpose(0, 2, 1, 3).reshape(b, s, nq * d)
    if operation == "gated_mlp":
        gate = operands @ weights["gate_proj.weight"].T
        activated = gate * np.exp(-np.logaddexp(0, -gate))
        return (activated * (operands @ weights["up_proj.weight"].T)) @ weights["down_proj.weight"].T
    if operation == "linear":
        return operands @ weights["weight"].T
    raise ValueError(f"Unknown operator: {operation}")


def toy_weights(c, seed):
    rng = np.random.default_rng(seed)
    h, m, v = c["hidden_size"], c["intermediate_size"], c["vocab_size"]
    q, kv, d = c["num_attention_heads"] * c["head_dim"], c["num_key_value_heads"] * c["head_dim"], c["head_dim"]
    w = {"model.embed_tokens.weight": rng.normal(0, 0.15, (v, h)),
         "model.norm.weight": rng.uniform(0.9, 1.1, h)}
    for i in range(c["num_hidden_layers"]):
        p = f"model.layers.{i}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            w[p + name + ".weight"] = rng.uniform(0.9, 1.1, h)
        for name, shape in (("q_proj", (q, h)), ("k_proj", (kv, h)),
                            ("v_proj", (kv, h)), ("o_proj", (h, q))):
            w[p + "self_attn." + name + ".weight"] = rng.normal(0, 0.08, shape)
        for name in ("q_norm", "k_norm"):
            w[p + "self_attn." + name + ".weight"] = rng.uniform(0.9, 1.1, d)
        for name, shape in (("gate_proj", (m, h)), ("up_proj", (m, h)), ("down_proj", (h, m))):
            w[p + "mlp." + name + ".weight"] = rng.normal(0, 0.08, shape)
    return w


def make_case(case):
    weights = toy_weights(case["config"], case["weight_seed"])
    ids = np.asarray(case["token_ids"], dtype=np.int64)
    embeddings = weights["model.embed_tokens.weight"][ids]
    return weights, embeddings, forward(case["config"], weights, embeddings)
